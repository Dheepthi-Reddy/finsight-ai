"""
OpenAI GPT-4 client for the FinSight RAG pipeline.

Why use both Claude and GPT-4 in the same project
---------------------------------------------------
Claude (via AWS Bedrock) and GPT-4 (via OpenAI) are not interchangeable — they
have different strengths, pricing, and availability characteristics:

  CLAUDE (Bedrock)
    • Runs inside AWS — no data leaves your VPC if using a VPC endpoint.
    • Better for regulated industries: AWS BAA, SOC 2, HIPAA eligibility.
    • Slightly stronger on long-document reasoning and instruction-following.
    • Pricing is per-token; no subscription required.

  GPT-4 (OpenAI)
    • Broader ecosystem: function calling, Assistants API, fine-tuning.
    • Often faster on short-form generation tasks.
    • gpt-4o is multimodal (text + images) if document screenshots are needed.
    • Requires data leaving AWS — may conflict with strict data-residency rules.

Having both wired in lets you:
  1. A/B test answer quality on the same questions (use ModelBenchmark).
  2. Fall back to GPT-4 if Bedrock has a regional outage.
  3. Route different query types: use Claude for long 10-K analysis, GPT-4 for
     short factual lookups where its speed advantage matters.
  4. Satisfy stakeholders who specifically request GPT-4 outputs for comparison.

How the shared interface works
--------------------------------
Both OpenAIClient and BedrockClient expose identical public methods:
  invoke(prompt, system_prompt, max_tokens, temperature) → InvokeResponse
  invoke_stream(prompt, ...)                             → Iterator[str | InvokeResponse]
  invoke_batch(prompts, ...)                             → list[InvokeResponse]

FinSightChain, AgenticRAGChain, LLMJudge, and ModelBenchmark all type-hint
their client parameter as BedrockClient but only call .invoke() — so passing
an OpenAIClient works without any changes to those classes. This is the
duck-typing equivalent of a protocol/interface.

Authentication
--------------
OpenAI reads the API key from the OPENAI_API_KEY environment variable
automatically. Set it in your .env file or CI secrets:

    OPENAI_API_KEY=sk-proj-...

For production, store the key in AWS Secrets Manager and inject it as an
environment variable at container startup — never commit it to source control.

Usage
-----
    from rag_pipeline.generation.openai_client import OpenAIClient
    from rag_pipeline.chain import FinSightChain

    # Drop-in replacement for BedrockClient
    chain = FinSightChain(bedrock=OpenAIClient())
    result = chain.query("What are JPMorgan's AML obligations?")

    # Side-by-side comparison
    from rag_pipeline.generation.bedrock_client import BedrockClient
    from rag_pipeline.optimization.model_benchmark import ModelBenchmark

    bench = ModelBenchmark(models=[
        ("gpt-4o",                                   "GPT-4o",         5.00, 15.00),
        ("us.anthropic.claude-haiku-4-5-20251001-v1:0", "Claude Haiku", 0.80,  4.00),
    ])
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Generator

# ── OpenAI SDK ─────────────────────────────────────────────────────────────────
# The openai package provides a typed Python client for the OpenAI API.
# It reads OPENAI_API_KEY from the environment automatically.
try:
    from openai import OpenAI
    from openai import APIConnectionError, APIStatusError, AuthenticationError
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

# Re-use the InvokeResponse dataclass from bedrock_client so the rest of the
# pipeline handles responses from both providers identically — same fields,
# same __repr__, same is_stub flag.
from rag_pipeline.generation.bedrock_client import InvokeResponse

# ── Configuration ─────────────────────────────────────────────────────────────
# gpt-4o is the recommended default: it is OpenAI's most capable and
# cost-efficient GPT-4 class model as of 2025, replacing gpt-4-turbo.
# Other options: "gpt-4o-mini" (faster, cheaper), "gpt-4-turbo" (legacy).
DEFAULT_MODEL_ID = "gpt-4o"

DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.0   # deterministic for factual compliance answers


# ─────────────────────────────────────────────────────────────────────────────
# OPENAI CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class OpenAIClient:
    """
    Drop-in replacement for BedrockClient that routes requests to OpenAI GPT-4.

    Exposes the same three public methods as BedrockClient:
      invoke()        — single synchronous request
      invoke_stream() — token-by-token streaming generator
      invoke_batch()  — concurrent multi-prompt dispatch

    All three return InvokeResponse objects (imported from bedrock_client),
    so FinSightChain, AgenticRAGChain, LLMJudge, and ModelBenchmark work with
    either client without modification.

    Stub mode mirrors BedrockClient's behaviour: when OPENAI_API_KEY is absent
    or OPENAI_STUB=true is set, all calls return canned responses locally with
    no network traffic. This keeps CI and local development working without
    spending API credits.
    """

    def __init__(
        self,
        model_id:  str = DEFAULT_MODEL_ID,
        stub_mode: bool | None = None,
    ) -> None:
        """
        Parameters
        ----------
        model_id  : str
            OpenAI model name. Examples: "gpt-4o", "gpt-4o-mini", "gpt-4-turbo".
        stub_mode : bool | None
            True  → always return stub responses (no API calls).
            False → always attempt live calls (raises if key missing).
            None  → auto-detect: stub if OPENAI_API_KEY is absent or
                    OPENAI_STUB=true is set.
        """
        if not _OPENAI_AVAILABLE:
            raise ImportError(
                "openai package is not installed.\n"
                "Install with: pip install openai"
            )

        self.model_id = model_id
        self._client: OpenAI | None = None

        # Explicit stub override.
        if stub_mode is True or os.environ.get("OPENAI_STUB", "").lower() == "true":
            self._stub_mode = True
            print("OpenAIClient: stub mode ON (forced via OPENAI_STUB env or constructor)")
            return

        if stub_mode is False:
            self._stub_mode = False
            self._client = OpenAI()
            return

        # Auto-detect: stub when no API key is configured.
        self._stub_mode, self._client = self._auto_init()

        status = "stub" if self._stub_mode else "live"
        print(f"OpenAIClient ready  model={model_id}  mode={status}")

    # ── Public: invoke ────────────────────────────────────────────────────────

    def invoke(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> InvokeResponse:
        """
        Send a single prompt to GPT-4 and return a structured response.

        The system_prompt is passed as the OpenAI "system" role message, which
        sets GPT-4's persona and constraints for the conversation — equivalent
        to the system field in the Bedrock/Anthropic Messages API.

        Parameters
        ----------
        prompt        : str   User question or instruction.
        system_prompt : str   Persistent system-level instruction (persona, rules).
        max_tokens    : int   Hard ceiling on generated tokens.
        temperature   : float 0.0 = deterministic; higher = more varied output.

        Returns
        -------
        InvokeResponse with text, token counts, model_id, and is_stub flag.
        """
        if self._stub_mode:
            return self._stub_response(prompt, system_prompt, max_tokens, temperature)
        return self._live_invoke(prompt, system_prompt, max_tokens, temperature)

    # ── Public: invoke_stream ─────────────────────────────────────────────────

    def invoke_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> Generator[str | InvokeResponse, None, None]:
        """
        Stream GPT-4's response token-by-token.

        Yields str chunks as tokens arrive, then yields a final InvokeResponse
        with the complete accumulated text and token counts. This mirrors
        BedrockClient.invoke_stream() exactly so streaming callers work with
        either provider.

        In stub mode, yields the full stub text as a single chunk then the
        summary InvokeResponse — same behaviour as BedrockClient stub streaming.

        Yields
        ------
        str           : Incremental text delta from the model.
        InvokeResponse: Final summary (last item yielded).
        """
        if self._stub_mode:
            stub = self._stub_response(prompt, system_prompt, max_tokens, temperature)
            yield stub.text
            yield stub
            return

        messages = _build_messages(prompt, system_prompt)

        # stream=True enables server-sent events; OpenAI yields ChatCompletionChunk
        # objects, each containing a delta.content string (possibly None/empty).
        stream = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
        )

        full_text: list[str] = []
        input_tokens = 0
        output_tokens = 0
        finish_reason = "stop"

        for chunk in stream:
            # Token deltas arrive in choices[0].delta.content.
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                full_text.append(delta)
                yield delta

            # finish_reason is set on the last content chunk.
            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

            # Usage is included in the final chunk when stream_options is set.
            if chunk.usage:
                input_tokens = chunk.usage.prompt_tokens
                output_tokens = chunk.usage.completion_tokens

        yield InvokeResponse(
            text=("".join(full_text)),
            model_id=self.model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=finish_reason,
        )

    # ── Public: invoke_batch ─────────────────────────────────────────────────

    def invoke_batch(
        self,
        prompts: list[str],
        system_prompt: str = "",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        max_workers: int = 5,
    ) -> list[InvokeResponse]:
        """
        Send multiple prompts to GPT-4 concurrently and return all responses.

        Mirrors BedrockClient.invoke_batch() exactly — same signature, same
        error-isolation guarantee (a failed individual request returns a stub
        InvokeResponse, not an exception).

        OpenAI's rate limits are per-minute token and request counts. max_workers=5
        is conservative; increase it if your tier supports higher concurrency.
        Check your rate limit tier at platform.openai.com/account/limits.

        Returns
        -------
        list[InvokeResponse]  Same length as prompts, in the same order.
        """
        if not prompts:
            return []

        if self._stub_mode:
            return [
                self._stub_response(p, system_prompt, max_tokens, temperature)
                for p in prompts
            ]

        results: list[InvokeResponse | None] = [None] * len(prompts)

        def _invoke_indexed(index: int, prompt: str) -> tuple[int, InvokeResponse]:
            try:
                response = self._live_invoke(prompt, system_prompt, max_tokens, temperature)
                return index, response
            except Exception as exc:
                return index, InvokeResponse(
                    text=f"[BATCH ERROR at index {index}] {type(exc).__name__}: {exc}",
                    model_id=self.model_id,
                    input_tokens=0,
                    output_tokens=0,
                    is_stub=True,
                    stop_reason="error",
                )

        with ThreadPoolExecutor(max_workers=min(max_workers, len(prompts))) as executor:
            futures = {
                executor.submit(_invoke_indexed, i, p): i
                for i, p in enumerate(prompts)
            }
            for future in as_completed(futures):
                idx, response = future.result()
                results[idx] = response

        return results  # type: ignore[return-value]

    # ── Private: live invocation ──────────────────────────────────────────────

    def _live_invoke(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> InvokeResponse:
        """Call the OpenAI chat completions endpoint and return an InvokeResponse."""
        messages = _build_messages(prompt, system_prompt)

        completion = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        text = completion.choices[0].message.content or ""
        finish_reason = completion.choices[0].finish_reason or "stop"
        usage = completion.usage

        return InvokeResponse(
            text=text,
            model_id=self.model_id,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            stop_reason=finish_reason,
        )

    # ── Private: auto-init ────────────────────────────────────────────────────

    def _auto_init(self) -> tuple[bool, OpenAI | None]:
        """
        Detect whether a valid API key is available and build the client.

        Returns (stub_mode, client). Mirrors BedrockClient._auto_init() in
        structure so the two clients have consistent startup behaviour.
        """
        api_key = os.environ.get("OPENAI_API_KEY", "")

        if not api_key:
            print(
                "OpenAIClient: OPENAI_API_KEY not set — falling back to stub mode.\n"
                "Set OPENAI_API_KEY in your .env file or environment to use live GPT-4."
            )
            return True, None

        try:
            client = OpenAI(api_key=api_key)
            # OpenAI doesn't validate the key at client creation time.
            # Make a minimal models.list() call to confirm the key is valid.
            # This is cheap (~50ms) and catches invalid/revoked keys early.
            client.models.list()
            return False, client

        except AuthenticationError:
            print(
                "OpenAIClient: OPENAI_API_KEY is invalid or revoked "
                "— falling back to stub mode."
            )
            return True, None

        except APIConnectionError as exc:
            print(
                f"OpenAIClient: could not reach OpenAI API ({exc}) "
                "— falling back to stub mode."
            )
            return True, None

        except Exception as exc:
            print(
                f"OpenAIClient: unexpected error during init "
                f"({type(exc).__name__}: {exc}) — falling back to stub mode."
            )
            return True, None

    # ── Private: stub response ────────────────────────────────────────────────

    def _stub_response(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> InvokeResponse:
        """
        Return a canned InvokeResponse without calling OpenAI.

        Token counts are estimated at 4 chars ≈ 1 token to keep downstream
        budget logic working in tests — the same heuristic BedrockClient uses.
        """
        stub_text = (
            f"[STUB] This response was generated locally without calling OpenAI. "
            f"Model: {self.model_id}\n\n"
            f"Your question: \"{prompt[:120]}{'...' if len(prompt) > 120 else ''}\"\n\n"
            "In a live environment, GPT-4 would retrieve the relevant SEC filing "
            "excerpts and synthesise a detailed compliance answer here. To enable "
            "live inference, set the OPENAI_API_KEY environment variable."
        )

        estimated_input = (len(prompt) + len(system_prompt)) // 4
        estimated_output = len(stub_text) // 4

        return InvokeResponse(
            text=stub_text,
            model_id=self.model_id,
            input_tokens=estimated_input,
            output_tokens=estimated_output,
            is_stub=True,
            stop_reason="stop",
        )


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_messages(
    prompt: str,
    system_prompt: str,
) -> list[dict[str, str]]:
    """
    Build the OpenAI messages list from a prompt and optional system prompt.

    The OpenAI Chat Completions API uses a list of role/content dicts.
    The system role sets the model's persona and constraints; the user role
    carries the actual question. This maps directly to Bedrock's system +
    messages[{role: user}] structure.
    """
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


# ─────────────────────────────────────────────────────────────────────────────
# DEMO / CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from rag_pipeline.generation.prompt_templates import COMPLIANCE_SYSTEM_PROMPT
    from rag_pipeline.generation.bedrock_client import BedrockClient

    QUESTION = (
        "What are JPMorgan's AML obligations under the Bank Secrecy Act "
        "based on their 10-K disclosures?"
    )

    # ── Single provider demo ──────────────────────────────────────────────────
    print("── OpenAI GPT-4")
    gpt4 = OpenAIClient()
    r = gpt4.invoke(prompt=QUESTION, system_prompt=COMPLIANCE_SYSTEM_PROMPT)
    print(f"   {r}")
    print(f"   stub={r.is_stub}")

    # ── Side-by-side with Claude ──────────────────────────────────────────────
    print("\n── Claude (Bedrock)")
    claude = BedrockClient()
    r2 = claude.invoke(prompt=QUESTION, system_prompt=COMPLIANCE_SYSTEM_PROMPT)
    print(f"   {r2}")

    # ── FinSightChain with GPT-4 ──────────────────────────────────────────────
    print("\n── FinSightChain wired to GPT-4")
    from rag_pipeline.chain import FinSightChain
    chain = FinSightChain(bedrock=OpenAIClient())
    print(f"   Chain status: {chain.status()}")

    # ── Batch invoke ──────────────────────────────────────────────────────────
    print("\n── Batch invoke (3 prompts concurrently)")
    questions = [
        "What AML controls does JPMorgan disclose?",
        "What is JPMorgan's CET1 ratio?",
        "What fraud detection systems does Bank of America use?",
    ]
    responses = gpt4.invoke_batch(questions, system_prompt=COMPLIANCE_SYSTEM_PROMPT)
    for q, resp in zip(questions, responses):
        print(f"   [{resp.model_id}  {resp.input_tokens}+{resp.output_tokens} tok"
              f"  stub={resp.is_stub}]  {q[:50]}")
