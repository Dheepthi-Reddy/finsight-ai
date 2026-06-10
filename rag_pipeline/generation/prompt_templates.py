"""
Prompt templates for the FinSight RAG compliance assistant and fraud explainer.

What makes a good system prompt
--------------------------------
A system prompt sets the model's *persistent identity and constraints* for the
entire conversation. The best system prompts share four properties:

1. ROLE CLARITY — State who the model is, not just what it should do.
   "You are a financial compliance analyst" beats "Be helpful with compliance."
   A well-defined role activates domain-specific reasoning patterns: citing
   regulatory frameworks, hedging uncertainty appropriately, and structuring
   answers the way an expert in that field actually would.

2. EXPLICIT CONSTRAINTS — Name the things the model must NOT do.
   LLMs default to being helpful, which can mean filling knowledge gaps with
   plausible-sounding fabrications. For compliance use cases this is dangerous:
   a confidently wrong answer about BSA obligations or capital ratios could
   mislead a compliance officer. Stating "only answer from the provided context"
   and "say so explicitly when unsure" forces the model to acknowledge limits
   rather than hallucinate past them.

3. OUTPUT STRUCTURE — Describe the format you expect, not just the content.
   Structured output (citations, confidence ratings, numbered lists) is easier
   for downstream code to parse and easier for humans to scan quickly. Asking
   for a confidence rating (HIGH/MEDIUM/LOW) also forces the model to
   introspect on evidential strength before committing to an answer.

4. AUDIENCE CALIBRATION — Adjust vocabulary and tone to the reader.
   A compliance officer wants precise regulatory citations; a card-issuer's
   customer service agent wants plain English. The same underlying facts need
   very different framings. Audience-aware prompts produce output that can be
   used without further editing.

Template variables
------------------
Templates use Python str.format() syntax — {variable_name} — so callers can
render them with a single .format() call:

    from rag_pipeline.generation.prompt_templates import (
        COMPLIANCE_USER_TEMPLATE,
        FRAUD_EXPLANATION_USER_TEMPLATE,
    )

    user_msg = COMPLIANCE_USER_TEMPLATE.format(
        context=retrieved_chunks,
        question="What are JPM's AML obligations?",
    )

    explanation = FRAUD_EXPLANATION_USER_TEMPLATE.format(
        transaction_details=txn_dict,
        shap_features=top_drivers,
        fraud_probability=0.87,
    )
"""

# ─────────────────────────────────────────────────────────────────────────────
# COMPLIANCE ASSISTANT — SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

# Why this structure works:
#   - Opening role statement frames every subsequent instruction as coming from
#     an expert persona, not a generic assistant. Models follow persona-anchored
#     constraints more reliably than free-floating rules.
#   - The "ONLY use information" clause is placed early and repeated via the
#     uncertainty rule to reduce hallucination on regulatory specifics.
#   - Citation format is prescribed exactly (Ticker / Filing / Chunk) so
#     downstream parsing is deterministic — no regex guessing required.
#   - The confidence rating forces explicit epistemic calibration. A model that
#     must commit to HIGH/MEDIUM/LOW before answering tends to hedge more
#     accurately than one asked to "be appropriately uncertain".
#   - Prohibited behaviours are stated as negatives ("Do not speculate") rather
#     than positives ("Stay grounded") because negatives are more precisely
#     scoped and harder for the model to rationalise around.
COMPLIANCE_SYSTEM_PROMPT = """\
You are a senior financial compliance analyst with deep expertise in:
- SEC filing analysis (10-K annual reports, 10-Q quarterly reports, 8-K material events)
- Anti-money laundering (AML) and Bank Secrecy Act (BSA) regulations
- Basel III and IV capital adequacy frameworks (CET1, Tier 1, TLAC)
- FINRA and OCC supervisory requirements
- Fraud risk disclosures and internal control assessments

ANSWERING RULES — follow these precisely:

1. SOURCE RESTRICTION
   Answer ONLY using information present in the provided document excerpts.
   Do not draw on general knowledge, prior training data, or external sources.
   If the excerpts do not contain enough information to answer the question,
   say: "The provided filings do not contain sufficient information to answer
   this question."

2. CITATIONS
   Every factual claim must be followed by an inline citation in this exact
   format: [TICKER / FILING_TYPE / Chunk N of M]
   Example: JPMorgan maintains a CET1 ratio above 12.5% [JPM / 10-K / Chunk 4 of 31].
   Do not omit citations even for claims that seem obvious.

3. CONFIDENCE RATING
   End every response with a confidence assessment on its own line:
   CONFIDENCE: HIGH   — the excerpts directly and explicitly address the question
   CONFIDENCE: MEDIUM — the excerpts are relevant but incomplete or indirect
   CONFIDENCE: LOW    — the answer is inferred; direct evidence is absent

4. UNCERTAINTY FLAGGING
   When you are uncertain about a specific figure, date, or regulatory
   interpretation, flag it inline: (UNCERTAIN: reason).
   Example: The consent order was issued in 2022 (UNCERTAIN: exact date not in excerpts).

5. PROHIBITED BEHAVIOURS
   - Do not speculate beyond what the excerpts state.
   - Do not combine information from excerpts with external knowledge.
   - Do not give legal advice or recommend specific compliance actions.
   - Do not present inferred conclusions as confirmed facts.

6. FORMAT
   - Use numbered points for multi-part answers.
   - Use a brief summary sentence before detailed points for long answers.
   - Keep responses focused — omit content not directly relevant to the question.
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMPLIANCE ASSISTANT — USER TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

# Why context comes before question:
#   LLMs process input left-to-right and attend more strongly to content that
#   appears near the question. Placing context first "primes" the model with
#   the relevant text before it reads the question, improving retrieval fidelity
#   compared to reversed ordering (especially for long context windows).
#
# Why we label the section boundaries:
#   "DOCUMENT EXCERPTS:" and "QUESTION:" act as semantic anchors. The model can
#   use these markers to distinguish source material from the task instruction,
#   reducing the risk of treating the question text as part of the retrievable
#   corpus (a subtle source of hallucination in RAG systems).
COMPLIANCE_USER_TEMPLATE = """\
DOCUMENT EXCERPTS:
{context}

QUESTION:
{question}

Using only the document excerpts above, provide a precise, well-cited answer.
"""


# ─────────────────────────────────────────────────────────────────────────────
# FRAUD EXPLAINER — SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

# Why this prompt differs fundamentally from the compliance prompt:
#   - Audience: a cardholder or a customer service agent — not a data scientist.
#     Technical language (SHAP values, log-odds, XGBoost) would confuse rather
#     than inform. The prompt explicitly bans jargon and demands analogies.
#   - Tone: empathetic and non-accusatory. A fraud flag is embarrassing for a
#     legitimate customer; the explanation should acknowledge that fraud models
#     make mistakes and that flagging is protective, not punitive.
#   - Actionability: compliance answers are informational; fraud explanations
#     must end with a concrete next step. A user who understands why their
#     transaction was flagged but doesn't know what to do next is still stuck.
#   - Numerical guardrails: the model is given the exact fraud probability and
#     instructed to reflect it honestly. Without this, models tend to either
#     catastrophise ("this is definitely fraud") or minimise ("probably fine"),
#     neither of which matches the calibrated model output.
FRAUD_EXPLANATION_SYSTEM_PROMPT = """\
You are a clear, empathetic fraud prevention specialist explaining a transaction
risk assessment to a customer or customer service agent who has no technical
background. Your goal is to help them understand why a transaction was flagged
and what they can do next.

COMMUNICATION RULES:

1. PLAIN ENGLISH ONLY
   Never use technical terms like SHAP values, XGBoost, log-odds, feature
   importance, probability threshold, or model score. Instead, describe the
   same concepts in everyday language: "the system noticed", "this is unusual
   because", "compared to your normal spending".

2. HONEST PROBABILITY FRAMING
   Reflect the fraud probability accurately using natural language thresholds:
   - 85%+  → "very high likelihood of fraud — we recommend blocking this"
   - 60–84% → "elevated risk — this warrants a closer look"
   - 30–59% → "some suspicious signals — worth a quick confirmation"
   - <30%   → "low risk — but one or two things stood out"
   Do not use the raw percentage in the explanation unless the reader explicitly
   asked for it. Use the qualitative framing above instead.

3. EXPLAIN THE DRIVERS IN HUMAN TERMS
   Each risk factor from the model should be translated into a relatable
   observation:
   - amount_x_risk      → "large amount at a high-risk type of merchant"
   - is_night / hour    → "happened in the early hours of the morning"
   - txn_count_1h       → "multiple transactions in a short window"
   - account_age_bucket → "account is relatively new"
   - merchant_category  → "type of merchant associated with higher fraud rates"
   Lead with the factor that had the biggest impact. Limit to the top 3
   factors — listing more overwhelms rather than informs.

4. EMPATHY AND NON-ACCUSATORY TONE
   Acknowledge that the system flags some legitimate transactions. Avoid language
   that implies the customer has done something wrong. Use "the system detected"
   rather than "you did" or "your transaction looks suspicious".

5. ACTION GUIDANCE
   Always close with one clear recommended action tailored to the risk level:
   - CRITICAL/HIGH  → contact fraud team immediately / transaction has been blocked
   - MEDIUM         → verify via a second authentication channel before approving
   - LOW            → approve with a note; no action required from customer

6. LENGTH
   Keep the explanation between 3–5 short paragraphs. Brevity increases
   comprehension; a confused customer service agent cannot act quickly.
"""


# ─────────────────────────────────────────────────────────────────────────────
# FRAUD EXPLAINER — USER TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

# Why we pass shap_features as structured text rather than raw SHAP numbers:
#   The model is instructed not to use technical language, but if we pass raw
#   SHAP values (e.g., amount_x_risk: +2.14) the model must first interpret
#   them before re-expressing them — introducing a translation step where
#   errors can creep in. Callers should format shap_features as a human-readable
#   list before injecting (e.g. "amount_x_risk = 3040.0 (large amount at
#   high-risk merchant)") so the model can quote them directly.
#
# Why fraud_probability is passed explicitly:
#   Without grounding, the model might describe a 40% probability as "likely
#   fraud" (over-alarming) or a 90% probability as "might be fraud" (under-
#   alarming). Passing the numeric value lets the system prompt's threshold
#   rules anchor the qualitative framing to the calibrated model output.
FRAUD_EXPLANATION_USER_TEMPLATE = """\
TRANSACTION DETAILS:
{transaction_details}

KEY RISK FACTORS IDENTIFIED (most impactful first):
{shap_features}

FRAUD PROBABILITY SCORE: {fraud_probability:.1%}

Please explain in plain English why this transaction was flagged, which factors
contributed most to the risk assessment, and what the customer or agent should
do next.
"""


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def build_compliance_prompt(context: str, question: str) -> str:
    """
    Render the compliance user template with retrieved context and a question.

    Parameters
    ----------
    context  : str
        Concatenated SEC filing excerpts from the FAISS retriever.
        Typically formatted as one excerpt per line with source labels.
    question : str
        The user's natural-language compliance question.

    Returns
    -------
    str
        Ready-to-send user message. Pass COMPLIANCE_SYSTEM_PROMPT separately
        as the system prompt to BedrockClient.invoke().
    """
    return COMPLIANCE_USER_TEMPLATE.format(context=context, question=question)


def build_fraud_explanation_prompt(
    transaction_details: str | dict,
    shap_features: str | list,
    fraud_probability: float,
) -> str:
    """
    Render the fraud explanation user template.

    Accepts either pre-formatted strings or raw objects (dict / list of dicts)
    — converts them to readable text automatically.

    Parameters
    ----------
    transaction_details : str | dict
        Human-readable transaction summary, or a dict that will be converted
        to "key: value" lines.
    shap_features : str | list
        Formatted risk factor list, or a list of dicts from SHAPAnalyzer
        (expects keys: feature, shap_value, feature_value, direction).
    fraud_probability : float
        Raw model output in [0.0, 1.0] from FraudPredictor.score().

    Returns
    -------
    str
        Ready-to-send user message. Pass FRAUD_EXPLANATION_SYSTEM_PROMPT
        separately as the system prompt to BedrockClient.invoke().
    """
    if isinstance(transaction_details, dict):
        transaction_details = "\n".join(
            f"  {k}: {v}" for k, v in transaction_details.items()
        )

    if isinstance(shap_features, list):
        lines = []
        for i, f in enumerate(shap_features, 1):
            direction = "↑ increases fraud risk" if f.get(
                "shap_value", 0) > 0 else "↓ decreases fraud risk"
            lines.append(
                f"  {i}. {f.get('feature', '?')}: value={f.get('feature_value', '?')}  {direction}"
            )
        shap_features = "\n".join(lines)

    return FRAUD_EXPLANATION_USER_TEMPLATE.format(
        transaction_details=transaction_details,
        shap_features=shap_features,
        fraud_probability=fraud_probability,
    )


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Compliance prompt preview ─────────────────────────────────────────────
    sample_context = (
        "[JPM / 10-K / Chunk 4 of 31]\n"
        "JPMorgan Chase maintains comprehensive AML compliance programs governed "
        "by the Bank Secrecy Act and OFAC regulations. The firm files Suspicious "
        "Activity Reports (SARs) for transactions exhibiting patterns consistent "
        "with money laundering or terrorist financing...\n\n"
        "[BAC / 10-K / Chunk 7 of 28]\n"
        "Bank of America's AML program includes transaction monitoring systems "
        "that screen for structuring, rapid movement of funds, and geographic "
        "anomalies. The program is overseen by the Chief Compliance Officer "
        "and reported quarterly to the Board Risk Committee..."
    )
    compliance_user = build_compliance_prompt(
        context=sample_context,
        question="What AML controls do large US banks maintain under the Bank Secrecy Act?",
    )
    print("── COMPLIANCE SYSTEM PROMPT (first 300 chars) ─────────────────")
    print(COMPLIANCE_SYSTEM_PROMPT[:300], "...\n")
    print("── COMPLIANCE USER PROMPT ─────────────────────────────────────")
    print(compliance_user)

    # ── Fraud explanation prompt preview ──────────────────────────────────────
    txn = {
        "transaction_id": "txn-8821",
        "amount": "$3,200.00",
        "merchant": "Global Wire Services",
        "merchant_category": "wire_transfer",
        "time": "2:17 AM on Sunday",
        "location": "Miami, FL  (home city: Austin, TX)",
    }
    shap_top3 = [
        {"feature": "amount_x_risk",
         "shap_value": +2.41,
         "feature_value": 3040.0,
         "direction": "increases_fraud_risk"},
        {"feature": "is_night", "shap_value": +1.18,
            "feature_value": 1, "direction": "increases_fraud_risk"},
        {"feature": "txn_count_1h", "shap_value": +0.93,
            "feature_value": 6, "direction": "increases_fraud_risk"},
    ]
    fraud_user = build_fraud_explanation_prompt(
        transaction_details=txn,
        shap_features=shap_top3,
        fraud_probability=0.91,
    )
    print("\n── FRAUD SYSTEM PROMPT (first 300 chars) ──────────────────────")
    print(FRAUD_EXPLANATION_SYSTEM_PROMPT[:300], "...\n")
    print("── FRAUD USER PROMPT ──────────────────────────────────────────")
    print(fraud_user)
