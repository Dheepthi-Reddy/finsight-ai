"""
Downloads SEC filings for major financial companies and returns their text
for embedding into the compliance RAG pipeline.

Three filing types are collected because they cover different compliance surfaces:

  10-K  Annual report (filed once/year) — the most comprehensive document.
        Contains risk factors, internal controls, regulatory disclosures, and
        full audited financials. Primary source for questions about capital
        adequacy, AML programs, data privacy obligations, and litigation risk.

  10-Q  Quarterly report (filed 3×/year) — condensed financials and MD&A for
        each of the first three quarters. Useful for tracking how risk exposures
        and compliance postures evolve across reporting periods.

  8-K   Current report (filed within 4 business days of a material event) —
        covers earnings announcements, regulatory actions, executive changes,
        and M&A. Important for compliance Q&A because enforcement actions,
        consent orders, and settlements are disclosed here first.

Usage
-----
    from rag_pipeline.ingestion.sec_loader import load_filings

    docs = load_filings()           # default 10 tickers, all 3 filing types
    docs = load_filings(tickers=["JPM", "GS"], filing_types=["10-K"])

Each returned dict has keys: text, ticker, filing_type, path.
"""

import os
import re
import time
import textwrap
from html.parser import HTMLParser
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "JPM", "BAC", "GS", "MS", "WFC", "C", "BLK"]
DEFAULT_FILING_TYPES = ["10-K", "10-Q", "8-K"]
DOWNLOAD_DIR = Path("data/sec_filings")

# SEC EDGAR requires a User-Agent header identifying the requester.
# These values are used if the real downloader is available.
SEC_USER_COMPANY = os.getenv("SEC_USER_COMPANY", "FinSight AI Research")
SEC_USER_EMAIL = os.getenv("SEC_USER_EMAIL", "research@finsight-ai.dev")

# Limits per ticker per filing type — keep download size reasonable for dev
FILING_LIMITS = {"10-K": 2, "10-Q": 4, "8-K": 5}

# Truncate extracted text to this many characters per filing.
# SEC 10-Ks can exceed 1M characters; embeddings work best on focused chunks.
MAX_TEXT_CHARS = 80_000

# Full legal names for stub generation (no API needed to look these up)
COMPANY_NAMES: dict[str, str] = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.",
    "JPM": "JPMorgan Chase & Co.",
    "BAC": "Bank of America Corporation",
    "GS": "The Goldman Sachs Group, Inc.",
    "MS": "Morgan Stanley",
    "WFC": "Wells Fargo & Company",
    "C": "Citigroup Inc.",
    "BLK": "BlackRock, Inc.",
}


# ── HTML stripping ─────────────────────────────────────────────────────────────
# SEC filings are often delivered as HTML. stdlib's HTMLParser avoids the
# BeautifulSoup dependency while being sufficient for plain-text extraction.

class _HTMLStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        stripped = data.strip()
        if stripped:
            self._parts.append(stripped)

    def get_text(self) -> str:
        return " ".join(self._parts)


def _strip_html(raw: str) -> str:
    parser = _HTMLStripper()
    parser.feed(raw)
    text = parser.get_text()
    # Collapse whitespace runs left over from table/list structure
    text = re.sub(r"\s{3,}", "  ", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# REAL DOWNLOADER (requires: pip install sec-edgar-downloader)
# ─────────────────────────────────────────────────────────────────────────────

def _download_real(
    tickers: list[str],
    filing_types: list[str],
    download_dir: Path,
) -> list[dict]:
    """
    Downloads filings via the sec-edgar-downloader library and returns
    extracted plain text. Rate-limited to 0.5 s between requests to stay
    within SEC EDGAR's fair-use guidelines (max 10 req/s).
    """
    from sec_edgar_downloader import Downloader  # noqa: PLC0415

    dl = Downloader(SEC_USER_COMPANY, SEC_USER_EMAIL, str(download_dir))
    docs: list[dict] = []

    for ticker in tickers:
        for ftype in filing_types:
            limit = FILING_LIMITS.get(ftype, 3)
            print(f"  Downloading {ticker} {ftype} (limit={limit})...")
            try:
                dl.get(ftype, ticker, limit=limit)
            except Exception as exc:
                print(f"  [WARN] {ticker} {ftype} download failed: {exc}")
                continue
            time.sleep(0.5)   # respect SEC fair-use rate limit

            # sec-edgar-downloader writes to: <download_dir>/<ticker>/<ftype>/<accession>/
            filing_root = download_dir / ticker / ftype
            if not filing_root.exists():
                continue

            for accession_dir in sorted(filing_root.iterdir()):
                text = _extract_text_from_dir(accession_dir)
                if not text:
                    continue
                docs.append({
                    "text": text[:MAX_TEXT_CHARS],
                    "ticker": ticker,
                    "filing_type": ftype,
                    "path": str(accession_dir),
                })

    return docs


def _extract_text_from_dir(directory: Path) -> str:
    """
    Reads the primary document from an EDGAR accession directory.
    Prefers .htm/.html (full filing), falls back to .txt (SGML wrapper).
    """
    # Prefer .htm files — they contain the full formatted filing
    candidates = sorted(directory.glob("*.htm")) + sorted(directory.glob("*.html"))
    if not candidates:
        candidates = sorted(directory.glob("*.txt"))
    if not candidates:
        return ""

    raw = candidates[0].read_text(encoding="utf-8", errors="replace")
    # Strip HTML tags if this looks like HTML
    if raw.lstrip().startswith("<") or "<html" in raw[:500].lower():
        return _strip_html(raw)
    return raw.strip()


# ─────────────────────────────────────────────────────────────────────────────
# STUB GENERATOR (no external library required)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_stubs(tickers: list[str], filing_types: list[str], out_dir: Path) -> list[dict]:
    """
    Writes realistic stub filing text files to disk and returns them in the
    same format as the real downloader. Stubs contain actual regulatory
    language from standard SEC disclosure templates so the RAG embeddings
    reflect the domain the compliance chatbot will be questioned about.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    docs: list[dict] = []

    for ticker in tickers:
        company = COMPANY_NAMES.get(ticker, f"{ticker} Financial Corp.")
        for ftype in filing_types:
            limit = FILING_LIMITS.get(ftype, 3)
            for i in range(1, limit + 1):
                year = 2024 - (i - 1)
                period = f"Q{(i % 4) + 1} {year}" if ftype == "10-Q" else str(year)
                path = out_dir / ticker / ftype / f"stub_{i:02d}.txt"
                path.parent.mkdir(parents=True, exist_ok=True)

                text = _render_stub(ticker, company, ftype, year, period, i)
                path.write_text(text, encoding="utf-8")

                docs.append({
                    "text": text,
                    "ticker": ticker,
                    "filing_type": ftype,
                    "path": str(path),
                })

    return docs


def _render_stub(
    ticker: str, company: str, ftype: str, year: int, period: str, idx: int
) -> str:
    """Build a stub filing from templates appropriate to the filing type."""
    if ftype == "10-K":
        return _stub_10k(ticker, company, year)
    if ftype == "10-Q":
        return _stub_10q(ticker, company, period)
    return _stub_8k(ticker, company, year, idx)


def _stub_10k(ticker: str, company: str, year: int) -> str:
    return textwrap.dedent(f"""\
        UNITED STATES SECURITIES AND EXCHANGE COMMISSION
        Washington, D.C. 20549

        FORM 10-K
        ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d)
        OF THE SECURITIES EXCHANGE ACT OF 1934

        For the fiscal year ended December 31, {year}
        Commission File Number: 001-{abs(hash(ticker)) % 90000 + 10000}

        {company.upper()}
        (Exact name of registrant as it appears in its charter)

        =====================================================================
        PART I
        =====================================================================

        ITEM 1. BUSINESS

        {company} ("{ticker}") is a global financial services company providing a
        diversified range of banking, investment management, and financial services
        to consumers, corporations, governments, and institutions worldwide. The
        Company operates through multiple business segments subject to regulation by
        federal and state banking authorities, the SEC, FINRA, and international
        regulatory bodies.

        ITEM 1A. RISK FACTORS

        Regulatory and Compliance Risk
        The Company is subject to extensive regulation and supervision by federal and
        state regulators. Changes in laws or regulations, or in their interpretation
        or enforcement, could adversely affect the Company's business. The Company's
        compliance programs are designed to comply with the Bank Secrecy Act (BSA),
        Anti-Money Laundering (AML) requirements, the USA PATRIOT Act, Office of
        Foreign Assets Control (OFAC) regulations, and Know Your Customer (KYC)
        obligations. Failure to maintain adequate AML controls could result in
        regulatory sanctions, fines, and reputational harm.

        The Company maintains an enterprise-wide AML compliance program that
        includes policies, procedures, transaction monitoring systems, suspicious
        activity report (SAR) filing processes, and ongoing employee training.
        The program is overseen by a designated Bank Secrecy Act Officer and
        reviewed annually by internal audit.

        Technology and Cybersecurity Risk
        The Company relies on complex information technology systems. Cybersecurity
        incidents, including unauthorized access, ransomware, and data breaches,
        could compromise customer data and disrupt operations. The Company maintains
        a cybersecurity framework aligned with the NIST Cybersecurity Framework and
        conducts annual penetration testing and tabletop exercises.

        Fraud Risk
        The Company faces fraud risk across all business lines, including account
        takeover, synthetic identity fraud, payment fraud, and insider fraud. The
        Company employs machine learning-based transaction monitoring, behavioral
        analytics, and multi-factor authentication to detect and prevent fraud.
        Real-time scoring of transactions allows the Company to block suspicious
        activity before settlement. Despite these controls, fraud losses remain a
        material risk factor.

        Credit Risk
        Credit risk arises from the potential that a borrower or counterparty will
        fail to perform on its contractual obligations. The Company manages credit
        risk through underwriting standards, credit concentration limits, collateral
        requirements, and portfolio diversification.

        Interest Rate Risk
        Changes in interest rates affect the Company's net interest margin and the
        fair value of its assets and liabilities. The Company uses interest rate
        derivatives, gap analysis, and economic value of equity modeling to manage
        this exposure.

        ITEM 1B. UNRESOLVED STAFF COMMENTS — None.

        =====================================================================
        PART II
        =====================================================================

        ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION
        AND RESULTS OF OPERATIONS

        Overview
        For the fiscal year ended December 31, {year}, {company} reported net
        revenues of approximately ${"%.1f" % (abs(hash(ticker + str(year))) % 60 + 20)}B
        and net income attributable to common shareholders of approximately
        ${"%.1f" % (abs(hash(ticker + str(year))) % 15 + 5)}B. Return on equity was
        {"%.1f" % (abs(hash(ticker)) % 12 + 8)}%.

        Capital and Liquidity
        The Company maintains capital ratios in excess of regulatory minimums under
        Basel III requirements. Common Equity Tier 1 (CET1) capital ratio was
        {"%.1f" % (abs(hash(ticker + str(year + 1))) % 4 + 12)}% as of December 31,
        {year}, compared to the 4.5% minimum plus the 2.5% capital conservation
        buffer required by regulation. The Liquidity Coverage Ratio (LCR) and Net
        Stable Funding Ratio (NSFR) both exceeded 100% throughout the year.

        Anti-Money Laundering Program
        The Company filed {abs(hash(ticker + str(year + 2))) % 2000 + 500} Suspicious
        Activity Reports (SARs) with FinCEN during {year}. The Company's transaction
        monitoring system reviewed over {abs(hash(ticker)) % 500 + 100} million
        transactions and escalated {abs(hash(ticker + str(year + 3))) % 50000 + 10000}
        alerts for analyst review. No material AML enforcement actions were taken
        against the Company during {year}.

        ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK
        Market risk is managed through value-at-risk (VaR) models, stress testing,
        and scenario analysis consistent with Basel Committee on Banking Supervision
        guidelines.

        ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA
        The Consolidated Financial Statements and Notes thereto are incorporated
        herein by reference from the Annual Report to Shareholders.

        =====================================================================
        PART III
        =====================================================================

        ITEM 10. DIRECTORS, EXECUTIVE OFFICERS AND CORPORATE GOVERNANCE
        The Company has adopted a Code of Business Conduct and Ethics applicable to
        all employees, officers, and directors. The Audit Committee oversees
        compliance with the Foreign Corrupt Practices Act (FCPA), Sarbanes-Oxley
        Act internal controls requirements, and the Company's whistleblower program.

        ITEM 14. PRINCIPAL ACCOUNTANT FEES AND SERVICES
        Audit Fees: ${abs(hash(ticker)) % 40 + 10}M
        Audit-Related Fees: ${abs(hash(ticker + str(year))) % 8 + 2}M
        Tax Fees: ${abs(hash(ticker + str(year + 1))) % 5 + 1}M
    """)


def _stub_10q(ticker: str, company: str, period: str) -> str:
    return textwrap.dedent(f"""\
        UNITED STATES SECURITIES AND EXCHANGE COMMISSION
        Washington, D.C. 20549

        FORM 10-Q
        QUARTERLY REPORT PURSUANT TO SECTION 13 OR 15(d)
        OF THE SECURITIES EXCHANGE ACT OF 1934

        For the quarterly period ended: {period}

        {company.upper()} ({ticker})

        =====================================================================
        PART I — FINANCIAL INFORMATION
        =====================================================================

        ITEM 1. FINANCIAL STATEMENTS (UNAUDITED)

        The accompanying unaudited condensed consolidated financial statements
        have been prepared in accordance with generally accepted accounting
        principles (GAAP) and applicable SEC regulations for interim financial
        information.

        Net revenues for {period}: ${abs(hash(ticker + period)) % 20 + 5}B
        Provision for credit losses: ${abs(hash(ticker + period + "p")) % 3 + 1}B
        Net income: ${abs(hash(ticker + period + "n")) % 5 + 1}B

        ITEM 2. MANAGEMENT'S DISCUSSION AND ANALYSIS

        Regulatory Update
        During {period}, the Company received no material regulatory orders or
        consent agreements. The Company continues to cooperate with ongoing
        examinations by the OCC, FDIC, and Federal Reserve. Capital ratios
        remained above all regulatory minimums and internal targets.

        AML and Sanctions Compliance
        Transaction monitoring enhancements deployed during the quarter reduced
        false-positive alert rates by {abs(hash(ticker + period + "aml")) % 15 + 5}%
        while maintaining detection coverage. The Company updated its OFAC
        screening lists to reflect new Treasury SDN designations. No SAR filing
        deficiencies were identified during the quarter.

        Fraud Metrics
        Fraud losses for {period} were ${abs(hash(ticker + period + "f")) % 200 + 50}M,
        representing {"%.2f" % (abs(hash(ticker + period + "fr")) % 10 / 100 + 0.01)}%
        of total transaction volume. The Company's machine learning fraud models
        maintained precision above 90% and recall above 85% for card-not-present
        fraud detection.

        ITEM 3. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK
        No material changes from the Annual Report on Form 10-K.

        ITEM 4. CONTROLS AND PROCEDURES
        Management evaluated the effectiveness of the Company's disclosure controls
        and procedures as of the end of {period}. Based on that evaluation, the
        CEO and CFO concluded that the disclosure controls and procedures are
        effective in ensuring that information required to be disclosed is recorded,
        processed, summarized, and reported within required time periods.

        =====================================================================
        PART II — OTHER INFORMATION
        =====================================================================

        ITEM 1. LEGAL PROCEEDINGS
        The Company is party to various legal proceedings arising in the ordinary
        course of business. Management does not believe these proceedings will have
        a material adverse effect on the Company's financial condition.

        ITEM 1A. RISK FACTORS
        No material changes from the risk factors disclosed in the Annual Report
        on Form 10-K for the fiscal year ended December 31.
    """)


def _stub_8k(ticker: str, company: str, year: int, idx: int) -> str:
    # Rotate through representative material-event types to give the RAG
    # pipeline diverse compliance content from current-report disclosures.
    events = [
        (
            "REGULATORY ACTION AND CONSENT ORDER",
            f"{company} entered into a formal agreement with the Office of the "
            f"Comptroller of the Currency (OCC) relating to the Company's Bank "
            f"Secrecy Act / Anti-Money Laundering compliance program. Under the "
            f"agreement, the Company has committed to enhance its transaction "
            f"monitoring systems, increase AML staffing, and engage a third-party "
            f"consultant to review its customer due diligence procedures. The Company "
            f"has established a dedicated remediation team and expects to complete "
            f"all required enhancements within 18 months. The agreement does not "
            f"require the payment of a civil money penalty at this time.",
        ),
        (
            "QUARTERLY EARNINGS AND FRAUD DISCLOSURE",
            f"{company} today reported financial results for the quarter. "
            f"The Company also disclosed that it identified a fraud incident affecting "
            f"approximately {abs(hash(ticker + str(year + idx))) % 5000 + 500} customer "
            f"accounts. The incident involved unauthorized access through compromised "
            f"credentials. The Company has notified affected customers, reset "
            f"credentials, and filed required SARs with FinCEN. Estimated losses "
            f"are ${abs(hash(ticker + str(year))) % 50 + 5}M, substantially covered "
            f"by the Company's fraud loss reserve. The Company has engaged a "
            f"cybersecurity firm to conduct a forensic investigation.",
        ),
        (
            "CHANGE IN CHIEF COMPLIANCE OFFICER",
            f"{company} announced that its Chief Compliance Officer (CCO) will "
            f"retire effective {year}-12-31. The Company has appointed an interim "
            f"CCO and commenced an external search for a permanent replacement. "
            f"The retiring CCO oversaw the Company's global compliance program "
            f"including AML, sanctions screening, FCPA, and regulatory affairs. "
            f"The Board's Risk Committee will maintain enhanced oversight of "
            f"compliance functions during the transition period.",
        ),
        (
            "FINRA INVESTIGATION AND SETTLEMENT",
            f"{company} reached a settlement with FINRA regarding supervisory "
            f"deficiencies in its broker-dealer operations. The Company agreed to "
            f"pay a fine of ${abs(hash(ticker + str(year + idx + 1))) % 20 + 5}M and "
            f"undertake remedial measures including enhanced trade surveillance, "
            f"updated written supervisory procedures, and targeted training. "
            f"The settlement relates to the period {year - 2} through {year - 1} and "
            f"does not constitute an admission of wrongdoing by the Company.",
        ),
        (
            "DATA BREACH NOTIFICATION",
            f"{company} became aware of a data security incident in which an "
            f"unauthorized third party accessed certain customer data. The Company "
            f"immediately engaged outside counsel and a cybersecurity firm, notified "
            f"affected customers, and made required notifications to state attorneys "
            f"general and federal regulators. Affected data may include names, "
            f"account numbers, and transaction history for approximately "
            f"{abs(hash(ticker + str(year + idx + 2))) % 100000 + 10000} customers. "
            f"The Company has enhanced its security controls and implemented "
            f"additional monitoring. This incident is not expected to have a "
            f"material adverse effect on the Company's financial condition.",
        ),
    ]
    item_name, body = events[(idx - 1) % len(events)]

    return textwrap.dedent(f"""\
        UNITED STATES SECURITIES AND EXCHANGE COMMISSION
        Washington, D.C. 20549

        FORM 8-K
        CURRENT REPORT PURSUANT TO SECTION 13 OR 15(d)
        OF THE SECURITIES EXCHANGE ACT OF 1934

        Date of Report: {year}-{(idx * 2):02d}-15

        {company.upper()} ({ticker})

        ITEM 8.01 — {item_name}

        {body}

        SIGNATURES
        Pursuant to the requirements of the Securities Exchange Act of 1934,
        the registrant has duly caused this report to be signed on its behalf
        by the undersigned hereunto duly authorized.

        {company}
        By: /s/ Chief Legal Officer
        Date: {year}-{(idx * 2):02d}-15
    """)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def load_filings(
    tickers: list[str] = DEFAULT_TICKERS,
    filing_types: list[str] = DEFAULT_FILING_TYPES,
    download_dir: Path | str = DOWNLOAD_DIR,
    force_stubs: bool = False,
) -> list[dict]:
    """
    Download SEC filings and return them as a list of text documents.

    Attempts to use sec-edgar-downloader for real EDGAR data. Falls back to
    pre-rendered stub filings if the library is not installed or if
    force_stubs=True (useful for unit testing without network access).

    Parameters
    ----------
    tickers : list[str]
        Stock tickers to download filings for.
    filing_types : list[str]
        SEC form types: "10-K", "10-Q", and/or "8-K".
    download_dir : Path
        Local directory for downloaded or generated files.
    force_stubs : bool
        Skip the real downloader even if sec-edgar-downloader is installed.

    Returns
    -------
    list[dict]
        Each dict: {"text": str, "ticker": str, "filing_type": str, "path": str}
        text is truncated to MAX_TEXT_CHARS characters per filing.
    """
    download_dir = Path(download_dir)

    use_stubs = force_stubs
    if not use_stubs:
        try:
            import sec_edgar_downloader  # noqa: F401
        except ImportError:
            print(
                "[INFO] sec-edgar-downloader not installed — using stub filings.\n"
                "       Install with: pip install sec-edgar-downloader"
            )
            use_stubs = True

    if use_stubs:
        print(f"Generating stub filings for {len(tickers)} tickers × {len(filing_types)} types...")
        docs = _generate_stubs(tickers, filing_types, download_dir)
    else:
        print(f"Downloading real SEC filings for: {', '.join(tickers)}")
        docs = _download_real(tickers, filing_types, download_dir)

    print(f"Loaded {len(docs)} filing documents.")
    _print_summary(docs)
    return docs


def _print_summary(docs: list[dict]) -> None:
    from collections import Counter
    by_type = Counter(d["filing_type"] for d in docs)
    by_ticker = Counter(d["ticker"] for d in docs)
    print(f"  By filing type : {dict(by_type)}")
    print(f"  By ticker      : {len(by_ticker)} tickers, "
          f"{min(by_ticker.values())}–{max(by_ticker.values())} docs each")
    avg_chars = sum(len(d["text"]) for d in docs) / max(len(docs), 1)
    print(f"  Avg text length: {avg_chars:,.0f} chars per filing")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download or stub SEC filings.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--types", nargs="+", default=DEFAULT_FILING_TYPES,
                        dest="filing_types")
    parser.add_argument("--stubs", action="store_true",
                        help="Force stub generation even if sec-edgar-downloader is installed")
    parser.add_argument("--out", default=str(DOWNLOAD_DIR),
                        help="Output directory (default: data/sec_filings)")
    args = parser.parse_args()

    docs = load_filings(
        tickers=args.tickers,
        filing_types=args.filing_types,
        download_dir=Path(args.out),
        force_stubs=args.stubs,
    )

    # Spot-check: print the first 400 characters of the first document
    if docs:
        print(f"\n── Sample: {docs[0]['ticker']} {docs[0]['filing_type']} ──")
        print(docs[0]["text"][:400].strip())
        print("...")
