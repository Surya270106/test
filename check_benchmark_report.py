#!/usr/bin/env python3
"""Small report-level smoke checker for the real public GitHub benchmark cases."""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "benchmarks" / "real_project_cases.json"


def load_cases() -> dict[str, dict]:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    return {case["id"]: case for case in data["cases"]}



def _normalize_benchmark_transcript(report: str) -> str:
    """Remove terminal decoration and reconstruct wrapped report paragraphs.

    Gravity output can contain ANSI colors, HTML entities, Rich box gutters, and
    terminal-wrapped Markdown. Strip presentation-only framing while preserving
    blank paragraph boundaries so evidence in unrelated sections cannot satisfy
    a target-local check.
    """
    text = html.unescape(report or "")
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    lines: list[str] = []
    border_only = re.compile(r"^[╭╰╮╯┌┐└┘├┤┬┴┼─━═╴╶╵╷+\- ]+$")
    for raw in text.splitlines():
        line = raw.translate(str.maketrans({"│": " ", "║": " ", "┃": " "}))
        line = re.sub(r"^\s*[╭╰┌└].*?[╮╯┐┘]\s*$", "", line)
        line = re.sub(r"^\s*[╭╰╮╯┌┐└┘├┤┬┴┼─━═]+\s*$", "", line)
        line = line.strip()
        if border_only.fullmatch(line):
            line = ""
        lines.append(line)

    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if not line:
            if current:
                paragraphs.append(" ".join(current).strip())
                current = []
            elif paragraphs and paragraphs[-1] != "":
                paragraphs.append("")
        else:
            current.append(line)
    if current:
        paragraphs.append(" ".join(current).strip())
    return "\n\n".join(item for item in paragraphs if item)


# Cases whose required evidence may legitimately appear in more than one report
# section, provided every section is explicitly anchored to the target artifact.
# Identity checks (URL, title, author, status) always stay on the primary block.
TARGET_LINKED_EVIDENCE_CASES = frozenset({"fastapi-15953"})

_OTHER_ARTIFACT_PATH = re.compile(
    r"^https?://github\.com/[^/\s]+/[^/\s]+/(?:pull|issues|commit|commits|releases|discussions|actions/runs)/",
)
_TEST_QUALIFIERS = (
    "author-reported", "pr description", "not independently verified", "not separately verified",
    "not independently reproduced", "source-reported",
)
_CI_SUCCESS_CLAIM = re.compile(
    r"(?:\b52\s+(?:ci\s+)?checks?\s+passed\b|\bci\s+(?:checks?\s+)?(?:passed|succeeded|is green|was green)\b"
    r"|\bchecks\s+(?:are\s+)?(?:all\s+)?green\b|\bverified\s+(?:by|in)\s+ci\b)"
)
_CI_NEGATION = re.compile(r"(?:\bnot\b|\bno\b|\bnor\b|\bwithout\b|\bunless\b|\bnever\b|n't)[^.;]{0,60}$")


def _flexible_pattern(text: str) -> str:
    """Match text tolerating whitespace introduced by terminal wrapping."""
    return r"\s+".join(re.escape(part) for part in text.split())


def _block_names_target(case: dict, block: str) -> bool:
    """True when a block carries an explicit identity anchor for the target.

    Anchors: the canonical URL (exact, not a longer number), the exact title, or
    the repository together with the PR number written as ``#N``/``pull/N``.
    """
    lower = block.casefold()
    url = case["artifact_url"].casefold().rstrip("/")
    url_pattern = r"\s*".join(re.escape(char) for char in url) + r"(?![\w\-])"
    if re.search(url_pattern, lower):
        return True
    if re.search(r"(?<![\w])" + _flexible_pattern(case["title"].casefold()) + r"(?![\w])", lower):
        return True
    number = str(case.get("artifact_number", ""))
    repo = str(case.get("repo", "")).casefold()
    return bool(number and repo and repo in lower
                and re.search(rf"(?:#|pull/){re.escape(number)}(?!\d)", lower))


def _block_mentions_other_artifact(case: dict, block: str) -> bool:
    """True when a block also references a different GitHub artifact.

    Such a block is ambiguous about which artifact a test count belongs to, so
    it is never accepted as target-linked evidence (fail closed).
    """
    lower = block.casefold()
    target_url = case["artifact_url"].casefold().rstrip("/")
    number = str(case.get("artifact_number", ""))
    for raw in re.findall(r"https?://[^\s<>\]\)\}\"']+", lower):
        candidate = raw.rstrip(".,;:!?").rstrip("/")
        if candidate == target_url or candidate.startswith((target_url + "/", target_url + "#", target_url + "?")):
            continue
        if _OTHER_ARTIFACT_PATH.match(candidate):
            return True
    for ref in re.findall(r"(?:#|\bpull/)(\d{3,})(?!\d)", lower):
        if ref != number:
            return True
    return False


def target_linked_blocks(case: dict, blocks: list[str]) -> list[str]:
    """Return report blocks explicitly tied to the target and to no other artifact."""
    return [block for block in blocks
            if block.strip() and _block_names_target(case, block)
            and not _block_mentions_other_artifact(case, block)]


def _contains_fact(block: str, fact: str) -> bool:
    return bool(re.search(r"(?<![\w.,])" + _flexible_pattern(fact.casefold()) + r"(?!\w)", block.casefold()))


def _asserts_unqualified_ci_success(block: str) -> bool:
    lower = block.casefold()
    for match in _CI_SUCCESS_CLAIM.finditer(lower):
        if not _CI_NEGATION.search(lower[max(0, match.start() - 80):match.start()]):
            return True
    return False


def evaluate(case: dict, report: str) -> list[tuple[bool, str]]:
    normalized_report = _normalize_benchmark_transcript(report)
    lower = normalized_report.casefold()
    compact_lower = re.sub(r"\s+", "", lower)
    results: list[tuple[bool, str]] = []
    url = case["artifact_url"].casefold().rstrip("/")
    compact_url = re.sub(r"\s+", "", url)
    title = case["title"].casefold()
    compact_title = re.sub(r"\s+", "", title)
    url_present = compact_url in compact_lower
    title_present = title in lower or compact_title in compact_lower
    url_candidates = re.findall(r"https?://[^\s<>\]\)\}\"']+", lower)
    expected_parent = url.rsplit("/", 1)[0] + "/"
    wrong_url = next((candidate.rstrip(".,;:!?") for candidate in url_candidates
                      if candidate.rstrip(".,;:!?").startswith(expected_parent)
                      and candidate.rstrip(".,;:!?").rstrip("/") != url), None)
    results.append((url_present, f"target URL is canonical: {case['artifact_url']}" if url_present else
                    f"incorrect target URL: found {wrong_url}; expected {case['artifact_url']}" if wrong_url else
                    f"missing target URL: expected {case['artifact_url']}"))

    # Bind all artifact-specific checks to the paragraph containing the exact
    # target URL. Rich gutters and wrapped terminal lines are presentation only.
    # Prefer a block containing both the exact URL and title; otherwise use the
    # URL block and let the target-local title assertion fail closed.
    blocks = normalized_report.split("\n\n")
    target_paragraph = next((
        block for block in blocks
        if compact_url in re.sub(r"\s+", "", block.casefold())
        and compact_title in re.sub(r"\s+", "", block.casefold())
    ), None)
    if target_paragraph is None:
        target_paragraph = next((
            block for block in blocks
            if compact_url in re.sub(r"\s+", "", block.casefold())
        ), None)
    if target_paragraph is None:
        target_paragraph = next((
            block for block in blocks
            if compact_title in re.sub(r"\s+", "", block.casefold())
        ), None)
    target_context = target_paragraph.casefold() if target_paragraph else ""
    compact_target_context = re.sub(r"\s+", "", target_context)
    title_local = title in target_context or compact_title in compact_target_context
    results.append((title_present and title_local,
                    f"exact target title present in target-linked block: {case['title']}" if title_present and title_local
                    else f"exact target title missing or not target-local: expected {case['title']}"))
    author = case["artifact_author"].casefold()
    results.append((bool(target_context) and author in target_context,
                    f"target-local author attribution present: {case['artifact_author']}"))

    status = case["status_on_verified_date"].casefold()
    if status in {"open", "closed_unmerged"}:
        if status == "open":
            has_status_signal = any(term in target_context for term in ("open", "proposed", "pending review"))
            label = "open/proposed status is explicit"
            state_desc = "open PR"
        else:
            has_status_signal = any(term in target_context for term in ("closed", "closed without merge", "not merged", "closed unmerged", "superseded"))
            label = "closed-without-merge status is explicit"
            state_desc = "closed unmerged PR"
        wrong_state_terms = ("merged", "shipped", "released", "landed")
        wrong_state = any(term in target_context for term in wrong_state_terms) and not any(
            phrase in target_context for phrase in ("not merged", "not been merged", "unmerged", "without merge", "not shipped", "not released", "not landed")
        ) if status == "open" else any(term in target_context for term in ("merged", "shipped", "released", "landed")) and not any(
            phrase in target_context for phrase in ("not merged", "not been merged", "unmerged", "without merge", "not shipped", "not released", "not landed")
        )
        results.append((has_status_signal, f"{label} (missing if absent)"))
        # Require both a correct target-local state and no contradictory landed
        # claim in that same target record. Diagnostics distinguish wrong state.
        positive_landed = re.search(r"\b(?:was merged|has been merged|merged into (?:master|main)|shipped|released|landed in (?:master|main))\b", target_context)
        wrong_status = bool(positive_landed) or wrong_state
        results.append((not target_context or not wrong_status,
                        f"target status does not contradict {state_desc} (contradicted if failed)"))
    elif status == "merged":
        has_merged_signal = bool(re.search(r"\b(?:was merged|has been merged|merged on|merged\b.{0,45}\binto (?:main|master))\b", target_context))
        results.append((has_merged_signal, "merged status is explicit"))
        if case["id"] == "pydantic-13013-merged":
            results.append(("83 checks passed" in target_context, "target page's 83 checks passed evidence is present"))
            caveat = any(term in target_context for term in ("not independently verified", "not a production metric", "not production", "cross-environment", "different runtime environments"))
            results.append((caveat, "target-local cross-environment performance caveat is retained"))
    elif status == "published":
        results.append((any(term in target_context for term in ("published", "released on", "release published")),
                        "published release status is explicit"))

    # Case-specific evidence is required in the cited target record, rather than
    # anywhere in the report where unrelated artifacts can create false passes.
    required_groups = {
        "fastapi-15641": (("errors=replace", "replacement character", "decode errors with replacement"),),
        "fastapi-15953": (("52 passed", "52 tests passed", "52 passing"),),
        "fastapi-16159": (("24.6",), ("19.1",)),
        "fastapi-4089-closed-unmerged": (("not merged", "without merge", "unmerged"),),
        "pydantic-v2-13-0-release-context": (("verified signature", "verified commit", "verified on github"),
                                                ("contributors", "multiple contributors", "individual contributors")),
    }.get(case["id"], ())
    linked_blocks = target_linked_blocks(case, blocks) if case["id"] in TARGET_LINKED_EVIDENCE_CASES else []
    for alternatives in required_groups:
        if case["id"] in TARGET_LINKED_EVIDENCE_CASES:
            # Evidence may sit in any section that is explicitly anchored to the
            # target and mentions no other artifact; the whole report is never searched.
            present = any(_contains_fact(block, fact) for block in linked_blocks for fact in alternatives)
        else:
            present = any(fact in target_context or re.sub(r"\s+", "", fact.casefold()) in compact_target_context for fact in alternatives)
        results.append((present, f"target-specific required evidence present: {' / '.join(alternatives)} (missing if absent)"))

    if case["id"] == "fastapi-16159":
        qualified = any(term in target_context for term in ("author-reported", "reported in the pr", "not independently verified", "not reproduced"))
        results.append((qualified, "benchmark metrics are qualified as author-reported/unreproduced"))
    if case["id"] == "fastapi-15953":
        count_facts = required_groups[0] if required_groups else ()
        # The qualification must sit in the same target-linked block as the test
        # count it qualifies, and no linked block may claim independent CI success.
        qualified_blocks = [block for block in linked_blocks
                            if any(_contains_fact(block, fact) for fact in count_facts)
                            and any(term in block.casefold() for term in _TEST_QUALIFIERS)]
        ci_overclaim = any(_asserts_unqualified_ci_success(block) for block in linked_blocks)
        results.append((bool(qualified_blocks) and not ci_overclaim,
                        "test count is distinguished from separately verified CI"))
    if case["id"] == "pydantic-v2-13-0-release-context":
        disclaimers = ("does not establish that", "does not prove that", "not evidence that",
                       "must not be credited with", "do not attribute", "not attributable to",
                       "not enough to claim", "not a claim that")
        has_disclaimer = any(term in target_context for term in disclaimers)
        contradicts = any(term in target_context for term in ("authored every change", "authored all changes", "every change was authored by"))
        results.append((has_disclaimer and not contradicts,
                        "target-local caveat prevents attributing all release changes to publisher"))
    if case["id"] == "pydantic-13013-merged":
        has_metric = bool(re.search(r"\b30\.07(?:%| percent)(?!\w)", target_context))
        has_production_claim = any(term in target_context for term in ("production performance improved", "production metric", "production faster", "in production"))
        overclaim = has_metric and has_production_claim
        results.append((not overclaim, "30.07% benchmark is not presented as a verified production result"))
    if case["id"] == "fastapi-16159":
        production_overclaim = bool(re.search(r"(?is)(?:production latency|production performance|in production).{0,120}(?:24\.6|19\.1|improv|faster)", target_context))
        results.append((not production_overclaim, "author-reported microbenchmarks are not claimed as measured production impact"))
    return results


def evaluate_full_backfill_audit(report: str) -> list[tuple[bool, str]]:
    """Require a clean structured completeness audit for a live benchmark run.

    This is separate from the target-local content checks so synthetic report
    fixtures remain useful unit tests. The live benchmark runner always enables
    this gate and records the model/CLI exit code separately.
    """
    decoded = html.unescape(report or "")
    decoded = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", decoded)
    normalized = _normalize_benchmark_transcript(decoded)
    lower = normalized.casefold()
    marker = "full-backfill completeness audit"
    marker_index = lower.find(marker)
    if marker_index < 0:
        return [
            (False, "full-backfill completeness audit section is present"),
            (False, "full-backfill audit is COMPLETE with no unresolved candidates or limitations"),
            (False, "rendered report explicitly states RESULT: COMPLETE"),
        ]

    tail = normalized[marker_index + len(marker):]
    start = tail.find("{")
    parsed: dict | None = None
    if start >= 0:
        fragment = tail[start:]
        fragment = "".join(char for char in fragment if char not in "│║┃")
        # Rich may wrap JSON across rendered terminal rows. Whitespace outside
        # strings is legal JSON; compacting all whitespace also re-joins a token
        # split by a presentation-only line wrap, while keeping JSON escapes.
        compact = re.sub(r"\s+", "", fragment)
        depth = 0
        in_string = False
        escaped = False
        end = None
        for index, char in enumerate(compact):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end is not None:
            try:
                value = json.loads(compact[:end])
                if isinstance(value, dict):
                    parsed = value
            except (json.JSONDecodeError, TypeError):
                parsed = None

    audit_ok = parsed is not None and parsed.get("status") == "COMPLETE"
    unresolved_ok = parsed is not None and parsed.get("unresolved_candidate_urls") == []
    critical_ok = parsed is not None and parsed.get("critical_failures") == []
    citations_ok = parsed is not None and parsed.get("unsupported_citations") == []
    limitations_ok = parsed is not None and parsed.get("limitations") == []
    if parsed is not None and parsed.get("qualifying_artifact_count") == 0:
        zero_ok = parsed.get("zero_result_is_audited") is True
    else:
        zero_ok = True
    rendered_complete = bool(re.search(r"\bresult:\s*complete\b", lower)) and not bool(
        re.search(r"\bresult:\s*(?:partial|failed)\b", lower)
    )
    checks = [
        (parsed is not None, "full-backfill completeness audit JSON parses successfully"),
        (audit_ok, "full-backfill completeness audit status is COMPLETE"),
        (unresolved_ok, "full-backfill audit has no unresolved candidate URLs"),
        (critical_ok, "full-backfill audit has no critical failures"),
        (citations_ok, "full-backfill audit has no unsupported citations"),
        (limitations_ok, "full-backfill audit has no remaining limitations"),
        (zero_ok, "zero-contribution result is explicitly audited when qualifying count is zero"),
        (rendered_complete, "rendered report explicitly states RESULT: COMPLETE and not PARTIAL/FAILED"),
    ]
    return checks


def evaluate_smoke(report: str) -> list[tuple[bool, str]]:
    """Validate the dedicated structured audit, including HTML-rendered logs."""
    decoded = html.unescape(report)
    decoded = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", decoded)
    audit = _parse_smoke_audit(decoded)
    # Rich renders each boxed line separately; collapse line wrapping and
    # remove gutters so a phrase split across terminal rows remains readable.
    rendered = decoded.translate(str.maketrans({"│": " ", "║": " ", "┃": " "}))
    lower = re.sub(r"\s+", " ", rendered).casefold()
    aliases = audit.get("tool_aliases", []) if audit else []
    alias_map = {item.get("raw_name"): item for item in aliases if isinstance(item, dict)}
    alias_checks = [
        (alias_map.get(raw, {}).get("logical_name") == logical and alias_map.get(raw, {}).get("authorized") is True,
         f"authorized runtime alias maps {raw} to {logical}")
        for raw, logical in (("web_search", "web.search"), ("web_fetch", "web.fetch"))
    ]
    fields: list[tuple[str, Any, str]] = [
        ("mode", "focused_retrieval_smoke", "focused retrieval smoke mode"),
        ("diagnostic_only", True, "diagnostic-only mode"),
        ("planned_query_count", 1, "exactly one planned query"),
        ("planned_query_statuses", ["succeeded"], "planned query succeeded"),
        ("permission_mapping_valid", True, "runtime tools map to authorized manifest permissions"),
        ("search_status", "search_succeeded", "targeted search succeeded"),
        ("search_probe_matches_plan", True, "search matches explicit plan"),
        ("fetch_status", "fetch_succeeded", "fetch succeeded"),
        ("fetch_schema_validated", True, "fetch schema validated"),
        ("fetch_request_fields", ["urls"], "fetch uses required urls argument"),
        ("fetch_urls_argument_count", 1, "one URL passed to fetch"),
        ("fetch_url_observed_in_search_response", True, "fetch URL was observed in search response"),
        ("fetch_urls_argument_matches_search_result", True, "fetch argument matches search URL"),
        ("fetch_url_matches", True, "parsed fetch URL matches request"),
        ("fetch_title_matches_body", True, "parsed title matches fetched body"),
        ("fetch_body_observed", True, "fetched body observed"),
        ("fetch_parse_error", None, "fetch parser reported no error"),
        ("final_status", "SMOKE_PASS", "structured smoke status is SMOKE_PASS"),
    ]
    checks = [(audit is not None, "machine-readable retrieval smoke audit is present and valid JSON")]
    checks.extend((audit is not None and audit.get(key) == expected, label) for key, expected, label in fields)
    checks.extend((audit is not None and passed, label) for passed, label in alias_checks)
    checks.append(("result: smoke pass" in lower, "rendered report marks RESULT: SMOKE PASS"))
    checks.append(("result: partial" not in lower and "research unavailable" not in lower,
                   "rendered report contains no partial/unavailable result"))
    scope_ok = audit is not None and audit.get("mode") == "focused_retrieval_smoke" \
        and audit.get("diagnostic_only") is True and _has_safe_smoke_scope(lower)
    checks.append((scope_ok,
                   "report states diagnostic-only scope and makes no contribution/impact claim"))
    checks.extend((phrase in lower, f"smoke output includes {phrase}") for phrase in (
        "- retrieved url:", "- page title:", "fetched page excerpt"))
    return checks


def _has_safe_smoke_scope(text: str) -> bool:
    """Require clear limitations and reject positive claims contradicting them."""
    not_full_backfill = bool(re.search(
        r"\b(?:not|is not|isn't|isn't a|not a|not intended as a|not intended to be a)\s+(?:a\s+)?full backfill\b",
        text,
    ))
    no_contribution_impact_claim = any(phrase in text for phrase in (
        "no contribution or impact claim",
        "no contributions or impact claims",
        "does not make a contribution or impact claim",
        "does not make contribution or impact claims",
        "makes no claims about contribution or impact",
        "makes no claims about contributions or impact",
        "does not claim an individual contribution or impact",
        "does not claim individual contributions or impact",
    ))
    positive_backfill = bool(re.search(
        r"\b(?:is|was|completes?|completed|performs?|performed|provides)\s+(?:a\s+)?full backfill\b",
        text,
    ))
    claim_matches = list(re.finditer(
        r"\b(?:claims?|demonstrates?|establishes?)\s+(?:an?\s+)?(?:individual\s+)?(?:contributions?|impact)\b",
        text,
    ))
    positive_claim = any(not re.search(
        r"\b(?:does not|do not|doesn't|don't|never|no)\s+$",
        text[max(0, match.start() - 24):match.start()],
    ) for match in claim_matches)
    contradictory = positive_backfill or positive_claim
    return not_full_backfill and no_contribution_impact_claim and not contradictory


def _parse_smoke_audit(report: str) -> dict | None:
    """Parse an audit object from plain, HTML-escaped, ANSI, or Rich-box output.

    The producer emits a compact object whose allowed success values contain no
    whitespace. Removing wrapping whitespace therefore restores terminal-wrapped
    JSON. If the record is truncated, malformed, or not an object, return None.
    """
    marker = "RETRIEVAL_SMOKE_AUDIT_JSON"
    marker_positions = [match.start() for match in re.finditer(re.escape(marker), report)]
    for position in reversed(marker_positions):
        tail = report[position + len(marker):]
        start = tail.find("{")
        if start < 0:
            continue
        depth = 0
        in_string = False
        escaped = False
        end = None
        for index, char in enumerate(tail[start:], start=start):
            if char in "│║┃":
                continue
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end is None:
            continue
        fragment = tail[start:end]
        fragment = "".join(char for char in fragment if char not in "│║┃")
        fragment = re.sub(r"\s+", "", fragment)
        try:
            candidate = json.loads(fragment)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(candidate, dict):
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List benchmark case IDs and live-run commands")
    parser.add_argument("--case", choices=sorted(load_cases()), help="Benchmark case to check")
    parser.add_argument("--report", type=Path, help="Markdown output saved from an actual agent run")
    parser.add_argument("--smoke", action="store_true", help="Check a smoke report for explicit partial/failure markers")
    parser.add_argument("--require-complete-audit", action="store_true",
                        help="Also require a parsed COMPLETE full-backfill audit with no unresolved candidates/limitations")
    args = parser.parse_args()
    cases = load_cases()
    if args.list:
        for case in cases.values():
            print(f"{case['id']}: {case['github_username']} — {case['title']} [{case['status_on_verified_date']}]")
            print(f"  {case['live_run_command']}")
        if not args.case and not args.report:
            return 0
    if args.smoke:
        if not args.report:
            parser.error("--report is required with --smoke")
        if not args.report.is_file():
            parser.error(f"Report file not found: {args.report}")
        checks = evaluate_smoke(args.report.read_text(encoding="utf-8"))
        for passed, label in checks:
            print(("PASS  " if passed else "FAIL  ") + label)
        return 0 if all(passed for passed, _ in checks) else 1
    if not args.case or not args.report:
        parser.error("--case and --report are both required unless using --list alone")
    if not args.report.is_file():
        parser.error(f"Report file not found: {args.report}")
    case = cases[args.case]
    report = args.report.read_text(encoding="utf-8")
    checks = evaluate(case, report)
    if args.require_complete_audit:
        checks.extend(evaluate_full_backfill_audit(report))
    print(f"Benchmark: {case['id']} — {case['title']}")
    for passed, label in checks:
        if passed:
            print(f"PASS  {label}")
        else:
            category = "UNAVAILABLE TOOL" if any(term in report.casefold() for term in ("http 501", "web.fetch unavailable", "tool unavailable")) else (
                "INCORRECT EVIDENCE" if any(term in label for term in ("contradicted", "incorrect", "overclaim")) else "MISSING EVIDENCE"
            )
            print(f"FAIL [{category}]  {label}")
    failed = sum(not passed for passed, _ in checks)
    print(f"Result: {len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
