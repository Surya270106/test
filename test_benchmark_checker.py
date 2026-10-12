from __future__ import annotations

import importlib.util
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_benchmark_report.py"
spec = importlib.util.spec_from_file_location("check_benchmark_report", SCRIPT)
checker = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(checker)


def test_benchmark_checker_passes_good_open_pr_report():
    case = checker.load_cases()["fastapi-15641"]
    report = """# Contribution Ledger
## Contribution ledger
- **Fix jsonable_encoder crash on invalid UTF-8 bytes** — Open PR authored by SAURABHSALVE.
  Proposed invalid UTF-8 handling using decode errors=replace and a regression test. This is open and not merged.
  Source: https://github.com/fastapi/fastapi/pull/15641
"""
    assert all(passed for passed, _ in checker.evaluate(case, report))


def test_smoke_checker_rejects_partial_or_failed_retrieval_reports():
    audit = {
        "mode": "focused_retrieval_smoke", "diagnostic_only": True,
        "planned_query_count": 1, "planned_query_statuses": ["succeeded"],
        "tool_aliases": [
            {"raw_name": "web_search", "logical_name": "web.search", "authorized": True},
            {"raw_name": "web_fetch", "logical_name": "web.fetch", "authorized": True},
        ],
        "permission_mapping_valid": True, "search_status": "search_succeeded",
        "search_probe_matches_plan": True, "fetch_status": "fetch_succeeded",
        "fetch_schema_validated": True, "fetch_request_fields": ["urls"],
        "fetch_urls_argument_count": 1, "fetch_url_observed_in_search_response": True,
        "fetch_urls_argument_matches_search_result": True, "fetch_url_matches": True,
        "fetch_title_matches_body": True, "fetch_body_observed": True,
        "fetch_parse_error": None, "final_status": "SMOKE_PASS",
    }
    valid_smoke = """RESULT: SMOKE PASS
This run is diagnostic-only, not a full backfill, and makes no contribution or impact claim.
- Retrieved URL: https://github.com/octocat/Hello-World
- Page title: README
Fetched page excerpt: # README
RETRIEVAL_SMOKE_AUDIT_JSON: """ + html.escape(json.dumps(audit, separators=(",", ":")))
    passed_complete = checker.evaluate_smoke(valid_smoke)
    assert all(passed for passed, _ in passed_complete)
    assert not all(passed for passed, _ in checker.evaluate_smoke("A report lacking an audit result."))
    for report in (
        "**RESULT: PARTIAL — evidence or retrieval validation failed.**",
        "Research unavailable because fetch_failed.",
        "retrieval probe failed",
    ):
        assert not all(passed for passed, _ in checker.evaluate_smoke(report))


def test_smoke_checker_requires_every_structured_field_and_success_value():
    audit = {
        "mode": "focused_retrieval_smoke", "diagnostic_only": True,
        "planned_query_count": 1, "planned_query_statuses": ["succeeded"],
        "tool_aliases": [
            {"raw_name": "web_search", "logical_name": "web.search", "authorized": True},
            {"raw_name": "web_fetch", "logical_name": "web.fetch", "authorized": True},
        ],
        "permission_mapping_valid": True, "search_status": "search_succeeded",
        "search_probe_matches_plan": True, "fetch_status": "fetch_succeeded",
        "fetch_schema_validated": True, "fetch_request_fields": ["urls"],
        "fetch_urls_argument_count": 1, "fetch_url_observed_in_search_response": True,
        "fetch_urls_argument_matches_search_result": True, "fetch_url_matches": True,
        "fetch_title_matches_body": True, "fetch_body_observed": True,
        "fetch_parse_error": None, "final_status": "SMOKE_PASS",
    }
    body = "RESULT: SMOKE PASS\nnot a full backfill; no contribution or impact claim\n- Retrieved URL: x\n- Page title: y\nFetched page excerpt\n"
    for key, failed_value in (("fetch_schema_validated", False), ("fetch_status", "fetch_failed"),
                              ("fetch_url_matches", False), ("fetch_parse_error", "invalid"),
                              ("planned_query_statuses", ["failed"])):
        broken = dict(audit)
        broken[key] = failed_value
        report = body + "RETRIEVAL_SMOKE_AUDIT_JSON: " + json.dumps(broken)
        assert not all(passed for passed, _ in checker.evaluate_smoke(report)), key

    missing = dict(audit)
    missing.pop("fetch_body_observed")
    report = body + "RETRIEVAL_SMOKE_AUDIT_JSON: " + json.dumps(missing)
    assert not all(passed for passed, _ in checker.evaluate_smoke(report))


def test_smoke_checker_parses_ansi_html_and_rich_wrapped_audit_capture():
    audit = {
        "mode": "focused_retrieval_smoke", "diagnostic_only": True,
        "planned_query_count": 1, "planned_query_statuses": ["succeeded"],
        "tool_aliases": [
            {"raw_name": "web_search", "logical_name": "web.search", "authorized": True},
            {"raw_name": "web_fetch", "logical_name": "web.fetch", "authorized": True},
        ],
        "permission_mapping_valid": True, "search_status": "search_succeeded",
        "search_probe_matches_plan": True, "fetch_status": "fetch_succeeded",
        "fetch_schema_validated": True, "fetch_request_fields": ["urls"],
        "fetch_urls_argument_count": 1, "fetch_url_observed_in_search_response": True,
        "fetch_urls_argument_matches_search_result": True, "fetch_url_matches": True,
        "fetch_title_matches_body": True, "fetch_body_observed": True,
        "fetch_parse_error": None, "final_status": "SMOKE_PASS",
    }
    compact = json.dumps(audit, separators=(",", ":"))
    wrapped = "\n".join("│ " + compact[i:i + 23] + " │" for i in range(0, len(compact), 23))
    report = (
        "\x1b[36m╭─ live report ─╮\x1b[0m\n"
        "│ RESULT: SMOKE PASS\n"
        "│ This diagnostic-only run validates one targeted search and one fetched public page.\n"
        "│ It is not a full backfill and makes no contribution or impact claim.\n"
        "│ - Retrieved URL: https://github.com/octocat/Hello-World\n"
        "│ - Page title: README\n"
        "│ Fetched page excerpt: # README\n"
        "│ RETRIEVAL_SMOKE_AUDIT_JSON:\n"
        + html.escape(wrapped)
        + "\n╰───────────────╯\n"
    )
    assert all(passed for passed, _ in checker.evaluate_smoke(report))


def test_smoke_scope_requires_valid_audit_and_clear_nonclaim_limitations():
    audit = {
        "mode": "focused_retrieval_smoke", "diagnostic_only": True,
        "planned_query_count": 1, "planned_query_statuses": ["succeeded"],
        "tool_aliases": [
            {"raw_name": "web_search", "logical_name": "web.search", "authorized": True},
            {"raw_name": "web_fetch", "logical_name": "web.fetch", "authorized": True},
        ],
        "permission_mapping_valid": True, "search_status": "search_succeeded",
        "search_probe_matches_plan": True, "fetch_status": "fetch_succeeded",
        "fetch_schema_validated": True, "fetch_request_fields": ["urls"],
        "fetch_urls_argument_count": 1, "fetch_url_observed_in_search_response": True,
        "fetch_urls_argument_matches_search_result": True, "fetch_url_matches": True,
        "fetch_title_matches_body": True, "fetch_body_observed": True,
        "fetch_parse_error": None, "final_status": "SMOKE_PASS",
    }
    prefix = "RESULT: SMOKE PASS\n- Retrieved URL: x\n- Page title: y\nFetched page excerpt\n"
    audit_line = "RETRIEVAL_SMOKE_AUDIT_JSON: " + json.dumps(audit)

    def scope_passes(report: str) -> bool:
        return next(passed for passed, label in checker.evaluate_smoke(report)
                    if label.startswith("report states diagnostic-only scope"))

    exact_transcript_paragraph = (
        "This run validates one targeted search and one fetched public page. "
        "It is not a full backfill and makes no contribution or impact claim."
    )
    assert scope_passes(prefix + exact_transcript_paragraph + "\n" + audit_line)

    equivalent = (
        "Focused diagnostic only: one targeted search and one fetched page, "
        "not intended as a full backfill; it does not claim individual contributions or impact."
    )
    assert scope_passes(prefix + equivalent + "\n" + audit_line)

    missing_limitations = "This is a diagnostic-only run that fetches a public page."
    assert not scope_passes(prefix + missing_limitations + "\n" + audit_line)

    contradictory = exact_transcript_paragraph + " This run also completes a full backfill and claims impact."
    assert not scope_passes(prefix + contradictory + "\n" + audit_line)

    no_audit = prefix + exact_transcript_paragraph
    assert not scope_passes(no_audit)
    false_audit = dict(audit, diagnostic_only=False)
    assert not scope_passes(prefix + exact_transcript_paragraph + "\n"
                            + "RETRIEVAL_SMOKE_AUDIT_JSON: " + json.dumps(false_audit))


def test_smoke_checker_rejects_malformed_truncated_and_contradictory_audits():
    for audit_text in (
        '{"mode":"focused_retrieval_smoke"',
        '{"mode":"focused_retrieval_smoke","final_status":"SMOKE_PASS","fetch_status":"fetch_failed"}',
    ):
        report = "RESULT: SMOKE PASS\nRETRIEVAL_SMOKE_AUDIT_JSON: " + audit_text
        assert not all(passed for passed, _ in checker.evaluate_smoke(report))

    # A contradictory later record must not be hidden by an earlier passing one.
    base = {
        "mode": "focused_retrieval_smoke", "diagnostic_only": True,
        "planned_query_count": 1, "planned_query_statuses": ["succeeded"],
        "tool_aliases": [
            {"raw_name": "web_search", "logical_name": "web.search", "authorized": True},
            {"raw_name": "web_fetch", "logical_name": "web.fetch", "authorized": True},
        ],
        "permission_mapping_valid": True, "search_status": "search_succeeded",
        "search_probe_matches_plan": True, "fetch_status": "fetch_succeeded",
        "fetch_schema_validated": True, "fetch_request_fields": ["urls"],
        "fetch_urls_argument_count": 1, "fetch_url_observed_in_search_response": True,
        "fetch_urls_argument_matches_search_result": True, "fetch_url_matches": True,
        "fetch_title_matches_body": True, "fetch_body_observed": True,
        "fetch_parse_error": None, "final_status": "SMOKE_PASS",
    }
    contradictory = dict(base, final_status="PARTIAL")
    report = (
        "RESULT: SMOKE PASS\nnot a full backfill; no contribution or impact claim\n"
        "- Retrieved URL: x\n- Page title: y\nFetched page excerpt\n"
        "RETRIEVAL_SMOKE_AUDIT_JSON: " + json.dumps(base) + "\n"
        "RETRIEVAL_SMOKE_AUDIT_JSON: " + json.dumps(contradictory)
    )
    assert not all(passed for passed, _ in checker.evaluate_smoke(report))


def test_benchmark_checker_fails_if_open_pr_is_claimed_as_shipped():
    case = checker.load_cases()["fastapi-15641"]
    report = """- **Fix jsonable_encoder crash on invalid UTF-8 bytes** by SAURABHSALVE.
  This was merged into master and shipped in FastAPI.
  https://github.com/fastapi/fastapi/pull/15641
"""
    checks = checker.evaluate(case, report)
    landed_check = [passed for passed, label in checks if "status does not contradict" in label]
    assert landed_check == [False]


def test_benchmark_checker_requires_qualification_for_self_reported_benchmark():
    case = checker.load_cases()["fastapi-16159"]
    report = """# Skip included routers whose prefix cannot match the request path
- Author: Kludex. Open proposal. https://github.com/fastapi/fastapi/pull/16159
- 24.6x faster routing and 19.1x faster 404.
"""
    checks = checker.evaluate(case, report)
    assert any(not passed and "author-reported/unreproduced" in label for passed, label in checks)



def test_pydantic_release_negative_control_requires_attribution_caveat():
    case = checker.load_cases()["pydantic-v2-13-0-release-context"]
    report = """# Pydantic v2.13.0
Viicos published the release, which proves Viicos authored every change.
https://github.com/pydantic/pydantic/releases/tag/v2.13.0
"""
    checks = checker.evaluate(case, report)
    attribution_check = [passed for passed, label in checks if "target-local caveat" in label]
    assert attribution_check == [False]



def test_closed_unmerged_pr_cannot_be_claimed_shipped():
    case = checker.load_cases()["fastapi-4089-closed-unmerged"]
    report = """# Fix issue with double model parsing
- Author: uriyyo. This closed PR was merged into master and shipped.
https://github.com/fastapi/fastapi/pull/4089
"""
    checks = checker.evaluate(case, report)
    landed_check = [passed for passed, label in checks if "status does not contradict" in label]
    assert landed_check == [False]


def test_release_caveat_elsewhere_cannot_replace_missing_target_artifact():
    case = checker.load_cases()["pydantic-v2-13-0-release-context"]
    report = """Pydantic's publisher does not establish that it authored every change.

https://github.com/pydantic/pydantic/pull/13013
"""
    checks = checker.evaluate(case, report)
    assert not next(passed for passed, label in checks if "target URL" in label)
    assert not next(passed for passed, label in checks if "exact target title" in label)
    assert not next(passed for passed, label in checks if "target-local caveat" in label)


def test_case_specific_metrics_must_be_local_to_the_cited_target():
    case = checker.load_cases()["fastapi-16159"]
    report = """# Skip included routers whose prefix cannot match the request path

Author: Kludex. Open proposal. https://github.com/fastapi/fastapi/pull/16159

Unrelated benchmark: 24.6x and 19.1x.
"""
    checks = checker.evaluate(case, report)
    metric_checks = [passed for passed, label in checks if "target-specific required evidence" in label]
    assert metric_checks == [False, False]


def test_open_target_with_target_local_merged_claim_fails():
    case = checker.load_cases()["fastapi-15641"]
    report = """# Fix jsonable_encoder crash on invalid UTF-8 bytes

SAURABHSALVE; open PR. The target was merged and shipped.
https://github.com/fastapi/fastapi/pull/15641
"""
    checks = checker.evaluate(case, report)
    assert not next(passed for passed, label in checks if "status does not contradict" in label)


def test_wrong_artifact_url_is_reported_as_incorrect_not_missing():
    case = checker.load_cases()["fastapi-15641"]
    report = "Fix jsonable_encoder crash on invalid UTF-8 bytes by SAURABHSALVE; open proposal.\nhttps://github.com/fastapi/fastapi/pull/15953"
    check = next((passed, label) for passed, label in checker.evaluate(case, report) if "target URL" in label)
    assert not check[0]
    assert "incorrect target URL" in check[1]


def test_pydantic_cross_environment_caveat_does_not_excuse_production_overclaim():
    case = checker.load_cases()["pydantic-13013-merged"]
    report = """Allow default factories of private attributes to take validated model data — Viicos.
    Merged into main; 83 checks passed. The benchmark is cross-environment, and production performance improved 30.07%.
    https://github.com/pydantic/pydantic/pull/13013
    """
    checks = checker.evaluate(case, report)
    metric = next(passed for passed, label in checks if "not claimed as a production" in label or "not presented as a verified production" in label)
    assert not metric


def test_benchmark_checker_handles_rich_wrapped_target_block_and_open_outcome_caveat():
    case = checker.load_cases()["fastapi-15641"]
    target = (
        "**Exact fetched target:** [Fix jsonable_encoder crash on invalid UTF-8 bytes] "
        "(https://github.com/fastapi/fastapi/pull/15641); Observed title: "
        "Fix jsonable_encoder crash on invalid UTF-8 bytes; Displayed author/login: SAURABHSALVE; "
        "Observed status: open; source says decode errors=replace and uses a replacement character. "
        "Outcome limit: this is an open proposal; the fetched evidence does not establish a merge, shipment, or release."
    )
    wrapped = "\n".join("│ " + target[index:index + 29] + " │" for index in range(0, len(target), 29))
    report = "╭─ result ─╮\n" + wrapped + "\n╰──────────╯\n"
    checks = checker.evaluate(case, report)
    failures = [label for passed, label in checks if not passed]
    assert not failures, failures


def test_benchmark_checker_keeps_exact_title_and_author_target_local_after_rich_wrapping():
    case = checker.load_cases()["fastapi-15641"]
    report = """│ ## Report
│ The title Fix jsonable_encoder crash on invalid UTF-8 bytes belongs to another artifact.
│ SAURABHSALVE is discussed elsewhere in the report.
│ Source page did not contain the requested artifact URL.
│
│ Target lookup: https://github.com/fastapi/fastapi/pull/15953
│ Document custom headers for JSONL and SSE streams; author Deadshot1831; open.
│
│ **Exact target:** https://github.com/fastapi/fastapi/pull/15641
│
"""
    checks = checker.evaluate(case, report)
    assert not next(passed for passed, label in checks if "exact target title" in label)
    assert not next(passed for passed, label in checks if "target-local author" in label)


def test_full_backfill_audit_checker_accepts_html_and_rich_wrapped_complete_audit():
    audit = {
        "status": "COMPLETE",
        "unresolved_candidate_urls": [],
        "critical_failures": [],
        "unsupported_citations": [],
        "limitations": [],
        "qualifying_artifact_count": 0,
        "zero_result_is_audited": True,
    }
    encoded = html.escape(json.dumps(audit, indent=2))
    audit_lines = "\n".join("│ " + line for line in encoded.splitlines())
    report = (
        "╭─ result ─╮\n"
        "│ ### Full-backfill completeness audit\n"
        "│ <pre>\n" + audit_lines + "\n│ </pre>\n"
        "│ **RESULT: COMPLETE — full-backfill evidence and coverage checks passed.**\n"
        "╰──────────╯\n"
    )
    checks = checker.evaluate_full_backfill_audit(report)
    assert all(passed for passed, _ in checks), [label for passed, label in checks if not passed]


def test_full_backfill_audit_checker_rejects_partial_or_incomplete_zero_result():
    audit = {
        "status": "PARTIAL",
        "unresolved_candidate_urls": ["https://github.com/fastapi/fastapi/pull/123"],
        "critical_failures": [],
        "unsupported_citations": [],
        "limitations": ["candidate not fetched"],
        "qualifying_artifact_count": 0,
        "zero_result_is_audited": False,
    }
    report = (
        "### Full-backfill completeness audit\n<pre>" + json.dumps(audit) + "</pre>\n"
        "**RESULT: PARTIAL — evidence or retrieval validation failed.**"
    )
    checks = checker.evaluate_full_backfill_audit(report)
    assert not all(passed for passed, _ in checks)
    assert not next(passed for passed, label in checks if "status is COMPLETE" in label)
    assert not next(passed for passed, label in checks if "zero-contribution result" in label)


# --- target-linked evidence for fastapi-15953 ------------------------------------------------------------

_P_URL = "https://github.com/fastapi/fastapi/pull/15953"
_P_TITLE = "Document custom headers for JSONL and SSE streams"
_P_SUMMARY = (f"**Target:** [{_P_TITLE}]({_P_URL}) — PR #15953 by Deadshot1831, currently open "
              "(proposed, not merged).")
_P_SIGNAL = ("## Engineering and collaboration signal\n- The PR page reports `+263 -0 in 6 files`. The author's "
             "validation statement: 52 tests passed, Ruff checks passed. These results are author-reported, not "
             f"independently reproduced here. [PR #15953]({_P_URL})")
_P_ADDENDUM = (f"### Exact target artifact verification\n\n**Exact fetched target:** [{_P_TITLE}]({_P_URL}); "
               f"Observed title: {_P_TITLE}; Displayed author/login: Deadshot1831; Observed status: open; "
               "Verbatim fetched source excerpt (untrusted page text): \"pytest ... 52 passed\"; Evidence limit: "
               "Test counts are source-reported and not independently verified unless separately checked.")
_TEST_LABEL = "52 passed / 52 tests passed / 52 passing"
_QUAL_LABEL = "test count is distinguished from separately verified CI"


def _p_report(*blocks: str) -> str:
    return "# Contribution Ledger\n\n" + "\n\n".join(blocks) + "\n"


def _p_status(report: str) -> dict[str, bool]:
    case = checker.load_cases()["fastapi-15953"]
    return {label: passed for passed, label in checker.evaluate(case, report)}


def _p_evidence_checks(report: str) -> tuple[bool, bool]:
    results = _p_status(report)
    count = next(v for k, v in results.items() if _TEST_LABEL in k)
    qual = next(v for k, v in results.items() if _QUAL_LABEL in k)
    return count, qual


def test_fastapi_15953_test_evidence_found_in_later_target_linked_sections():
    report = _p_report(_P_SUMMARY, "Filler paragraph.", _P_SIGNAL, _P_ADDENDUM)
    assert all(_p_status(report).values())


def test_fastapi_15953_test_evidence_accepted_from_signal_section_alone():
    assert _p_evidence_checks(_p_report(_P_SUMMARY, _P_SIGNAL)) == (True, True)


def test_fastapi_15953_test_evidence_accepted_from_addendum_alone():
    assert _p_evidence_checks(_p_report(_P_SUMMARY, _P_ADDENDUM)) == (True, True)


def test_fastapi_15953_missing_test_count_still_fails():
    signal = _P_SIGNAL.replace("52 tests passed", "all tests passed")
    addendum = _P_ADDENDUM.replace("52 passed", "tests ran")
    count, _ = _p_evidence_checks(_p_report(_P_SUMMARY, signal, addendum))
    assert count is False


def test_fastapi_15953_unqualified_test_claim_still_fails():
    signal = ("## Signal\n- Validation: 52 tests passed. [PR #15953](" + _P_URL + ")")
    count, qual = _p_evidence_checks(_p_report(_P_SUMMARY, signal))
    assert count is True and qual is False


def test_fastapi_15953_qualifier_must_be_in_same_block_as_the_count():
    count_only = f"- Validation: 52 tests passed. [PR #15953]({_P_URL})"
    qualifier_only = f"- The author's claims are author-reported and unverified. [PR #15953]({_P_URL})"
    count, qual = _p_evidence_checks(_p_report(_P_SUMMARY, count_only, qualifier_only))
    assert count is True and qual is False


def test_fastapi_15953_unrelated_artifact_cannot_supply_test_count_or_qualification():
    other = ("## Other work\n- A different PR reports 52 tests passed; this is author-reported. "
             "[PR #15954](https://github.com/fastapi/fastapi/pull/15954)")
    assert _p_evidence_checks(_p_report(_P_SUMMARY, other)) == (False, False)


def test_fastapi_15953_unanchored_paragraph_cannot_supply_test_count():
    unanchored = "## Elsewhere\n- 52 tests passed, author-reported."
    assert _p_evidence_checks(_p_report(_P_SUMMARY, unanchored)) == (False, False)


def test_fastapi_15953_block_mixing_target_and_other_artifact_is_not_accepted():
    mixed = (f"- [PR #15953]({_P_URL}) and [PR #15954](https://github.com/fastapi/fastapi/pull/15954): "
             "52 tests passed, author-reported.")
    assert _p_evidence_checks(_p_report(_P_SUMMARY, mixed)) == (False, False)


def test_fastapi_15953_longer_pr_number_url_is_not_the_target():
    wrong = ("- 52 tests passed, author-reported. [PR](https://github.com/fastapi/fastapi/pull/159530)")
    assert _p_evidence_checks(_p_report(_P_SUMMARY, wrong)) == (False, False)


def test_fastapi_15953_repo_and_number_anchor_without_url_is_accepted():
    block = "- In fastapi/fastapi, PR #15953 reports 52 tests passed; this is author-reported."
    assert _p_evidence_checks(_p_report(_P_SUMMARY, block)) == (True, True)


def test_fastapi_15953_independent_ci_success_claim_fails_qualification():
    claim = f"- All 52 CI checks passed and tests were author-reported. [PR #15953]({_P_URL})"
    _, qual = _p_evidence_checks(_p_report(_P_SUMMARY, _P_SIGNAL, claim))
    assert qual is False


def test_fastapi_15953_negated_ci_statement_does_not_fail_qualification():
    note = f"- No independent CI result was verified; 52 tests passed is author-reported. [PR #15953]({_P_URL})"
    assert _p_evidence_checks(_p_report(_P_SUMMARY, note)) == (True, True)


def _p_check(report: str, fragment: str) -> bool:
    case = checker.load_cases()["fastapi-15953"]
    matches = [passed for passed, label in checker.evaluate(case, report) if fragment in label]
    assert len(matches) == 1, (fragment, matches)
    return matches[0]


def test_fastapi_15953_primary_identity_checks_remain_strict_with_good_later_sections():
    # A valid later signal section cannot rescue a bad primary target block. (The machine-generated
    # addendum is deliberately omitted here: it carries the observed identity itself and is a valid primary.)
    wrong_title = f"**Target:** [Some other title]({_P_URL}) by Deadshot1831, open."
    no_author = f"**Target:** [{_P_TITLE}]({_P_URL}) is open."
    no_status = f"**Target:** [{_P_TITLE}]({_P_URL}) by Deadshot1831."
    merged = f"**Target:** [{_P_TITLE}]({_P_URL}) by Deadshot1831 was merged into main and shipped."

    assert _p_check(_p_report(wrong_title, _P_SIGNAL), "exact target title") is False
    assert _p_check(_p_report(no_author, _P_SIGNAL), "author attribution") is False
    assert _p_check(_p_report(no_status, _P_SIGNAL), "open/proposed status") is False
    assert _p_check(_p_report(merged, _P_SIGNAL), "does not contradict open PR") is False
    # Control: the valid primary block passes all four identity/status checks.
    good = _p_report(_P_SUMMARY, _P_SIGNAL, _P_ADDENDUM)
    for fragment in ("exact target title", "author attribution", "open/proposed status", "does not contradict open PR"):
        assert _p_check(good, fragment) is True
