"""Regression tests: a COMPLETE completeness audit must not be turned into PARTIAL by validator false positives,
and genuine report-quality defects must still produce PARTIAL."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import agent
from core import report_quality_note
from tool_contract import validate_target_report

URL = "https://github.com/fastapi/fastapi/pull/15953"
TITLE = "Document custom headers for JSONL and SSE streams"
SOURCE = f"{TITLE} #15953 Open Deadshot1831 wants to merge fastapi/fastapi. pytest -> 52 passed. {URL}"
TARGET = {
    "requested_url": URL, "url": URL, "title": TITLE, "author": "Deadshot1831", "status": "open",
    "source_text": SOURCE, "retrieval_outcome": "retrieved", "fetch_ok": True,
    "artifact_type": "pull_request", "merged": False, "evidence_timestamp_utc": "2026-10-12T00:00:00",
}
TARGET_PARA = f"**Target:** [{TITLE}]({URL}) — PR #15953 by Deadshot1831, currently open (proposed, not merged)."
SIGNAL = (f"- The author's validation statement: 52 tests passed. These results are author-reported, "
          f"not independently reproduced here. [PR #15953]({URL})")
COMPLETE_AUDIT = {
    "status": "COMPLETE", "limitations": [], "critical_failures": [], "zero_result_is_audited": False,
    "unresolved_candidate_urls": [], "unsupported_citations": [], "qualifying_artifact_count": 1,
}


def _run_audit(report: str, *, with_addendum: bool = True, audit: dict | None = None) -> dict:
    full = report.rstrip() + (agent._target_artifact_addendum(TARGET) if with_addendum else "")
    state = {
        "report": full, "tool_evidence": [SOURCE], "target_evidence": dict(TARGET),
        "normalized_inputs": {"output_focus": "Full backfill", "repositories": ["fastapi/fastapi"],
                              "github_username": "Deadshot1831"},
        "planned_search_queries": [], "planned_search_operations": [], "tool_failures": [],
        "tool_availability_issues": [], "provenance": [], "candidate_urls": [],
    }
    with patch.object(agent, "assess_full_backfill", return_value=audit or COMPLETE_AUDIT):
        return asyncio.run(agent._audit_report(state))


# --- validate_target_report: complete block chosen by completeness, not by first mention -------------

def test_early_brief_url_mention_does_not_hide_complete_target_citation():
    report = f"Scope: see [the PR]({URL}).\n\n{TARGET_PARA}\n"
    assert validate_target_report(URL, TARGET, report) == []


def test_deterministic_addendum_satisfies_target_citation_after_earlier_mention():
    report = f"Scope: see [the PR]({URL}).\n\n## Notes\nNothing else.\n" + agent._target_artifact_addendum(TARGET)
    assert validate_target_report(URL, TARGET, report) == []


def test_target_citation_still_fails_when_no_single_block_has_every_field():
    # URL+title in one block, author+status in another: fields must co-occur in ONE block.
    report = f"[{TITLE}]({URL})\n\nDeadshot1831 opened this; it is open."
    problems = validate_target_report(URL, TARGET, report)
    assert any("missing the observed author" in p for p in problems)
    assert any("missing the observed status" in p for p in problems)


def test_target_citation_fails_when_title_is_wrong_everywhere():
    report = f"**Target:** [Some other title]({URL}) by Deadshot1831, open."
    assert any("missing the observed title" in p for p in validate_target_report(URL, TARGET, report))


def test_target_citation_fails_without_any_url_block():
    problems = validate_target_report(URL, TARGET, "No links at all.")
    assert any("missing the observed URL" in p for p in problems)


# --- report_quality_note: ledger section detection ---------------------------------------------------

def test_h1_title_containing_contribution_ledger_is_not_a_ledger_section():
    report = f"# Contribution Ledger — Deadshot1831\n\nScope note without a link.\n\n## 1. Target\n{TARGET_PARA}\n"
    assert report_quality_note(report, SOURCE) == ""


def test_h1_ledger_heading_with_uncited_list_item_still_warns():
    report = f"# Contribution Ledger\n- Claimed work with no link\n\n## Notes\n{TARGET_PARA}\n"
    assert "contribution-ledger section has no direct artifact link" in report_quality_note(report, SOURCE)


def test_real_ledger_section_without_link_still_warns():
    report = f"# Report\n\n## Contribution ledger\n- Fixed a bug.\n\n## Notes\n{TARGET_PARA}\n"
    assert "contribution-ledger section has no direct artifact link" in report_quality_note(report, SOURCE)


def test_unobserved_cited_url_still_warns():
    report = f"{TARGET_PARA}\n\nSee https://github.com/fastapi/fastapi/pull/99999"
    assert "not found in the web-tool responses" in report_quality_note(report, SOURCE)


# --- end-to-end final status derived from real validations ---------------------------------------------

def test_complete_audit_and_valid_report_yield_complete_even_with_earlier_url_mention():
    report = f"# Contribution Ledger — Deadshot1831\n\nScope: [PR]({URL}).\n\n{TARGET_PARA}\n\n## Signal\n{SIGNAL}\n"
    out = _run_audit(report)
    assert out["report_quality_passed"] is True
    assert "**RESULT: COMPLETE" in out["result"]
    assert "RESULT: PARTIAL" not in out["result"]


def test_complete_audit_does_not_force_complete_when_ledger_item_is_uncited():
    report = f"## Contribution ledger\n- {TARGET_PARA}\n- Reviewed discussion (no link)\n"
    out = _run_audit(report)
    assert out["report_quality_passed"] is False
    assert "RESULT: PARTIAL" in out["result"]
    assert "report-quality gate failed on: report-quality warnings" in out["result"]
    assert "no direct artifact citation" in out["result"]


def test_complete_audit_does_not_force_complete_when_target_block_is_incomplete():
    report = f"Scope: [PR]({URL}) only."
    out = _run_audit(report, with_addendum=False)
    assert out["report_quality_passed"] is False
    assert "target-local citation problems" in out["result"]


def test_non_complete_audit_still_yields_partial_for_otherwise_valid_report():
    audit = {**COMPLETE_AUDIT, "status": "PARTIAL", "limitations": ["one candidate unresolved"]}
    out = _run_audit(f"{TARGET_PARA}\n\n{SIGNAL}\n", audit=audit)
    assert out["report_quality_passed"] is False
    assert "RESULT: PARTIAL" in out["result"]
