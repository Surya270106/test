"""Fail-closed helpers for inspecting and invoking runtime-discovered MCP tools."""
from __future__ import annotations

import json
import asyncio
import re
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit
from collections.abc import Mapping
from typing import Any


CANONICAL_TOOL_NAMES = frozenset({"web.search", "web.fetch"})
# Exact names confirmed in the authenticated Gateway tools/list response. The
# logical values remain the only names authorized by manifest.yaml.
TOOL_NAME_ALIASES: dict[str, str] = {
    "web_search": "web.search",
    "web_fetch": "web.fetch",
}


def _as_json_schema(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    method = getattr(value, "model_json_schema", None)
    if callable(method):
        try:
            schema = method()
        except Exception:
            return None
        return dict(schema) if isinstance(schema, Mapping) else None
    return None


def tool_input_schema(tool: Any) -> dict[str, Any] | None:
    """Read the schema supplied by the live adapter; never synthesize one."""
    candidates = [getattr(tool, "args_schema", None)]
    get_schema = getattr(tool, "get_input_schema", None)
    if callable(get_schema):
        try:
            candidates.append(get_schema())
        except Exception:
            pass
    for candidate in candidates:
        schema = _as_json_schema(candidate)
        if schema is not None:
            return schema
    return None


def normalize_tool_name(raw_name: Any, aliases: Mapping[str, str] | None = None) -> str:
    """Resolve exact canonical names or an explicitly configured observed alias."""
    name = str(raw_name or "").strip()
    mapped = (aliases if aliases is not None else TOOL_NAME_ALIASES).get(name, name)
    return mapped if mapped in CANONICAL_TOOL_NAMES else ""


def match_authorized_tools(tools: list[Any], aliases: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Map authorized logical names to the original runtime tool objects."""
    matched: dict[str, Any] = {}
    for tool in tools:
        logical_name = normalize_tool_name(getattr(tool, "name", ""), aliases)
        if logical_name and logical_name not in matched:
            matched[logical_name] = tool
    return matched


def register_search_query(query: Any, seen_queries: set[str], *, max_queries: int = 4) -> tuple[bool, str]:
    """Enforce a bounded set of distinct non-empty search queries per run."""
    normalized = re.sub(r"\s+", " ", str(query or "")).strip().casefold()
    if not normalized:
        return False, "Search request failed: query is empty."
    if normalized in seen_queries:
        return False, "Search request failed: duplicate query blocked by the per-run limit."
    if len(seen_queries) >= max_queries:
        return False, "Search request failed: the per-run search query limit was reached."
    seen_queries.add(normalized)
    return True, "Search query registered."


def build_focused_smoke_report(probe: Mapping[str, Any]) -> str | None:
    """Build a limited report only from a fully validated one-fetch probe."""
    evidence = probe.get("fetch_evidence")
    if (probe.get("fetch_status") != "fetch_succeeded" or not isinstance(evidence, Mapping)
            or not all(isinstance(evidence.get(key), str) and evidence[key].strip()
                       for key in ("url", "title", "source_text"))):
        return None
    excerpt = evidence["source_text"].strip()[:900]
    quoted_excerpt = "\n".join("> " + line for line in excerpt.splitlines())
    return (
        "## Focused retrieval smoke — limited scope\n\n"
        "This diagnostic-only run validates one targeted search and one fetched public page. It is not a full backfill and makes no contribution or impact claim.\n\n"
        f"- Retrieved URL: {evidence['url']}\n- Page title: {evidence['title']}\n\n"
        f"Fetched page excerpt (untrusted source content):\n\n{quoted_excerpt}"
    )


def focused_smoke_audit_record(
    *, discovery: Mapping[str, Any] | None, probe: Mapping[str, Any] | None,
    planned_queries: list[Mapping[str, Any]] | None, final_status: str,
) -> dict[str, Any]:
    """Return a credential-safe, machine-readable focused retrieval audit."""
    discovery = discovery or {}
    probe = probe or {}
    runtime_tools = discovery.get("tools", [])
    aliases = []
    if isinstance(runtime_tools, list):
        for item in runtime_tools:
            if not isinstance(item, Mapping):
                continue
            raw = str(item.get("raw_name", ""))
            logical = str(item.get("normalized_name", ""))
            aliases.append({"raw_name": raw, "logical_name": logical,
                            "authorized": bool(item.get("matches_allowed_tool"))})
    queries = planned_queries if isinstance(planned_queries, list) else []
    return {
        "mode": "focused_retrieval_smoke",
        "diagnostic_only": True,
        "planned_query_count": len(queries),
        "planned_query_statuses": [str(item.get("status", "not_executed"))
                                   for item in queries if isinstance(item, Mapping)],
        "tool_aliases": aliases,
        "permission_mapping_valid": all(
            item["authorized"] and item["logical_name"] in {"web.search", "web.fetch"}
            for item in aliases if item["raw_name"] in {"web_search", "web_fetch", "web.search", "web.fetch"}
        ) and {item["logical_name"] for item in aliases if item["authorized"]}
        >= {"web.search", "web.fetch"},
        "search_status": str(probe.get("status", "not_run")),
        "search_probe_matches_plan": probe.get("search_probe_matches_plan") is True,
        "fetch_status": str(probe.get("fetch_status", "not_run")),
        "fetch_schema_validated": probe.get("fetch_schema_validated") is True,
        "fetch_request_fields": probe.get("fetch_request_fields", []),
        "fetch_urls_argument_count": probe.get("fetch_urls_argument_count"),
        "fetch_url_observed_in_search_response": probe.get("fetch_url_observed_in_search_response") is True,
        "fetch_urls_argument_matches_search_result": probe.get("fetch_urls_argument_matches_search_result") is True,
        "fetch_url_matches": probe.get("fetch_url_matches") is True,
        "fetch_title_matches_body": probe.get("fetch_title_matches_body") is True,
        "fetch_body_observed": probe.get("fetch_body_observed") is True,
        "fetch_parse_error": probe.get("fetch_parse_error"),
        "final_status": final_status,
    }


def _safe_diagnostic_text(value: Any, private_values: list[str] | None = None, limit: int = 1200) -> str:
    text = str(value or "")
    for secret in sorted((v for v in (private_values or []) if v), key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [REDACTED]", text)
    text = re.sub(r"(?i)(token|api[_-]?key|authorization|password|secret)(\s*[:=]\s*)\S+", r"\1\2[REDACTED]", text)
    return text[:limit]


def describe_runtime_tool(
    tool: Any, aliases: Mapping[str, str] | None = None,
    private_values: list[str] | None = None,
) -> dict[str, Any]:
    """Create a credential-safe catalog diagnostic for one adapter tool."""
    raw_name = str(getattr(tool, "name", "") or "")
    schema = tool_input_schema(tool)
    properties = schema.get("properties", {}) if isinstance(schema, Mapping) else {}
    required = schema.get("required", []) if isinstance(schema, Mapping) else []
    if not isinstance(required, list):
        required = []
    property_details: dict[str, Any] = {}
    if isinstance(properties, Mapping):
        for key, spec in properties.items():
            if isinstance(spec, Mapping):
                property_details[str(key)] = {
                    "type": spec.get("type"),
                    "required": key in required,
                    "description": _safe_diagnostic_text(spec.get("description", ""), private_values, limit=400),
                }
            else:
                property_details[str(key)] = {"type": None, "required": key in required}
    normalized = normalize_tool_name(raw_name, aliases)
    return {
        "raw_name": _safe_diagnostic_text(raw_name, private_values, limit=200),
        "description": _safe_diagnostic_text(getattr(tool, "description", ""), private_values, limit=1200),
        "argument_schema": {
            "type": schema.get("type") if isinstance(schema, Mapping) else None,
            "properties": property_details,
            "required": [str(key) for key in required],
            "schema_available": schema is not None,
        },
        "normalized_name": normalized,
        "matches_allowed_tool": bool(normalized),
    }


def summarize_tool_discovery(
    tools: list[Any], aliases: Mapping[str, str] | None = None,
    private_values: list[str] | None = None,
) -> dict[str, Any]:
    """Summarize a successful tools/list response, including a legitimate empty list."""
    records = [describe_runtime_tool(tool, aliases, private_values) for tool in tools]
    return {
        "status": "available" if records else "empty",
        "tool_count": len(records),
        "tools": records,
        "matched_allowed_names": sorted({record["normalized_name"] for record in records if record["matches_allowed_tool"]}),
    }


def discovery_failure(diagnostic: Mapping[str, Any]) -> str | None:
    """Return a distinct safe explanation for protocol error, empty, or unmatched tools."""
    status = diagnostic.get("status")
    if status == "connection_or_protocol_error":
        error = diagnostic.get("error") or {}
        error_type = error.get("error_type", "UnknownError") if isinstance(error, Mapping) else "UnknownError"
        return f"MCP tools/list failed: {error_type}."
    if status == "empty":
        return "MCP tools/list succeeded but returned zero tools."
    if not diagnostic.get("matched_allowed_names"):
        return "MCP tools/list returned tools, but none matched authorized names web.search or web.fetch."
    return None


def report_passes_audit(
    *, retrieval_error: str = "", quality_note: str = "", problems: list[str] | None = None,
    failures: list[str] | None = None, availability_issues: list[str] | None = None,
) -> bool:
    """Central fail-closed decision used by the graph's final report gate."""
    return not bool(retrieval_error or quality_note or problems or failures or availability_issues)


def safe_discovery_error(exc: BaseException, private_values: list[str] | None = None) -> dict[str, str]:
    """Expose a useful, sanitized connection/protocol error without env values."""
    return {
        "error_type": type(exc).__name__,
        "error_message": _safe_diagnostic_text(exc, private_values, limit=1000),
    }


def build_search_request(tool: Any, query: str) -> tuple[dict[str, Any] | None, str]:
    """Construct a search request from a discovered query-like string field."""
    schema = tool_input_schema(tool)
    if schema is None or not isinstance(schema.get("properties"), Mapping):
        return None, "The live search tool did not expose a readable properties schema."
    properties = schema["properties"]
    query_key = next((key for key, spec in properties.items()
                      if re.sub(r"[^a-z]", "", str(key).casefold()) in {"query", "searchquery", "q"}
                      and isinstance(spec, Mapping) and spec.get("type") == "string"), None)
    if query_key is None:
        return None, "The live search schema has no supported string query field."
    required = schema.get("required", [])
    if not isinstance(required, list) or any(key != query_key for key in required):
        return None, "The live search schema has additional or invalid required arguments."
    return {str(query_key): query}, "Request matches the live search schema."


def build_fetch_request(tool: Any, target_url: str | list[str] | tuple[str, ...]) -> tuple[dict[str, Any] | None, str]:
    """Construct a fetch request only when the live schema proves it is valid.

    The supported request follows the observed Gateway contract: required
    ``urls`` array containing strings. Unknown required fields fail closed.
    """
    schema = tool_input_schema(tool)
    if schema is None:
        return None, "The live tool did not expose a readable input schema."
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return None, "The live tool schema has no properties object."
    urls_property = properties.get("urls")
    if not isinstance(urls_property, Mapping) or urls_property.get("type") != "array":
        return None, "The live tool schema does not declare an array `urls` argument."
    item_schema = urls_property.get("items")
    if not isinstance(item_schema, Mapping) or item_schema.get("type") != "string":
        return None, "The live `urls` schema does not declare string array items."
    required = schema.get("required", [])
    if not isinstance(required, list) or "urls" not in required or any(name != "urls" for name in required):
        return None, "The live tool schema requires additional or invalid arguments."
    targets = [target_url] if isinstance(target_url, str) else list(target_url)
    if not targets or any(not isinstance(url, str) for url in targets):
        return None, "The fetch targets must be a non-empty list of URLs."
    if any(urlsplit(url).scheme != "https" or not urlsplit(url).netloc for url in targets):
        return None, "Every fetch target must be an absolute HTTPS URL."
    return {"urls": targets}, "Request matches the live tool schema."


def normalize_tool_response(value: Any) -> tuple[str, bool]:
    """Convert MCP text/structured content to text and preserve error flags."""
    is_error = getattr(value, "isError", False) is True
    if isinstance(value, str):
        return value.strip(), is_error
    if isinstance(value, Mapping):
        is_error = is_error or value.get("isError") is True
        if "content" in value:
            extra_keys = set(value) - {"content", "isError"}
            if extra_keys:
                try:
                    return json.dumps(value, ensure_ascii=False, sort_keys=True), is_error
                except (TypeError, ValueError):
                    return "", True
            text, nested_error = normalize_tool_response(value["content"])
            return text, is_error or nested_error
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True), is_error
        except (TypeError, ValueError):
            return "", True
    if isinstance(value, (list, tuple)):
        pieces: list[str] = []
        for block in value:
            if isinstance(block, Mapping) and isinstance(block.get("text"), str):
                pieces.append(block["text"].strip())
            elif isinstance(block, str):
                pieces.append(block.strip())
        return "\n".join(piece for piece in pieces if piece), is_error
    structured = getattr(value, "structuredContent", None)
    if isinstance(structured, Mapping):
        blocks = getattr(value, "content", None)
        text, nested_error = normalize_tool_response(blocks) if blocks is not None else ("", False)
        try:
            combined = json.dumps({"content": text, "structuredContent": structured}, ensure_ascii=False, sort_keys=True)
            return combined, is_error or nested_error
        except (TypeError, ValueError):
            return "", True
    artifact = getattr(value, "artifact", None)
    if isinstance(artifact, Mapping):
        blocks = getattr(value, "content", None)
        text, nested_error = normalize_tool_response(blocks) if blocks is not None else ("", False)
        try:
            combined = json.dumps({"content": text, "artifact": artifact}, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            return text, True
        return combined, is_error or nested_error
    blocks = getattr(value, "content", None)
    if blocks is not None:
        text, nested_error = normalize_tool_response(blocks)
        return text, is_error or nested_error
    return "", True


def is_http_501_failure(text: str, *, is_error: bool = False) -> bool:
    """Recognize an explicit HTTP 501 error without treating arbitrary text as success."""
    lowered = (text or "").casefold()
    return is_error or "http 501" in lowered or "status code: 501" in lowered


def is_tool_failure(text: str, *, is_error: bool = False) -> bool:
    """Fail closed on explicit transport/status failures, not page prose."""
    lowered = (text or "").lstrip().casefold()
    return (
        is_error
        or is_http_501_failure(text)
        or bool(re.search(r"\bhttp\s+[45]\d\d\b", lowered[:500]))
        or any(marker in lowered[:300] for marker in ("error:", "request failed", "fetch failed", "timed out"))
    )


async def invoke_tool_payload(tool: Any, request: dict[str, Any], *, timeout_seconds: float = 30) -> tuple[Any, bool, str | None]:
    """Invoke while retaining the unredacted adapter response object for parsing."""
    try:
        task = asyncio.create_task(tool.ainvoke(request))
        done, _ = await asyncio.wait({task}, timeout=timeout_seconds)
        if not done:
            task.cancel()
            task.add_done_callback(_consume_task_exception)
            return None, True, "TimeoutError"
        raw = task.result()
    except Exception as exc:
        return None, True, type(exc).__name__
    text, response_error = normalize_tool_response(raw)
    return raw, response_error or is_tool_failure(text) or not text, None


async def invoke_tool(tool: Any, request: dict[str, Any], *, timeout_seconds: float = 30) -> tuple[str, bool, str | None]:
    """Text-normalizing convenience wrapper for callers that need display text."""
    raw, failed, error_type = await invoke_tool_payload(tool, request, timeout_seconds=timeout_seconds)
    text = normalize_tool_response(raw)[0] if raw is not None else ""
    return text, failed, error_type


def _consume_task_exception(task: asyncio.Task[Any]) -> None:
    """Observe a cancelled/timed-out tool task if it finishes after the caller returns."""
    if task.cancelled():
        return
    try:
        task.exception()
    except (asyncio.CancelledError, Exception):
        pass


def extract_target_fields(response_text: str) -> dict[str, str]:
    """Extract only explicit fields from a structured response or labeled text.

    This is intentionally conservative: unlabelled prose is retained as source
    text but does not establish URL/title/status/author fields.
    """
    fields: dict[str, str] = {}
    try:
        payload = json.loads(response_text)
    except (TypeError, ValueError):
        payload = None

    def visit(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
            return
        if not isinstance(value, Mapping):
            return
        for key, item in value.items():
            normalized_key = re.sub(r"[^a-z]", "", str(key).casefold())
            if isinstance(item, str):
                dest = {
                    "url": "url", "canonicalurl": "url", "sourceurl": "url", "finalurl": "url",
                    "title": "title", "pagetitle": "title", "status": "status", "state": "status",
                    "author": "author", "login": "author", "content": "source_text",
                    "text": "source_text", "excerpt": "source_text",
                }.get(normalized_key)
                if dest and item.strip() and dest not in fields:
                    fields[dest] = item.strip()
            elif isinstance(item, (Mapping, list, tuple)):
                visit(item)

    if payload is not None:
        visit(payload)
    if "source_text" not in fields:
        fields["source_text"] = response_text.strip()
    for line in response_text.splitlines():
        match = re.match(r"^\s*(canonical url|source url|final url|url|page title|title|status|state|author|login)\s*:\s*(.+?)\s*$", line, re.I)
        if not match:
            continue
        label = re.sub(r"\s+", "", match.group(1).casefold())
        dest = {
            "canonicalurl": "url", "sourceurl": "url", "finalurl": "url", "url": "url",
            "pagetitle": "title", "title": "title", "status": "status", "state": "status",
            "author": "author", "login": "author",
        }[label]
        fields.setdefault(dest, match.group(2).strip())
    return fields



def _split_flattened_github_metadata(body: str) -> str:
    """Restore boundaries lost when a provider flattens rendered GitHub pages."""
    text = str(body or "")
    text = re.sub(r"[ \t]+(?=#{1,6}\s)", "\n", text)
    labels = (
        r"state|status|author|login|actor|created(?:\s+at)?|updated(?:\s+at)?|"
        r"repository|number|committed(?:\s+at)?|commit(?:\s+date)?|"
        r"committer date|author date|date|parent|tree|reviewed(?:\s+at)?|"
        r"submitted(?:\s+at)?|commented(?:\s+at)?|published|released|"
        r"merged(?:\s+at)?|title|subject|commit message|artifact type|"
        r"contribution type|association|source branch|target branch|mergeable|commits"
    )
    text = re.sub(
        rf"[ \t]+(?:[-|][ \t]*)?(?=(?:{labels})\s*:)",
        "\n",
        text,
        flags=re.IGNORECASE,
    )
    return text


def _commit_identity_is_body_backed(url: str, body: str, source_metadata: Mapping[str, Any]) -> bool:
    """Recognize a commit page when its own body proves URL identity and metadata.

    Some fetch providers use a generic page title that is not the Git commit
    subject. For commits only, accept that mismatch when the fetched page itself
    contains the URL's SHA (or its first 12 hex characters), author, date, and
    the repository encoded in the URL. This avoids trusting an unrelated title
    while retaining a verifiable, source-backed commit record.
    """
    parts = [part for part in urlsplit((url or "").strip()).path.split("/") if part]
    if len(parts) < 4 or parts[2].casefold() != "commit":
        return False
    sha = parts[3].casefold()
    if not re.fullmatch(r"[0-9a-f]{7,40}", sha):
        return False
    body_lower = (body or "").casefold()
    sha_tokens = {sha}
    if len(sha) >= 12:
        sha_tokens.add(sha[:12])
    if not any(re.search(rf"(?<![0-9a-f]){re.escape(token)}(?![0-9a-f])", body_lower) for token in sha_tokens):
        return False
    expected_repo = f"{parts[0]}/{parts[1]}".casefold()
    observed_repo = str(source_metadata.get("repository") or "").strip().casefold()
    author = str(source_metadata.get("author") or "").strip()
    committed_at = source_metadata.get("committed_at") or source_metadata.get("created_at")
    return bool(author and committed_at and observed_repo == expected_repo)


def _title_matches_body(title: str, body: str, artifact_type: str | None = None) -> bool:
    """Require a title match appropriate to the artifact's rendered body."""
    normalized_title = re.sub(r"\s+", " ", title or "").strip().casefold()
    if not normalized_title or not body.strip():
        return False
    matching_body = _split_flattened_github_metadata(body)
    meaningful: list[str] = []
    commit_message_next = False
    metadata_labels = re.compile(
        r"^(?:state|status|author|login|actor|created|created at|updated|updated at|repository|number|"
        r"committed|committed at|commit|sha|date|parent|tree|reviewed|submitted|commented|published|released)\s*:",
        re.I,
    )
    for raw_line in matching_body.splitlines():
        line = re.sub(r"^\s*(?:[-*+]\s+|#{1,6}\s*)", "", raw_line).strip()
        if not line:
            continue
        if artifact_type == "commit":
            labeled_message = re.match(r"^(?:commit message|subject)\s*:\s*(.*?)\s*$", line, re.I)
            if labeled_message:
                if labeled_message.group(1):
                    meaningful.append(labeled_message.group(1))
                    commit_message_next = False
                else:
                    commit_message_next = True
                continue
        if metadata_labels.match(line):
            continue
        if commit_message_next:
            meaningful.append(line)
            commit_message_next = False
        else:
            meaningful.append(line)
    if artifact_type == "commit":
        # Commit fetch envelopes may render the subject and message body as
        # separate lines (the structured title can correspond to either). An
        # exact complete line is required; unrelated prose or substring matches
        # do not satisfy the integrity check.
        return any(re.sub(r"\s+", " ", line).casefold() == normalized_title for line in meaningful)
    first_lines = [line.strip() for line in matching_body.splitlines() if line.strip()][:3]
    exact_headings = []
    for raw_line in matching_body.splitlines():
        heading = re.match(r"^\s*#{1,6}\s+(.+?)\s*#*\s*$", raw_line)
        if heading:
            exact_headings.append(
                re.sub(r"\s+", " ", heading.group(1)).strip().casefold()
            )
    return (
        normalized_title in exact_headings
        or any(
            re.sub(r"^#{1,6}\s+", "", line).strip().strip("#").strip().casefold()
            == normalized_title
            for line in first_lines
        )
        or bool(first_lines and first_lines[0].casefold() == normalized_title)
    )



def _body_backed_pull_request_title(
    title: str, url: str, body: str, source_metadata: Mapping[str, Any],
) -> str | None:
    """Accept a provider PR-title suffix only when URL and page body agree."""
    parts = [part for part in urlsplit((url or "").strip()).path.split("/") if part]
    if len(parts) < 4 or parts[2].casefold() != "pull":
        return None

    expected_repo = f"{parts[0]}/{parts[1]}"
    number = parts[3]
    match = re.fullmatch(
        rf"(?P<core>.+?)\s+·\s+Pull Request\s+#{re.escape(number)}"
        rf"\s+·\s+{re.escape(expected_repo)}",
        (title or "").strip(),
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    observed_repo = str(source_metadata.get("repository") or "").strip()
    if observed_repo.casefold() != expected_repo.casefold():
        return None

    identity_pattern = (
        rf"(?<![A-Za-z0-9_.-]){re.escape(expected_repo)}"
        rf"\s+#{re.escape(number)}(?!\d)"
    )
    if not re.search(identity_pattern, body or "", flags=re.IGNORECASE):
        return None

    core_title = match.group("core").strip()
    source_title = source_metadata.get("title")
    if not isinstance(source_title, str) or not source_title.strip():
        return None

    normalize_title = lambda value: re.sub(r"\s+", " ", value).strip().casefold()
    if normalize_title(core_title) != normalize_title(source_title):
        return None
    if not _title_matches_body(source_title, body, "pull_request"):
        return None
    return source_title.strip()


def is_search_results_payload(text: str) -> bool:
    """Detect a JSON search-results envelope masquerading as fetched page text."""
    try:
        payload = json.loads((text or "").strip())
    except (TypeError, ValueError):
        return False
    if not isinstance(payload, Mapping):
        return False
    for key in ("search_results", "organic_results", "web_results"):
        value = payload.get(key)
        if isinstance(value, list):
            return True
    results = payload.get("results")
    if isinstance(results, list) and any(
        isinstance(item, Mapping) and any(k in item for k in ("snippet", "link", "url"))
        for item in results
    ):
        return True
    data = payload.get("data")
    if isinstance(data, Mapping) and data is not payload:
        return is_search_results_payload(json.dumps(data, ensure_ascii=False))
    return False


def _metadata_string(item: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, Mapping):
            login = value.get("login") or value.get("name")
            if isinstance(login, str) and login.strip():
                return login.strip()
    return None



def _normalize_source_github_identity(value: str) -> str:
    """Normalize an explicit GitHub login without trusting unrelated links."""
    text = str(value or "").strip()
    match = re.fullmatch(
        r"\[@(.+?)\]\((https?://github\.com/[^)]+)\)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        visible_login = match.group(1).strip()
        profile = urlsplit(match.group(2))
        profile_parts = [part for part in profile.path.split("/") if part]
        if (
            profile.netloc.casefold() == "github.com"
            and len(profile_parts) == 1
            and profile_parts[0].casefold() == visible_login.casefold()
            and not profile.query
            and not profile.fragment
        ):
            return visible_login

    plain_login = text[1:] if text.startswith("@") else text
    if re.fullmatch(r"[A-Za-z0-9_.\-\[\]]+", plain_login):
        return plain_login
    return text


def _source_metadata(body: str, url: str) -> dict[str, Any]:
    """Read explicit labels and headings from fetched GitHub artifact text."""
    fields: dict[str, Any] = {}
    label_map = {
        "state": "status", "status": "status", "review state": "status", "author": "author", "login": "author",
        "actor": "actor", "created": "created_at", "created at": "created_at",
        "committed": "committed_at", "committed at": "committed_at", "commit date": "committed_at",
        "committer date": "committed_at", "author date": "committed_at", "date": "event_date",
        "updated": "updated_at", "updated at": "updated_at", "repository": "repository",
        "reviewed": "reviewed_at", "reviewed at": "reviewed_at", "submitted": "reviewed_at",
        "submitted at": "reviewed_at", "commented": "reviewed_at", "commented at": "reviewed_at",
        "published": "published_at", "released": "published_at", "merged at": "merged_at",
        "merged": "merged", "title": "title", "subject": "title", "commit message": "title",
        "artifact type": "artifact_type", "contribution type": "artifact_type",
    }
    for raw_line in _split_flattened_github_metadata(body).splitlines():
        line = re.sub(r"^\s*(?:[-*+]\s+)", "", raw_line).strip()
        match = re.match(r"^([^:]{1,40})\s*:\s*(.*?)\s*$", line)
        if not match:
            if "title" not in fields:
                heading = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
                if heading:
                    fields["title"] = heading.group(1).strip()
            continue
        label = re.sub(r"\s+", " ", match.group(1).strip().casefold())
        destination = label_map.get(label)
        value = match.group(2).strip()
        if destination == "repository":
            repo_match = re.match(r"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)\b", value)
            if repo_match:
                value = repo_match.group(1)
        if destination and value and destination not in fields:
            fields[destination] = value
    for identity_field in ("author", "actor"):
        identity_value = fields.get(identity_field)
        if isinstance(identity_value, str) and identity_value.strip():
            fields[identity_field] = _normalize_source_github_identity(identity_value)
    if isinstance(fields.get("merged"), str):
        merged_value = fields["merged"].casefold()
        if merged_value in {"true", "yes"}:
            fields["merged"] = True
        elif merged_value in {"false", "no"}:
            fields["merged"] = False
        else:
            fields.pop("merged", None)
    if fields.get("status"):
        fields["status"] = str(fields["status"]).strip().casefold()
    declared_artifact_type = fields.get("artifact_type")
    path = [part for part in urlsplit(url).path.split("/") if part]
    if len(path) >= 4 and path[2].casefold() == "pull":
        inferred_artifact_type = "pull_request"
    elif len(path) >= 4 and path[2].casefold() == "issues":
        inferred_artifact_type = "issue"
    elif len(path) >= 4 and path[2].casefold() in {"commit", "commits"}:
        inferred_artifact_type = "commit"
    elif len(path) >= 5 and path[2].casefold() == "releases":
        inferred_artifact_type = "release"
    elif len(path) >= 4 and path[2].casefold() == "discussions":
        inferred_artifact_type = "discussion"
    else:
        inferred_artifact_type = None
    fields["artifact_type"] = declared_artifact_type or inferred_artifact_type
    event_date = fields.pop("event_date", None)
    if event_date:
        event_field = {
            "commit": "committed_at", "release": "published_at",
            "review": "reviewed_at", "comment": "reviewed_at",
        }.get(str(fields.get("artifact_type", "")), "created_at")
        fields.setdefault(event_field, event_date)
    artifact_kind = str(fields.get("artifact_type", "" )).casefold().replace(" ", "_")
    if artifact_kind in {"review", "comment"} and not fields.get("reviewed_at") and fields.get("created_at"):
        fields["reviewed_at"] = fields["created_at"]
    if artifact_kind == "commit" and not fields.get("committed_at") and fields.get("created_at"):
        fields["committed_at"] = fields["created_at"]
    return fields


def _source_activity_events(body: str, parent_url: str) -> list[dict[str, Any]]:
    """Extract only clearly attributed and dated review/comment timeline events.

    Gateway fetches commonly render GitHub timelines as bold actor lines such
    as ``**octocat** commented on 2026-06-04T16:30:00Z:`` rather than dedicated
    headings. Support that explicit format and compact, dated ``Review by``
    entries while refusing to promote undated or unattributed text.
    """
    lines = (body or "").splitlines()
    events: list[dict[str, Any]] = []
    start: int | None = None
    current_kind: str | None = None
    current_actor: str | None = None
    current_event_at: str | None = None
    in_timeline = False

    def finish(end: int) -> None:
        if start is None or current_kind is None:
            return
        block = lines[start:end]
        block_text = "\n".join(block).strip()
        fields = _source_metadata(block_text, parent_url)
        # Explicit actor/date parsed from the event marker outrank labels that
        # may appear inside user-authored comment text.
        actor = current_actor or fields.get("actor") or fields.get("author")
        event_at = current_event_at or fields.get("reviewed_at") or fields.get("created_at")
        if not actor or not event_at:
            return
        events.append({
            "artifact_type": current_kind,
            "actor": actor,
            "reviewed_at": event_at,
            "url": parent_url,
            "event_id": fields.get("event_id") or f"{current_kind}:{actor}:{event_at}",
            "title": fields.get("title"),
            "body": fields.get("body") or block_text,
            "repository": fields.get("repository"),
            "source_text": block_text,
        })

    def begin(index: int, kind: str, actor: str, event_at: str | None) -> None:
        nonlocal start, current_kind, current_actor, current_event_at
        if start is not None:
            finish(index)
        start = index
        current_kind = kind
        current_actor = actor
        current_event_at = event_at

    iso_timestamp = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})"
    compact_timestamp = r"\d{8}T\d{6}Z"
    for index, line in enumerate(lines):
        if re.match(r"^\s*(?:#{1,6}\s+)?(?:timeline|activity events?)\s*:?\s*$", line, re.I):
            if start is not None and current_kind is not None:
                finish(index)
            start = None
            current_kind = None
            current_actor = None
            current_event_at = None
            in_timeline = True
            continue
        if not in_timeline:
            continue
        # Native markdown rendered by the live Gateway, e.g.:
        # **octocat** commented on 2026-06-04T16:30:00Z:
        timeline = re.match(
            rf"^\s{{0,3}}\*\*([A-Za-z0-9-]+)\*\*\s+(.+?)\s+(?:on|at)\s+({iso_timestamp})\s*:?\s*$",
            line, re.I,
        )
        if timeline:
            actor, action, event_at = timeline.groups()
            action_lower = action.casefold()
            if "comment" in action_lower:
                begin(index, "comment", actor, event_at)
                continue
            if any(term in action_lower for term in ("review", "approved", "request changes", "requested changes")):
                begin(index, "review", actor, event_at)
                continue

        # Some Gateway pages render review activity as a compact timeline row:
        # - Review by octocat: approved 20260604T163000Z
        review_row = re.match(
            rf"^\s*(?:[-*+]\s*)?Review by @?([A-Za-z0-9-]+)\s*:\s*(.*?)(?:\s+({iso_timestamp}|{compact_timestamp}))?\s*$",
            line, re.I,
        )
        if review_row:
            actor, details, event_at = review_row.groups()
            if not event_at:
                timestamp_match = re.search(rf"\b({iso_timestamp}|{compact_timestamp})\b", details)
                event_at = timestamp_match.group(1) if timestamp_match else None
            if event_at and re.fullmatch(compact_timestamp, event_at):
                event_at = datetime.strptime(event_at, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
            begin(index, "review", actor, event_at)
            continue

        heading = re.match(r"^\s{0,3}#{1,6}\s+(review|comment)(?:\s+event)?(?:\s+by\s+@?([A-Za-z0-9-]+))?\s*$", line, re.I)
        if heading:
            kind, actor = heading.group(1).casefold(), heading.group(2)
            begin(index, kind, actor or "", None)
            continue
        if start is not None and current_kind is None:
            event_line = re.match(r"^\s*(review|comment)(?:\s+event)?(?:\s+by\s+@?([A-Za-z0-9-]+))?\s*:?\s*$", line, re.I)
            if event_line:
                current_kind = event_line.group(1).casefold()
                current_actor = event_line.group(2)
                current_event_at = None
    if start is not None and current_kind is not None:
        finish(len(lines))
    return events


def _activity_events(item: Mapping[str, Any], body: str, parent_url: str) -> list[dict[str, Any]]:
    """Combine explicit structured event records with clearly delimited text events."""
    events = _source_activity_events(body, parent_url)
    raw_events = item.get("events")
    if not isinstance(raw_events, list):
        return events
    for raw in raw_events:
        if not isinstance(raw, Mapping):
            continue
        kind_value = str(raw.get("artifact_type") or raw.get("type") or raw.get("event") or "").casefold()
        kind = "review" if "review" in kind_value else "comment" if "comment" in kind_value else None
        actor = _metadata_string(raw, "actor", "user", "author", "login")
        event_at = _metadata_string(raw, "reviewed_at", "submitted_at", "commented_at", "created_at", "date")
        if not kind or not actor or not event_at:
            continue
        url = _metadata_string(raw, "url", "html_url", "source_url") or parent_url
        body_text = _metadata_string(raw, "body", "text", "content")
        events.append({
            "artifact_type": kind,
            "actor": actor,
            "reviewed_at": event_at,
            "url": url,
            "event_id": _metadata_string(raw, "event_id", "id", "node_id") or f"{kind}:{actor}:{event_at}",
            "title": _metadata_string(raw, "title") or raw.get("state"),
            "body": body_text or json.dumps(dict(raw), ensure_ascii=False, sort_keys=True),
            "repository": _metadata_string(raw, "repository", "repo"),
            "source_text": body_text or json.dumps(dict(raw), ensure_ascii=False, sort_keys=True),
        })
    # A structured payload may repeat an event already rendered in the body.
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for event in events:
        key = (str(event.get("artifact_type")), str(event.get("event_id")))
        unique.setdefault(key, event)
    return list(unique.values())


def parse_fetch_response(response: Any, expected_url: str) -> dict[str, Any]:
    """Parse the observed MCP fetch envelope and validate its actual page result.

    ``response`` is the normalized MCP response text produced by
    ``invoke_tool``; its JSON envelope is retained intact for URL comparisons.
    Diagnostic redaction must happen only after this function.
    """
    raw_text, response_error = normalize_tool_response(response)
    try:
        payload = json.loads(raw_text)
    except (TypeError, ValueError):
        return {"fetch_ok": False, "fetch_error": "malformed_response", "source_text": raw_text}
    if not isinstance(payload, Mapping):
        return {"fetch_ok": False, "fetch_error": "malformed_response", "source_text": raw_text}

    # Adapter ToolMessage envelopes can hold JSON text in content/artifact;
    # prefer actual structured payloads, otherwise unwrap content JSON.
    outer: Mapping[str, Any] = payload
    if "data" not in outer and isinstance(outer.get("structuredContent"), Mapping):
        outer = outer["structuredContent"]
    if "data" not in outer and isinstance(outer.get("artifact"), Mapping):
        artifact = outer["artifact"]
        if isinstance(artifact.get("structured_content"), Mapping):
            outer = artifact["structured_content"]
        elif "data" in artifact or "results" in artifact:
            outer = artifact
    content = outer.get("content")
    if "data" not in outer and isinstance(content, str):
        try:
            nested = json.loads(content)
            if isinstance(nested, Mapping):
                outer = nested
        except ValueError:
            pass

    if response_error:
        return {"fetch_ok": False, "fetch_error": "tool_error", "source_text": raw_text}
    if outer.get("error") not in (None, False, ""):
        return {"fetch_ok": False, "fetch_error": "response_error", "source_text": raw_text}
    if outer.get("successful") is False or outer.get("success") is False:
        return {"fetch_ok": False, "fetch_error": "unsuccessful_response", "source_text": raw_text}

    data = outer.get("data", outer)
    if not isinstance(data, Mapping):
        return {"fetch_ok": False, "fetch_error": "malformed_data", "source_text": raw_text}
    statuses = data.get("statuses", outer.get("statuses", []))
    if statuses is not None:
        if not isinstance(statuses, list):
            return {"fetch_ok": False, "fetch_error": "malformed_statuses", "source_text": raw_text}
        for entry in statuses:
            if not isinstance(entry, Mapping):
                return {"fetch_ok": False, "fetch_error": "malformed_status_entry", "source_text": raw_text}
            state = str(entry.get("status", "")).casefold()
            if entry.get("successful") is False or entry.get("success") is False or entry.get("error") not in (None, False, ""):
                return {"fetch_ok": False, "fetch_error": "failed_status", "source_text": raw_text}
            if state and state not in {"ok", "success", "succeeded", "complete", "completed", "200"}:
                return {"fetch_ok": False, "fetch_error": "failed_status", "source_text": raw_text}

    results = data.get("results")
    if not isinstance(results, list) or not results:
        return {"fetch_ok": False, "fetch_error": "empty_results" if results == [] else "malformed_results", "source_text": raw_text}
    if any(not isinstance(item, Mapping) for item in results):
        return {"fetch_ok": False, "fetch_error": "malformed_result_item", "source_text": raw_text}

    expected = _canonical_fetch_url(expected_url)
    candidates: list[tuple[Mapping[str, Any], str]] = []
    for item in results:
        url_value = item.get("url")
        if not isinstance(url_value, str) or not url_value.strip():
            identifier = item.get("id")
            url_value = identifier if isinstance(identifier, str) and urlsplit(identifier).scheme in {"http", "https"} else ""
        if url_value:
            candidates.append((item, url_value))
    chosen = next(((item, url) for item, url in candidates if _canonical_fetch_url(url) == expected), None)
    if chosen is None:
        mismatch_item, mismatch_url = candidates[0] if candidates else ({}, "")
        return {
            "fetch_ok": False,
            "fetch_error": "url_mismatch" if mismatch_url else "missing_url",
            "url": mismatch_url,
            "title": str(mismatch_item.get("title") or ""),
            "source_text": str(mismatch_item.get("text") or mismatch_item.get("content") or ""),
        }

    item, actual_url = chosen
    title = item.get("title")
    body = item.get("text", item.get("content"))
    if not isinstance(body, str) or not body.strip():
        return {"fetch_ok": False, "fetch_error": "empty_body", "url": actual_url,
                "title": str(title or ""), "source_text": ""}
    source_metadata = _source_metadata(body, actual_url)
    if not isinstance(title, str) or not title.strip():
        title = source_metadata.get("title")
    if not isinstance(title, str) or not title.strip():
        return {"fetch_ok": False, "fetch_error": "missing_title", "url": actual_url, "source_text": str(body or "")}
    if is_search_results_payload(body):
        return {"fetch_ok": False, "fetch_error": "nested_search_payload", "url": actual_url,
                "title": title.strip(), "source_text": ""}
    artifact_type = _metadata_string(item, "artifact_type", "contribution_type", "type") or source_metadata.get("artifact_type")
    if not _title_matches_body(title, body, str(artifact_type or "")):
        provider_pr_title = _body_backed_pull_request_title(
            title, actual_url, body, source_metadata,
        )
        if provider_pr_title:
            title = provider_pr_title
        elif str(artifact_type or "").casefold().replace(" ", "_") == "commit" and _commit_identity_is_body_backed(actual_url, body, source_metadata):
            # The exact commit URL, SHA, author, date, and repository are all
            # grounded in fetched page text; replace an unrelated provider/page
            # title with a neutral identifier rather than carrying the mismatch.
            path = [part for part in urlsplit(actual_url).path.split("/") if part]
            sha = path[3]
            source_title = source_metadata.get("title")
            title = source_title.strip() if isinstance(source_title, str) and source_title.strip() and _title_matches_body(source_title, body, "commit") else f"Commit {sha[:12]}"
        else:
            return {"fetch_ok": False, "fetch_error": "title_body_mismatch", "url": actual_url, "title": title.strip(), "source_text": body.strip()}
    author = _metadata_string(item, "author", "login", "user") or source_metadata.get("author")
    actor = _metadata_string(item, "actor") or source_metadata.get("actor") or author
    status = _metadata_string(item, "status", "state") or source_metadata.get("status")
    metadata = {
        "artifact_type": _metadata_string(item, "artifact_type", "contribution_type", "type") or source_metadata.get("artifact_type"),
        "created_at": _metadata_string(item, "created_at", "created", "date", "authored_at") or source_metadata.get("created_at"),
        "committed_at": _metadata_string(item, "committed_at", "commit_date", "authored_at") or source_metadata.get("committed_at"),
        "merged_at": _metadata_string(item, "merged_at") or source_metadata.get("merged_at"),
        "published_at": _metadata_string(item, "published_at", "published", "released_at") or source_metadata.get("published_at"),
        "reviewed_at": _metadata_string(item, "reviewed_at", "submitted_at", "commented_at") or source_metadata.get("reviewed_at"),
        "updated_at": _metadata_string(item, "updated_at", "updated") or source_metadata.get("updated_at"),
        "repository": _metadata_string(item, "repository", "repo") or source_metadata.get("repository"),
    }
    metadata["artifact_type"] = artifact_type
    normalized_type = re.sub(r"[^a-z]", "", str(artifact_type or "").casefold())
    if normalized_type in {"review", "comment"} and not metadata["reviewed_at"]:
        metadata["reviewed_at"] = metadata["created_at"]
    if normalized_type == "commit" and not metadata["committed_at"]:
        metadata["committed_at"] = metadata["created_at"]
    merged = item.get("merged") if isinstance(item.get("merged"), bool) else source_metadata.get("merged")
    if status is None and merged is True:
        status = "merged"
    structured_values = {
        "author": _metadata_string(item, "author", "login", "user"),
        "actor": _metadata_string(item, "actor"),
        "status": _metadata_string(item, "status", "state"),
        "created_at": _metadata_string(item, "created_at", "created", "date", "authored_at"),
        "committed_at": _metadata_string(item, "committed_at", "commit_date", "authored_at"),
        "repository": _metadata_string(item, "repository", "repo"),
    }
    conflicts: list[str] = []
    for field, structured_value in structured_values.items():
        source_value = source_metadata.get(field)
        if structured_value and source_value:
            left, right = structured_value.strip(), str(source_value).strip()
            if field in {"author", "actor", "status", "repository"}:
                same = left.casefold() == right.casefold()
            elif field.endswith("_at"):
                try:
                    left_dt = datetime.fromisoformat(left.replace("Z", "+00:00"))
                    right_dt = datetime.fromisoformat(right.replace("Z", "+00:00"))
                    if left_dt.tzinfo:
                        left_dt = left_dt.astimezone(timezone.utc)
                    if right_dt.tzinfo:
                        right_dt = right_dt.astimezone(timezone.utc)
                    same = left_dt == right_dt
                except ValueError:
                    same = left == right
            else:
                same = left == right
            if not same:
                conflicts.append(field)
    if conflicts:
        return {"fetch_ok": False, "fetch_error": "metadata_conflict", "metadata_conflicts": conflicts,
                "url": actual_url.strip(), "title": title.strip(), "source_text": body.strip()}
    return {
        "url": actual_url.strip(),
        "title": title.strip(),
        "source_text": body.strip(),
        "author": author,
        "actor": actor,
        "status": status,
        "merged": merged if isinstance(merged, bool) else None,
        "metadata_conflicts": [],
        "activity_events": _activity_events(item, body, actual_url),
        **metadata,
        "fetch_ok": True,
        "fetch_error": None,
    }


def _canonical_url(url: str) -> str:
    parts = urlsplit((url or "").strip())
    if not parts.scheme or not parts.netloc:
        return ""
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), "", ""))


def _canonical_fetch_url(url: str) -> str:
    """Compare fetch URLs conservatively, preserving query and path identity."""
    parts = urlsplit((url or "").strip())
    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
        return ""
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), parts.query, ""))


def validate_target_report(expected_url: str, evidence: Mapping[str, str], report: str) -> list[str]:
    """Deterministically require retrieved provenance fields in the report."""
    problems: list[str] = []
    actual_url = evidence.get("url", "")
    title = evidence.get("title", "")
    status = evidence.get("status", "")
    author = evidence.get("author", "")
    source_text = evidence.get("source_text", "")
    if not source_text.strip():
        problems.append("retrieved source text is missing")
    if not actual_url:
        problems.append("retrieval response did not identify its canonical URL")
    elif _canonical_url(actual_url) != _canonical_url(expected_url):
        problems.append("retrieved canonical URL does not match the requested artifact")
    if not title:
        problems.append("retrieved source title is missing")
    if not status:
        problems.append("observed source status is missing")
    if not author:
        problems.append("observed source author is missing")
    for field_name, value in (("title", title), ("status", status), ("author", author)):
        if value and value.casefold() not in source_text.casefold():
            problems.append(f"observed {field_name} is not present in retrieved source text")
    required_fields = (("URL", expected_url), ("title", title), ("status", status), ("author", author))

    def missing_fields(paragraph: str) -> list[str]:
        return [name for name, value in required_fields if value and value.casefold() not in paragraph]

    # Every required field must sit together in ONE block that cites the target.
    # The block is chosen by completeness, not by being the first URL mention: an
    # earlier brief mention (e.g. an executive summary) must not hide a complete
    # target-local citation, while a report with no single complete block still fails.
    best_missing = [name for name, value in required_fields if value]
    if actual_url:
        for block in re.split(r"\n\s*\n", report or ""):
            lowered = block.casefold()
            if actual_url.casefold() in lowered or expected_url.casefold() in lowered:
                missing = missing_fields(lowered)
                if len(missing) < len(best_missing):
                    best_missing = missing
                if not best_missing:
                    break
    for field_name in best_missing:
        problems.append(f"target-local report citation is missing the observed {field_name}")
    return problems


def sanitize_tool_arguments(arguments: Any, private_values: list[str] | None = None) -> dict[str, Any]:
    """Keep useful request shape while redacting credentials and supplied user data."""
    if not isinstance(arguments, Mapping):
        return {"arguments": "[unavailable]"}
    private = [value for value in (private_values or []) if isinstance(value, str) and value]
    result: dict[str, Any] = {}
    for key, value in arguments.items():
        lowered = str(key).casefold()
        if any(marker in lowered for marker in ("token", "auth", "cookie", "password", "secret", "api_key")):
            result[str(key)] = "[REDACTED]"
        elif isinstance(value, str):
            cleaned = value
            for private_value in sorted(private, key=len, reverse=True):
                cleaned = re.sub(re.escape(private_value), "[USER_INPUT]", cleaned, flags=re.I)
            cleaned = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [REDACTED]", cleaned)
            cleaned = re.sub(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", "[EMAIL]", cleaned, flags=re.I)
            result[str(key)] = cleaned[:500]
        elif isinstance(value, (int, float, bool)) or value is None:
            result[str(key)] = value
        else:
            result[str(key)] = "[omitted]"
    return result
