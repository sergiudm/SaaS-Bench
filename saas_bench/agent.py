"""browser-use agent wrapper for SaaS-Bench."""

import asyncio
import json
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import urllib.request
from contextvars import ContextVar
from dataclasses import dataclass as _dataclass
from pathlib import Path

import httpx
from browser_use import Agent, Browser, ChatOpenAI
from browser_use.tools.service import Tools
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage
from playwright.async_api import async_playwright
from typing import Any


# All transient per-task data (chrome user-data dirs, agent workdirs) lives under
# $SAAS_BENCH_TMP, defaulting to a stable subdir of the system tempdir so
# `_global_cleanup` in run.py can sweep stale files after a hard kill.
_SLOT_PREFIX = os.environ.get("SAAS_SLOT_PREFIX", "rollout")
_TMP_BASE = os.environ.get(
    "SAAS_BENCH_TMP",
    os.path.join(tempfile.gettempdir(), f"saas_bench_{_SLOT_PREFIX}"),
)
os.makedirs(_TMP_BASE, exist_ok=True)


def _patch_xterm_fill() -> None:
    """Monkey-patch Element.fill to use CDP insertText for xterm.js terminals.

    xterm.js handles both 'keyDown' and 'char' CDP events, causing each
    keystroke to appear twice when browser-use types character-by-character.
    insertText bypasses the keydown chain entirely, so no doubling occurs.
    """
    from browser_use.actor.element import Element

    original_fill = Element.fill

    async def _xterm_aware_fill(self, value: str, clear: bool = True) -> None:
        is_xterm = False
        try:
            result = await self._client.send.DOM.describeNode(
                params={"backendNodeId": self._backend_node_id},
                session_id=self._session_id,
            )
            attrs = result.get("node", {}).get("attributes", [])
            # attributes is a flat list: [name, value, name, value, ...]
            for i in range(0, len(attrs) - 1, 2):
                if attrs[i] == "class" and "xterm" in attrs[i + 1]:
                    is_xterm = True
                    break
        except Exception:
            pass

        if is_xterm:
            # Focus via DOM.focus to avoid triggering xterm mouse handlers
            try:
                await self._client.send.DOM.focus(
                    params={"backendNodeId": self._backend_node_id},
                    session_id=self._session_id,
                )
                await asyncio.sleep(0.02)
            except Exception:
                pass
            await self._client.send.Input.insertText(
                params={"text": value},
                session_id=self._session_id,
            )
        else:
            await original_fill(self, value, clear)

    Element.fill = _xterm_aware_fill  # type: ignore[method-assign]


_patch_xterm_fill()


def _patch_xterm_send_keys() -> None:
    """Monkey-patch DefaultActionWatchdog.on_SendKeysEvent for xterm.js terminals.

    The existing _patch_xterm_fill only covers the `input` action (Element.fill path).
    When the agent uses `send_keys` with plain text (e.g. typing a shell command),
    the text goes through on_SendKeysEvent which dispatches keyDown+char+keyUp per
    character.  xterm.js processes both keyDown and char, causing each character to
    appear twice.

    This patch intercepts plain-text send_keys: if the browser's focused element is
    inside an xterm container, we use CDP Input.insertText instead (same fix as fill).
    Modifier combos (Ctrl+C) and special keys (Enter, Escape) are left untouched.
    """
    from browser_use.browser.watchdogs.default_action_watchdog import DefaultActionWatchdog

    _original_on_send_keys = DefaultActionWatchdog.on_SendKeysEvent

    _SPECIAL_KEYS = frozenset({
        "Enter", "Tab", "Delete", "Backspace", "Escape",
        "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
        "PageUp", "PageDown", "Home", "End",
        "Control", "Alt", "Meta", "Shift",
        "F1", "F2", "F3", "F4", "F5", "F6",
        "F7", "F8", "F9", "F10", "F11", "F12",
    })

    _KEY_ALIASES = {
        "enter": "Enter", "return": "Enter", "tab": "Tab",
        "escape": "Escape", "esc": "Escape", "backspace": "Backspace",
        "delete": "Delete", "space": " ",
        "up": "ArrowUp", "down": "ArrowDown",
        "left": "ArrowLeft", "right": "ArrowRight",
        "pageup": "PageUp", "pagedown": "PageDown",
        "home": "Home", "end": "End",
    }

    async def on_SendKeysEvent(self, event):  # type: ignore[override]
        keys: str = event.keys

        # Key combos (Ctrl+C, etc.) → always use original handler
        if "+" in keys:
            return await _original_on_send_keys(self, event)

        # Normalise single key name
        normalised = _KEY_ALIASES.get(keys.strip().lower(), keys)

        # Special / modifier keys → always use original handler
        if normalised in _SPECIAL_KEYS:
            return await _original_on_send_keys(self, event)

        # Plain text — check whether the focused element lives inside xterm.js
        try:
            cdp_session = await self.browser_session.get_or_create_cdp_session(
                focus=True,
            )
            result = await cdp_session.cdp_client.send.Runtime.evaluate(
                params={
                    "expression": (
                        "!!(document.activeElement && "
                        "document.activeElement.closest('.xterm'))"
                    ),
                },
                session_id=cdp_session.session_id,
            )
            is_xterm = result.get("result", {}).get("value") is True
        except Exception:
            is_xterm = False

        if is_xterm:
            # Bypass keyDown+char and use insertText (no doubling)
            await cdp_session.cdp_client.send.Input.insertText(
                params={"text": keys},
                session_id=cdp_session.session_id,
            )
            self.logger.info(f"⌨️ Sent keys via insertText (xterm): {keys}")
            if "enter" in keys.lower() or "\n" in keys or "\r" in keys:
                await asyncio.sleep(0.1)
            return

        # Not xterm → fall back to original handler
        return await _original_on_send_keys(self, event)

    DefaultActionWatchdog.on_SendKeysEvent = on_SendKeysEvent  # type: ignore[assignment]


_patch_xterm_send_keys()


def _strip_tool_call_wrapper(content: str) -> str:
    """Strip common LLM output decorations so we get raw JSON.

    Handles (in order):
      - ```json ... ``` / ``` ... ``` markdown fences
      - <thinking>...</thinking> blocks (Claude reasoning leaks)
      - <tool_call>/<tool_calls>/<json_schema> XML wrappers
      - Leading natural-language text before the first '{'
      - Trailing characters after the last balanced '}'
    """
    if not content:
        return content

    # 1. Extract from markdown code fence if present.
    fence_match = re.search(r'```(?:json|JSON)?\s*\n?(.*?)\n?```', content, re.DOTALL)
    if fence_match:
        content = fence_match.group(1)

    # 2. Drop <thinking> blocks entirely (content inside is reasoning, not JSON).
    content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL | re.IGNORECASE)

    # 3. Drop XML tag wrappers Claude sometimes emits.
    content = re.sub(
        r'</?(?:tool_calls?|json_schema|response|output|answer)\s*/?>',
        '',
        content,
        flags=re.IGNORECASE,
    )

    # 4. Trim to first '{'.
    idx = content.find('{')
    if idx > 0:
        content = content[idx:]

    # 5. Trim trailing junk after last balanced '}'.
    content = content.strip()
    if content.startswith('{'):
        depth = 0
        last_close = -1
        in_str = False
        escape = False
        for i, ch in enumerate(content):
            if escape:
                escape = False
                continue
            if ch == '\\' and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    last_close = i
                    break
        if last_close > 0 and last_close < len(content) - 1:
            content = content[: last_close + 1]

    return content.strip()


_BANNED_ACTIONS = {"evaluate"}


def _rewrite_banned_actions(text: str) -> str:
    """Rewrite banned actions (e.g. evaluate) to 'think' so Pydantic validation passes."""
    try:
        obj = json.loads(text)
        actions = obj.get("action", [])
        if not isinstance(actions, list):
            return text
        changed = False
        for act in actions:
            if not isinstance(act, dict):
                continue
            for banned in _BANNED_ACTIONS:
                if banned in act:
                    payload = act.pop(banned)
                    act["think"] = (
                        f"{banned} action is disabled. "
                        f"Use click/input/send_keys instead. Wanted: {str(payload)[:200]}"
                    )
                    changed = True
                    break
        if changed:
            return json.dumps(obj, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    return text


# ContextVar that each httpx response hook writes the request id into.
# One slot per async task — safe under concurrent workers.
_current_request_id: ContextVar[str | None] = ContextVar("_current_request_id", default=None)
_current_response_usage: ContextVar[ChatInvokeUsage | None] = ContextVar("_current_response_usage", default=None)


# Header names that providers use for the per-call request id.  Listed in
# preference order; the first non-empty match wins.  Note: httpx normalises
# headers to lowercase, so we compare lower-cased names.
_REQUEST_ID_HEADERS = (
    "http_x_reqid",     # qnaigc / APISIX gateway (literal header name)
    "x-reqid",          # alternative gateway form
    "x-request-id",     # OpenAI / Anthropic standard
    "x-amzn-requestid", # AWS Bedrock
    "request-id",
    "x-tt-logid",       # ByteDance / Volc
)


def _make_request_id_hook():
    """Return an httpx event hook that captures request metadata into ContextVars."""
    async def _capture(response: httpx.Response) -> None:
        for name in _REQUEST_ID_HEADERS:
            rid = response.headers.get(name)
            if rid:
                _current_request_id.set(rid)
                break

        usage = await _usage_from_response(response)
        if usage is not None:
            _current_response_usage.set(usage)
    return _capture


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _usage_from_payload(payload: Any) -> ChatInvokeUsage | None:
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None

    prompt_tokens = _int_or_none(
        usage.get("prompt_tokens", usage.get("input_tokens"))
    )
    completion_tokens = _int_or_none(
        usage.get("completion_tokens", usage.get("output_tokens"))
    )
    total_tokens = _int_or_none(usage.get("total_tokens"))

    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None
    prompt_tokens = prompt_tokens or 0
    completion_tokens = completion_tokens or 0
    total_tokens = total_tokens if total_tokens is not None else prompt_tokens + completion_tokens

    prompt_details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details")
    prompt_cached_tokens = usage.get("prompt_cached_tokens")
    if prompt_cached_tokens is None and isinstance(prompt_details, dict):
        prompt_cached_tokens = prompt_details.get("cached_tokens")

    return ChatInvokeUsage(
        prompt_tokens=prompt_tokens,
        prompt_cached_tokens=_int_or_none(prompt_cached_tokens),
        prompt_cache_creation_tokens=_int_or_none(
            usage.get("prompt_cache_creation_tokens")
            or usage.get("cache_creation_input_tokens")
        ),
        prompt_image_tokens=_int_or_none(usage.get("prompt_image_tokens")),
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


async def _usage_from_response(response: httpx.Response) -> ChatInvokeUsage | None:
    try:
        await response.aread()
        payload = json.loads(response.content.decode(response.encoding or "utf-8"))
    except Exception:
        return None
    return _usage_from_payload(payload)


@_dataclass(frozen=True)
class _ModelPricing:
    input_per_1m_tokens_usd: float
    output_per_1m_tokens_usd: float
    cached_input_per_1m_tokens_usd: float | None = None
    cache_creation_input_per_1m_tokens_usd: float | None = None
    source: str = "unknown"


_BUILTIN_PRICING: dict[str, _ModelPricing] = {
    "claude-opus-4-6": _ModelPricing(5.0, 25.0, 0.50, source="built_in"),
    "claude-opus-4.6": _ModelPricing(5.0, 25.0, 0.50, source="built_in"),
    "claude-sonnet-4-6": _ModelPricing(3.0, 15.0, 0.30, source="built_in"),
    "claude-sonnet-4.6": _ModelPricing(3.0, 15.0, 0.30, source="built_in"),
    "gemini-2.0-flash": _ModelPricing(0.15, 0.60, source="built_in"),
    "gemini-3-flash-preview": _ModelPricing(0.50, 3.00, 0.05, source="built_in"),
    "gemini-3.5-flash": _ModelPricing(1.50, 9.00, source="built_in"),
    "qwen3.6-plus": _ModelPricing(0.33, 1.95, source="built_in"),
    "qwen/qwen3.6-plus": _ModelPricing(0.33, 1.95, source="built_in"),
}


def _float_env(*names: str) -> float | None:
    for name in names:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            continue
        try:
            return float(raw)
        except ValueError:
            continue
    return None


def _pricing_for_model(model_name: str) -> _ModelPricing | None:
    env_input = _float_env("LLM_INPUT_PRICE_PER_1M_TOKENS", "LLM_INPUT_COST_PER_1M_TOKENS")
    env_output = _float_env("LLM_OUTPUT_PRICE_PER_1M_TOKENS", "LLM_OUTPUT_COST_PER_1M_TOKENS")
    if env_input is not None and env_output is not None:
        return _ModelPricing(
            input_per_1m_tokens_usd=env_input,
            output_per_1m_tokens_usd=env_output,
            cached_input_per_1m_tokens_usd=_float_env(
                "LLM_CACHED_INPUT_PRICE_PER_1M_TOKENS",
                "LLM_CACHED_INPUT_COST_PER_1M_TOKENS",
            ),
            cache_creation_input_per_1m_tokens_usd=_float_env(
                "LLM_CACHE_CREATION_INPUT_PRICE_PER_1M_TOKENS",
                "LLM_CACHE_CREATION_INPUT_COST_PER_1M_TOKENS",
            ),
            source="env",
        )

    normalized = model_name.lower()
    if ":" in normalized:
        base, suffix = normalized.rsplit(":", 1)
        if suffix in {"minimal", "low", "medium", "high"}:
            normalized = base
    return _BUILTIN_PRICING.get(normalized)


def _pricing_to_dict(pricing: _ModelPricing | None) -> dict:
    if pricing is None:
        return {
            "source": "unavailable",
            "reason": (
                "Set LLM_INPUT_PRICE_PER_1M_TOKENS and "
                "LLM_OUTPUT_PRICE_PER_1M_TOKENS to enable spending estimates."
            ),
        }
    return {
        "source": pricing.source,
        "input_per_1m_tokens_usd": pricing.input_per_1m_tokens_usd,
        "output_per_1m_tokens_usd": pricing.output_per_1m_tokens_usd,
        "cached_input_per_1m_tokens_usd": pricing.cached_input_per_1m_tokens_usd,
        "cache_creation_input_per_1m_tokens_usd": pricing.cache_creation_input_per_1m_tokens_usd,
    }


def _usage_to_dict(usage: ChatInvokeUsage | None) -> dict | None:
    if usage is None:
        return None
    return {
        "prompt_tokens": usage.prompt_tokens,
        "prompt_cached_tokens": usage.prompt_cached_tokens or 0,
        "prompt_cache_creation_tokens": usage.prompt_cache_creation_tokens or 0,
        "prompt_image_tokens": usage.prompt_image_tokens or 0,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


def _spending_for_usage(usage: ChatInvokeUsage | None, pricing: _ModelPricing | None) -> float | None:
    if usage is None or pricing is None:
        return None

    cached_tokens = max(usage.prompt_cached_tokens or 0, 0)
    uncached_prompt_tokens = max(usage.prompt_tokens - cached_tokens, 0)
    cache_creation_tokens = max(usage.prompt_cache_creation_tokens or 0, 0)
    cached_price = (
        pricing.cached_input_per_1m_tokens_usd
        if pricing.cached_input_per_1m_tokens_usd is not None
        else pricing.input_per_1m_tokens_usd
    )
    cache_creation_price = (
        pricing.cache_creation_input_per_1m_tokens_usd
        if pricing.cache_creation_input_per_1m_tokens_usd is not None
        else pricing.input_per_1m_tokens_usd
    )

    total = (
        uncached_prompt_tokens * pricing.input_per_1m_tokens_usd
        + cached_tokens * cached_price
        + cache_creation_tokens * cache_creation_price
        + usage.completion_tokens * pricing.output_per_1m_tokens_usd
    ) / 1_000_000
    return total


def _empty_llm_stats(model_name: str) -> dict:
    pricing = _pricing_for_model(model_name)
    return {
        "model": model_name,
        "llm_call_count": 0,
        "usage_call_count": 0,
        "prompt_tokens": 0,
        "prompt_cached_tokens": 0,
        "prompt_cache_creation_tokens": 0,
        "prompt_image_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "spending_usd": 0.0,
        "pricing": _pricing_to_dict(pricing),
        "priced_call_count": 0,
        "unpriced_call_count": 0,
        "calls": [],
    }




@_dataclass
class _CleanOutputChatOpenAI(ChatOpenAI):
    """ChatOpenAI that:
    - cleans Claude's decorated output before JSON validation
    - captures the per-call request id (X-Reqid / x-request-id / etc.) from
      the HTTP response headers via an httpx event hook

    request_ids is populated in order: one entry per ainvoke call, None when
    no known header was present.  _extract_trajectory zips these with history
    steps so each trajectory step carries the id of the API call that
    produced it.

    New flow:
      1. Try the parent's normal path (fast happy path).
      2. On any parsing failure, fetch the raw completion (output_format=None),
         clean it, rewrite banned actions, then validate ourselves.
    """

    def __post_init__(self):
        self.request_ids: list[str | None] = []
        self.llm_call_records: list[dict] = []
        self._llm_call_count = 0
        self._pricing = _pricing_for_model(str(self.model))
        # Build a single reusable httpx.AsyncClient with the capture hook so
        # get_client() returns the same transport every call within this instance.
        self._http_client = httpx.AsyncClient(
            event_hooks={"response": [_make_request_id_hook()]},
        )
        # Inject into the dataclass field that ChatOpenAI.get_client() reads.
        object.__setattr__(self, "http_client", self._http_client)

    async def aclose(self) -> None:
        await self._http_client.aclose()

    async def _invoke_and_record(
        self,
        messages,
        output_format,
        call_mode: str,
        **kwargs,
    ) -> tuple[ChatInvokeCompletion, dict]:
        self._llm_call_count += 1
        call_no = self._llm_call_count
        _current_request_id.set(None)
        _current_response_usage.set(None)

        error: str | None = None
        result: ChatInvokeCompletion | None = None
        try:
            result = await super().ainvoke(messages, output_format=output_format, **kwargs)
            return result, self._record_call(call_no, call_mode, result.usage, None)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if result is None:
                self._record_call(call_no, call_mode, _current_response_usage.get(), error)

    def _record_call(
        self,
        call_no: int,
        call_mode: str,
        usage: ChatInvokeUsage | None,
        error: str | None,
    ) -> dict:
        if usage is None:
            usage = _current_response_usage.get()
        usage_dict = _usage_to_dict(usage)
        spending = _spending_for_usage(usage, self._pricing)
        record = {
            "call": call_no,
            "mode": call_mode,
            "request_id": _current_request_id.get(),
            "spending_usd": round(spending, 8) if spending is not None else None,
        }
        if usage_dict is not None:
            record.update(usage_dict)
        if error:
            record["error"] = error[:500]
        self.llm_call_records.append(record)
        return record

    def usage_summary(self, requested_model: str | None = None) -> dict:
        totals = {
            "prompt_tokens": 0,
            "prompt_cached_tokens": 0,
            "prompt_cache_creation_tokens": 0,
            "prompt_image_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        usage_call_count = 0
        spending_total = 0.0
        priced_call_count = 0

        for record in self.llm_call_records:
            if "total_tokens" not in record:
                continue
            usage_call_count += 1
            for key in totals:
                totals[key] += int(record.get(key) or 0)
            if record.get("spending_usd") is not None:
                priced_call_count += 1
                spending_total += float(record["spending_usd"])

        summary = {
            "model": str(self.model),
            "llm_call_count": len(self.llm_call_records),
            "usage_call_count": usage_call_count,
            **totals,
            "spending_usd": (
                round(spending_total, 6)
                if self._pricing is not None or usage_call_count == 0
                else None
            ),
            "pricing": _pricing_to_dict(self._pricing),
            "priced_call_count": priced_call_count,
            "unpriced_call_count": usage_call_count - priced_call_count,
            "calls": self.llm_call_records,
        }
        if requested_model and requested_model != str(self.model):
            summary["requested_model"] = requested_model
        return summary

    async def ainvoke(self, messages, output_format=None, **kwargs) -> Any:
        if output_format is None:
            result, record = await self._invoke_and_record(
                messages, None, "raw", **kwargs
            )
            self.request_ids.append(record.get("request_id"))
            return result

        # Happy path: let parent try first.
        try:
            result, record = await self._invoke_and_record(
                messages, output_format, "structured", **kwargs
            )
            if not isinstance(result.completion, str):
                self.request_ids.append(record.get("request_id"))
                return result
            # Parent returned raw string (e.g. dont_force_structured_output=True path)
            cleaned = _strip_tool_call_wrapper(result.completion)
            cleaned = _rewrite_banned_actions(cleaned)
            parsed = output_format.model_validate_json(cleaned)
            self.request_ids.append(record.get("request_id"))
            return ChatInvokeCompletion(
                completion=parsed,
                usage=result.usage,
                stop_reason=result.stop_reason,
            )
        except Exception:
            # Fall through to recovery: re-fetch as raw string and clean.
            pass

        raw_result, raw_record = await self._invoke_and_record(
            messages, None, "raw_recovery", **kwargs
        )
        raw_completion = raw_result.completion
        if not isinstance(raw_completion, str):
            raw_completion = str(raw_completion)

        cleaned = _strip_tool_call_wrapper(raw_completion)
        cleaned = _rewrite_banned_actions(cleaned)
        parsed = output_format.model_validate_json(cleaned)
        self.request_ids.append(raw_record.get("request_id"))
        return ChatInvokeCompletion(
            completion=parsed,
            usage=raw_result.usage,
            stop_reason=raw_result.stop_reason,
        )


LLM_BASE_URL = os.environ.get("LLM_BASE_URL")
LLM_API_KEY  = os.environ.get("LLM_API_KEY")
if not (LLM_BASE_URL and LLM_API_KEY):
    raise RuntimeError(
        "LLM_BASE_URL and LLM_API_KEY must be set in the environment "
        "(see .env.example for the expected variables)."
    )


TASK_G_RULES = """\
## SaaS-Bench Task-G Operating Rules

You are operating mock SaaS applications running in isolated Docker containers.
The exact access URLs are provided in <user_request> under
"Application Access URLs".

### URL Discipline (CRITICAL)
- NEVER guess or construct URLs from app names, brand names, or documentation.
  Default ports in vendor docs (e.g. bigcapital:8080, hrms:8000) are WRONG here.
- NEVER navigate to URLs like http://hrms.local, http://bigcapital, http://twenty.com.
- ONLY use URLs listed under "Application Access URLs" in the user request.
- For first-time access to an app, use the `navigate` action with the EXACT URL.
- After landing on an app, do NOT use `navigate` again for that app.

### Navigation Discipline (CRITICAL)
- After the initial `navigate` to an app, ALL subsequent navigation within that
  app MUST be done via UI clicks: menu items, sidebar links, breadcrumbs,
  in-page buttons or links.
- Do NOT navigate by typing path URLs like /vendors, /admin/users, /reports/xyz.
- Do NOT call APIs or trigger router changes via JavaScript.
- If a menu is hidden, look for hamburger icons, user dropdowns, expand arrows,
  or scroll the sidebar.
- Exception: only when the page is fully broken (404 / JS error) and the
  navigation menu is unreachable, you may navigate by URL once as a last
  resort, and you MUST record this fallback in `memory`.

### Required Field Handling (CRITICAL)
Task descriptions sometimes omit required fields. When a form requires a field
the task does not specify:
- Fill it with a REASONABLE DEFAULT and proceed:
    * Names/titles    -> derive from task context
    * Dates           -> today's date (see <step_info>)
    * Amounts         -> 0 if neutral, else derive from task numbers
    * Dropdowns       -> first valid option (often "Default" / "General")
    * Email           -> noreply@example.com
    * Phone           -> 0000000000
    * Description     -> short summary of the parent entity
- Do NOT loop searching for the missing info — fill, submit, move on.
- Record the assumption in `memory` so it appears in the trajectory.
### Multi-App Workflow
- Complete ALL subtasks for one app before switching to the next.
- Expect to log in again after switching apps.
- Keep one tab per app; avoid opening duplicate tabs of the same app.

### Data Fidelity
- Use EXACT values from the task: numbers, dates, IDs, names, currency symbols
  (₹, $, €, £). Do NOT round, paraphrase, or substitute symbols.
- Date format must match what the form requires (ISO vs MM/DD/YYYY).
- Copy long strings verbatim from the task description.

### Failure Recovery
- If an action fails 2 times in a row on the same target, switch strategy:
  scroll, switch view, open the parent record, or skip the subtask.
- If a subtask remains stuck after 3 alternative attempts, mark it skipped in
  todo.md and proceed to the next subtask. Do NOT loop indefinitely.
- In a code-server integrated terminal, if characters appear doubled (e.g. "ls" → "llss"), stop typing there and edit/run via the file editor instead.

### Frappe HRMS — Shadow DOM Workaround
Frappe HRMS forms use Shadow DOM components for Link, Select, and Date fields.
Standard UI input often fails on these fields. When a Frappe form field does not
accept typed text after 2 attempts, use this workaround:
1. Open the browser address bar and append `#` to stay on the same page.
2. Use `send_keys` to run this in the browser console (Ctrl+Shift+J):
     cur_frm.set_value('fieldname', 'value');
     cur_frm.save();
   Replace `fieldname` with the actual field API name (lowercase, underscored,
   e.g. `gender`, `date_of_birth`, `department`, `expense_approver`).
   For date fields use ISO format: `cur_frm.set_value('date_of_birth','1995-11-14')`.
3. After save, refresh the page to confirm the value was set.
This bypasses Shadow DOM entirely via Frappe's client-side API.

### Output Format
- Output RAW JSON only. No ```json``` fences, no <thinking> tags, no
  <tool_call>, no XML wrappers of any kind.
- The `evaluate` action is DISABLED. You cannot run JavaScript.
- Available actions (use ONLY these exact names):
    Browser UI:    click, input, scroll, send_keys, select_dropdown,
                   dropdown_options, upload_file, go_back, switch, close,
                   wait, search, navigate, extract, find_elements,
                   find_text, search_page, save_as_pdf
    File system:   read_file, write_file, replace_file
    Completion:    done
- Common parameter reminders (these EXACT names are required):
    click         -> {"index": N}
    input         -> {"index": N, "text": "...", "clear": true}
    scroll        -> {"down": true, "pages": 1.0, "index": N (optional)}
    send_keys     -> {"keys": "Control+a"}  (Escape, Enter, PageDown, Control+o)
    select_dropdown -> {"index": N, "text": "exact option text"}
    dropdown_options -> {"index": N}
    navigate      -> {"url": "...", "new_tab": false}
    switch        -> {"tab_id": "abcd"}  (4-char id)
    close         -> {"tab_id": "abcd"}
    upload_file   -> {"index": N, "path": "/abs/path"}
    go_back       -> {}     (no params)
    wait          -> {"seconds": 3}
    extract       -> {"query": "...", "extract_links": false, "extract_images": false}
    find_text     -> {"text": "..."}
    find_elements -> {"selector": "css selector", "attributes": ["href"]}
    search_page   -> {"pattern": "...", "regex": false, "case_sensitive": false}
    search        -> {"query": "...", "engine": "duckduckgo"}
    read_file     -> {"file_name": "todo.md"}
    write_file    -> {"file_name": "...", "content": "..."}
    replace_file  -> {"file_name": "...", "old_str": "...", "new_str": "..."}
    done          -> {"text": "...", "success": true, "files_to_display": []}
- Input image files listed in <available_file_paths> MUST be read with
  `read_file` (e.g. {"file_name": "/abs/path/image.jpg"}). The image will
  be rendered visually for you. Do NOT use `navigate` to open file:// URLs
  — that is blocked by security policy.
- Parameter name traps — these are WRONG, use the names above:
    extract uses `query` (NOT `instruction`, `prompt`, `task`, `description`)
    find_text uses `text` (NOT `query`, `pattern`, `search`)
    search_page uses `pattern` (NOT `query`, `text`)
    find_elements uses `selector` (NOT `query`, `xpath`, `css`)
    select_dropdown uses `text` (NOT `value`, `option`)
    scroll uses `down` (boolean, NOT `direction`); `pages` (float, NOT `amount`)
- For keyboard shortcuts ALWAYS use `send_keys`. Do NOT invent action
  names like `key_press`, `press_key`, `keyboard`, `type`, `screenshot`,
  `take_screenshot`, `evaluate`, `execute_script`, or `js`.
- `navigate` is reserved for first-time entry to an app URL from the
  "Application Access URLs" list. Do NOT navigate again within the same app.
"""


def _build_llm(model_name: str) -> _CleanOutputChatOpenAI:
    reasoning_effort = None
    if ":" in model_name:
        base, suffix = model_name.rsplit(":", 1)
        if suffix in {"minimal", "low", "medium", "high"}:
            model_name = base
            reasoning_effort = suffix
    kwargs: dict = {}
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    return _CleanOutputChatOpenAI(
        model=model_name,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        timeout=600,
        max_retries=5,
        dont_force_structured_output=True,
        add_schema_to_system_prompt=True,
        frequency_penalty=None,
        **kwargs,
    )


def _free_port() -> int:
    """Pick a random port in 40000-59999 that is not currently in use."""
    for _ in range(100):
        port = random.randint(40000, 59999)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("Could not find a free port for Chrome CDP")


_CHROME_TMP_BASE = os.path.join(_TMP_BASE, "chrome")


def _start_chrome(executable_path: str, port: int) -> tuple[subprocess.Popen, str]:
    user_data = f"{_CHROME_TMP_BASE}_{port}_{int(time.time())}"
    os.makedirs(user_data, exist_ok=True)
    proc = subprocess.Popen(
        [
            executable_path,
            f"--remote-debugging-port={port}",
            f"--user-data-dir={user_data}",
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    # Poll until Chrome CDP is ready (up to 120s)
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=1)
            return proc, user_data
        except Exception:
            time.sleep(0.5)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        proc.kill()
    shutil.rmtree(user_data, ignore_errors=True)
    raise RuntimeError(f"Chrome CDP port {port} not ready after 120s")


async def _kill(proc, browser) -> None:
    if proc:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    if browser:
        try:
            await asyncio.wait_for(browser.close(), timeout=2)
        except Exception:
            pass


def _extract_trajectory(history, request_ids: list[str | None] | None = None) -> list[dict]:
    """Serialize history.history into a list of step dicts for analysis."""
    steps = []
    for i, step in enumerate(history.history):
        entry: dict = {"step": i + 1}

        # Browser state: URL + title
        if hasattr(step, "state") and step.state:
            entry["url"] = getattr(step.state, "url", None)
            entry["title"] = getattr(step.state, "title", None)

        # Agent thought
        if step.model_output and hasattr(step.model_output, "current_state"):
            brain = step.model_output.current_state
            entry["thought"] = {
                "evaluation": getattr(brain, "evaluation_previous_goal", None),
                "memory": getattr(brain, "memory", None),
                "next_goal": getattr(brain, "next_goal", None),
            }

        # Actions taken this step
        if step.model_output and hasattr(step.model_output, "action"):
            actions = []
            for action in step.model_output.action:
                try:
                    actions.append(action.model_dump(exclude_none=True))
                except Exception:
                    actions.append(str(action))
            entry["actions"] = actions

        # Results (errors, extracted content, done signal)
        results = step.result if isinstance(step.result, list) else ([step.result] if step.result else [])
        entry["results"] = [
            {
                k: v for k, v in {
                    "error": getattr(r, "error", None),
                    "extracted_content": getattr(r, "extracted_content", None),
                    "is_done": getattr(r, "is_done", None),
                    "success": getattr(r, "success", None),
                }.items() if v is not None
            }
            for r in results
        ]

        # API request id (x-request-id from HTTP response header)
        if request_ids is not None and i < len(request_ids) and request_ids[i] is not None:
            entry["request_id"] = request_ids[i]

        steps.append(entry)
    return steps


async def run_task(
    task: dict,
    model_name: str,
    prompt: str,
    result_dir: str,
    max_steps: int = 80,
    slot_id: int | None = None,
    todo_md: str | None = None,
    run_idx: int | None = None,
    input_files: list[str] | None = None,
) -> dict:
    """Run one task with browser-use. Returns result dict."""
    task_id = task["task_id"]
    run_suffix = f"_r{run_idx}" if run_idx is not None else ""
    tag = f"[slot {slot_id}][{task_id}{run_suffix}]" if slot_id is not None else f"[{task_id}{run_suffix}]"
    port = _free_port()
    chrome_proc = None
    chrome_user_data = None
    browser = None
    llm: _CleanOutputChatOpenAI | None = None

    # Per-task working dir for browser-use file system (todo.md, results.md, etc.)
    workdir = Path(_TMP_BASE) / f"fs_{task_id}_{port}_{int(time.time())}"
    workdir.mkdir(parents=True, exist_ok=True)
    if todo_md:
        (workdir / "todo.md").write_text(todo_md, encoding="utf-8")

    try:
        async with async_playwright() as p:
            executable_path = p.chromium.executable_path

        chrome_proc, chrome_user_data = _start_chrome(executable_path, port)
        llm = _build_llm(model_name)
        browser = Browser(
            cdp_url=f"http://127.0.0.1:{port}",
            keep_alive=True,
            disable_security=True,
        )

        tools = Tools(exclude_actions=["evaluate"])
        agent = Agent(
            task=prompt,
            llm=llm,
            browser=browser,
            tools=tools,
            use_vision=True,
            generate_gif=False,
            save_conversation_path=None,
            max_failures=5,
            judge=None,
            file_system_path=str(workdir),
            extend_system_message=TASK_G_RULES,
            available_file_paths=input_files or [],
            llm_timeout=150,
        )

        history = await agent.run(max_steps=max_steps)
        raw = history.final_result() or ""
        output = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        trajectory = _extract_trajectory(history, request_ids=llm.request_ids)

        result = {
            "task_id": task_id,
            "status": "completed",
            "agent_output": output,
            "trajectory": trajectory,
            "llm_stats": llm.usage_summary(requested_model=model_name),
        }

    except Exception as e:
        result = {
            "task_id": task_id,
            "status": "error",
            "agent_output": "",
            "trajectory": [],
            "error_steps": [str(e)],
            "llm_stats": (
                llm.usage_summary(requested_model=model_name)
                if llm is not None
                else _empty_llm_stats(model_name)
            ),
        }

    finally:
        await _kill(chrome_proc, browser)
        if llm is not None:
            await llm.aclose()
        if chrome_user_data:
            shutil.rmtree(chrome_user_data, ignore_errors=True)
        shutil.rmtree(workdir, ignore_errors=True)

    Path(result_dir).mkdir(parents=True, exist_ok=True)
    out = Path(result_dir) / f"{task_id}{run_suffix}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"{tag} done — status={result['status']}", flush=True)
    return result
