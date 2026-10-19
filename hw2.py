#!/usr/bin/env python3
"""FTEC5660 HW2 student starter: build an agent that verifies CVs via MCP."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


MCP_URL = "https://ftec5660.ngrok.app/mcp"
MODEL_NAME = "deepseek-v4-flash"
THRESHOLD = 0.5


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def cv_files(folder: Path) -> list[Path]:
    """Return PDFs directly inside *folder*, sorted numerically (CV_2 before CV_10)."""

    def key(path: Path) -> tuple[int, str]:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return (int(digits) if digits else math.inf, path.name)

    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"),
        key=key,
    )


def cv_text(path: Path) -> str:
    """Convert one CV PDF to markdown text."""
    from markitdown import MarkItDown

    return MarkItDown(enable_plugins=False).convert(str(path)).text_content


async def load_mcp_tools() -> list[Any]:
    """Connect to the course MCP server and return its tools as LangChain tools."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "social_graph": {
                "transport": "http",
                "url": MCP_URL,
                "headers": {"ngrok-skip-browser-warning": "true"},
            }
        }
    )
    return await client.get_tools()


import difflib
import os
import sys
import time

MAX_PROFILES = 10
TOOL_TIMEOUT = 25
TOOL_ATTEMPTS = 2
CV_BUDGET = 150
GLOBAL_BUDGET = 1200
SCORE_STRONG_VALID = 0.96
SCORE_VALID = 0.9
SCORE_UNCLEAR = 0.6
SCORE_UNRESOLVED = 0.22
SCORE_DISCREPANCY = 0.07
FIT_FLOOR = 0.45
FIT_CONFIDENT = 0.78
DEBUG = bool(os.environ.get("HW2_DEBUG"))

_PUNCT_RE = re.compile(r"[^a-z0-9/+]+")

EXTRACT_SYSTEM = (
    "You convert one CV into a strict JSON record. You copy every value exactly "
    "as it appears in the CV: you never invent, normalise or 'fix' anything. "
    "You answer with one JSON object and nothing else."
)

EXTRACT_PROMPT = """CV text (plain text extracted from a PDF, possibly a mangled markdown table):
---
{cv}
---

Return ONE JSON object with exactly these keys:
{{
  "name": "full name shown on the CV",
  "city": "the candidate's city",
  "country": "the candidate's country",
  "hometown": "the hometown, or null",
  "headline": "the one-line self description, e.g. 'Logistics Professional', or null",
  "jobs": [{{"company": str, "title": str, "start_year": int|null, "end_year": int|null, "is_current": bool}}],
  "education": [{{"school": str, "degree": str, "field": str|null, "grad_year": int|null}}],
  "skills": [str]
}}

Rules:
- The text may be a broken markdown table: read across rows and cells, ignore empty cells, "|" characters and dashed lines.
- "Present", "Current" or "Now" as an end date means end_year is null and is_current is true.
- "2020 - Present" gives start_year 2020; "2015 - 2019" gives 2015 and 2019.
- "Graduated 2016" or a bare year next to the school is grad_year 2016.
- "BSc in Design" gives degree "BSc" and field "Design"; "Master of Business Administration in Logistics" gives degree "Master of Business Administration" and field "Logistics".
- Keep the job title words exactly, including any "Senior"/"Junior"/"Lead" prefix.
- skills: every item listed under a Skills section, exactly as written.
- Use null (or [] for the lists) when a section is absent."""

JUDGE_SYSTEM = (
    "You are a meticulous compliance reviewer comparing a CV claim against a "
    "candidate's LinkedIn profile. You answer with one JSON object only."
)

JUDGE_PROMPT = """Rules of this comparison:
- Word-only differences are NOT discrepancies: "Bachelor of Science" vs "BSc", "Master of Business Administration" vs "MBA", "UI/UX Design" vs "UI/UX", "PowerPoint" vs "Microsoft PowerPoint", a partial skill list (listing fewer skills than the profile), and "Senior Engineer" for an "Engineer" role whose seniority is senior.
- Real discrepancies ARE: an inflated job title or seniority, shifted employment or graduation years, an upgraded degree, a fake school or employer, a wrong city, or a skill the candidate does not have.

CV claim: {cv}
LinkedIn profile value: {profile}

Decide whether this difference is only wording (allowed) or a real discrepancy.
Reply with ONE JSON object: {{"allowed": true|false, "reason": "short justification"}}"""


def _dbg(*args: Any) -> None:
    if DEBUG:
        print("[hw2]", *args, file=sys.stderr, flush=True)


# --- small text helpers ----------------------------------------------------

def _norm(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).lower().replace("&", " and ")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", " ", _PUNCT_RE.sub(" ", text)).strip()


def _tokens(value: Any) -> list[str]:
    return _norm(value).split()


def _similar(a: Any, b: Any, threshold: float = 0.85) -> bool:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    return difflib.SequenceMatcher(None, na, nb).ratio() >= threshold


_CITY_ALIASES = {
    "hk": "hong kong", "hongkong": "hong kong", "nyc": "new york",
    "sf": "san francisco", "sg": "singapore", "hcmc": "ho chi minh city",
    "kl": "kuala lumpur", "bkk": "bangkok", "syd": "sydney", "ldn": "london",
}


def _city_key(value: Any) -> str:
    text = _norm(value)
    return _CITY_ALIASES.get(text, text)


def _city_match(a: Any, b: Any) -> bool:
    ka, kb = _city_key(a), _city_key(b)
    if not ka or not kb:
        return False
    return ka == kb or ka in kb or kb in ka or _similar(ka, kb, 0.9)


_COUNTRY_ALIASES = {
    "uk": "united kingdom", "usa": "united states", "us": "united states",
    "hksar": "hong kong", "prc": "china",
}


def _country_match(a: Any, b: Any) -> bool:
    ka = _COUNTRY_ALIASES.get(_norm(a), _norm(a))
    kb = _COUNTRY_ALIASES.get(_norm(b), _norm(b))
    if not ka or not kb:
        return False
    return ka == kb or ka in kb or kb in ka


_INDUSTRY_SKIP = {"professional", "specialist", "expert", "consultant", "practitioner"}


def _industry(value: Any) -> str:
    return " ".join(t for t in _tokens(value) if t not in _INDUSTRY_SKIP)


def _name_match(a: Any, b: Any) -> bool:
    ka, kb = _norm(a).replace(" ", ""), _norm(b).replace(" ", "")
    if not ka or not kb:
        return False
    if ka == kb:
        return True
    if difflib.SequenceMatcher(None, ka, kb).ratio() >= 0.9:
        return True
    # nicknames such as "Kel" for "Kelly": same surname and a shared first-name prefix
    ta, tb = _norm(a).split(), _norm(b).split()
    if len(ta) >= 2 and len(tb) >= 2:
        if ta[-1] == tb[-1] and ta[0][:3] == tb[0][:3]:
            return True
    return False


_DEGREE_LEVELS = (
    ("phd", 3), ("doctor", 3), ("doctorate", 3), ("dphil", 3),
    ("master", 2), ("msc", 2), ("mba", 2), ("meng", 2), ("mphil", 2), ("ma", 2), ("ms", 2),
    ("bachelor", 1), ("bsc", 1), ("beng", 1), ("bcom", 1), ("llb", 1), ("ba", 1), ("bs", 1),
    ("associate", 0.5), ("diploma", 0.5),
)


def _degree_level(value: Any) -> float | None:
    toks = _tokens(value)
    if not toks:
        return None
    for token, level in _DEGREE_LEVELS:
        if token in toks:
            return level
    return None


def _degree_subtype(value: Any) -> str | None:
    toks = set(_tokens(value))
    if {"mba"} & toks or {"business", "administration"} <= toks:
        return "mba"
    if {"phd", "doctor", "doctorate", "dphil"} & toks:
        return "phd"
    if {"msc", "master", "ms", "mphil"} & toks:
        return "msc"
    if {"bsc", "bachelor", "bs"} & toks:
        return "bsc"
    if {"beng", "meng"} & toks:
        return "eng"
    return None


_FIELD_ALIASES = {
    "ai": "artificial intelligence", "ml": "machine learning",
    "cs": "computer science", "hr": "human resources",
    "fin": "finance", "econ": "economics",
}


def _field_match(a: Any, b: Any) -> bool:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return True
    na = _FIELD_ALIASES.get(na, na)
    nb = _FIELD_ALIASES.get(nb, nb)
    if na == nb or na in nb or nb in na:
        return True
    ta, tb = set(na.split()), set(nb.split())
    if ta and tb and len(ta & tb) / min(len(ta), len(tb)) >= 0.6:
        return True
    return _similar(na, nb, 0.85)


_SCHOOL_ALIASES = {
    "hku": "university of hong kong",
    "the university of hong kong": "university of hong kong",
    "hkust": "hong kong university of science and technology",
    "the hong kong university of science and technology":
        "hong kong university of science and technology",
    "cuhk": "chinese university of hong kong",
    "the chinese university of hong kong": "chinese university of hong kong",
    "polyu": "hong kong polytechnic university",
    "the hong kong polytechnic university": "hong kong polytechnic university",
    "cityu": "city university of hong kong",
    "nus": "national university of singapore",
    "ntu": "nanyang technological university",
    "mit": "massachusetts institute of technology",
    "ucl": "university college london",
    "imperial": "imperial college london",
    "lse": "london school of economics",
}


def _school_key(value: Any) -> str:
    text = _norm(value)
    if text.startswith("the "):
        text = text[4:]
    return _SCHOOL_ALIASES.get(text, _SCHOOL_ALIASES.get(_norm(value), text))


def _school_match(a: Any, b: Any) -> bool:
    ka, kb = _school_key(a), _school_key(b)
    if not ka or not kb:
        return False
    return ka == kb or ka in kb or kb in ka or _similar(ka, kb, 0.88)


_SKILL_STOP = {
    "design", "microsoft", "advanced", "basic", "professional", "management",
    "engineering", "development", "tools", "skills", "using", "with",
}


def _skill_key(value: Any) -> str:
    return _norm(value)


def _skill_tokens(value: Any) -> set[str]:
    toks = _tokens(value)
    kept = [t for t in toks if t not in _SKILL_STOP]
    return set(kept or toks)


def _skill_match(a: Any, b: Any) -> bool:
    ka, kb = _skill_key(a), _skill_key(b)
    if ka and ka == kb:
        return True
    ta, tb = _skill_tokens(a), _skill_tokens(b)
    if not ta or not tb:
        return False
    if ta == tb:
        return True
    return len(ta & tb) / len(ta | tb) >= 0.75


def _split_skills(values: list[str]) -> list[str]:
    """'Docker, Prototyping' is two skills, not one: split the list separators."""
    out: list[str] = []
    for value in values:
        for part in re.split(r"[,;|•·]+", str(value)):
            part = part.strip(" \t-–—")
            if part:
                out.append(part)
    return out


_COMPANY_SUFFIX = {
    "ltd", "limited", "inc", "incorporated", "plc", "llc", "co", "corp",
    "corporation", "company", "group", "holdings", "pte", "gmbh",
}


def _company_key(value: Any) -> str:
    return " ".join(t for t in _tokens(value) if t not in _COMPANY_SUFFIX)


def _company_match(a: Any, b: Any) -> bool:
    ka, kb = _company_key(a), _company_key(b)
    if not ka or not kb:
        return False
    return ka == kb or ka in kb or kb in ka or _similar(ka, kb, 0.88)


_SENIORITY_WORDS = {
    "senior": "senior", "sr": "senior", "lead": "senior", "principal": "senior",
    "head": "senior", "staff": "senior", "chief": "senior",
    "junior": "junior", "jr": "junior", "intern": "junior", "trainee": "junior",
}


def _split_title(value: Any) -> tuple[str | None, str]:
    toks = _norm(value).split()
    if not toks:
        return None, ""
    level = _SENIORITY_WORDS.get(toks[0])
    if level:
        return level, " ".join(toks[1:])
    return None, " ".join(toks)


def _seniority_level(value: Any) -> str:
    if value is None:
        return ""
    verbatim = str(value).strip().lower()
    return _SENIORITY_WORDS.get(verbatim, verbatim)


def _year_eq(a: Any, b: Any) -> bool:
    try:
        return int(a) == int(b)
    except (TypeError, ValueError):
        return False


# --- claim extraction ------------------------------------------------------

def _content_text(result: Any) -> str:
    content = getattr(result, "content", result)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _parse_json(text: str) -> Any:
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[A-Za-z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    start = cleaned.find("{")
    if start < 0:
        return None
    depth = 0
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start:index + 1])
                except Exception:
                    return None
    return None


async def _llm_json(model: Any, system: str, human: str, attempts: int = 3) -> Any:
    last_error: Any = None
    for attempt in range(attempts):
        try:
            result = await model.ainvoke([("system", system), ("human", human)])
            data = _parse_json(_content_text(result))
            if isinstance(data, dict):
                return data
            last_error = f"unparsable output: {_content_text(result)[:200]!r}"
        except Exception as exc:  # network / API failure: retry, never crash the run
            last_error = repr(exc)
            await asyncio.sleep(1.0 + attempt)
    _dbg("LLM call failed:", last_error)
    return None


async def _extract_claims(model: Any, cv_text: str) -> dict[str, Any]:
    data = await _llm_json(model, EXTRACT_SYSTEM, EXTRACT_PROMPT.format(cv=cv_text[:12000]))
    if not isinstance(data, dict):
        return {}
    claims: dict[str, Any] = {
        "name": data.get("name"),
        "city": data.get("city"),
        "country": data.get("country"),
        "hometown": data.get("hometown"),
        "headline": data.get("headline"),
        "jobs": [j for j in data.get("jobs") or [] if isinstance(j, dict)],
        "education": [e for e in data.get("education") or [] if isinstance(e, dict)],
        "skills": _split_skills([str(s) for s in data.get("skills") or [] if s]),
    }
    if not claims["name"]:
        lines = [line.strip() for line in cv_text.splitlines() if line.strip()]
        if lines:
            claims["name"] = lines[0]
    _dbg("claims:", json.dumps(claims, ensure_ascii=False))
    return claims


# --- MCP helpers -----------------------------------------------------------

def _tool_payload(result: Any) -> Any:
    blocks = result if isinstance(result, (list, tuple)) else [result]
    for block in blocks:
        text = block.get("text") if isinstance(block, dict) else block
        if isinstance(text, str):
            try:
                return json.loads(text)
            except Exception:
                continue
    return None


class _BudgetExceeded(Exception):
    """Raised when one CV has used up its time or tolerated failures."""


class _CvBudget:
    """Per-CV guard so one slow or broken MCP call cannot sink the whole run."""

    def __init__(self, seconds: float = CV_BUDGET, failures: int = 3) -> None:
        self.deadline = time.monotonic() + seconds
        self.initial = failures
        self.failures = failures

    @property
    def used_failures(self) -> int:
        return self.initial - self.failures

    def check(self) -> None:
        if self.failures <= 0 or time.monotonic() > self.deadline:
            raise _BudgetExceeded()

    def fail(self) -> None:
        self.failures -= 1


async def _call_tool(tools: dict[str, Any], name: str, budget: Any = None, **kwargs: Any) -> Any:
    tool = tools.get(name)
    if tool is None:
        return None
    for attempt in range(TOOL_ATTEMPTS):
        if budget is not None:
            budget.check()
        try:
            result = await asyncio.wait_for(tool.ainvoke(kwargs), timeout=TOOL_TIMEOUT)
            return _tool_payload(result)
        except Exception as exc:  # hung or failed MCP call: retry once, then move on
            _dbg("tool error", name, repr(exc))
            if budget is not None:
                budget.fail()
            if attempt + 1 < TOOL_ATTEMPTS:
                await asyncio.sleep(0.5)
    return None


async def _search_linkedin(
    tools: dict[str, Any],
    q: str,
    location: str | None = None,
    industry: str | None = None,
    budget: Any = None,
) -> list[dict[str, Any]]:
    kwargs: dict[str, Any] = {"q": q, "limit": 20}
    if location:
        kwargs["location"] = location
    if industry:
        kwargs["industry"] = industry
    data = await _call_tool(tools, "search_linkedin_people", budget=budget, **kwargs)
    return [row for row in data or [] if isinstance(row, dict) and row.get("id") is not None]


async def _get_profile(
    tools: dict[str, Any],
    cache: dict[int, Any],
    person_id: int,
    budget: Any = None,
) -> Any:
    if person_id in cache:
        return cache[person_id]
    profile = await _call_tool(tools, "get_linkedin_profile", budget=budget, person_id=person_id)
    if not isinstance(profile, dict) or "error" in profile:
        return None  # never cache a failure: a retry later may succeed
    cache[person_id] = profile
    return profile


# --- entity resolution -----------------------------------------------------

def _best_job(job: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any] | None:
    best = None
    best_key = None
    for item in profile.get("experience") or []:
        if not _company_match(job.get("company"), item.get("company")):
            continue
        key = abs((job.get("start_year") or 0) - (item.get("start_year") or 0))
        if best is None or best_key is None or key < best_key:
            best, best_key = item, key
    return best


def _best_edu(edu: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any] | None:
    for item in profile.get("education") or []:
        if _school_match(edu.get("school"), item.get("school")):
            return item
    return None


def _fit_score(claims: dict[str, Any], profile: dict[str, Any]) -> float:
    """How much identifying evidence (not one field) links the CV to the profile."""
    points = 0.0
    weight = 0.0

    weight += 3.0
    if _city_match(claims.get("city"), profile.get("city")):
        points += 3.0
    elif _country_match(claims.get("country"), profile.get("country")):
        points += 1.5

    industry = _industry(claims.get("headline"))
    weight += 2.0
    if industry and _norm(industry) == _norm(profile.get("industry")):
        points += 2.0
    elif industry and _similar(industry, profile.get("industry"), 0.9):
        points += 1.0

    for job in claims.get("jobs") or []:
        weight += 4.0
        match = _best_job(job, profile)
        if match is None:
            continue
        points += 2.5
        if _year_eq(job.get("start_year"), match.get("start_year")):
            points += 1.0
        if _title_verdict(job.get("title"), match.get("title"), match.get("seniority"))[1] is False:
            points += 0.5

    for edu in claims.get("education") or []:
        weight += 5.0
        match = _best_edu(edu, profile)
        if match is None:
            continue
        points += 3.0
        if _field_match(edu.get("field"), match.get("field")):
            points += 1.0
        if _degree_level(edu.get("degree")) == _degree_level(match.get("degree")):
            points += 1.0

    skills = [s for s in (claims.get("skills") or []) if s]
    if skills:
        weight += 3.0
        known = [item.get("name") for item in profile.get("skills") or []]
        hits = sum(1 for skill in skills if any(_skill_match(skill, name) for name in known))
        points += 3.0 * hits / len(skills)

    return points / weight if weight else 0.0


def _cv_years(claims: dict[str, Any]) -> float | None:
    starts, ends = [], []
    for job in claims.get("jobs") or []:
        try:
            starts.append(int(job.get("start_year")))
        except (TypeError, ValueError):
            pass
        if job.get("is_current"):
            ends.append(time.gmtime().tm_year)
        else:
            try:
                ends.append(int(job.get("end_year")))
            except (TypeError, ValueError):
                pass
    if not starts or not ends:
        return None
    return max(ends) - min(starts)


def _prefit(claims: dict[str, Any], row: dict[str, Any]) -> float:
    """Rank search hits before spending a profile call on them."""
    points = 0.0
    if _name_match(claims.get("name"), row.get("name")):
        points += 2.0
    if _city_match(claims.get("city"), row.get("location")):
        points += 3.0
    industry = _industry(claims.get("headline"))
    if industry and _norm(industry) == _norm(row.get("industry")):
        points += 2.0
    expected = _cv_years(claims)
    if expected is not None and isinstance(row.get("years_experience"), (int, float)):
        points += 1.0 / (1.0 + abs(expected - float(row["years_experience"])) / 3.0)
    return points


async def _resolve(
    tools: dict[str, Any],
    cache: dict[int, Any],
    claims: dict[str, Any],
    budget: Any = None,
) -> tuple[dict[str, Any] | None, float, int]:
    """Find the right person among the many namesakes. Returns (profile, fit, seen)."""
    name = (claims.get("name") or "").strip()
    if not name:
        return None, 0.0, 0
    city = (claims.get("city") or "").strip() or None
    country = (claims.get("country") or "").strip() or None
    industry = _industry(claims.get("headline")) or None

    tiers: list[dict[str, Any]] = []
    if city and industry:
        tiers.append({"q": name, "location": city, "industry": industry})
    if city:
        tiers.append({"q": name, "location": city})
    if country and industry:
        tiers.append({"q": name, "location": country, "industry": industry})
    if industry:
        tiers.append({"q": name, "industry": industry})
    tiers.append({"q": name})

    seen_ids: list[int] = []
    best_profile: dict[str, Any] | None = None
    best_fit = 0.0

    for tier_index, tier in enumerate(tiers):
        if len(seen_ids) >= MAX_PROFILES:
            break
        rows = await _search_linkedin(tools, budget=budget, **tier)
        rows = [row for row in rows if _name_match(claims.get("name"), row.get("name"))]
        rows.sort(key=lambda row: _prefit(claims, row), reverse=True)
        # the narrowest query gets the deepest look; fallback queries only a few
        per_tier = 6 if tier_index == 0 else 2
        taken = 0
        for row in rows:
            if taken >= per_tier or len(seen_ids) >= MAX_PROFILES:
                break
            person_id = row.get("id")
            if person_id in seen_ids:
                continue
            seen_ids.append(person_id)
            taken += 1
            profile = await _get_profile(tools, cache, person_id, budget=budget)
            if not isinstance(profile, dict):
                continue
            if not _name_match(claims.get("name"), profile.get("name")):
                continue
            fit = _fit_score(claims, profile)
            if fit > best_fit:
                best_profile, best_fit = profile, fit
        if best_fit >= FIT_CONFIDENT:  # convincing match: stop asking the shared server
            break

    _dbg(f"resolve: {len(seen_ids)} namesakes checked, best fit {best_fit:.2f}")
    return best_profile, best_fit, len(seen_ids)


# --- claim-by-claim comparison --------------------------------------------

def _title_verdict(
    cv_title: Any,
    prof_title: Any,
    prof_seniority: Any,
) -> tuple[bool, bool | None, str]:
    """Return (matched, is_hard_discrepancy, note)."""
    cv_norm, prof_norm = _norm(cv_title), _norm(prof_title)
    if not cv_norm:
        return True, False, ""
    level, base = _split_title(cv_title)
    same_role = cv_norm == prof_norm or base == prof_norm
    if not same_role:
        cv_tokens, prof_tokens = set(base.split()), set(prof_norm.split())
        same_role = bool(cv_tokens and prof_tokens and (cv_tokens <= prof_tokens or prof_tokens <= cv_tokens))
    if not same_role:
        if _similar(base, prof_norm, 0.9):
            return True, None, f"title wording {cv_title!r} vs {prof_title!r} (looks equivalent)"
        return False, True, f"CV title {cv_title!r} vs profile {prof_title!r} ({prof_seniority})"
    if level is None:
        return True, False, ""
    if level == _seniority_level(prof_seniority):
        return True, False, ""
    return (
        False,
        True,
        f"CV title {cv_title!r} claims {level} seniority but profile has {prof_title!r} ({prof_seniority})",
    )


def _compare(claims: dict[str, Any], profile: dict[str, Any]) -> tuple[list, list]:
    hard: list[tuple[str, str]] = []
    soft: list[tuple[str, str]] = []

    if not _name_match(claims.get("name"), profile.get("name")):
        soft.append(("name", f"CV name {claims.get('name')!r} vs profile {profile.get('name')!r}"))

    if claims.get("city") and profile.get("city") and not _city_match(claims.get("city"), profile.get("city")):
        hard.append((
            "city",
            f"CV location {claims.get('city')!r} vs profile {profile.get('city')!r}, {profile.get('country')!r}",
        ))

    for job in claims.get("jobs") or []:
        company = job.get("company") or "?"
        match = _best_job(job, profile)
        if match is None:
            hard.append(("jobs", f"{company}: employer not found on LinkedIn"))
            continue
        _, bad, note = _title_verdict(job.get("title"), match.get("title"), match.get("seniority"))
        if bad:
            hard.append(("jobs/title", f"{company}: {note}"))
        elif note:
            soft.append(("jobs/title", f"{company}: {note}"))

        problems = []
        cv_start, cv_end, cv_current = job.get("start_year"), job.get("end_year"), bool(job.get("is_current"))
        p_start = match.get("start_year")
        p_current = bool(match.get("is_current"))
        if not _year_eq(cv_start, p_start):
            problems.append(f"start year {cv_start} vs {p_start}")
        if p_current:
            if not cv_current:
                problems.append(f"CV ends {cv_end} but profile says this job is current")
        else:
            if cv_current:
                problems.append("CV says current but profile has an end date")
            elif not _year_eq(cv_end, match.get("end_year")):
                problems.append(f"end year {cv_end} vs {match.get('end_year')}")
        if problems:
            hard.append(("jobs/dates", f"{company} ({match.get('title')}): " + "; ".join(problems)))

    for edu in claims.get("education") or []:
        school = edu.get("school") or "?"
        match = _best_edu(edu, profile)
        if match is None:
            hard.append(("education", f"{school}: school not found on LinkedIn"))
            continue
        cv_level = _degree_level(edu.get("degree"))
        prof_level = _degree_level(match.get("degree"))
        if cv_level is not None and prof_level is not None and cv_level != prof_level:
            hard.append((
                "education/degree",
                f"{school}: CV degree {edu.get('degree')!r} vs profile {match.get('degree')!r}",
            ))
        elif (
            _degree_subtype(edu.get("degree"))
            and _degree_subtype(match.get("degree"))
            and _degree_subtype(edu.get("degree")) != _degree_subtype(match.get("degree"))
        ):
            soft.append((
                "education/degree",
                f"{school}: CV degree {edu.get('degree')!r} vs profile {match.get('degree')!r}",
            ))
        if not _field_match(edu.get("field"), match.get("field")):
            hard.append((
                "education/field",
                f"{school}: CV field {edu.get('field')!r} vs profile {match.get('field')!r}",
            ))
        grad_year = edu.get("grad_year")
        if grad_year is not None and match.get("end_year") is not None and not _year_eq(grad_year, match.get("end_year")):
            hard.append((
                "education/year",
                f"{school}: CV graduation {grad_year} vs profile {match.get('end_year')}",
            ))

    known_skills = [item.get("name") for item in profile.get("skills") or []]
    for skill in claims.get("skills") or []:
        if skill and not any(_skill_match(skill, name) for name in known_skills):
            hard.append(("skills", f"CV lists skill {skill!r} that is not on LinkedIn"))

    return hard, soft


async def _adjudicate_soft(
    model: Any,
    claims: dict[str, Any],
    profile: dict[str, Any],
    soft: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Ask the model about genuinely ambiguous differences; return the real ones."""
    confirmed: list[tuple[str, str]] = []
    for field, note in soft[:4]:
        verdict = await _llm_json(
            model,
            JUDGE_SYSTEM,
            JUDGE_PROMPT.format(cv=f"{field}: {note}", profile=json.dumps(profile, ensure_ascii=False)[:2500]),
            attempts=2,
        )
        if isinstance(verdict, dict) and verdict.get("allowed") is False:
            confirmed.append((field, note + f" [{verdict.get('reason', '')}]"))
    return confirmed


# --- one CV end to end -----------------------------------------------------

async def _verify_one(model: Any, tools: dict[str, Any], cache: dict[int, Any], text: str) -> float:
    claims = await _extract_claims(model, text)
    if not claims or not claims.get("name"):
        _dbg("no claims extracted; scoring as unresolved")
        return SCORE_UNRESOLVED

    budget = _CvBudget()
    try:
        profile, fit, seen = await _resolve(tools, cache, claims, budget=budget)
    except _BudgetExceeded:
        _dbg("per-CV budget exhausted")
        return SCORE_UNCLEAR if budget.used_failures else SCORE_UNRESOLVED

    if profile is None or fit < FIT_FLOOR:
        if budget.used_failures >= 2:
            # the shared server, not the CV, blocked the check: do not call it a lie
            _dbg(f"resolution blocked by {budget.used_failures} MCP failures -> neutral score")
            return SCORE_UNCLEAR
        _dbg(f"entity resolution failed (seen={seen}, fit={fit:.2f}) -> {SCORE_UNRESOLVED}")
        return SCORE_UNRESOLVED

    try:
        hard, soft = _compare(claims, profile)
        if soft:
            hard = hard + await _adjudicate_soft(model, claims, profile, soft)
    except _BudgetExceeded:
        _dbg("per-CV budget exhausted during adjudication")
        return SCORE_UNCLEAR

    _dbg(f"fit={fit:.2f} hard={hard} soft={soft}")
    if hard:
        return SCORE_DISCREPANCY
    return SCORE_STRONG_VALID if fit >= FIT_CONFIDENT else SCORE_VALID


def build_agent(tools: list[Any]) -> Any:
    """Create and return your agent once.

    ``tools`` are the six SocialGraph MCP tools (Facebook + LinkedIn search and
    profile lookup), already wrapped as LangChain tools. You may add your own
    local tools as well.

    Suggested imports:
        from langchain_deepseek import ChatDeepSeek
        from langchain.agents import create_agent

    Use the DeepSeek model named by ``MODEL_NAME``. The API key is loaded
    from .env.
    """
    from langchain_deepseek import ChatDeepSeek

    tool_map: dict[str, Any] = {}
    for index, tool in enumerate(tools):
        tool_map[getattr(tool, "name", f"tool_{index}")] = tool

    model = ChatDeepSeek(model=MODEL_NAME, temperature=0, max_tokens=8000)

    return {"model": model, "tools": tool_map, "cache": {}}


async def score_cvs(agent: Any, cvs: dict[str, str]) -> dict[str, float | None]:
    """Run your agent and return one reliability score per CV.

    ``cvs`` maps each file name to its text, e.g. ``{"CV_1.pdf": "...", ...}``.
    Return a float in [0, 1] for every file name: higher means the CV is more
    likely consistent with the candidate's LinkedIn/Facebook data. A score
    above 0.5 counts as "valid", 0.5 or below counts as "has discrepancy".

        {"CV_1.pdf": 0.9, "CV_4.pdf": 0.1, ...}

    Catch errors per CV (e.g. a failed API call) and still return a score for
    it: an exception here means no results.csv, which scores zero.

    MCP tools are async, so call your agent with ``await agent.ainvoke(...)``.
    You may verify CVs in parallel (e.g. ``asyncio.gather``), but keep at most
    about 3 CVs in flight (e.g. with ``asyncio.Semaphore(3)``): the MCP server is
    shared by the whole class.
    """
    model = agent["model"]
    tools = agent.get("tools") or {}
    cache = agent.get("cache") if isinstance(agent.get("cache"), dict) else {}
    semaphore = asyncio.Semaphore(3)  # the MCP server is shared by the whole class
    scores: dict[str, float] = {}

    async def worker(name: str, text: str) -> None:
        async with semaphore:
            try:
                scores[name] = float(await asyncio.wait_for(_verify_one(model, tools, cache, text), timeout=CV_BUDGET))
            except asyncio.TimeoutError:
                _dbg(f"budget exceeded for {name}")
                scores[name] = SCORE_UNRESOLVED
            except Exception as exc:  # one bad CV must never lose the whole run
                _dbg(f"verify failed for {name}: {exc!r}")
                scores[name] = 0.6

    tasks = [asyncio.create_task(worker(name, text)) for name, text in cvs.items()]
    try:
        await asyncio.wait(tasks, timeout=GLOBAL_BUDGET)
    finally:
        # Hard wall-clock stop: a hung MCP call must never cost the whole run.
        for task in tasks:
            if not task.done():
                task.cancel()
    for name in cvs:
        scores.setdefault(name, SCORE_UNRESOLVED)
    return scores


# Everything below is provided runner/scoring code. No edits are needed.

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_score(value: Any) -> float | None:
    """Accept a float/int, or text containing exactly one number, in [0, 1]."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        score = float(value)
    else:
        text = str(getattr(value, "content", value))
        matches = _NUMBER_RE.findall(text)
        if len(matches) != 1:
            return None
        score = float(matches[0])
    if math.isnan(score) or not 0.0 <= score <= 1.0:
        return None
    return score


def read_ground_truth(folder: Path) -> dict[str, dict[str, Any]]:
    """Read labels (1 = valid CV, 0 = has discrepancy) and reasons from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    return {
        name: entry if isinstance(entry, dict) else {"label": entry}
        for name, entry in json.loads(path.read_text(encoding="utf-8")).items()
    }


def correctness_text(score: float | None, expected: dict[str, Any] | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if score is None:
        return "incorrect: score is missing or not a number in [0, 1]"
    if expected is None:
        return "not graded: no ground truth for this CV"
    label = int(expected["label"])
    predicted = 1 if score > THRESHOLD else 0
    if predicted == label:
        return "correct"
    reason = f" ({expected['reason']})" if expected.get("reason") else ""
    return f"incorrect: expected {label}{reason}, predicted {predicted}"


def write_results(names: list[str], scores: dict[str, Any], truth: dict[str, dict[str, Any]]) -> tuple[Path, int]:
    """Write the required results.csv file and return how many CVs were correct."""
    output = Path("results.csv")
    correct = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cv", "score", "correctness"])
        for name in names:
            score = parse_score(scores.get(name))
            verdict = correctness_text(score, truth.get(name))
            correct += verdict == "correct"
            writer.writerow([name, "" if score is None else f"{score:.4f}", verdict])
    return output, correct


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW2 on CV PDFs")
    parser.add_argument(
        "--cv-folder",
        required=True,
        type=Path,
        help="folder containing CV PDF files",
    )
    return parser.parse_args()


async def run(folder: Path) -> int:
    paths = cv_files(folder)
    if not paths:
        raise SystemExit(f"no PDF files found in {folder}")

    load_env_file()
    cvs = {path.name: cv_text(path) for path in paths}
    tools = await load_mcp_tools()
    agent = build_agent(tools)
    scores = await score_cvs(agent, cvs)
    if not isinstance(scores, dict):
        raise TypeError("score_cvs() must return a dictionary")

    truth = read_ground_truth(folder)
    output, correct = write_results(list(cvs), scores, truth)
    summary = f" Accuracy: {correct}/{len(cvs)}." if truth else ""
    print(f"Processed {len(cvs)} CV(s). Wrote {output}.{summary}")
    return 0


def main() -> int:
    args = parse_args()
    if not args.cv_folder.is_dir():
        raise SystemExit(f"not a folder: {args.cv_folder}")
    return asyncio.run(run(args.cv_folder))


if __name__ == "__main__":
    raise SystemExit(main())
