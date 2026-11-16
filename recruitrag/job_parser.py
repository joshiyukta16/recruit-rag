"""Turn a free-text job description into structured requirements."""
import re

from .models import JobRequirements
from .skills import extract_skills

_PREFERRED_RE = re.compile(r"nice to have|preferred|bonus|good to have|a plus|desirable", re.I)
_REQUIRED_RE = re.compile(r"requirements|must have|required|qualifications|responsibilities|what you", re.I)
_YEARS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:-\s*\d+\s*)?years?", re.I)


def _heading_kind(line: str):
    if len(line) < 60:
        if _PREFERRED_RE.search(line):
            return "preferred"
        if _REQUIRED_RE.search(line):
            return "required"
    return None


def parse_job(text: str) -> JobRequirements:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    title = re.sub(r"^(?:#+|job title\s*:|title\s*:)\s*", "", lines[0], flags=re.I).strip() if lines else ""

    # If the JD has section headings, text before the first heading is company/role blurb,
    # not a requirement ("...our recruitment platform" must not demand recruiting skills).
    has_headings = any(_heading_kind(l) for l in lines[1:])
    state = "intro" if has_headings else "required"

    required, preferred = [], []
    for line in lines[1:]:
        state = _heading_kind(line) or state
        if state == "required":
            required.extend(extract_skills(line))
        elif state == "preferred":
            preferred.extend(extract_skills(line))
    if not has_headings:
        required = extract_skills(" ".join(lines))  # single-paragraph JD: take everything

    required = list(dict.fromkeys(required))
    preferred = [s for s in dict.fromkeys(preferred) if s not in required]
    years = _YEARS_RE.search(text)
    first = next((i for i, l in enumerate(lines) if i > 0 and _heading_kind(l)), None)
    req_lines = lines[first:] if first is not None else lines[1:]
    requirements_text = "\n".join([title] + req_lines).strip()
    return JobRequirements(
        title=title, required_skills=required, preferred_skills=preferred,
        min_years=float(years.group(1)) if years else 0.0, raw_text=text,
        requirements_text=requirements_text,
    )
