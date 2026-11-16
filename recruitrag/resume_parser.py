"""Rule-based resume parsing: sections, contact info, skills, experience, education.

Deliberately deterministic (no LLM): fast, free, auditable, and testable. An LLM
extraction step can be layered on top later for messy layouts.
"""
import re
from datetime import date
from typing import Dict, List, Optional, Tuple

from .models import ResumeProfile
from .skills import extract_skills

SECTION_ALIASES = {
    "summary": ["summary", "professional summary", "profile", "objective", "career objective", "about me", "about"],
    "experience": ["experience", "work experience", "professional experience", "employment",
                   "employment history", "work history", "career history", "internships", "internship"],
    "education": ["education", "academic background", "academics", "qualifications", "education & training"],
    "skills": ["skills", "technical skills", "core competencies", "key skills", "skills & tools",
               "technologies", "tools"],
    "projects": ["projects", "personal projects", "academic projects", "key projects"],
    "certifications": ["certifications", "certificates", "licenses", "courses", "training"],
    "achievements": ["achievements", "awards", "honors", "accomplishments", "publications"],
}
_HEADING_LOOKUP = {alias: sec for sec, aliases in SECTION_ALIASES.items() for alias in aliases}

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
_MON = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
_RANGE_RE = re.compile(
    rf"(?:(?P<sm>{_MON})[a-z]*\.?,?\s+)?(?P<sy>(?:19|20)\d{{2}})\s*(?:-|–|—|to)\s*"
    rf"(?:(?:(?P<em>{_MON})[a-z]*\.?,?\s+)?(?P<ey>(?:19|20)\d{{2}})|(?P<now>present|current|now|till date|ongoing))",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\+?\d[\d \t().-]{8,16}\d")
_DEGREE_RE = re.compile(
    r"\b(ph\.?d|doctorate|master|m\.?sc|m\.?tech|m\.?e\b|mba|mca|bachelor|b\.?sc|b\.?tech|b\.?e\b|bca|b\.?com|diploma)",
    re.IGNORECASE,
)
_CLAIMED_YEARS_RE = re.compile(r"(\d{1,2})\+?\s*years?\s+(?:of\s+)?(?:professional\s+)?experience", re.IGNORECASE)


def _clean_heading(line: str) -> str:
    return re.sub(r"^[#*_\s]+|[#*_:\s]+$", "", line).lower()


def split_sections(text: str) -> Dict[str, str]:
    """Split a resume into {section: text}. Text before the first heading is 'header'."""
    sections: Dict[str, List[str]] = {"header": []}
    current = "header"
    for line in text.splitlines():
        cleaned = _clean_heading(line)
        if cleaned and len(line.strip()) <= 40 and cleaned in _HEADING_LOOKUP:
            current = _HEADING_LOOKUP[cleaned]
            sections.setdefault(current, [])
            continue
        sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items() if "\n".join(v).strip()}


# ---------------------------------------------------------------- name detection
_NOT_A_NAME = {
    "introduction", "contact", "contact me", "contact details", "contact information", "personal details",
    "personal information", "personal profile", "biodata", "curriculum vitae", "resume", "cv", "profile",
    "summary", "objective", "about me",
}
_DOC_WORDS_RE = re.compile(r"\b(curriculum vitae|resume|cv|biodata)\b(\s+of)?", re.I)
_LABEL_RE = re.compile(
    r"^\s*(?:e-?mail|phone|mobile|mob|tel(?:ephone)?|contact|address|location|linkedin|github|website|"
    r"portfolio|dob|date of birth|gender|nationality|city|state|country)\b", re.I)
_TITLE_WORDS = {  # job-title words: a line like "SENIOR BACKEND ENGINEER" is a headline, not a name
    "engineer", "developer", "analyst", "manager", "intern", "student", "scientist", "designer",
    "consultant", "specialist", "recruiter", "officer", "executive", "architect", "director",
    "administrator", "associate", "assistant", "programmer", "researcher", "graduate", "fresher",
    "undergraduate", "professional", "lead", "head", "trainee",
}
_PARTICLES = {"de", "da", "di", "del", "der", "van", "von", "bin", "binti", "al", "el", "la", "le", "du", "dos", "ibn"}


def _name_from_line(line: str) -> Optional[str]:
    """Reduce one line to a name candidate, or None. Handles markdown, Docling '<!-- image -->',
    'Name:' labels, 'Resume of X', 'X - Resume', 'X | email | phone', and 'X, MBA'."""
    t = re.sub(r"<[^>]*>", " ", line).strip()
    t = t.strip("|").strip().lstrip("#*_> ").rstrip("*_ ")
    t = re.sub(r"^name\s*:\s*", "", t, flags=re.I)
    if _LABEL_RE.match(t):
        return None
    t = _DOC_WORDS_RE.sub(" ", t)
    t = re.sub(r"^[\s\-–—:|,]+", "", t)
    t = re.split(r"\s*[|•·]\s*", t)[0]
    t = re.split(r"\s[-–—](?:\s|$)", t + " ")[0]
    t = t.split(",")[0].strip(" -–—:|")
    return t or None


def _is_valid_name(t: str) -> bool:
    if not t or "@" in t or any(c.isdigit() for c in t):
        return False
    if not all(c.isalpha() or c in " .'’-" for c in t):
        return False
    words = t.split()
    if not 1 <= len(words) <= 5:
        return False
    low = t.lower().rstrip(":")
    if low in _NOT_A_NAME or low in _HEADING_LOOKUP:
        return False
    if {w.lower().strip(".") for w in words} & _TITLE_WORDS:
        return False
    # real names are capitalised (or all caps); this rejects sentence fragments like "seeking a role in AI"
    return all(w[0].isupper() or w.lower() in _PARTICLES for w in words)


def _name_from_filename(filename: str) -> str:
    stem = re.sub(r"[_\-.]+", " ", filename.rsplit(".", 1)[0])
    stem = re.sub(r"\b(cv|resume|curriculum|vitae|final|updated|new|latest|copy|v\d+)\b|\d+", " ", stem, flags=re.I)
    return " ".join(stem.split()).title() or "Unknown Candidate"


def _extract_name(text: str, filename: str) -> str:
    """Pick the first plausible name among the first ~10 non-empty lines.
    Skips headings (INTRODUCTION, RESUME, Contact...), labels, job titles and sentences;
    prefers 2-5 word names; joins 'YUKTA' / 'JOSHI' split over two lines; falls back to the filename."""
    prev_single = None            # (word, line index) of an immediately preceding one-word candidate
    first_single = ""
    lines = [l for l in text.splitlines() if l.strip()][:10]
    for idx, line in enumerate(lines):
        heading = _clean_heading(line)
        if heading in _HEADING_LOOKUP and heading not in _NOT_A_NAME:
            break                                  # a real section (SKILLS, EXPERIENCE...) began: no name below it
        cand = _name_from_line(line)
        if not cand or not _is_valid_name(cand):
            continue
        if len(cand.split()) >= 2:
            return cand.title() if cand.isupper() else cand
        if prev_single and prev_single[1] == idx - 1:       # two consecutive one-word lines
            joined = f"{prev_single[0]} {cand}"
            return joined.title() if joined.isupper() else joined
        first_single = first_single or cand
        prev_single = (cand, idx)
    if first_single:
        return first_single.title() if first_single.isupper() else first_single
    return _name_from_filename(filename)


def _extract_phone(text: str) -> str:
    for m in _PHONE_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        if 10 <= len(digits) <= 13:
            return m.group().strip()
    return ""


def _month_index(year: int, month: Optional[str]) -> int:
    # Year-only dates are treated as mid-year so "2018-2021" => ~3 years, not 4.
    m = _MONTHS[month[:3].lower()] if month else 6
    return year * 12 + m


def compute_experience_years(text: str, today: Optional[date] = None) -> float:
    """Union of all date ranges in `text` (overlaps merged), in years."""
    today = today or date.today()
    intervals: List[Tuple[int, int]] = []
    for m in _RANGE_RE.finditer(text):
        start = _month_index(int(m.group("sy")), m.group("sm"))
        end = today.year * 12 + today.month if m.group("now") else _month_index(int(m.group("ey")), m.group("em"))
        if end >= start:
            intervals.append((start, end))
    intervals.sort()
    total, cur_s, cur_e = 0, None, None
    for s, e in intervals:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return round(total / 12, 1)


def _extract_education(sections: Dict[str, str]) -> List[str]:
    lines = sections.get("education", "").splitlines()
    return [l.strip(" -•*\t") for l in lines if l.strip() and _DEGREE_RE.search(l)][:4]


def make_candidate_id(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0]
    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_").lower()


def parse_resume(text: str, filename: str, today: Optional[date] = None) -> Tuple[ResumeProfile, Dict[str, str]]:
    sections = split_sections(text)
    header = sections.get("header", "")
    exp_text = sections.get("experience") or "\n".join(
        v for k, v in sections.items() if k not in ("education", "header"))
    years = compute_experience_years(exp_text, today)
    if years == 0:
        claimed = _CLAIMED_YEARS_RE.search(text)
        years = float(claimed.group(1)) if claimed else 0.0
    email = _EMAIL_RE.search(header) or _EMAIL_RE.search(text)
    profile = ResumeProfile(
        candidate_id=make_candidate_id(filename),
        filename=filename,
        name=_extract_name(text, filename),
        email=email.group() if email else "",
        phone=_extract_phone(header) or _extract_phone(text),
        skills=extract_skills(text),
        years_experience=years,
        education=_extract_education(sections),
        sections_found=[s for s in sections if s != "header"],
    )
    return profile, sections
