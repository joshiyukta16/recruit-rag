"""Blind-screening mode: mask names / contact details so reviewers judge on skills."""
import re
from dataclasses import replace
from typing import Dict, Iterable, List

from .models import CandidateMatch, ResumeProfile, SearchHit

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\+?\d[\d \t().-]{8,16}\d")
_URL_RE = re.compile(r"(?:https?://|www\.|linkedin\.com/|github\.com/)\S+", re.I)


class Anonymizer:
    def __init__(self, profiles: Iterable[ResumeProfile]):
        self.profiles = sorted(profiles, key=lambda p: p.candidate_id)
        self.aliases: Dict[str, str] = {}
        for i, p in enumerate(self.profiles):
            letter = chr(ord("A") + i % 26) * (i // 26 + 1)
            self.aliases[p.candidate_id] = f"Candidate {letter}"

    def label(self, candidate_id: str) -> str:
        return self.aliases.get(candidate_id, "Candidate")

    def mask_text(self, text: str) -> str:
        for p in self.profiles:
            alias = self.aliases[p.candidate_id]
            if p.name:
                text = re.sub(re.escape(p.name), alias, text, flags=re.I)
                for part in p.name.split():
                    if len(part) >= 3:
                        text = re.sub(rf"\b{re.escape(part)}\b", alias, text, flags=re.I)
        text = _EMAIL_RE.sub("[EMAIL]", text)
        text = _URL_RE.sub("[LINK]", text)
        return _PHONE_RE.sub("[PHONE]", text)

    def mask_hits(self, hits: List[SearchHit]) -> List[SearchHit]:
        return [
            replace(h, content=self.mask_text(h.content),
                    source=f"{self.label(h.candidate_id)}:{h.source.rsplit(':', 1)[-1]}",
                    candidate_id=self.label(h.candidate_id))
            for h in hits
        ]

    def mask_matches(self, matches: List[CandidateMatch]) -> List[CandidateMatch]:
        return [
            replace(m, name=self.label(m.candidate_id), candidate_id=self.label(m.candidate_id),
                    evidence=[self.mask_text(e) for e in m.evidence])
            for m in matches
        ]
