"""Resume indexer: load -> parse profile -> section-aware chunks."""
import os
import re
from datetime import date
from typing import Dict, List, Optional, Tuple

from ..interface.base_indexer import BaseIndexer
from ..interface.models import DataItem, ResumeProfile
from ..util.resume_parser import parse_resume
from ..util.text_loader import SUPPORTED_EXTENSIONS, load_document

MAX_CHUNK_CHARS = 900


def _pack(text: str, limit: int) -> List[str]:
    """Pack blank-line separated blocks (e.g. one job entry each) into <= limit char chunks."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    pieces: List[str] = []
    for b in blocks:  # split oversize blocks line by line
        if len(b) <= limit:
            pieces.append(b)
        else:
            buf = ""
            for line in b.splitlines():
                if buf and len(buf) + len(line) + 1 > limit:
                    pieces.append(buf)
                    buf = ""
                buf += ("\n" if buf else "") + line
            if buf:
                pieces.append(buf)
    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + len(p) + 2 > limit:
            chunks.append(cur)
            cur = ""
        cur += ("\n\n" if cur else "") + p
    if cur:
        chunks.append(cur)
    return chunks


def chunk_resume(profile: ResumeProfile, sections: Dict[str, str], limit: int = MAX_CHUNK_CHARS) -> List[DataItem]:
    """Every chunk is prefixed with candidate + section so it is self-describing at retrieval time."""
    items: List[DataItem] = []
    for section, body in sections.items():
        for piece in _pack(body, limit):
            items.append(DataItem(
                content=f"Candidate: {profile.name} | Section: {section}\n{piece}",
                source=f"{profile.candidate_id}:{len(items)}",
                candidate_id=profile.candidate_id,
                section=section,
            ))
    return items


class Indexer(BaseIndexer):
    def __init__(self, today: Optional[date] = None):
        self.today = today

    def index(self, document_paths: List[str]) -> Tuple[List[DataItem], List[ResumeProfile]]:
        items: List[DataItem] = []
        profiles: List[ResumeProfile] = []
        for path in document_paths:
            if os.path.splitext(path)[1].lower() not in SUPPORTED_EXTENSIONS:
                print(f"⚠️  Skipping unsupported file: {path}")
                continue
            text = load_document(path)
            profile, sections = parse_resume(text, os.path.basename(path), self.today)
            profiles.append(profile)
            items.extend(chunk_resume(profile, sections))
        return items, profiles
