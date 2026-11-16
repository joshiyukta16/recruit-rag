"""LanceDB vector store + JSON sidecar for structured resume profiles."""
import json
import os
from typing import Dict, List, Optional, Set

from ..interface.base_datastore import BaseDatastore
from ..interface.models import DataItem, ResumeProfile, SearchHit
from .embedder import Embedder, get_embedder


def _sql_in(ids: Set[str]) -> str:
    return ", ".join("'" + i.replace("'", "''") + "'" for i in sorted(ids))


class Datastore(BaseDatastore):
    DB_PATH = "data/recruit-lancedb"
    PROFILE_PATH = "data/profiles.json"
    TABLE = "resumes"

    def __init__(self, embedder: Optional[Embedder] = None):
        import lancedb
        self.embedder = embedder or get_embedder()
        os.makedirs(self.DB_PATH, exist_ok=True)
        self.db = lancedb.connect(self.DB_PATH)
        self.table = self.db.open_table(self.TABLE) if self.TABLE in self._table_names() else self._create_table()

    def _table_names(self) -> List[str]:
        names = self.db.table_names()
        return list(getattr(names, "tables", names))

    def _create_table(self):
        import pyarrow as pa
        schema = pa.schema([
            pa.field("vector", pa.list_(pa.float32(), self.embedder.dim)),
            pa.field("content", pa.utf8()),
            pa.field("source", pa.utf8()),
            pa.field("candidate_id", pa.utf8()),
            pa.field("section", pa.utf8()),
        ])
        return self.db.create_table(self.TABLE, schema=schema)

    # -- lifecycle
    def reset(self) -> None:
        if self.TABLE in self._table_names():
            self.db.drop_table(self.TABLE)
        self.table = self._create_table()
        if os.path.exists(self.PROFILE_PATH):
            os.remove(self.PROFILE_PATH)
        print("✅ Database reset complete")

    # -- chunks
    def get_vector(self, content: str) -> List[float]:
        return self.embedder.encode(content)

    def add_items(self, items: List[DataItem]) -> None:
        if not items:
            return
        vectors = self.embedder.encode_many([i.content for i in items])
        rows = [
            {"vector": v, "content": i.content, "source": i.source,
             "candidate_id": i.candidate_id, "section": i.section}
            for v, i in zip(vectors, items)
        ]
        (self.table.merge_insert("source")
            .when_matched_update_all().when_not_matched_insert_all().execute(rows))

    def delete_candidate(self, candidate_id: str) -> None:
        self.table.delete(f"candidate_id IN ({_sql_in({candidate_id})})")

    def search(self, query: str, top_k: int = 5, candidate_ids: Optional[Set[str]] = None) -> List[SearchHit]:
        q = self.table.search(self.get_vector(query))
        q = q.distance_type("cosine") if hasattr(q, "distance_type") else q.metric("cosine")
        if candidate_ids:
            q = q.where(f"candidate_id IN ({_sql_in(candidate_ids)})", prefilter=True)
        rows = q.limit(top_k).to_list()
        return [
            SearchHit(content=r["content"], source=r["source"], candidate_id=r["candidate_id"],
                      section=r["section"], score=1.0 - float(r.get("_distance", 1.0)))
            for r in rows
        ]

    # -- profiles
    def add_profiles(self, profiles: List[ResumeProfile]) -> None:
        data = {k: v.to_dict() for k, v in self.get_profiles().items()}
        data.update({p.candidate_id: p.to_dict() for p in profiles})
        os.makedirs(os.path.dirname(self.PROFILE_PATH), exist_ok=True)
        with open(self.PROFILE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def get_profiles(self) -> Dict[str, ResumeProfile]:
        if not os.path.exists(self.PROFILE_PATH):
            return {}
        with open(self.PROFILE_PATH, "r", encoding="utf-8") as f:
            return {k: ResumeProfile(**v) for k, v in json.load(f).items()}
