"""Hybrid retriever: BM25 (always on) + dense FAISS (when EMBEDDINGS is configured), fused with
reciprocal-rank fusion. Supports metadata filters (company, doc_type)."""
from __future__ import annotations

import json
import re
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

from placement_agent import config

TOKEN = re.compile(r"[a-z]+|\d+(?:[.,]\d+)*")  # split letter/digit runs: "bonus370000" -> bonus, 370000


def tokenize(text: str) -> list[str]:
    return [t.replace(",", "") for t in TOKEN.findall(text.lower())]


def load_docs() -> list[dict]:
    path = config.DATA_DIR / "docs.jsonl"
    if not path.exists():
        raise FileNotFoundError("No index found. Run: python -m placement_agent.ingest.build")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _embedder():
    from langchain.embeddings import init_embeddings

    return init_embeddings(config.EMBEDDINGS)


def build_dense_index(docs: list[dict]) -> int:
    if not config.EMBEDDINGS:
        return 0
    import faiss

    vecs = np.array(_embedder().embed_documents([d["text"] for d in docs]), dtype="float32")
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.INDEX_DIR / "faiss.index"))
    (config.INDEX_DIR / "meta.json").write_text(json.dumps({"embeddings": config.EMBEDDINGS, "n": len(docs)}))
    return len(docs)


class HybridRetriever:
    def __init__(self):
        self.docs = load_docs()
        self.bm25 = BM25Okapi([tokenize(d["text"]) for d in self.docs])
        self.index = self.embed = None
        meta = config.INDEX_DIR / "meta.json"
        if config.EMBEDDINGS and meta.exists() and json.loads(meta.read_text())["embeddings"] == config.EMBEDDINGS:
            import faiss

            self.index = faiss.read_index(str(config.INDEX_DIR / "faiss.index"))
            self.embed = _embedder()

    @property
    def mode(self) -> str:
        return "hybrid (BM25 + dense)" if self.index is not None else "BM25"

    def _allowed(self, company: str | None, doc_type: str | None, location_terms: list[str] | None = None) -> np.ndarray:
        from placement_agent.ingest.build import same_company

        mask = np.ones(len(self.docs), dtype=bool)
        for i, d in enumerate(self.docs):
            md = d["metadata"]
            if doc_type and md.get("doc_type") != doc_type:
                mask[i] = False
            elif company:
                if md.get("company"):
                    mask[i] = same_company(company, md["company"])
                else:  # stats docs: keep if they mention the company
                    mask[i] = company.lower() in d["text"].lower()
            if mask[i] and location_terms:
                mask[i] = any(t in d["text"].lower() for t in location_terms)
        return mask

    def search(self, query: str, k: int = 5, company: str | None = None, doc_type: str | None = None,
               location_terms: list[str] | None = None) -> list[dict]:
        mask = self._allowed(company, doc_type, location_terms)
        if not mask.any():
            return []
        rankings = []
        bm = self.bm25.get_scores(tokenize(query))
        bm[~mask] = -np.inf
        rankings.append([i for i in np.argsort(-bm) if mask[i]][: max(k * 4, 20)])
        if self.index is not None:
            import faiss

            q = np.array([self.embed.embed_query(query)], dtype="float32")
            faiss.normalize_L2(q)
            _, ids = self.index.search(q, min(len(self.docs), max(k * 8, 40)))
            rankings.append([i for i in ids[0] if i >= 0 and mask[i]])
        fused: dict[int, float] = {}
        for ranking in rankings:  # reciprocal-rank fusion
            for rank, i in enumerate(ranking):
                fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (60 + rank)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [{**self.docs[i], "score": round(fused[i], 4)} for i in top]


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever()
