"""Retrieval-Augmented Generation (RAG) travel assistant.

1. The Markdown files in rag_knowledge/ are split into sections (one chunk per "##" heading).
2. Each chunk is embedded and stored in a FAISS index (inner product on L2-normalised vectors = cosine).
   - Embeddings: Sentence Transformers (all-MiniLM-L6-v2) when installed and downloadable,
     otherwise a TF-IDF fallback (words + character n-grams), so the assistant also works fully offline.
3. A question retrieves the top-k chunks. If GROQ_API_KEY is set, an LLM writes the answer from
   those chunks only; otherwise the assistant returns the best passage (extractive answer).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = ROOT / "rag_knowledge"


@dataclass
class Chunk:
    source: str
    title: str
    text: str


def load_chunks(folder: Path = KNOWLEDGE_DIR) -> list[Chunk]:
    chunks = []
    for path in sorted(folder.glob("*.md")):
        topic = path.stem.replace("_", " ")
        for block in re.split(r"\n(?=## )", path.read_text(encoding="utf-8")):
            lines = [line for line in block.strip().splitlines() if not line.startswith("# ")]
            if not lines:
                continue
            title = lines[0].lstrip("# ").strip() if lines[0].startswith("##") else topic
            body = " ".join(line.strip() for line in lines[1:] if line.strip())
            if body:
                chunks.append(Chunk(source=path.name, title=title, text=body))
    return chunks


class _TfidfEmbedder:
    name = "tfidf"

    def __init__(self, corpus: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_union

        # word TF-IDF + character n-grams (robust to plurals and small spelling differences)
        self.vec = make_union(
            TfidfVectorizer(sublinear_tf=True, stop_words="english"),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True),
        ).fit(corpus)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.vec.transform(texts).toarray().astype("float32")


class _SentenceTransformerEmbedder:
    name = "all-MiniLM-L6-v2"

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts), dtype="float32")


def _make_embedder(corpus: list[str]):
    if os.environ.get("RAG_EMBEDDINGS", "auto") != "tfidf":
        try:
            return _SentenceTransformerEmbedder()
        except Exception:  # package missing or model not downloadable (offline)
            pass
    return _TfidfEmbedder(corpus)


class TravelAssistant:
    def __init__(self, folder: Path = KNOWLEDGE_DIR):
        self.chunks = load_chunks(folder)
        corpus = [f"{c.source.removesuffix('.md').replace('_', ' ')}. {c.title}. {c.text}" for c in self.chunks]
        self.embedder = _make_embedder(corpus)
        vectors = self.embedder.encode(corpus)
        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def search(self, query: str, k: int = 3) -> list[dict]:
        q = self.embedder.encode([query])
        faiss.normalize_L2(q)
        scores, ids = self.index.search(q, min(k, len(self.chunks)))
        return [
            {"score": round(float(s), 3), "source": self.chunks[i].source,
             "title": self.chunks[i].title, "text": self.chunks[i].text}
            for s, i in zip(scores[0], ids[0]) if i >= 0
        ]

    # below this cosine similarity the question is considered out of scope (tuned for the TF-IDF fallback)
    min_score = 0.18

    def answer(self, question: str, k: int = 3) -> dict:
        hits = self.search(question, k)
        confidence = hits[0]["score"] if hits else 0.0
        if not hits or confidence < self.min_score:
            return {"answer": "I don't have information about that in my travel knowledge base.",
                    "confidence": confidence, "sources": [], "mode": "none"}
        llm_answer = _ask_llm(question, hits)
        return {
            "answer": llm_answer or hits[0]["text"],
            "confidence": confidence,
            "sources": [{"source": h["source"], "title": h["title"], "score": h["score"]} for h in hits],
            "mode": "llm" if llm_answer else "extractive",
        }

    def categories(self) -> list[str]:
        return sorted({c.source.removesuffix(".md").replace("_", " ") for c in self.chunks})


def _ask_llm(question: str, hits: list[dict]) -> str | None:
    """Optional generation step with Groq (OpenAI-compatible API). Returns None if not configured."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return None
    try:
        import httpx

        context = "\n\n".join(f"[{h['title']}] {h['text']}" for h in hits)
        resp = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": "You are a travel assistant. Answer in 2-4 sentences using ONLY "
                                                  "the context. If the context does not contain the answer, say so."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
            },
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None
