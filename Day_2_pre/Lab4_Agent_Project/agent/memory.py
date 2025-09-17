from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import re, math, time, uuid

class ShortMemory:
    def __init__(self, k: int = 5):
        self.buf = deque(maxlen=k)
    def add(self, role: str, text: str):
        self.buf.append({"ts": time.time(), "role": role, "text": text})
    def recent(self) -> List[Dict[str, Any]]:
        return list(self.buf)
    def formatted(self) -> str:
        return "\n".join(f"{x['role']}: {x['text']}" for x in self.buf)

@dataclass
class MemoryItem:
    id: str
    text: str
    kind: str
    meta: Dict[str, Any]

class InMemoryVectorStore:
    def __init__(self):
        self.items: Dict[str, MemoryItem] = {}
        self.vectors: Dict[str, Dict[str, float]] = {}
        self.df: Dict[str, int] = {}
        self._N = 0
    def _tokens(self, text: str):
        return re.findall(r"[a-z0-9]+", text.lower())
    def _vectorize(self, text: str) -> Dict[str, float]:
        toks = self._tokens(text)
        tf: Dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        vec: Dict[str, float] = {}
        for t, c in tf.items():
            df = self.df.get(t, 1)
            import math
            idf = math.log((1 + self._N) / df)
            vec[t] = c * idf
        norm = (sum(v*v for v in vec.values()) ** 0.5) or 1.0
        return {t: v/norm for t, v in vec.items()}
    def _cosine(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        return sum(a.get(t,0.0)*b.get(t,0.0) for t in set(a)|set(b))
    def upsert(self, item: MemoryItem):
        is_new = item.id not in self.items
        self.items[item.id] = item
        if is_new:
            self._N += 1
            for tok in set(self._tokens(item.text)):
                self.df[tok] = self.df.get(tok, 0) + 1
        self.vectors[item.id] = self._vectorize(item.text)
    def search(self, query: str, k: int = 3, kind: Optional[str] = None) -> List[MemoryItem]:
        qvec = self._vectorize(query)
        scored = []
        for mid, item in self.items.items():
            if kind and item.kind != kind:
                continue
            score = self._cosine(qvec, self.vectors[mid])
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:k]]

def redact_pii(text: str) -> str:
    return text

def is_storable(text: str) -> bool:
    t = text.lower().strip()
    if t.startswith("remember "): return True
    if any(t.startswith(pfx) for pfx in ("my preference", "i prefer", "note that")): return True
    if len(t) < 15: return False
    if t.endswith("?"): return False
    return True

class LongMemory:
    def __init__(self):
        self.store = InMemoryVectorStore()
    def remember(self, text: str, *, kind: str = "preference", meta: Optional[Dict[str, Any]] = None) -> str:
        clean = text
        item = MemoryItem(id=str(uuid.uuid4()), text=clean, kind=kind, meta=meta or {"ts": time.time(), "source": "user"})
        self.store.upsert(item)
        return f"Stored: {clean}"
    def retrieve(self, query: str, *, k: int = 3, kind: Optional[str] = None) -> List[MemoryItem]:
        return self.store.search(query, k=k, kind=kind)
