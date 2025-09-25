import json
from pathlib import Path
from datetime import datetime
from dataclasses import is_dataclass, asdict

def _safe_json(x):
    """Recursively make objects JSON-safe and keep traces small."""
    # Primitives pass through
    if x is None or isinstance(x, (bool, int, float, str)):
        return x

    # Messages from our Agent base class
    if hasattr(x, "type") and hasattr(x, "payload"):
        return {"type": x.type, "payload": _safe_json(x.payload)}

    # Dataclasses → dict
    if is_dataclass(x):
        return _safe_json(asdict(x))

    # Dicts: sanitize values and redact heavy fields
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k == "fleet":
                try:
                    # Summarize instead of dumping 2,000 devices
                    out[k] = {"_type": "Fleet", "population": len(v.devices)}
                except Exception:
                    out[k] = {"_type": "Fleet"}
            elif k == "cohort":
                # Only store the size (and a tiny sample) to keep traces small
                if isinstance(v, (list, tuple)):
                    out[k] = {
                        "_type": "Cohort",
                        "size": len(v),
                        "sample": list(v[:5])  # tiny peek
                    }
                else:
                    out[k] = _safe_json(v)
            else:
                out[k] = _safe_json(v)
        return out

    # Lists/tuples
    if isinstance(x, (list, tuple)):
        # Cap length to avoid huge traces
        return [_safe_json(i) for i in list(x)[:20]]

    # Fallback: string representation
    return str(x)

class Tracer:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, agent_name, incoming, decision, outgoing):
        event = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "agent": agent_name,
            "incoming": _safe_json(incoming),
            "decision": _safe_json(decision),
            "outgoing": _safe_json(outgoing),
        }
        with self.path.open("a") as f:
            f.write(json.dumps(event) + "\n")
