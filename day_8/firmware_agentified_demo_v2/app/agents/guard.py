from .base import Agent, Message
from anomaly import gate_pass

class SafetyGuardAgent(Agent):
    def __init__(self, name, baseline: float, threshold: float = 0.0020, trace=None):
        super().__init__(name, trace=trace)
        self.baseline = baseline
        self.threshold = threshold

    def think(self, state):
        current = state["kpi"]["crash_rate"] if "kpi" in state else state["rows"][-1]["crash_rate"]
        ok = gate_pass(self.baseline, current, threshold=self.threshold)
        return Message("GateResult", {"pass": ok, "baseline": self.baseline, "current": current, "threshold": self.threshold})
