from .base import Agent, Message
from rollout import make_cohorts, wave_name

class PlannerAgent(Agent):
    def __init__(self, name, device_ids, trace=None):
        super().__init__(name, trace=trace)
        self.cohorts = make_cohorts(device_ids)
        self.idx = 0

    def think(self, state):
        if self.idx >= len(self.cohorts):
            return Message("Noop", {})
        cohort = self.cohorts[self.idx]
        name = wave_name(self.idx)
        self.idx += 1
        return Message("ApplyWave", {"cohort": cohort, "wave_name": name, "version": "1.0.1"})
