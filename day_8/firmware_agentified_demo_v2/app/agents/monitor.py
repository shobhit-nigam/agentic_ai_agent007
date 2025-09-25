from .base import Agent, Message

class MonitorAgent(Agent):
    """Reads KPIs from the fleet and emits Observation messages."""
    def think(self, state):
        fleet = state["fleet"]
        k = fleet.tick()
        return Message("Observation", {"kpi": k, "epoch": state.get("epoch", 0)})
