from .base import Agent, Message

class ActuatorAgent(Agent):
    def think(self, state):
        cmd = state["cmd"]
        fleet = state["fleet"]
        if cmd["type"] == "ApplyWave":
            fleet.apply_firmware(cmd["cohort"], cmd["version"])
            return Message("WaveApplied", {"wave_name": cmd["wave_name"], "size": len(cmd["cohort"])})
        if cmd["type"] == "Rollback":
            fleet.apply_firmware(cmd["cohort"], "1.0.0")
            return Message("RollbackDone", {"wave_name": cmd["wave_name"], "size": len(cmd["cohort"])})
        return Message("Noop", {})
