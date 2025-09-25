import random
from dataclasses import dataclass

@dataclass
class Device:
    device_id: str
    model: str
    firmware: str = "1.0.0"
    crashed: bool = False
    health: float = 1.0

@dataclass
class Fleet:
    devices: list
    baseline_crash_p: float = 0.002
    newfw_crash_p: float = 0.0035
    recovery_p: float = 0.15

    def tick(self):
        crashes = 0
        for d in self.devices:
            if d.crashed and random.random() < self.recovery_p:
                d.crashed = False
                d.health = min(1.0, d.health + 0.05)
            p = self.newfw_crash_p if d.firmware != "1.0.0" else self.baseline_crash_p
            if d.model == "SensorX-200":
                p *= 1.1
            if not d.crashed and random.random() < p:
                d.crashed = True
                d.health = max(0.0, d.health - 0.2)
                crashes += 1
        total = len(self.devices)
        crash_rate = crashes / total if total else 0.0
        return {"crash_rate": crash_rate, "crashes": crashes, "population": total}

    def apply_firmware(self, cohort_ids, version):
        ids = set(cohort_ids)
        for d in self.devices:
            if d.device_id in ids:
                d.firmware = version
