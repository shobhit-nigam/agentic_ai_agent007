
import random
from dataclasses import dataclass, field

@dataclass
class Device:
    device_id: str
    model: str
    firmware: str = "1.0.0"
    crashed: bool = False
    health: float = 1.0  # 0..1

@dataclass
class Fleet:
    devices: list
    baseline_crash_p: float = 0.002  # 0.2% per epoch
    newfw_crash_p: float = 0.0035    # 0.35% per epoch (worse to trigger demo)
    recovery_p: float = 0.15

    def tick(self):
        """Advance one epoch; compute KPIs."""
        crashes = 0
        for d in self.devices:
            # If previously crashed, some recover
            if d.crashed and random.random() < self.recovery_p:
                d.crashed = False
                d.health = min(1.0, d.health + 0.05)

            # Crash dynamics depend on firmware
            p = self.newfw_crash_p if d.firmware != "1.0.0" else self.baseline_crash_p
            # Slight per-model variation
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
        for d in self.devices:
            if d.device_id in cohort_ids:
                d.firmware = version
