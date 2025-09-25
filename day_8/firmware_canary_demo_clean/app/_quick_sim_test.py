from simulator import Device, Fleet

fleet = Fleet([Device(f"D{i:04d}", model=("SensorX-200" if i % 4 == 0 else "SensorX-100")) for i in range(1000)])
print("Epoch 0:", fleet.tick())           # baseline crash rate
fleet.apply_firmware([f"D{i:04d}" for i in range(10)], "1.0.1")
print("Epoch 1:", fleet.tick())           # small change
