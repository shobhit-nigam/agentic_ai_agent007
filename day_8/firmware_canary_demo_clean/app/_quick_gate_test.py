from anomaly import gate_pass

print("PASS?", gate_pass(0.0020, 0.0030))  # True
print("PASS?", gate_pass(0.0020, 0.0050))  # False
