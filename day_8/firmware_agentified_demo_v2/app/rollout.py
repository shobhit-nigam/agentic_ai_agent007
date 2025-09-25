import math, random

def make_cohorts(device_ids, waves=(0.01, 0.10, 0.50, 1.00)):
    ids = list(device_ids)
    random.shuffle(ids)
    n = len(ids)
    cohorts, start = [], 0
    for w in waves:
        size = n if w == 1.0 else max(1, math.floor(n * w))
        cohorts.append(ids[start:start+size])
        start += size
    return cohorts

def wave_name(i):
    return ["1%", "10%", "50%", "100%"][i] if i < 4 else f"wave{i+1}"
