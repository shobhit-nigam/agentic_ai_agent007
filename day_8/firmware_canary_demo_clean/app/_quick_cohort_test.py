from rollout import make_cohorts, wave_name

cohorts = make_cohorts(range(1000))
print([len(c) for c in cohorts], [wave_name(i) for i in range(4)])
