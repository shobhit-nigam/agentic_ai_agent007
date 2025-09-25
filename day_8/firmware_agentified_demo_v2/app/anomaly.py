def gate_pass(baseline_cr, current_cr, threshold=0.0020):
    return (current_cr - baseline_cr) <= threshold
