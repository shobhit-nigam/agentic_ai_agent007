def gate_pass(baseline_cr, current_cr, threshold=0.0020):
    """
    Allow wave if crash-rate increase (absolute delta) is within threshold.
    Example: baseline 0.0020 → allowed up to 0.0040.
    """
    return (current_cr - baseline_cr) <= threshold
