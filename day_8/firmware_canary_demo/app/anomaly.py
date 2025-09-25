
def gate_pass(baseline_cr, current_cr, threshold=0.0020):
    """
    Gate logic: allow if crash_rate increase <= threshold (absolute delta).
    Example: baseline 0.2% (0.0020) → current up to 0.4% (0.0040) allowed.
    """
    return (current_cr - baseline_cr) <= threshold
