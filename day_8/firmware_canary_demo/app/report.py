
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def make_report(kpi_csv: str, plot_path: str, summary_path: str, baseline: float):
    df = pd.read_csv(kpi_csv)
    # Plot crash rate over epochs
    plt.figure()
    plt.plot(df["epoch"], df["crash_rate"])
    plt.xlabel("Epoch")
    plt.ylabel("Crash Rate")
    plt.title("Crash Rate Over Time (Canary Waves)")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    last = df.iloc[-1]
    rolled_back = bool(df["rolled_back"].any())
    decision = "ROLLED BACK" if rolled_back else "ROLLOUT COMPLETED"

    lines = [
        f"Decision: {decision}",
        f"Baseline crash rate: {baseline:.4f}",
        f"Final crash rate: {last['crash_rate']:.4f}",
        f"Total epochs: {int(last['epoch'])+1}",
        f"Notes: Threshold gate is absolute delta = 0.0020",
    ]
    Path(summary_path).write_text("\n".join(lines))
    return decision
