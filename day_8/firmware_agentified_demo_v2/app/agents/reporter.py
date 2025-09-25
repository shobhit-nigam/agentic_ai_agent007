from .base import Agent, Message
from report import make_report
from pathlib import Path
import pandas as pd

class ReporterAgent(Agent):
    def __init__(self, name, out_dir: str, trace=None):
        super().__init__(name, trace=trace)
        self.out = Path(out_dir)

    def think(self, state):
        df = pd.DataFrame(state["rows"])
        csv = self.out / "kpis.csv"
        df.to_csv(csv, index=False)
        decision = make_report(
            kpi_csv=str(csv),
            plot_path=str(self.out / "report.png"),
            summary_path=str(self.out / "summary.txt"),
            baseline=state["baseline"],
        )
        return Message("ReportReady", {"decision": decision, "csv": str(csv)})
