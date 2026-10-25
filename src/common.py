from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "results" / "figures"
DATA = ROOT / "results"
FIG.mkdir(parents=True, exist_ok=True)
