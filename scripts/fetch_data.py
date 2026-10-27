"""Download and build every dataset used in the paper from its public source (SHA-256 pinned where the source is fixed).

  python scripts/fetch_data.py            # writes data/raw and data/processed
"""
from pathlib import Path

from medhyperdiff.data import build_nhanes, build_nhanes_cbc, build_nhanes_cycle, build_thyroid, download_uci

ROOT = Path(__file__).resolve().parents[1] / "data"


def main():
    for name in ("ilpd", "heart_failure", "ckd", "diabetes"):
        print(download_uci(name, ROOT))
    print(build_thyroid(ROOT))
    print(build_nhanes(ROOT))
    print(build_nhanes_cbc(ROOT))
    for panel, cycles in (("cbc", "hil"), ("chem", "gil")):
        for cycle in cycles:
            print(build_nhanes_cycle(ROOT, panel, cycle))


if __name__ == "__main__":
    main()
