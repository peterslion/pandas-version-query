"""Load the car fuel efficiency CSV and print how many records it contains."""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "data" / "car_fuel_efficiency_2026.csv"


def main() -> None:
    frame = pd.read_csv(DATA_PATH)
    print(f"pandas {pd.__version__}")
    print(f"records: {len(frame)}")


if __name__ == "__main__":
    main()
