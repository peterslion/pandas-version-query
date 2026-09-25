"""Calculate homework questions Q3 through Q7 for the 2026 fuel-efficiency dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "data" / "car_fuel_efficiency_2026.csv"


def main() -> None:
    frame = pd.read_csv(DATA_PATH)

    fuel_types = frame["fuel_type"].nunique()
    columns_with_missing = int(frame.isna().any().sum())
    max_asia_efficiency = frame.loc[frame["origin"] == "Asia", "fuel_efficiency_mpg"].max()

    horsepower = frame["horsepower"]
    median_before = horsepower.median()
    most_frequent = horsepower.mode().iloc[0]
    frame["horsepower"] = horsepower.fillna(most_frequent)
    median_after = frame["horsepower"].median()
    if median_after > median_before:
        median_change = "Yes, it increased"
    elif median_after < median_before:
        median_change = "Yes, it decreased"
    else:
        median_change = "No"

    asia = frame.loc[frame["origin"] == "Asia", ["vehicle_weight", "model_year"]].head(7)
    x = asia.to_numpy()
    xtx = x.T @ x
    xtx_inverse = np.linalg.inv(xtx)
    y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
    w = xtx_inverse @ x.T @ y
    weight_sum = w.sum()

    print(f"Q3. fuel types: {fuel_types}")
    print(f"Q4. columns with missing values: {columns_with_missing}")
    print(f"Q5. maximum fuel efficiency from Asia: {max_asia_efficiency}")
    print(f"Q6. median horsepower before: {median_before}")
    print(f"Q6. most frequent horsepower: {most_frequent}")
    print(f"Q6. median horsepower after fillna: {median_after}")
    print(f"Q6. has the median changed: {median_change}")
    print(f"Q7. sum of weights: {weight_sum}")


if __name__ == "__main__":
    main()
