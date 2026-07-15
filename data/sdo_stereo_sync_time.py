import argparse
from pathlib import Path

import pandas as pd


START_TIME = "2010-05-13 00:00:00"


def load_times(path, wavelength):
    df = pd.read_csv(path, usecols=["FileName", "DateObs", "Polar"])
    df["DateObs"] = pd.to_datetime(df["DateObs"], errors="coerce")
    df["Polar"] = pd.to_numeric(df["Polar"], errors="coerce")
    return (
        df.loc[df["Polar"] == wavelength, ["DateObs", "FileName"]]
        .dropna()
        .drop_duplicates("DateObs")
        .sort_values("DateObs")
        .rename(columns={"DateObs": "time", "FileName": "filename"})
    )


def make_sync_time_csv(summary_dir, output_path, wavelength=304):
    stereo_a = load_times(summary_dir / "STEREO-A.csv", wavelength)
    stereo_b = load_times(summary_dir / "STEREO-B.csv", wavelength)

    target = pd.DataFrame({
        "sdo_time": pd.date_range(START_TIME, stereo_b["time"].max(), freq="12h")
    })
    result = pd.merge_asof(
        target,
        stereo_a.rename(columns={
            "time": "stereo_a_time",
            "filename": "stereo_a_filename",
        }),
        left_on="sdo_time",
        right_on="stereo_a_time",
        direction="nearest",
    )
    result = pd.merge_asof(
        result,
        stereo_b.rename(columns={
            "time": "stereo_b_time",
            "filename": "stereo_b_filename",
        }),
        left_on="sdo_time",
        right_on="stereo_b_time",
        direction="nearest",
    )
    result["stereo_a_diff_minutes"] = (
        result["stereo_a_time"] - result["sdo_time"]
    ).abs().dt.total_seconds() / 60
    result["stereo_b_diff_minutes"] = (
        result["stereo_b_time"] - result["sdo_time"]
    ).abs().dt.total_seconds() / 60
    result["matched"] = (
        (result["stereo_a_diff_minutes"] <= 30)
        & (result["stereo_b_diff_minutes"] <= 30)
    )
    result.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_dir", type=Path, help="STEREO summary CSV 폴더")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--wavelength", type=int, default=304)
    args = parser.parse_args()

    output_path = (
        args.output
        or args.summary_dir / f"sdo_stereo_sync_time_{args.wavelength}.csv"
    )
    make_sync_time_csv(args.summary_dir, output_path, args.wavelength)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
