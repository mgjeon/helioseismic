"""Match regular SDO target times with nearby STEREO observations."""

from pathlib import Path

import pandas as pd


DATA_ROOT = Path(r"E:\helioseismology\sdo_stereo_euvi")
CATALOG_ROOT = DATA_ROOT / "catalog"
MATCH_ROOT = DATA_ROOT / "matches"
WAVELENGTHS = (304, 195, 171)
START_TIME = "2010-05-13 00:00:00"
MAX_TIME_DIFFERENCE = pd.Timedelta(minutes=30)


def load_stereo_catalog(path):
    """Load and normalize the filename, observation time, and wavelength columns."""
    catalog = pd.read_csv(path, usecols=["FileName", "DateObs", "Polar"])
    catalog["DateObs"] = pd.to_datetime(catalog["DateObs"], errors="coerce")
    catalog["Polar"] = pd.to_numeric(catalog["Polar"], errors="coerce")
    return catalog.dropna()


def select_wavelength(catalog, wavelength):
    """Return unique chronological observation times for one EUVI wavelength."""
    return (
        catalog.loc[catalog["Polar"] == wavelength, ["DateObs", "FileName"]]
        .drop_duplicates("DateObs")
        .sort_values("DateObs")
        .rename(columns={"DateObs": "time", "FileName": "filename"})
    )


def build_match_table(stereo_a_catalog, stereo_b_catalog, wavelength):
    """Match 12-hour SDO targets to STEREO-A/B and write a wavelength CSV."""
    stereo_a = select_wavelength(stereo_a_catalog, wavelength)
    stereo_b = select_wavelength(stereo_b_catalog, wavelength)
    matches = pd.DataFrame({
        "sdo_time": pd.date_range(START_TIME, stereo_b["time"].max(), freq="12h")
    })

    matches = pd.merge_asof(
        matches,
        stereo_a.rename(columns={
            "time": "stereo_a_time",
            "filename": "stereo_a_filename",
        }),
        left_on="sdo_time",
        right_on="stereo_a_time",
        direction="nearest",
    )
    matches = pd.merge_asof(
        matches,
        stereo_b.rename(columns={
            "time": "stereo_b_time",
            "filename": "stereo_b_filename",
        }),
        left_on="sdo_time",
        right_on="stereo_b_time",
        direction="nearest",
    )

    stereo_a_diff = (matches["stereo_a_time"] - matches["sdo_time"]).abs()
    stereo_b_diff = (matches["stereo_b_time"] - matches["sdo_time"]).abs()
    matches["stereo_a_diff_minutes"] = stereo_a_diff.dt.total_seconds() / 60
    matches["stereo_b_diff_minutes"] = stereo_b_diff.dt.total_seconds() / 60
    matches["matched"] = (
        (stereo_a_diff <= MAX_TIME_DIFFERENCE)
        & (stereo_b_diff <= MAX_TIME_DIFFERENCE)
    )

    output_path = MATCH_ROOT / f"matches_{wavelength}.csv"
    matches.to_csv(output_path, index=False)
    print(f"Saved {output_path}")


def main():
    """Build match tables for all configured STEREO wavelengths."""
    MATCH_ROOT.mkdir(parents=True, exist_ok=True)
    stereo_a_catalog = load_stereo_catalog(CATALOG_ROOT / "sta_euvi.csv")
    stereo_b_catalog = load_stereo_catalog(CATALOG_ROOT / "stb_euvi.csv")
    for wavelength in WAVELENGTHS:
        build_match_table(stereo_a_catalog, stereo_b_catalog, wavelength)


if __name__ == "__main__":
    main()
