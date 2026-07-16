"""Match regular SDO target times with nearby STEREO observations."""

from pathlib import Path

import pandas as pd


DATA_ROOT = Path(r"E:\helioseismology\sdo_stereo_euvi")
CATALOG_ROOT = DATA_ROOT / "catalog"
MATCH_ROOT = DATA_ROOT / "matches"
WAVELENGTHS = (304, 195, 171)
START_TIME = "2010-05-13 00:00:00"
MAX_TIME_DIFFERENCE = pd.Timedelta(minutes=30)
MIN_IMAGE_SIZE = 2048
CATALOG_COLUMNS = [
    "FileName", "DateObs", "Polar", "Xsize", "Ysize", "NMISS", "Prog",
]


def load_stereo_catalog(path):
    """Load and normalize the STEREO matching and quality columns."""
    catalog = pd.read_csv(path, usecols=CATALOG_COLUMNS)
    catalog["DateObs"] = pd.to_datetime(catalog["DateObs"], errors="coerce")
    for column in ("Polar", "Xsize", "Ysize", "NMISS"):
        catalog[column] = pd.to_numeric(catalog[column], errors="coerce")
    return catalog.dropna(subset=CATALOG_COLUMNS)


def select_wavelength(catalog, wavelength):
    """Select full-size, complete, normal EUVI observations at one wavelength."""
    quality = (
        catalog["Xsize"].ge(MIN_IMAGE_SIZE)
        & catalog["Ysize"].ge(MIN_IMAGE_SIZE)
        & catalog["NMISS"].eq(0)
        & catalog["Prog"].astype("string").str.casefold().eq("norm").fillna(False)
    )
    return (
        catalog.loc[
            (catalog["Polar"] == wavelength) & quality,
            ["DateObs", "FileName", "Xsize", "Ysize", "NMISS", "Prog"],
        ]
        .drop_duplicates("DateObs")
        .sort_values("DateObs")
        .rename(columns={
            "DateObs": "time",
            "FileName": "filename",
            "Xsize": "xsize",
            "Ysize": "ysize",
            "NMISS": "nmiss",
            "Prog": "program",
        })
    )


def build_match_table(stereo_a_catalog, stereo_b_catalog, wavelength):
    """Match 12-hour SDO targets to STEREO-A/B and write a wavelength CSV."""
    stereo_a = select_wavelength(stereo_a_catalog, wavelength).add_prefix("stereo_a_")
    stereo_b = select_wavelength(stereo_b_catalog, wavelength).add_prefix("stereo_b_")
    matches = pd.DataFrame({
        "sdo_time": pd.date_range(
            START_TIME, stereo_b["stereo_b_time"].max(), freq="12h"
        )
    })

    matches = pd.merge_asof(
        matches,
        stereo_a,
        left_on="sdo_time",
        right_on="stereo_a_time",
        direction="nearest",
    )
    matches = pd.merge_asof(
        matches,
        stereo_b,
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
