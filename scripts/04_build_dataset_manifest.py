"""Inspect downloaded FITS files and build a dataset manifest."""

from pathlib import Path

import pandas as pd
from astropy.io import fits
from astropy.time import Time, TimeDelta
from tqdm import tqdm


DATA_ROOT = Path(r"E:\helioseismology\sdo_stereo_euvi")
FITS_ROOT = DATA_ROOT / "fits"


def read_observation_time(path, source):
    """Read an SDO time or the exposure-centered STEREO time from a FITS file."""
    try:
        with fits.open(path) as hdus:
            key = "T_OBS" if source == "sdo_aia" else "DATE-OBS"
            header = next(hdu.header for hdu in hdus if key in hdu.header)

        time = Time(header[key])
        if source != "sdo_aia":
            exposure = header.get("EXPOSURE_TIME", header.get("EXPTIME"))
            if exposure is None:
                raise KeyError("EXPOSURE_TIME or EXPTIME")
            time += TimeDelta(float(exposure) / 2, format="sec")
        return pd.Timestamp(time.datetime)
    except Exception as error:
        print(f"Invalid FITS: {path} ({error})")
        return pd.NaT


def build_dataset_manifest():
    """Write availability, paths, times, and time offsets for all FITS files."""
    rows = []

    for wavelength_dir in sorted(FITS_ROOT.iterdir()):
        if not wavelength_dir.is_dir() or not wavelength_dir.name.isdigit():
            continue

        directories = {
            source: wavelength_dir / source
            for source in ("sdo_aia", "sta_euvi", "stb_euvi")
        }
        filenames = sorted({
            path.name
            for directory in directories.values()
            if directory.exists()
            for path in directory.glob("*.fits")
        })

        for filename in tqdm(filenames, desc=wavelength_dir.name):
            paths = {
                source: directory / filename
                for source, directory in directories.items()
            }
            times = {
                source: (
                    read_observation_time(path, source) if path.exists() else pd.NaT
                )
                for source, path in paths.items()
            }
            rows.append({
                "wavelength": int(wavelength_dir.name),
                "filename": filename,
                "sdo_aia_available": pd.notna(times["sdo_aia"]),
                "sta_euvi_available": pd.notna(times["sta_euvi"]),
                "stb_euvi_available": pd.notna(times["stb_euvi"]),
                "sdo_aia_path": (
                    paths["sdo_aia"].relative_to(DATA_ROOT)
                    if paths["sdo_aia"].exists()
                    else None
                ),
                "sta_euvi_path": (
                    paths["sta_euvi"].relative_to(DATA_ROOT)
                    if paths["sta_euvi"].exists()
                    else None
                ),
                "stb_euvi_path": (
                    paths["stb_euvi"].relative_to(DATA_ROOT)
                    if paths["stb_euvi"].exists()
                    else None
                ),
                "sdo_aia_time": times["sdo_aia"],
                "sta_euvi_time": times["sta_euvi"],
                "stb_euvi_time": times["stb_euvi"],
                "sta_euvi_diff_minutes": abs(
                    times["sta_euvi"] - times["sdo_aia"]
                ).total_seconds() / 60,
                "stb_euvi_diff_minutes": abs(
                    times["stb_euvi"] - times["sdo_aia"]
                ).total_seconds() / 60,
            })

    output_path = DATA_ROOT / "dataset_manifest.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved {output_path}")


def main():
    """Build the manifest for the configured dataset root."""
    build_dataset_manifest()


if __name__ == "__main__":
    main()
