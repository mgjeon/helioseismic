import argparse
from pathlib import Path

import pandas as pd
from astropy.io import fits
from astropy.time import Time, TimeDelta
from tqdm import tqdm


def get_header(path, key):
    with fits.open(path) as hdus:
        return next(hdu.header for hdu in hdus if key in hdu.header)


def get_time(path, source):
    try:
        if source == "sdo":
            return pd.Timestamp(Time(get_header(path, "T_OBS")["T_OBS"]).datetime)

        header = get_header(path, "DATE-OBS")
        exposure = header.get("EXPOSURE_TIME", header.get("EXPTIME"))
        time = Time(header["DATE-OBS"]) + TimeDelta(float(exposure) / 2, format="sec")
        return pd.Timestamp(time.datetime)
    except Exception as error:
        print(f"Invalid FITS: {path} ({error})")
        return pd.NaT


def make_csv(root):
    rows = []

    for wavelength_dir in sorted(root.iterdir()):
        if not wavelength_dir.is_dir() or not wavelength_dir.name.isdigit():
            continue

        dirs = {source: wavelength_dir / source for source in ["sdo", "sta", "stb"]}
        filenames = sorted({
            path.name
            for directory in dirs.values() if directory.exists()
            for path in directory.glob("*.fits")
        })

        for filename in tqdm(filenames, desc=wavelength_dir.name):
            paths = {source: directory / filename for source, directory in dirs.items()}
            times = {
                source: get_time(path, source) if path.exists() else pd.NaT
                for source, path in paths.items()
            }
            rows.append({
                "wavelength": int(wavelength_dir.name),
                "filename": filename,
                "sdo_available": pd.notna(times["sdo"]),
                "sta_available": pd.notna(times["sta"]),
                "stb_available": pd.notna(times["stb"]),
                "sdo_path": paths["sdo"].relative_to(root) if paths["sdo"].exists() else None,
                "sta_path": paths["sta"].relative_to(root) if paths["sta"].exists() else None,
                "stb_path": paths["stb"].relative_to(root) if paths["stb"].exists() else None,
                "sdo_time": times["sdo"],
                "sta_time": times["sta"],
                "stb_time": times["stb"],
                "sta_diff_minutes": abs(times["sta"] - times["sdo"]).total_seconds() / 60,
                "stb_diff_minutes": abs(times["stb"] - times["sdo"]).total_seconds() / 60,
            })

    pd.DataFrame(rows).to_csv(root / "availability.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    make_csv(args.root)


if __name__ == "__main__":
    main()
