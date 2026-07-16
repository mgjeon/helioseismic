from pathlib import Path
from time import sleep

import pandas as pd
from astropy.io import fits
from astropy.time import Time
from tqdm import tqdm

from utils import _download, _fits_url, down_aia_euv


SUMMARY_ROOT = Path(r"E:\helioseismology\stereo_euvi")
OUTPUT_ROOT = Path(r"E:\helioseismology\sdo_stereo_sync")
WAVELENGTHS = [304, 195, 171]


def valid_fits(path):
    try:
        with fits.open(path, memmap=False) as hdus:
            for hdu in hdus:
                _ = hdu.data
        return True
    except Exception:
        return False


def download_stereo(url, path, retries=5):
    if valid_fits(path):
        return True
    path.unlink(missing_ok=True)
    part = path.with_suffix(path.suffix + ".part")

    for attempt in range(retries):
        try:
            part.unlink(missing_ok=True)
            _download(url, part, desc=path.name)
            if not valid_fits(part):
                raise OSError("Downloaded FITS file is incomplete")
            part.replace(path)
            return True
        except Exception as error:
            part.unlink(missing_ok=True)
            if attempt == retries - 1:
                print(f"Failed after {retries} attempts, skipping: {path.name} ({error})")
                return False
            print(f"Retry {attempt + 1}/{retries}: {path.name} ({error})")
            sleep(2 ** attempt)


def download_sdo(path, obstime, wavelength, retries=5):
    if valid_fits(path):
        return True
    path.unlink(missing_ok=True)

    for attempt in range(retries):
        downloaded = down_aia_euv(path.parent, obstime, wavelength=wavelength)
        if downloaded is not None and valid_fits(path):
            return True
        path.unlink(missing_ok=True)
        if attempt < retries - 1:
            print(f"Retry {attempt + 1}/{retries}: {path.name}")
            sleep(2 ** attempt)
    return False


def main():
    for wavelength in WAVELENGTHS:
        root = OUTPUT_ROOT / str(wavelength)
        dirs = {name: root / name for name in ["sdo", "sta", "stb"]}
        for directory in dirs.values():
            directory.mkdir(parents=True, exist_ok=True)

        sync = pd.read_csv(
            SUMMARY_ROOT / f"sdo_stereo_sync_time_{wavelength}.csv"
        )
        sync = sync[sync["matched"]]

        for row in tqdm(sync.itertuples(index=False), total=len(sync), desc=str(wavelength)):
            name = pd.Timestamp(row.sdo_time).strftime("%Y%m%d_%H%M%S.fits")
            sdo = dirs["sdo"] / name
            sta = dirs["sta"] / name
            stb = dirs["stb"] / name

            if not download_sdo(sdo, Time(row.sdo_time), wavelength):
                continue
            if not download_stereo(
                _fits_url(
                    "STEREO-A",
                    pd.Timestamp(row.stereo_a_time),
                    row.stereo_a_filename,
                ),
                sta,
            ):
                continue
            if not download_stereo(
                _fits_url(
                    "STEREO-B",
                    pd.Timestamp(row.stereo_b_time),
                    row.stereo_b_filename,
                ),
                stb,
            ):
                continue


if __name__ == "__main__":
    main()
