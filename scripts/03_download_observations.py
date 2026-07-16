"""Download matched SDO/AIA and STEREO/EUVI FITS observations."""

from pathlib import Path
from time import sleep
from urllib.request import urlretrieve

import astropy.units as u
import drms
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from tqdm import tqdm


DATA_ROOT = Path(r"E:\helioseismology\sdo_stereo_euvi")
MATCH_ROOT = DATA_ROOT / "matches"
FITS_ROOT = DATA_ROOT / "fits"
STEREO_WAVELENGTHS = (304, 195, 171)
AIA_WAVELENGTH = {304: 304, 195: 193, 171: 171}
STEREO_BASE_URL = (
    "https://stereo-ssc.nascom.nasa.gov/data/ins_data/secchi/L0_YMD"
)
JSOC_BASE_URL = "http://jsoc.stanford.edu"
RETRIES = 5


def is_valid_fits(path):
    """Return whether a FITS file exists and all of its HDUs can be read."""
    try:
        with fits.open(path, memmap=False) as hdus:
            for hdu in hdus:
                _ = hdu.data
        return True
    except Exception:
        return False


def download_stereo_euvi(url, path):
    """Download and validate one EUVI FITS file, retrying transient failures."""
    if is_valid_fits(path):
        return True

    path.unlink(missing_ok=True)
    part = path.with_suffix(".fits.part")

    for attempt in range(1, RETRIES + 1):
        try:
            part.unlink(missing_ok=True)
            urlretrieve(url, part)
            if not is_valid_fits(part):
                raise OSError("Downloaded FITS file is incomplete")
            part.replace(path)
            return True
        except Exception as error:
            part.unlink(missing_ok=True)
            if attempt == RETRIES:
                print(f"Failed: {path.name} ({error})")
                return False
            print(f"Retry {attempt}/{RETRIES}: {path.name} ({error})")
            sleep(2 ** (attempt - 1))


def download_sdo_aia(path, obstime, wavelength):
    """Download the nearest QUALITY=0 AIA FITS observation within 30 minutes."""
    if is_valid_fits(path):
        return True

    path.unlink(missing_ok=True)
    window = 30 * u.minute
    time_range = (
        f"{(obstime - window).strftime('%Y.%m.%d_%H:%M:%S')}-"
        f"{(obstime + window).strftime('%Y.%m.%d_%H:%M:%S')}"
    )

    for attempt in range(1, RETRIES + 1):
        try:
            keys, segments = drms.Client().query(
                f"aia.lev1_euv_12s[{time_range}][{wavelength}]",
                key=drms.JsocInfoConstants.all,
                seg="image",
            )
            quality_keys = keys[keys["QUALITY"] == 0]
            if quality_keys.empty:
                raise RuntimeError("No QUALITY=0 record")
            times = Time.strptime(
                quality_keys["T_OBS"].to_numpy().astype(str),
                format_string="%Y-%m-%dT%H:%M:%S.%fZ",
                scale="utc",
            )
            index = quality_keys.index[abs(times - obstime).argmin()]
            header = keys.loc[index].to_dict()
            image_url = JSOC_BASE_URL + segments.loc[index, "image"]

            with fits.open(
                image_url, cache=False, do_not_scale_image_data=True
            ) as hdus:
                for key, value in header.items():
                    if pd.isna(value):
                        continue
                    try:
                        hdus[1].header[key] = value
                    except (TypeError, ValueError):
                        pass
                hdus.writeto(path, overwrite=True, output_verify="silentfix")

            if not is_valid_fits(path):
                raise OSError("Downloaded FITS file is incomplete")
            return True
        except Exception as error:
            path.unlink(missing_ok=True)
            if attempt == RETRIES:
                print(f"Failed: AIA {wavelength} {path.name} ({error})")
                return False
            print(f"Retry {attempt}/{RETRIES}: AIA {wavelength} {path.name} ({error})")
            sleep(2 ** (attempt - 1))


def main():
    """Download every matched triplet for each configured wavelength."""
    for stereo_wavelength in STEREO_WAVELENGTHS:
        wavelength_root = FITS_ROOT / str(stereo_wavelength)
        sdo_aia_dir = wavelength_root / "sdo_aia"
        sta_euvi_dir = wavelength_root / "sta_euvi"
        stb_euvi_dir = wavelength_root / "stb_euvi"
        for directory in (sdo_aia_dir, sta_euvi_dir, stb_euvi_dir):
            directory.mkdir(parents=True, exist_ok=True)

        matches = pd.read_csv(MATCH_ROOT / f"matches_{stereo_wavelength}.csv")
        matches = matches[matches["matched"]]

        for row in tqdm(
            matches.itertuples(index=False),
            total=len(matches),
            desc=str(stereo_wavelength),
        ):
            sdo_time = Time(row.sdo_time)
            filename = sdo_time.strftime("%Y%m%d_%H%M%S.fits")

            if not download_sdo_aia(
                sdo_aia_dir / filename,
                sdo_time,
                AIA_WAVELENGTH[stereo_wavelength],
            ):
                continue

            sta_url = (
                f"{STEREO_BASE_URL}/a/img/euvi/"
                f"{pd.Timestamp(row.stereo_a_time):%Y/%m/%d}/"
                f"{row.stereo_a_filename}"
            )
            if not download_stereo_euvi(sta_url, sta_euvi_dir / filename):
                continue

            stb_url = (
                f"{STEREO_BASE_URL}/b/img/euvi/"
                f"{pd.Timestamp(row.stereo_b_time):%Y/%m/%d}/"
                f"{row.stereo_b_filename}"
            )
            download_stereo_euvi(stb_url, stb_euvi_dir / filename)


if __name__ == "__main__":
    main()
