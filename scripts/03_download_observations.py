"""Download matched SDO/AIA and STEREO/EUVI FITS observations."""

import re
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from time import sleep

import astropy.units as u
import drms
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from parfive import Downloader
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
JSOC_QUERY_KEYS = (
    f"{drms.JsocInfoConstants.all.value},{drms.JsocInfoConstants.recnum.value}"
)
RETRIES = 5
HEADER_ALIASES = {
    "DATE__OBS": "DATE-OBS",
    "T_REC_epoch": "TRECEPOC",
    "T_REC_step": "TRECSTEP",
    "T_REC_unit": "TRECUNIT",
    "NOISEMASK": "NOISEMAS",
}
SEGMENT_KEYWORD = re.compile(
    r"^(BUNIT|DATAKURT|DATAMAX|DATAMEAN|DATAMEDN|DATAMIN|DATARMS|"
    r"DATASKEW|DATAVALS|MISSVALS)_\d{3}$"
)
BLANK_EXPORT_KEYS = {
    "BKEYD1", "BKEYD2", "BKEYD3", "BKEYI1", "BKEYI2", "BKEYI3",
    "CALVER32", "CSYSER1", "CSYSER2",
}
STALE_AIA_KEYS = {
    "DATE_OBS", "DATE__OBS", "T_REC_epoch", "T_REC_step", "T_REC_unit",
    "T_REC_roun", "TRECROUN", "NOISEMASK", "BLD_VERS", "ROI_NWIN",
}
JSOC_TIME_PATTERN = re.compile(
    r"^(?P<base>\d{4}(?:-\d{2}-\d{2}T|\.\d{2}\.\d{2}_)\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d+))?(?P<suffix>Z|_TAI)?$"
)


def query_max_precision(client, *args, **kwargs):
    """Run a DRMS query with JSOC's maximum-precision ``M=1`` flag."""
    json_request = client._json._json_request

    def max_precision_request(url):
        """Append ``M=1`` to JSOC record-list requests."""
        if url.startswith(client._server.url_jsoc_info) and "?op=rs_list" in url:
            url += "&M=1"
        return json_request(url)

    client._json._json_request = max_precision_request
    try:
        return client.query(*args, **kwargs)
    finally:
        client._json._json_request = json_request


def format_export_time(value):
    """Round a maximum-precision JSOC timestamp to export milliseconds."""
    match = JSOC_TIME_PATTERN.match(value)
    if match is None:
        return value

    base = match.group("base")
    fraction = match.group("fraction") or "0"
    date_format = "%Y-%m-%dT%H:%M:%S" if "-" in base else "%Y.%m.%d_%H:%M:%S"
    milliseconds = int(
        (Decimal(f"0.{fraction}") * 1000).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
    )
    timestamp = datetime.strptime(base, date_format) + timedelta(
        milliseconds=milliseconds
    )
    suffix = "_TAI" if match.group("suffix") == "_TAI" else ""
    return (
        timestamp.strftime(date_format)
        + f".{timestamp.microsecond // 1000:03d}"
        + suffix
    )


def is_valid_fits(path):
    """Return whether a FITS file exists and all of its HDUs can be read."""
    try:
        with fits.open(path, memmap=False) as hdus:
            for hdu in hdus:
                _ = hdu.data
        return True
    except Exception:
        return False


def download_file(url, path):
    """Download one URL to an exact path with parfive and one network retry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    downloader = Downloader(max_conn=1, max_splits=1, overwrite=True)
    downloader.enqueue_file(url, path=path.parent, filename=path.name)
    result = downloader.download()
    if result.errors:
        result = downloader.retry(result)
    if result.errors or not path.is_file():
        raise RuntimeError(f"Failed to download {url}: {result.errors!r}")


def prepare_aia_fits(keywords, path):
    """Open a downloaded AIA segment and apply JSOC export header conventions."""
    hdus = fits.open(path, do_not_scale_image_data=True)
    header = hdus[1].header
    hdus[0].header["BITPIX"] = 8

    for key, value in keywords.items():
        if key in {"*recnum*", "T_REC_roun"} or SEGMENT_KEYWORD.match(key):
            continue

        if pd.isna(value):
            if key in {"CRDER1", "CRDER2"}:
                value = 0.0
            elif key not in header and key not in BLANK_EXPORT_KEYS:
                continue
            else:
                header.pop(key, None)
                value = None
        elif isinstance(value, str) and value.lower() in {"missing", "nan"}:
            value = None
        elif key.startswith("ROI_") and value == -(2**31):
            value = None

        export_key = HEADER_ALIASES.get(key, key)
        if export_key in {
            "T_REC", "TRECEPOC", "T_OBS", "DATE-OBS", "DATE", "DATE_ME",
            "DATE_S", "ISPPKTIM",
        } and isinstance(value, str):
            value = format_export_time(value)

        if export_key == "HISTORY":
            if value is None:
                continue
            for line in value.splitlines():
                header.add_history("".join(char for char in line if ord(char) >= 32))
            continue
        if isinstance(value, str):
            value = "".join(char for char in value if ord(char) >= 32)
        try:
            header[export_key] = value
        except (TypeError, ValueError):
            print(f"Skipped invalid FITS header: {export_key}={value!r}")

    recnum = keywords.get("*recnum*")
    if not pd.isna(recnum):
        recnum = int(recnum)
        header["DRMS_ID"] = f"aia.lev1_euv_12s:{recnum}:image"
        header["PRIMARYK"] = "T_REC, WAVELNTH"
        header["RECNUM"] = recnum
        header["LONGSTRN"] = "OGIP 1.0"

    for stale_key in STALE_AIA_KEYS:
        header.pop(stale_key, None)
    return hdus


def download_stereo_euvi(url, path):
    """Download and validate one EUVI FITS file, retrying transient failures."""
    if is_valid_fits(path):
        return True

    path.unlink(missing_ok=True)
    part = path.with_suffix(".fits.part")

    for attempt in range(1, RETRIES + 1):
        try:
            part.unlink(missing_ok=True)
            download_file(url, part)
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
            client = drms.Client()
            keys, segments = query_max_precision(
                client,
                f"aia.lev1_euv_12s[{time_range}][{wavelength}]",
                key=JSOC_QUERY_KEYS,
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
            keywords = keys.loc[index].to_dict()
            image_url = JSOC_BASE_URL + segments.loc[index, "image"]
            part = path.with_suffix(".fits.part")
            part.unlink(missing_ok=True)
            download_file(image_url, part)

            with prepare_aia_fits(keywords, part) as hdus:
                hdus.writeto(path, overwrite=True, output_verify="silentfix")
            part.unlink(missing_ok=True)

            if not is_valid_fits(path):
                raise OSError("Downloaded FITS file is incomplete")
            return True
        except Exception as error:
            path.unlink(missing_ok=True)
            path.with_suffix(".fits.part").unlink(missing_ok=True)
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
