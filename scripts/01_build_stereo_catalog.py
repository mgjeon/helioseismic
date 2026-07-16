"""Download STEREO summary files and build observation catalogs."""

import re
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
from parfive import Downloader
from tqdm import tqdm


DATA_ROOT = Path(r"E:\helioseismology\sdo_stereo_euvi")
CATALOG_ROOT = DATA_ROOT / "catalog"
SUMMARY_URLS = {
    "A": "https://stereo-ssc.nascom.nasa.gov/data/ins_data/secchi/L0_YMD/a/summary/",
    "B": "https://stereo-ssc.nascom.nasa.gov/data/ins_data/secchi/L0_YMD/b/summary/",
}
SUMMARY_COLUMNS = [
    "FileName", "DateObs", "Tel", "Exptime", "Xsize", "Ysize", "Filter",
    "Polar", "Prog", "OSnum", "Dest", "FPS", "LED", "CMPRS", "NMISS",
]


def download_monthly_summaries():
    """Download missing monthly summary files for STEREO-A and STEREO-B."""
    downloader = Downloader()

    for spacecraft, summary_url in SUMMARY_URLS.items():
        output_dir = CATALOG_ROOT / f"st{spacecraft.lower()}_euvi"
        output_dir.mkdir(parents=True, exist_ok=True)
        html = urlopen(summary_url).read().decode()

        for filename in re.findall(
            rf'href="(scc{spacecraft}\d{{6}}\.img\.eu)"', html
        ):
            if not (output_dir / filename).exists():
                downloader.enqueue_file(summary_url + filename, path=output_dir)

    result = downloader.download()
    if result.errors:
        result = downloader.retry(result)
    if result.errors:
        raise RuntimeError(f"Failed to download STEREO summaries: {result.errors!r}")


def build_stereo_catalogs():
    """Combine monthly summaries into one chronological CSV per spacecraft."""
    for spacecraft in SUMMARY_URLS:
        rows = []
        paths = sorted(
            (CATALOG_ROOT / f"st{spacecraft.lower()}_euvi").glob("*.img.eu")
        )

        for path in tqdm(paths, desc=f"STEREO-{spacecraft} catalog"):
            data = pd.read_csv(
                path,
                sep="|",
                skiprows=2,
                names=SUMMARY_COLUMNS,
                on_bad_lines="skip",
            )
            data = data.map(
                lambda value: value.strip() if isinstance(value, str) else value
            )
            data["DateObs"] = pd.to_datetime(
                data["DateObs"], format="%Y/%m/%d %H:%M:%S", errors="coerce"
            )
            rows.append(data.dropna(subset=["DateObs"]))

        if rows:
            catalog = pd.concat(rows, ignore_index=True).sort_values("DateObs")
            output_path = CATALOG_ROOT / f"st{spacecraft.lower()}_euvi.csv"
            catalog.to_csv(output_path, index=False)


def main():
    """Download monthly summaries and rebuild both STEREO catalogs."""
    CATALOG_ROOT.mkdir(parents=True, exist_ok=True)
    download_monthly_summaries()
    build_stereo_catalogs()


if __name__ == "__main__":
    main()
