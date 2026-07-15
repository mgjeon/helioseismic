import argparse
import re
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import pandas as pd
from parfive import Downloader
from tqdm import tqdm


SUMMARY_URLS = {
    "A": "https://stereo-ssc.nascom.nasa.gov/data/ins_data/secchi/L0_YMD/a/summary/",
    "B": "https://stereo-ssc.nascom.nasa.gov/data/ins_data/secchi/L0_YMD/b/summary/",
}

SUMMARY_COLUMNS = [
    "FileName", "DateObs", "Tel", "Exptime", "Xsize", "Ysize", "Filter",
    "Polar", "Prog", "OSnum", "Dest", "FPS", "LED", "CMPRS", "NMISS",
]


def download_summary_files(output_dir):
    downloader = Downloader(overwrite=True)

    for spacecraft, summary_url in SUMMARY_URLS.items():
        spacecraft_dir = output_dir / f"st{spacecraft.lower()}"
        spacecraft_dir.mkdir(parents=True, exist_ok=True)

        html = urlopen(summary_url).read().decode()
        filenames = re.findall(
            rf'href="(scc{spacecraft}\d{{6}}\.img\.eu)"', html
        )

        for filename in filenames:
            output_path = spacecraft_dir / filename
            file_url = urljoin(summary_url, filename)

            if output_path.exists():
                request = Request(file_url, method="HEAD")
                with urlopen(request) as response:
                    remote_size = int(response.headers["Content-Length"])
                if output_path.stat().st_size == remote_size:
                    print(f"Skipping {output_path}")
                    continue

            downloader.enqueue_file(
                file_url,
                path=spacecraft_dir,
                filename=filename,
            )

    downloader.download()


def make_csv_files(output_dir):
    for spacecraft in SUMMARY_URLS:
        rows = []
        paths = sorted((output_dir / f"st{spacecraft.lower()}").glob("*.img.eu"))
        for path in tqdm(paths, desc=f"STEREO-{spacecraft} CSV"):
            df = pd.read_csv(
                path,
                sep="|",
                skiprows=2,
                names=SUMMARY_COLUMNS,
                on_bad_lines="skip",
            )
            df = df.map(lambda value: value.strip() if isinstance(value, str) else value)
            df["DateObs"] = pd.to_datetime(
                df["DateObs"],
                format="%Y/%m/%d %H:%M:%S",
                errors="coerce",
            )
            rows.append(df.dropna(subset=["DateObs"]))

        if rows:
            result = pd.concat(rows, ignore_index=True).sort_values("DateObs")
            result.to_csv(output_dir / f"STEREO-{spacecraft}.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path, help="다운로드할 폴더")
    args = parser.parse_args()
    download_summary_files(args.output_dir)
    make_csv_files(args.output_dir)


if __name__ == "__main__":
    main()
