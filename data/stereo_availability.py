import argparse
import re
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from parfive import Downloader


SUMMARY_URLS = {
    "A": "https://stereo-ssc.nascom.nasa.gov/data/ins_data/secchi/L0_YMD/a/summary/",
    "B": "https://stereo-ssc.nascom.nasa.gov/data/ins_data/secchi/L0_YMD/b/summary/",
}


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path, help="다운로드할 폴더")
    args = parser.parse_args()
    download_summary_files(args.output_dir)


if __name__ == "__main__":
    main()
