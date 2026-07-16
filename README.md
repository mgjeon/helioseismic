# SDO–STEREO EUV dataset

SDO/AIA와 STEREO-A/B EUVI의 근접 관측을 12시간 간격으로 수집한다.

## 데이터 구조

모든 결과는 `E:\helioseismology\sdo_stereo_euvi` 아래에 저장한다.

```text
sdo_stereo_euvi/
├── catalog/
│   ├── sta_euvi/
│   ├── stb_euvi/
│   ├── sta_euvi.csv
│   └── stb_euvi.csv
├── matches/
│   ├── matches_304.csv
│   ├── matches_195.csv
│   └── matches_171.csv
├── fits/
│   ├── 304/
│   │   ├── sdo_aia/
│   │   ├── sta_euvi/
│   │   └── stb_euvi/
│   ├── 195/
│   └── 171/
└── dataset_manifest.csv
```

경로를 변경하려면 각 스크립트 상단의 `DATA_ROOT`를 수정한다.
기존 데이터는 새 경로로 이전하지 않는다.

## 실행

아래 명령을 순서대로 실행한다.

```powershell
uv run scripts/01_build_stereo_catalog.py
uv run scripts/02_match_observation_times.py
uv run scripts/03_download_observations.py
uv run scripts/04_build_dataset_manifest.py
```

1. 월별 STEREO summary를 내려받아 우주선별 카탈로그를 만든다.
2. 품질 조건을 만족하며 SDO 기준 시각에서 30분 이내인 STEREO-A/B 관측을 찾는다.
3. `parfive`로 SDO/AIA와 STEREO/EUVI FITS를 내려받고 손상 여부를 검사한다.
4. 파일 경로, 관측 시각, 가용 여부와 시각 차이를 `dataset_manifest.csv`에 기록한다.

STEREO 304/195/171 Å에는 각각 SDO/AIA 304/193/171 Å를 대응시킨다.
SDO/AIA FITS 헤더는 JSOC export 규칙에 맞게 정규화한다.
STEREO/EUVI는 normal 관측 중 크기가 2048×2048 이상이고 누락 블록이 0인
파일만 매칭하며, 다운로드 후 실제 FITS 헤더로 같은 조건을 다시 검사한다.
