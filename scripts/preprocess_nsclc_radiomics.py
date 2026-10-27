"""Preprocess NSCLC-Radiomics without cropping the lung bounding box.

Lung DICOM-SEG annotations are retained when available.  Cases without a
decodable lung annotation receive a TotalSegmentator lung mask instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.preprocess_nsclc import process_case, sync_split_config, write_report

RAW_ROOT = Path(r"D:\NSCLC-Radiomics")
DATASET_ROOT = ROOT_DIR / "data" / "nsclc_radiomics"
OLD_DATASET_ROOT = Path(r"D:\2.5D Lung Tumor Segmentation\data\nsclc-radiomics")
CASE_IDS: list[str] | None = None
HU_WINDOW_LOW = -700
HU_WINDOW_HIGH = 500
TOTALSEGMENTATOR_DEVICE: str | None = "cuda"
TOTALSEGMENTATOR_RESAMPLE_MM = 3.0
OVERWRITE = True
DRY_RUN = False


def main() -> int:
    sync_split_config(OLD_DATASET_ROOT, DATASET_ROOT)
    cases = sorted(path for path in RAW_ROOT.glob("LUNG*") if path.is_dir())
    requested = set(CASE_IDS or []) # 
    cases = [path for path in cases if not requested or path.name in requested]
    if requested and len(cases) != len(requested):
        raise SystemExit(
            f"requested cases not found: {', '.join(sorted(requested - {path.name for path in cases}))}"
        )
    rows: list[dict[str, str]] = []
    for case_dir in cases:
        try:
            row = process_case(
                case_dir,
                DATASET_ROOT / "processed",
                DATASET_ROOT / "nifti",
                DATASET_ROOT / "intermediate" / "totalsegmentator",
                radiogenomics=False,
                overwrite=OVERWRITE,
                dry_run=DRY_RUN,
                device=TOTALSEGMENTATOR_DEVICE,
                hu_low=HU_WINDOW_LOW,
                hu_high=HU_WINDOW_HIGH,
                output_z_spacing_mm=None,
                totalsegmentator_resample_mm=TOTALSEGMENTATOR_RESAMPLE_MM,
            )
        except Exception as error:
            row = {
                "raw_case_id": case_dir.name,
                "case_id": case_dir.name.replace("LUNG1-", "LUNG-", 1),
                "status": "failed",
                "message": f"{type(error).__name__}: {error}",
            }
        rows.append(row)
        print(f"{row['status'].upper()}: {case_dir.name} {row.get('message', '')}")
        write_report(rows, DATASET_ROOT / "processing_report.csv")
    return 0 if all(row["status"] != "failed" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
