"""Match HSPN CZI files to doctor ISKDC/Oxford case-level references.

The source CZI packages use both seven- and eight-digit forms of the same
2026 case number (for example, ``20260351.czi`` maps to ``K2026351``).  This
script creates a reproducible candidate and unique-case manifest so final
scoring validation is based on an explicit case mapping rather than directory
order.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-xlsx", type=Path, required=True)
    parser.add_argument("--atrophy-root", type=Path, required=True)
    parser.add_argument("--task2-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def case_key(value: object) -> str:
    """Normalize K/SK IDs and variable-width CZI case stems to one key."""
    text = re.sub(r"^(?:SK|K)", "", str(value).strip())
    if len(text) == 8 and text.startswith("20260") and text.isdigit():
        return text[:4] + text[5:]
    return text


def czi_case_stem(path: Path) -> str:
    matches = re.findall(r"20\d{5,6}", path.stem)
    if not matches:
        return ""
    return max(matches, key=len)


def load_scores(path: Path) -> pd.DataFrame:
    table = pd.read_excel(path, sheet_name=0, header=None).iloc[2:, :9].copy()
    table.columns = ["case_id", "sex", "age", "iskdc", "M", "E", "S", "T", "C"]
    table = table.dropna(subset=["case_id"])
    table["case_id"] = table["case_id"].astype(str).str.strip()
    table["case_key"] = table["case_id"].map(case_key)
    if table["case_key"].duplicated().any():
        duplicates = table.loc[table["case_key"].duplicated(), "case_id"].tolist()
        raise ValueError("Duplicate doctor-reference case keys: {}".format(duplicates))
    return table


def collect_candidates(root: Path, source_dataset: str, score_keys: set[str]) -> list[dict[str, str]]:
    rows = []
    for czi_path in sorted(root.rglob("*.czi")):
        stem = czi_case_stem(czi_path)
        key = case_key(stem)
        if key in score_keys:
            rows.append(
                {
                    "source_dataset": source_dataset,
                    "czi_stem": stem,
                    "case_key": key,
                    "czi_path": str(czi_path),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores = load_scores(args.score_xlsx)
    score_keys = set(scores["case_key"])
    candidates = pd.DataFrame(
        collect_candidates(args.atrophy_root, "tubular_atrophy_52", score_keys)
        + collect_candidates(args.task2_root, "intrinsic_cell_task2_94", score_keys)
    )
    if candidates.empty:
        raise SystemExit("No CZI files matched the doctor-reference case IDs.")

    candidates = candidates.merge(scores, on="case_key", how="left", validate="many_to_one")
    candidates["candidate_count_for_case"] = candidates.groupby("case_key")["czi_path"].transform("size")
    candidates.to_csv(args.output_dir / "candidate_czi_records.csv", index=False, encoding="utf-8-sig")

    unique = (
        candidates.sort_values(["case_key", "source_dataset", "czi_path"])
        .drop_duplicates("case_key", keep="first")
        .sort_values("case_id")
    )
    unique.to_csv(args.output_dir / "unique_case_manifest.csv", index=False, encoding="utf-8-sig")

    unmatched = scores.loc[~scores["case_key"].isin(set(unique["case_key"]))].sort_values("case_id")
    unmatched.to_csv(args.output_dir / "unmatched_doctor_cases.csv", index=False, encoding="utf-8-sig")

    summary = {
        "doctor_reference_cases": int(len(scores)),
        "candidate_czi_records": int(len(candidates)),
        "unique_cases_with_local_czi": int(len(unique)),
        "doctor_cases_without_local_czi": int(len(unmatched)),
        "duplicate_candidate_cases": int((unique["candidate_count_for_case"] > 1).sum()),
        "sources": unique["source_dataset"].value_counts().to_dict(),
    }
    (args.output_dir / "summary.json").write_text(
        pd.Series(summary).to_json(force_ascii=False, indent=2), encoding="utf-8"
    )
    print(pd.Series(summary).to_json(force_ascii=False, indent=2))


if __name__ == "__main__":
    main()
