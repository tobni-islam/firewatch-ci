import argparse
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


def generate_report(
    current: str = "logs/inference_log.csv",
    reference: str = "logs/reference_log.csv",
    output: str = "reports/drift_report.html",
) -> None:
    df_current = pd.read_csv(current)
    df_reference = pd.read_csv(reference)

    # Encode predicted_class as numeric so Evidently can track class distribution shifts
    class_map = {"none": 0, "fire": 1, "smoke": 2}
    for df in (df_current, df_reference):
        df["class_id"] = df["predicted_class"].map(class_map).fillna(-1).astype(int)

    target_columns = ["confidence", "class_id", "num_detections"]

    report = Report(
        metrics=[
            DataDriftPreset(columns=target_columns),
        ]
    )

    my_eval = report.run(reference_data=df_reference, current_data=df_current)

    Path(output).parent.mkdir(exist_ok=True, parents=True)

    my_eval.save_html(output)

    print(f"Drift report saved -> {output}")
    print(f"  Reference : {len(df_reference)} rows")
    print(f"  Current   : {len(df_current)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", default="logs/inference_log.csv")
    parser.add_argument("--reference", default="logs/reference_log.csv")
    parser.add_argument("--output", default="reports/drift_report.html")
    args = parser.parse_args()

    generate_report(current=args.current, reference=args.reference, output=args.output)
