from pathlib import Path
import time

import pandas as pd
import torch
from PIL import Image

from CV.preprocessing import (
    get_baseline_transform,
    get_grayscale_transform,
    get_grayscale_denoise_transform,
    get_grayscale_denoise_clahe_transform,
    get_grayscale_denoise_threshold_transform,
    get_grayscale_denoise_deskew_transform,
)

from ml.dataset import resolve_image_path


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "CV"
    / "preprocessing_experiments"
)


# ============================================================
# PREPROCESSING PIPELINES
# ============================================================

PIPELINES = {
    "P0_RGB_Baseline": get_baseline_transform,

    "P1_Grayscale": get_grayscale_transform,

    "P2_Grayscale_Denoise": get_grayscale_denoise_transform,

    "P3_Grayscale_Denoise_CLAHE":
        get_grayscale_denoise_clahe_transform,

    "P4_Grayscale_Denoise_Otsu":
        get_grayscale_denoise_threshold_transform,

    "P5_Grayscale_Denoise_Deskew":
        get_grayscale_denoise_deskew_transform,
}


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    df = pd.read_csv(CSV_PATH)

    return df


# ============================================================
# CALCULATE TENSOR STATISTICS
# ============================================================

def calculate_tensor_statistics(tensor):

    return {
        "channels": tensor.shape[0],
        "height": tensor.shape[1],
        "width": tensor.shape[2],
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "mean": float(tensor.mean()),
        "std": float(tensor.std()),
        "foreground_ratio": float(
            (tensor < 0.9).float().mean()
        ),
    }


# ============================================================
# RUN ONE PIPELINE
# ============================================================

def run_pipeline(
    pipeline_name,
    transform,
    df,
    max_samples=None,
):

    rows = []

    sample_df = df

    if max_samples is not None:
        sample_df = df.head(max_samples)

    print()
    print("=" * 70)
    print(pipeline_name)
    print("=" * 70)

    for index, row in sample_df.iterrows():

        image_path = resolve_image_path(
            row["image_path"]
        )

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            start_time = time.perf_counter()

            tensor = transform(image)

            elapsed = (
                time.perf_counter()
                - start_time
            )

            stats = calculate_tensor_statistics(
                tensor
            )

            rows.append({
                "pipeline": pipeline_name,
                "image_path": row["image_path"],
                "medicine_name": row.get(
                    "medicine_name",
                    ""
                ),
                "split": row.get(
                    "split",
                    ""
                ),
                "channels": stats["channels"],
                "height": stats["height"],
                "width": stats["width"],
                "min": stats["min"],
                "max": stats["max"],
                "mean": stats["mean"],
                "std": stats["std"],
                "foreground_ratio":
                    stats["foreground_ratio"],
                "processing_time_ms":
                    elapsed * 1000,
            })

        except Exception as error:

            print(
                f"Skipping {image_path}: {error}"
            )

    return rows


# ============================================================
# SUMMARIZE RESULTS
# ============================================================

def summarize_results(results_df):

    summary = (
        results_df
        .groupby("pipeline")
        .agg({
            "channels": "first",
            "height": "first",
            "width": "first",
            "min": "mean",
            "max": "mean",
            "mean": "mean",
            "std": "mean",
            "foreground_ratio": "mean",
            "processing_time_ms": "mean",
        })
        .reset_index()
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PREPROCESSING EXPERIMENTS")
    print("=" * 70)

    print()
    print("Loading dataset...")

    df = load_dataset()

    print(
        f"Total dataset images: {len(df)}"
    )

    print()
    print("Running all preprocessing pipelines...")

    all_results = []

    for pipeline_name, transform_function in PIPELINES.items():

        transform = transform_function()

        results = run_pipeline(
            pipeline_name,
            transform,
            df,
            max_samples=None,
        )

        all_results.extend(results)

    results_df = pd.DataFrame(
        all_results
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------

    detailed_path = (
        OUTPUT_DIR
        / "preprocessing_experiment_results.csv"
    )

    results_df.to_csv(
        detailed_path,
        index=False
    )

    # --------------------------------------------------------
    # Create summary
    # --------------------------------------------------------

    summary_df = summarize_results(
        results_df
    )

    summary_path = (
        OUTPUT_DIR
        / "preprocessing_experiment_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    print(
        f"Detailed results:\n{detailed_path}"
    )

    print()
    print(
        f"Summary:\n{summary_path}"
    )

    print()
    print("Preprocessing experiments completed.")


if __name__ == "__main__":
    main()