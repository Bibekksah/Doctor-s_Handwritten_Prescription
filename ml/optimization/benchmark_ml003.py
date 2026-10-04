from pathlib import Path
import json
import time

import torch

from ml.model_ml003 import TransferMedicineClassifier
from ml.dataloader import create_dataloaders
from ml.device import get_device, get_device_name
from CV.preprocessing import get_grayscale_transform


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "ml"
    / "checkpoints"
    / "best_model_ML003.pth"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "ml"
    / "results"
    / "optimization"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def get_model_size_mb(path):
    return path.stat().st_size / (1024 ** 2)


def benchmark_inference(
    model,
    device,
    input_tensor,
    warmup=20,
    iterations=100,
):

    model.eval()

    input_tensor = input_tensor.to(device)

    # Warm-up
    with torch.no_grad():

        for _ in range(warmup):
            _ = model(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():

        for _ in range(iterations):
            _ = model(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()

    total_time = end - start

    average_time = total_time / iterations

    return {
        "total_time_seconds": total_time,
        "average_latency_ms": average_time * 1000,
        "throughput_images_per_second": 1 / average_time,
        "iterations": iterations,
        "warmup": warmup,
    }


def main():

    print("=" * 80)
    print("ML-003 DEPLOYMENT BASELINE BENCHMARK")
    print("=" * 80)

    device = get_device()

    print(f"Device: {device}")
    print(f"Device name: {get_device_name(device)}")

    # ---------------------------------------------------------
    # Load checkpoint
    # ---------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    num_classes = len(
        checkpoint["class_to_idx"]
    )

    model = TransferMedicineClassifier(
        num_classes=num_classes
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.eval()

    # ---------------------------------------------------------
    # Model information
    # ---------------------------------------------------------

    parameters = count_parameters(model)

    model_size_mb = get_model_size_mb(
        CHECKPOINT_PATH
    )

    print("\nModel information")
    print("-" * 80)

    print(f"Model: ResNet-18")
    print(f"Classes: {num_classes}")
    print(f"Parameters: {parameters:,}")
    print(f"Checkpoint size: {model_size_mb:.2f} MB")

    # ---------------------------------------------------------
    # Test input
    # ---------------------------------------------------------

    transform = get_grayscale_transform()

    _, _, test_loader, _, _ = create_dataloaders(
        batch_size=1,
        train_transform=transform,
        validation_transform=transform,
        test_transform=transform,
    )

    batch = next(iter(test_loader))

    image = batch[0]

    print(f"\nInput shape: {tuple(image.shape)}")

    # ---------------------------------------------------------
    # GPU benchmark
    # ---------------------------------------------------------

    print("\nRunning inference benchmark...")

    gpu_result = None

    if torch.cuda.is_available():

        gpu_result = benchmark_inference(
            model,
            torch.device("cuda"),
            image,
        )

        print("\nGPU benchmark")
        print("-" * 80)

        print(
            f"Average latency: "
            f"{gpu_result['average_latency_ms']:.3f} ms"
        )

        print(
            f"Throughput: "
            f"{gpu_result['throughput_images_per_second']:.2f} images/sec"
        )

    # ---------------------------------------------------------
    # CPU benchmark
    # ---------------------------------------------------------

    cpu_model = TransferMedicineClassifier(
        num_classes=num_classes
    )

    cpu_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    cpu_model = cpu_model.cpu()

    cpu_result = benchmark_inference(
        cpu_model,
        torch.device("cpu"),
        image.cpu(),
    )

    print("\nCPU benchmark")
    print("-" * 80)

    print(
        f"Average latency: "
        f"{cpu_result['average_latency_ms']:.3f} ms"
    )

    print(
        f"Throughput: "
        f"{cpu_result['throughput_images_per_second']:.2f} images/sec"
    )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    results = {
        "experiment": "ML003_DEPLOYMENT_BASELINE",
        "model": "ResNet-18",
        "num_classes": num_classes,
        "parameters": parameters,
        "checkpoint_size_mb": model_size_mb,
        "input_shape": list(image.shape),
        "device": get_device_name(device),
        "gpu_benchmark": gpu_result,
        "cpu_benchmark": cpu_result,
    }

    output_path = (
        RESULTS_DIR
        / "ml003_baseline_benchmark.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)

    print(f"Results saved to:")
    print(output_path)


if __name__ == "__main__":
    main()
