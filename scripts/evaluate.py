from datetime import datetime
from pathlib import Path

import click
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_dataset(path: Path):
    return torch.load(path, weights_only=False)


def build_dataloaders(config):
    processed_dir = Path(config["data"]["processed_dir"])
    batch_size = config["dataloader"]["batch_size"]

    train_dataset = load_dataset(processed_dir / "train.pt")
    val_dataset = load_dataset(processed_dir / "val.pt")
    test_dataset = load_dataset(processed_dir / "test.pt")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


def evaluate(model, data_loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            predictions = torch.argmax(logits, dim=1)

            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)

    return correct / total


def save_metrics(
    model_name,
    train_accuracy,
    val_accuracy,
    test_accuracy,
    metrics_path,
):
    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = pd.DataFrame(
        [
            {
                "model": model_name,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
                "test_accuracy": test_accuracy,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        ]
    )

    if metrics_path.exists():
        result.to_csv(
            metrics_path,
            mode="a",
            header=False,
            index=False,
        )
    else:
        result.to_csv(
            metrics_path,
            index=False,
        )


@click.command()
@click.option(
    "--config",
    default="configs/model_config.yaml",
    help="Path to model config",
)
@click.option(
    "--data-config",
    default="configs/data_config.yaml",
    help="Path to data config",
)
def main(config, data_config):
    model_config = load_config(config)
    data_config = load_config(data_config)

    device_name = model_config["device"]

    if device_name == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders(data_config)
    checkpoint_path = Path(model_config["training"]["checkpoint_path"])
    model = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = model.to(device)

    train_accuracy = evaluate(
        model,
        train_loader,
        device,
    )

    val_accuracy = evaluate(
        model,
        val_loader,
        device,
    )

    test_accuracy = evaluate(
        model,
        test_loader,
        device,
    )

    print(f"Train accuracy: {train_accuracy:.4f}")
    print(f"Val accuracy:   {val_accuracy:.4f}")
    print(f"Test accuracy:  {test_accuracy:.4f}")

    metrics_path = Path(data_config["metrics"]["metrics_path"])

    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    save_metrics(
        model_name=model_config["model"]["name"],
        train_accuracy=train_accuracy,
        val_accuracy=val_accuracy,
        test_accuracy=test_accuracy,
        metrics_path=metrics_path,
    )

    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
