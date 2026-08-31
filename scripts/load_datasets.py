from pathlib import Path

import click
import torch
import yaml
from torch.utils.data import random_split
from torchvision import datasets
from torchvision.transforms import v2


@click.command()
@click.option(
    "--config", 
    default="configs/model_config.yaml", 
    help="Path to model config",
    type=click.Path(exists=True),
)
def main(config: str) -> None:
    with open(config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    data_config = config["data"]

    raw_dir = Path(data_config["raw_dir"])
    processed_dir = Path(data_config["processed_dir"])

    seed = data_config["seed"]
    train_ratio = data_config["train_ratio"]

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    train_dataset = datasets.FashionMNIST(
        root=raw_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.FashionMNIST(
        root=raw_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_size = int(len(train_dataset) * train_ratio)
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    torch.save(train_dataset, processed_dir / "train.pt")
    torch.save(val_dataset, processed_dir / "val.pt")
    torch.save(test_dataset, processed_dir / "test.pt")

    click.echo(f"Train dataset saved to {processed_dir / 'train.pt'}")
    click.echo(f"Validation dataset saved to {processed_dir / 'val.pt'}")
    click.echo(f"Test dataset saved to {processed_dir / 'test.pt'}")


if __name__ == "__main__":
    main()
