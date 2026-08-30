import click
import torch
import yaml
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.models import resnet18
from torchvision.transforms import v2
import torch.nn as nn


def load_config(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def create_dataloaders(config):
    dataset_config = config["dataset"]
    dataloader_config = config["dataloader"]

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    train_dataset = datasets.FashionMNIST(
        root=dataset_config["root"],
        train=True,
        download=dataset_config["download"],
        transform=transform,
    )

    train_size = int(
        len(train_dataset) * dataset_config["train"]["split"]
    )
    val_size = len(train_dataset) - train_size

    train_dataset, _ = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config["seed"]),
    )

    test_dataset = datasets.FashionMNIST(
        root=dataset_config["root"],
        train=False,
        download=dataset_config["download"],
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=dataloader_config["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=dataloader_config["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    return train_loader, test_loader


def build_model(config):
    model_config = config["model"]

    model = resnet18(
        weights=model_config["weights"],
    )

    model.conv1 = nn.Conv2d(
        in_channels=model_config["input_channels"],
        out_channels=model_config["conv1"]["out_channels"],
        kernel_size=model_config["conv1"]["kernel_size"],
        stride=model_config["conv1"]["stride"],
        padding=model_config["conv1"]["padding"],
        bias=model_config["conv1"]["bias"],
    )

    model.maxpool = nn.Identity()

    model.fc = nn.Sequential(
        nn.Dropout(model_config["dropout"]),
        nn.Linear(
            model.fc.in_features,
            model_config["num_classes"],
        ),
    )

    return model


def evaluate(model, loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            predictions = torch.argmax(logits, dim=1)

            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)

    return correct / total


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

    train_loader, test_loader = create_dataloaders(
        data_config
    )

    model = build_model(model_config)

    checkpoint = torch.load(
        model_config["training"]["checkpoint_path"],
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    train_accuracy = evaluate(
        model,
        train_loader,
        device,
    )

    test_accuracy = evaluate(
        model,
        test_loader,
        device,
    )

    print(f"Accuracy train: {train_accuracy:.4f}")
    print(f"Accuracy test: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()