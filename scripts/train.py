import copy
from pathlib import Path

import click
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision.models import resnet18


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_dataset(path: Path):
    return torch.load(path, weights_only=False)


def create_dataloaders(config):
    config = config

    processed_dir = Path(config["data"]["processed_dir"])

    train_dataset = load_dataset(processed_dir / "train.pt")
    val_dataset = load_dataset(processed_dir / "val.pt")

    batch_size = config["dataloader"]["batch_size"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader


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


def train(model, train_loader, val_loader, config, device):
    optimizer_config = config["optimizer"]
    training_config = config["training"]

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config["lr"],
        betas=tuple(optimizer_config["betas"]),
        weight_decay=optimizer_config["weight_decay"],
    )

    best_val_loss = float("inf")
    best_model_state = None
    counter = 0

    for epoch in range(training_config["epochs"]):
        model.train()

        train_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()

        val_loss = 0.0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(x_batch)
                loss = criterion(logits, y_batch)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(
            f"epoch - {epoch + 1}, "
            f"train_loss - {train_loss:.4f}, "
            f"val_loss - {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0

            best_model_state = copy.deepcopy(model.state_dict())

        else:
            counter += 1

            if counter >= training_config["early_stopping"]["patience"]:
                print("Early stopping!")
                break

    model.load_state_dict(best_model_state)

    return model, best_val_loss


@click.command()
@click.option(
    "--config",
    default="configs/model_config.yaml",
    help="Path to model config",
    type=click.Path(exists=True),
)
@click.option(
    "--data-config",
    default="configs/data_config.yaml",
    help="Path to data config",
    type=click.Path(exists=True),
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

    train_loader, val_loader = create_dataloaders(data_config)

    model = build_model(model_config)
    model = model.to(device)

    model, best_val_loss = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=model_config,
        device=device,
    )

    checkpoint_path = Path(model_config["training"]["checkpoint_path"])

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        model,
        checkpoint_path,
    )

    print(f"Best model saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
