from pathlib import Path

import torch
import torchvision
from engine import train
from torch import nn
from torchmetrics import Accuracy
from torchvision import transforms
from utils import get_device, plot_history, save_model, seed_everything

from data import create_data_loaders, fetch_data, unzip_data

DATA_DIR = Path("data")

if __name__ == "__main__":
    # Seed
    seed = 42
    seed_everything(seed)

    # Download and unzip data
    url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
    filepath = DATA_DIR / "pizza_steak_sushi.zip"
    img_dir = DATA_DIR / "pizza_steak_sushi"
    if not img_dir.exists():
        fetch_data(filepath=filepath, url=url)
        unzip_data(filepath=filepath, extract_dir=img_dir)
    else:
        print(f"Data already exists in {DATA_DIR / img_dir}")

    # Create data loaders
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # RGB
                std=[0.229, 0.224, 0.225],  # RGB
            ),
        ]
    )
    # Or automatically grab the transforms from torchvision for the model
    # weights = EfficientNet_B0_Weights.DEFAULT
    # auto_transform = weights.transforms()

    train_dir = img_dir / "train"
    test_dir = img_dir / "test"
    BATCH_SIZE = 32
    train_loader, test_loader = create_data_loaders(
        train_dir=train_dir,
        test_dir=test_dir,
        train_transform=transform,
        test_transform=transform,
        batch_size=BATCH_SIZE,
    )

    # Model
    device = get_device()
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights)

    # Freeze all model feature parametrs
    for param in model.parameters():
        param.requires_grad = False

    # Replace the last layer
    num_classes = len(train_loader.dataset.classes)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2, inplace=True),
        nn.Linear(1280, out_features=num_classes, bias=True),
    )
    model.to(device)

    # Loss, optimizer, scheduler
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    acc_fn = Accuracy(task="multiclass", num_classes=num_classes)

    # Training
    EPOCHS = 15
    history = train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        epochs=EPOCHS,
        loss_fn=loss_fn,
        acc_fn=acc_fn,
        optimizer=optimizer,
        device=device,
    )

    # Plot loss and accuracy
    plot_history(history)

    # Save model
    save_model(model, save_path="06_tuned_efficientnet_b0.pth")
