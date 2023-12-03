import torch
from engine import train
from model import TinyVGG
from torchmetrics import Accuracy
from torchvision import transforms
from utils import get_device, plot_history, save_model, seed_everything

from data import create_data_loaders, fetch_data, unzip_data

if __name__ == "__main__":
    # Seed
    seed = 42
    seed_everything(seed)

    # Download and unzip data
    url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
    filename = "pizza_steak_sushi.zip"
    img_dir = "pizza_steak_sushi"
    fetch_data(file_name=filename, url=url)
    unzip_data(file_name=filename, extract_dir=img_dir)

    # Create data loaders
    train_dir = img_dir + "/train"
    test_dir = img_dir + "/test"
    batch_size = 32
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128), antialias=True),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
        ]
    )
    train_loader, test_loader = create_data_loaders(
        train_dir=train_dir,
        test_dir=test_dir,
        train_transform=transform,
        test_transform=transform,
        batch_size=batch_size,
    )

    # Create model
    n_classes = len(train_loader.dataset.classes)
    model = TinyVGG(input_channels=1, hidden_dim=32, output_dim=n_classes)

    # Create loss function, accuracy function, optimizer
    device = get_device()
    loss_fn = torch.nn.CrossEntropyLoss()
    acc_fn = Accuracy(task="multiclass", num_classes=n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train model
    history = train(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        epochs=50,
        loss_fn=loss_fn,
        acc_fn=acc_fn,
        optimizer=optimizer,
        device=device,
    )

    # Plot history
    plot_history(history)

    # Save model
    save_model(model=model, save_path="05_tinyvgg.pth")
