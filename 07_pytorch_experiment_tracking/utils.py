import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() else device
    return device


def save_model(model, save_path):
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / save_path
    torch.save(model.state_dict(), save_path)


def plot_history(history):
    figs_dir = Path("figs")
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Plot loss
    plt.figure(figsize=(10, 7))
    plt.title("Training History")
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["test_loss"], label="test_loss")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # save
    plt.savefig(figs_dir / "06_training_history.png")

    # Plot accuracy
    plt.figure(figsize=(10, 7))
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["test_acc"], label="test_acc")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    # save
    plt.savefig(figs_dir / "06_training_accuracy.png")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
