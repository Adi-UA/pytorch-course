import torch
from tqdm import tqdm


def train_step(model, dataloader, loss_fn, acc_fn, optimizer, device):
    model.train()
    model.to(device)
    acc_fn.to(device)
    train_loss, train_acc = 0.0, 0.0
    for X, Y in dataloader:
        X, Y = X.to(device), Y.to(device)

        # forward
        Y_logits = model(X)
        Y_pred = torch.argmax(torch.softmax(Y_logits, dim=1), dim=1)
        batch_loss = loss_fn(Y_logits, Y)
        batch_acc = acc_fn(Y_pred, Y)

        # backward
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # metrics
        train_loss += batch_loss
        train_acc += batch_acc

    train_loss /= len(dataloader)
    train_acc = train_acc / len(dataloader) * 100

    return train_loss, train_acc


def test_step(model, dataloader, loss_fn, acc_fn, device):
    model.eval()
    model.to(device)
    with torch.inference_mode():
        test_loss, test_acc = 0.0, 0.0
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)

            # forward
            Y_logits = model(X)
            Y_pred = torch.argmax(torch.softmax(Y_logits, dim=1), dim=1)
            batch_loss = loss_fn(Y_logits, Y)
            batch_acc = acc_fn(Y_pred, Y)

            # metrics
            test_loss += batch_loss
            test_acc += batch_acc

        test_loss /= len(dataloader)
        test_acc = test_acc / len(dataloader) * 100

    return test_loss, test_acc


def train(
    model, train_dataloader, test_dataloader, epochs, loss_fn, acc_fn, optimizer, device
):
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }
    for epoch in tqdm(range(epochs), desc="Training", unit="epochs"):
        print(f"Epoch {epoch+1}/{epochs}")
        print("-" * 20)

        train_loss, train_acc = train_step(
            model, train_dataloader, loss_fn, acc_fn, optimizer, device
        )
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, acc_fn, device)

        history["train_loss"].append(train_loss.item())
        history["train_acc"].append(train_acc.item())
        history["test_loss"].append(test_loss.item())
        history["test_acc"].append(test_acc.item())

        print(
            f"Epoch {epoch+1}/{epochs} | Train loss: {train_loss:.3f} | Train acc: {train_acc:.2f}% | Test loss: {test_loss:.3f} | Test acc: {test_acc:.2f}%"
        )
        print("-" * 20)

    return history
