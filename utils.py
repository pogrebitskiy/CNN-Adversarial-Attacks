import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=50, device='gpu'):
    """ Train the model.

    Args:
        model (torch.nn.Module): The model to be trained.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer to be used.
        train_loader (torch.utils.data.DataLoader): The training data loader.
        val_loader (torch.utils.data.DataLoader): The validation data loader.
        num_epochs (int): The number of epochs to train the model.
        device (str): The device to be used for training.

    Returns:
        tuple: A tuple containing the trained model and the training history.
    """
    model = model.to(device)

    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []

    for epoch in tqdm(range(num_epochs)):
        correct = 0
        losses = []

        model.train()

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets).sum().item()

        train_acc.append(correct / len(train_loader.dataset))
        train_loss.append(np.mean(losses))

        val_correct = 0
        losses = []

        model.eval()

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                losses.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == targets).sum().item()

        val_acc.append(val_correct / len(val_loader.dataset))
        val_loss.append(np.mean(losses))

        # Print every 5 epochs with rounding
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, '
                  f'Train Accuracy: {train_acc[-1] * 100:.2f}%, '
                  f'Train Loss: {train_loss[-1]:.4f}, '
                  f'Val Accuracy: {val_acc[-1] * 100:.2f}%, '
                  f'Val Loss: {val_loss[-1]:.4f}')

        # Clear CUDA cache
        torch.cuda.empty_cache()

    return model, train_acc, train_loss, val_acc, val_loss


def test_model(model, test_loader, device='gpu'):
    """ Test the model.

    Args:
        model (torch.nn.Module): The model to be tested.
        test_loader (torch.utils.data.DataLoader): The test data loader.
        device (str): The device to be used for testing.

    Returns:
        float: The test accuracy of the model.
    """
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        accuracy = correct / total

    return accuracy


def plot_training_history(train_acc, train_loss, val_acc, val_loss, title='Training History', save_path=None):
    """ Plot the training history.

    Args:
        train_acc (list): The training accuracy history.
        train_loss (list): The training loss history.
        val_acc (list): The validation accuracy history.
        val_loss (list): The validation loss history.
        title (str): The title of the plot.
        save_path (str): The path to save the plot.
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.suptitle(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()
