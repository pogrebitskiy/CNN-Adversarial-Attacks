import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from art.attacks.evasion import ProjectedGradientDescent


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


def test_model(model, test_loader, device='cuda'):
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


def loader_to_numpy(loader):
    X = []
    y = []
    for data in loader:
        X.append(data[0].numpy())
        y.append(data[1].numpy())
    X = np.concatenate(X)
    y = np.concatenate(y)
    return X, y


def evaluate_attack(attacker, classifier, X_test, y_test):
    x_test_adv = attacker.generate(X_test, y_test)
    x_test_adv_pred = np.argmax(classifier.predict(x_test_adv), axis=1)
    nb_correct_adv_pred = np.sum(x_test_adv_pred == y_test)
    return nb_correct_adv_pred


def compare_classifiers(classifier1, classifier2, X_test, y_test, eps_values, batch_size):
    nb_correct_classifier1 = []
    nb_correct_classifier2 = []

    for eps in eps_values:
        attacker = ProjectedGradientDescent(classifier1, eps=eps, eps_step=0.01, max_iter=200, batch_size=batch_size)
        nb_correct_classifier1.append(evaluate_attack(attacker, classifier1, X_test, y_test))

        attacker = ProjectedGradientDescent(classifier2, eps=eps, eps_step=0.01, max_iter=200, batch_size=batch_size)
        nb_correct_classifier2.append(evaluate_attack(attacker, classifier2, X_test, y_test))

    plt.plot(eps_values, nb_correct_classifier1, 'b--', label='Clean Classifier')
    plt.plot(eps_values, nb_correct_classifier2, 'r--', label='Adversarial Classifier')
    plt.legend(loc='upper right', shadow=True, fontsize='large')
    plt.xlabel('Perturbation size (eps, L-Inf)')
    plt.ylabel('Classification Accuracy')
    plt.show()


def plot_images(X_test, y_test, clean_classifier, attack, title, n=5):
    # Plot the original and adversarial images based on an attack object
    fig, axs = plt.subplots(2, n, figsize=(15, 5))

    # Add a title to the figure
    fig.suptitle(title, fontsize=16)

    for i in range(n):
        axs[0, i].imshow(X_test[i].reshape(28, 28), cmap='gray')
        axs[0, i].set_title(f"Label: {y_test[i]}")
        axs[1, i].imshow(attack.generate(X_test[i].reshape(1, 1, 28, 28), np.array([y_test[i]])).reshape(28, 28),
                         cmap='gray')
        axs[1, i].set_title(
            f"Label: {np.argmax(clean_classifier.predict(attack.generate(X_test[i].reshape(1, 1, 28, 28), np.array([y_test[i]]))))}")

    # Add titles to the rows
    axs[0, 0].set_ylabel('Original Images', fontsize=12)
    axs[1, 0].set_ylabel('Adversarial Images', fontsize=12)

    # remove axis labels
    for ax in axs.flat:
        ax.label_outer()

    plt.show()