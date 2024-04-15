import torch.nn as nn
import torchvision.models as models


class MNIST_FC_500_100_10(nn.Module):
    """ A simple fully connected neural network with 3 hidden layers for MNIST classification.

    Attributes:
        input_size (int): The size of the input feature vector.
        num_classes (int): The number of classes for classification.
    """

    def __init__(self, num_classes=10):
        super(MNIST_FC_500_100_10, self).__init__()

        # Define the layers
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, num_classes)
        self.activation = nn.ReLU()

    def forward(self, x):
        """ Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        # Flatten the image
        x = x.flatten(start_dim=1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


class MNIST_LeNet(nn.Module):
    """ A simple LeNet model for MNIST dataset.

    Attributes:
        num_classes (int): The number of classes for classification.
    """

    def __init__(self, num_classes=10):
        super(MNIST_LeNet, self).__init__()

        # Define the layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.activation = nn.ReLU()

    def forward(self, x):
        """ Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        x = self.activation(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.activation(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = x.view(-1, 16 * 5 * 5)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


class MNIST_VGG(nn.Module):
    """ A simple VGG model for MNIST dataset.

    Attributes:
        num_classes (int): The number of classes for classification.
    """

    def __init__(self, num_classes=10):
        super(MNIST_VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        """ Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class MNIST_ResNet(nn.Module):
    """ A simple ResNet model for MNIST dataset.

    Attributes:
        num_classes (int): The number of classes for classification.
    """

    def __init__(self, num_classes=10):
        super(MNIST_ResNet, self).__init__()
        self.resnet = models.resnet18()
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                      bias=False)  # Change the first layer to accept grayscale images
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features,
                                   num_classes)  # Change the final layer to output 10 classes

    def forward(self, x):
        """ Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        x = self.resnet(x)
        return x


class MNIST_GoogLeNet(nn.Module):
    """ A simple GoogLeNet model for MNIST dataset.

    Attributes:
        num_classes (int): The number of classes for classification.
    """

    def __init__(self, num_classes=10):
        super(MNIST_GoogLeNet, self).__init__()
        self.googlenet = models.googlenet(aux_logits=False)  # Disable auxiliary outputs
        self.googlenet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                         bias=False)  # Change the first layer to accept grayscale images
        self.googlenet.fc = nn.Linear(self.googlenet.fc.in_features,
                                      num_classes)  # Change the final layer to output 10 classes

    def forward(self, x):
        """ Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        x = self.googlenet(x)
        return x


