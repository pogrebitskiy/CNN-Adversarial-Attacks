import torch.nn as nn
import torchvision.models as models


class Imagenette_FC500_100_10(nn.Module):
    def __init__(self, num_classes=10):
        super(Imagenette_FC500_100_10, self).__init__()
        self.fc1 = nn.Linear(3 * 224 * 224, 500)  # Adjust the input size to match Imagenette images
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, num_classes)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the image
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


class Imagenette_LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(Imagenette_LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.activation(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = x.view(-1, 16 * 53 * 53)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


class Imagenette_VGG(nn.Module):
    def __init__(self, num_classes=10):
        super(Imagenette_VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
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
            nn.Linear(128 * 56 * 56, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Imagenette_ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(Imagenette_ResNet, self).__init__()
        self.resnet = models.resnet18()
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        return x


class Imagenette_GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(Imagenette_GoogLeNet, self).__init__()
        self.googlenet = models.googlenet(aux_logits=False)  # Disable auxiliary outputs
        self.googlenet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.googlenet.fc = nn.Linear(self.googlenet.fc.in_features, num_classes)

    def forward(self, x):
        x = self.googlenet(x)
        return x
