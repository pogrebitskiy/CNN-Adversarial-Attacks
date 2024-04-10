import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
from PIL import ImageOps
import math
import torchvision.models as models
from PIL import Image
from deepfool import deepfool
import os

# With this line
net = models.resnet34(weights='IMAGENET1K_V1')

# Switch to evaluation mode
net.eval()

im_orig = Image.open('test_im2.jpg')

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Remove the mean
im = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,
                         std=std)])(im_orig)

r, loop_i, label_orig, label_pert, pert_image = deepfool(im, net)
r = torch.from_numpy(r)

labels = open(os.path.join('synset_words.txt'), 'r').read().split('\n')

str_label_orig = labels[int(label_orig)].split(',')[0]
str_label_pert = labels[int(label_pert)].split(',')[0]

print("Original label = ", str_label_orig)
print("Perturbed label = ", str_label_pert)


def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv * torch.ones(A.shape))
    A = torch.min(A, maxv * torch.ones(A.shape))
    return A


clip = lambda x: clip_tensor(x, 0, 255)

tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=[1 / x for x in std]),
                         transforms.Normalize(mean=[-x for x in mean], std=[1, 1, 1]),
                         transforms.Lambda(clip),
                         transforms.ToPILImage(),
                         transforms.CenterCrop(224)])


plt.figure()
plt.imshow(tf(pert_image.cpu()[0]))
plt.title(str_label_pert)
plt.show()


# Convert the tensor to a PIL image
noise_image = tf(r.cpu()[0])

# Apply histogram equalization
equalized = ImageOps.equalize(noise_image)

# Display the equalized image
plt.figure()
plt.imshow(equalized, cmap='gray')
plt.title('Equalized noise')
plt.show()

# Show the original image
plt.figure()
plt.imshow(tf(im_orig))
plt.title(str_label_orig)
plt.show()
