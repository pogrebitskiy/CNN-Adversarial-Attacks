# Protecting CV Models Against Adversarial Attacks

This project is an analysis of the paper "Towards Deep Learning Models Resistant to Adversarial Attacks". The project aims to recreate the experiments from the paper by training five different models on the MNIST dataset and testing their adversarial robustness using PGD, BIM, and FGSM to generate adversarial examples. The goal is to test a wider range of network architectures and determine how large of a role the structure plays in the model's ability to generalize better, thus being more robust to adversarial attacks.

You can view the project's website [here](https://expo.baulab.info/2024-Spring/pogrebitskiy/).

## Models

The project trains and tests the following models:

- 2-layer fully connected network
- VGG
- LeNet
- GoogLeNet
- ResNet

Each model is trained from scratch for 50 epochs on the MNIST dataset. The models' performances are then tested on the raw test set and adversarial images from each of the 3 types of attacks (PGD, BIM, FGSM).

## Adversarial Training

After evaluating the original and augmented test sets on the models, each model is further trained for an additional 10 epochs using an augmented dataset. This new training data comprises a 50/50 split between original data and augmented images generated using each of the three adversarial attacks (PGD, BIM, and FGSM).

## Experimental Findings

The project presents detailed findings from the experiments, including tables and figures showing the accuracy of each model on the original testing data and adversarial data, as well as the accuracy difference between the original model and the hardened model.

## Team Members

- David Pogrebitskiy: pogrebitskiy.d@northeastern.edu
- Benjamin Wyant: wyant.b@northeastern.edu

## References

- [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/pdf/1706.06083.pdf)
- [DeepFool: a simple and accurate method to fool deep neural networks](https://arxiv.org/pdf/1511.04599.pdf)
