import time
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.models import mobilenet_v3_small
from utils.dataset import TeamMateDataset
import random
import torchvision.transforms.functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# Function to apply log-softmax
def apply_log_softmax(outputs):
    return torch.nn.functional.log_softmax(outputs, dim=1)

# Function to plot and save confusion matrix
def plot_confusion_matrix(cm, epoch, phase='train'):
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Person 0', 'Person 1'], yticklabels=['Person 0', 'Person 1'], cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Epoch {epoch} ({phase})')
    plt.tight_layout()
    plt.savefig(f'lab8/confusion_matrix_epoch_{epoch}_{phase}.png')
    plt.close()

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Create the datasets and dataloaders without the transform in the dataset itself
    trainset = TeamMateDataset(n_images=50, train=True)
    testset = TeamMateDataset(n_images=10, train=False)
    trainloader = DataLoader(trainset, batch_size=25, shuffle=True)
    testloader = DataLoader(testset, batch_size=2, shuffle=False)

    # Create the model and optimizer
    model = mobilenet_v3_small(weights=None, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # Loss lists for storing loss values
    train_losses_cross_entropy = []
    test_losses_cross_entropy = []
    train_losses_nll = []
    test_losses_nll = []

    # Saving parameters
    best_train_loss_cross_entropy = 1e9
    best_train_loss_nll = 1e9

    # Random Horizontal Flip (for training loop)
    def random_horizontal_flip(images, p=0.5):
        if random.random() < p:  # flip with probability p
            images = torch.flip(images, [3])  # Flip along the horizontal axis (W dimension)
        return images

    # Random Rotation (for training loop)
    def random_rotation(images, max_degrees=10):
        angle = random.uniform(-max_degrees, max_degrees)
        images = torch.stack([F.rotate(image, angle) for image in images])  # Rotate each image in the batch
        return images

    # Epoch Loop - from 1 to 3
    for epoch in range(1, 4):  # Let's run for 3 epochs

        # Start timer
        t = time.time_ns()

        # Train with CrossEntropyLoss
        model.train()
        train_loss_cross_entropy = 0
        train_preds = []
        train_labels = []

        for i, (images, labels) in tqdm(enumerate(trainloader), total=len(trainloader), leave=False):
            images = random_horizontal_flip(images)
            images = random_rotation(images, max_degrees=10)
            images = images.permute(0, 3, 1, 2).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute loss using CrossEntropyLoss
            criterion_cross_entropy = torch.nn.CrossEntropyLoss()
            loss_cross_entropy = criterion_cross_entropy(outputs, labels)

            # Backward pass
            loss_cross_entropy.backward()
            optimizer.step()

            train_loss_cross_entropy += loss_cross_entropy.item()

            # Save predictions and labels for confusion matrix
            _, predicted = torch.max(outputs, 1)
            train_preds.extend(predicted.cpu().numpy())  # Move to CPU for numpy compatibility
            train_labels.extend(labels.cpu().numpy())

        # Compute and save confusion matrix for training set
        train_cm = confusion_matrix(train_labels, train_preds)
        plot_confusion_matrix(train_cm, epoch, phase='train')

        # Test with CrossEntropyLoss
        model.eval()
        test_loss_cross_entropy = 0
        correct = 0
        total = 0
        test_preds = []
        test_labels = []

        for images, labels in tqdm(testloader, total=len(testloader), leave=False):
            images = images.permute(0, 3, 1, 2).to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss_cross_entropy = criterion_cross_entropy(outputs, labels)

            test_loss_cross_entropy += loss_cross_entropy.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Save predictions and labels for confusion matrix
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

        # Compute and save confusion matrix for test set
        test_cm = confusion_matrix(test_labels, test_preds)
        plot_confusion_matrix(test_cm, epoch, phase='test')

        print(f'Epoch: {epoch}, Train Loss (CrossEntropy): {train_loss_cross_entropy / len(trainloader):.4f}, '
              f'Test Loss (CrossEntropy): {test_loss_cross_entropy / len(testloader):.4f}, '
              f'Test Accuracy (CrossEntropy): {correct / total:.4f}')

        # Save results for CrossEntropyLoss
        train_losses_cross_entropy.append(train_loss_cross_entropy / len(trainloader))
        test_losses_cross_entropy.append(test_loss_cross_entropy / len(testloader))

        if train_loss_cross_entropy < best_train_loss_cross_entropy:
            best_train_loss_cross_entropy = train_loss_cross_entropy
            torch.save(model.state_dict(), 'lab8/best_model_cross_entropy.pth')

        # Train with NLLLoss
        model.train()
        train_loss_nll = 0
        train_preds = []
        train_labels = []

        for i, (images, labels) in tqdm(enumerate(trainloader), total=len(trainloader), leave=False):
            images = random_horizontal_flip(images)
            images = random_rotation(images, max_degrees=10)
            images = images.permute(0, 3, 1, 2).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Apply LogSoftmax for NLLLoss
            outputs = apply_log_softmax(outputs)

            # Compute loss using NLLLoss
            criterion_nll = torch.nn.NLLLoss()
            loss_nll = criterion_nll(outputs, labels)

            # Backward pass
            loss_nll.backward()
            optimizer.step()

            train_loss_nll += loss_nll.item()

            # Save predictions and labels for confusion matrix
            _, predicted = torch.max(outputs, 1)
            train_preds.extend(predicted.cpu().numpy())  # Move to CPU for numpy compatibility
            train_labels.extend(labels.cpu().numpy())

        # Compute and save confusion matrix for training set
        train_cm = confusion_matrix(train_labels, train_preds)
        plot_confusion_matrix(train_cm, epoch, phase='train')

        # Test with NLLLoss
        model.eval()
        test_loss_nll = 0
        correct = 0
        total = 0
        test_preds = []
        test_labels = []

        for images, labels in tqdm(testloader, total=len(testloader), leave=False):
            images = images.permute(0, 3, 1, 2).to(device)
            labels = labels.to(device)

            outputs = model(images)
            outputs = apply_log_softmax(outputs)

            loss_nll = criterion_nll(outputs, labels)

            test_loss_nll += loss_nll.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Save predictions and labels for confusion matrix
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

        # Compute and save confusion matrix for test set
        test_cm = confusion_matrix(test_labels, test_preds)
        plot_confusion_matrix(test_cm, epoch, phase='test')

        print(f'Epoch: {epoch}, Train Loss (NLL): {train_loss_nll / len(trainloader):.4f}, '
              f'Test Loss (NLL): {test_loss_nll / len(testloader):.4f}, '
              f'Test Accuracy (NLL): {correct / total:.4f}')

        # Save results for NLLLoss
        train_losses_nll.append(train_loss_nll / len(trainloader))
        test_losses_nll.append(test_loss_nll / len(testloader))

        if train_loss_nll < best_train_loss_nll:
            best_train_loss_nll = train_loss_nll
            torch.save(model.state_dict(), 'lab8/best_model_nll.pth')

        # Save the model
        torch.save(model.state_dict(), 'lab8/current_model.pth')

        # Plot both loss curves
        plt.figure()
        plt.plot(train_losses_cross_entropy, label='Train Loss (CrossEntropy)')
        plt.plot(test_losses_cross_entropy, label='Test Loss (CrossEntropy)')
        plt.plot(train_losses_nll, label='Train Loss (NLL)')
        plt.plot(test_losses_nll, label='Test Loss (NLL)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('lab8/task7_loss_comparison_plot.png')
