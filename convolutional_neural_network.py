import numpy as np
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
from tqdm import tqdm
import matplotlib.pyplot as plt

# Function to parse filenames for stability labels
def parse_label_from_filename(filename):
    """Extract the stability label from the filename."""
    pattern = r'recurrence_matrix_U0_\d+\.\d+_segment_\d+_stability_(\d+)\.npy'
    match = re.match(pattern, filename)
    if match:
        return int(match.group(1))  # Stability label (0 or 1)
    return None

# Custom Dataset for recurrence matrices
class RecurrenceMatrixDataset(Dataset):
    def __init__(self, matrix_dir):
        self.files = list(Path(matrix_dir).glob('*.npy'))
        self.labels = [parse_label_from_filename(file.name) for file in self.files]
        self.files = [file for file, label in zip(self.files, self.labels) if label is not None]
        self.labels = [label for label in self.labels if label is not None]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        matrix = np.load(self.files[idx])
        label = self.labels[idx]
        matrix = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        label = torch.tensor(label, dtype=torch.long)
        return matrix, label

# Define the CNN model
class RecurrenceMatrixCNN(nn.Module):
    def __init__(self):
        super(RecurrenceMatrixCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # Input: 450x450 -> Output: 450x450
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: 225x225
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Output: 225x225
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: 112x112
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Output: 112x112
            nn.ReLU(),
            nn.MaxPool2d(2)  # Output: 56x56
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 56 * 56, 128),  # Adjust dimensions based on input size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)  # Binary classification
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

if __name__ == "__main__":
    matrix_dir = "pressure_segments/recurrence_matrices"

    # Load data
    print("Loading data...")
    dataset = RecurrenceMatrixDataset(matrix_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Create CNN model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RecurrenceMatrixCNN().to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Track metrics for visualization
    train_losses = []
    val_losses = []
    val_accuracies = []

    # Training loop
    num_epochs = 20
    print("Training the CNN...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for matrices, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Training]"):
            matrices, labels = matrices.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(matrices)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation loop
        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for matrices, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]"):
                matrices, labels = matrices.to(device), labels.to(device)
                outputs = model(matrices)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = val_correct / len(val_dataset)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    print("Training complete! Model saved as 'recurrence_matrix_cnn.pth'.")
    torch.save(model.state_dict(), "recurrence_matrix_cnn.pth")

    # Save validation accuracy to a notepad file
    with open("validation_accuracy.txt", "w") as f:
        f.write("Epoch\tValidation Accuracy\n")
        for epoch, accuracy in enumerate(val_accuracies, 1):
            f.write(f"{epoch}\t{accuracy:.4f}\n")

    # Visualization of metrics
    plt.figure(figsize=(15, 8))
    plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    #plt.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='red', linewidth=2)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Metrics', fontsize=14)
    #plt.title('Training and Validation Metrics Over Epochs', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig('training_validation_metrics_improved.png', dpi=300)
    plt.show()
