import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend, suitable for server environments
# Note: No image transformation or compression is applied as required.
# torchvision.transforms is intentionally not used.

# ==== Paths ====
img_dir = 'Dataset Binary'
label_file = 'Dataset_Labels.xlsx'

# ==== Load labels ====
df = pd.read_excel(label_file)
df.columns = ['Spine_Name', 'Spine_Label']

# Normalize names and labels: remove extra spaces/newlines and avoid duplicated extensions
df['Spine_Name'] = df['Spine_Name'].astype(str).str.strip()
df['Spine_Label'] = df['Spine_Label'].astype(str).str.strip()

# ==== Encode labels ====
encoder = LabelEncoder()
df['Label_Encoded'] = encoder.fit_transform(df['Spine_Label'])
num_classes = len(encoder.classes_)
print(f"Detected {num_classes} classes: {encoder.classes_}")

# ==== Resolve image paths ====
def resolve_image_path(name: str) -> str | None:
    # The name may already contain an extension; try common extensions uniformly
    name = str(name).strip()
    base, ext = os.path.splitext(name)
    candidates = []

    # If the original name already looks like a filename
    if ext:
        candidates.append(name)

    # Try common image extensions
    for e in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
        candidates.append(base + e)

    # Remove duplicates while preserving order
    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            uniq.append(c)
            seen.add(c)

    # Check file existence
    for filename in uniq:
        p = os.path.join(img_dir, filename)
        if os.path.exists(p):
            return p
    return None

df['Image_Path'] = df['Spine_Name'].apply(resolve_image_path)
missing_count = df['Image_Path'].isna().sum()
if missing_count:
    print(f"Skipping {missing_count} rows: corresponding image files not found")

df = df[df['Image_Path'].notna()].reset_index(drop=True)

# ==== Custom Dataset (no resizing/normalization, tensor conversion only) ====
class SpineDataset(Dataset):
    def __init__(self, df, img_dir):
        self.df = df
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = row['Label_Encoded']
        img_path = row['Image_Path']

        # Load image at original resolution
        image = Image.open(img_path).convert("RGB")

        # No image transformation: no resizing or normalization
        # Only convert to Tensor (C, H, W) with values in [0, 1]
        arr = np.array(image)  # (H, W, C), uint8
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return tensor, label

# No image transformation or compression is applied

# ==== Train / test split ====
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['Label_Encoded']
)

# Print split statistics
print("\nDataset split summary:")
print(f"Training set: {len(train_df)} samples, Test set: {len(test_df)} samples")
print("Training set class distribution:")
print(train_df['Spine_Label'].value_counts())
print("Test set class distribution:")
print(test_df['Spine_Label'].value_counts())

# Save split details to CSV
split_train = train_df.copy()
split_train['split'] = 'train'
split_test = test_df.copy()
split_test['split'] = 'test'
split_all = pd.concat([split_train, split_test], ignore_index=True)

split_all[['Spine_Name', 'Spine_Label', 'Image_Path', 'split']].to_csv(
    'split_result.csv',
    index=False
)
print("Split details saved to split_result.csv")

train_dataset = SpineDataset(train_df, img_dir)
test_dataset = SpineDataset(test_df, img_dir)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ==== Define CNN model ====
class CNN(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super(CNN, self).__init__()
        self.dropout = dropout

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Adaptive pooling to a fixed spatial size to avoid dependency on input resolution
        self.adapt_pool = nn.AdaptiveAvgPool2d((16, 16))

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adapt_pool(x)
        x = self.fc_layers(x)
        return x

# ==== Initialize model ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(num_classes).to(device)
print(f"Using device: {device}")

# ==== Loss function and optimizer ====
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ==== Training history ====
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

# ==== Training ====
EPOCHS = 20

import optuna

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(dataloader)
    acc = 100 * correct / total
    return avg_loss, acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(dataloader)
    val_acc = 100 * val_correct / val_total
    return avg_val_loss, val_acc

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    dropout = trial.suggest_uniform('dropout', 0.3, 0.7)
    max_epochs = trial.suggest_int('max_epochs', 5, 40)

    model_temp = CNN(num_classes, dropout=dropout).to(device)
    optimizer_temp = optim.Adam(model_temp.parameters(), lr=lr)

    temp_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    temp_val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for ep in range(max_epochs):
        train_one_epoch(model_temp, temp_train_loader, criterion, optimizer_temp, device)

    val_loss, val_acc = evaluate(model_temp, temp_val_loader, criterion, device)
    return val_loss

print("\n===== Hyperparameter tuning with Optuna (Bayesian Optimization) =====\n")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=12)

best = study.best_params
print("Best hyperparameters found by Optuna:", best)

EPOCHS = best['max_epochs']
print(f"Training epochs automatically selected by Optuna: {EPOCHS}")

optimizer = optim.Adam(model.parameters(), lr=best['lr'])
train_loader = DataLoader(train_dataset, batch_size=best['batch_size'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=best['batch_size'], shuffle=False)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    avg_train_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total
    avg_val_loss = val_loss / len(test_loader)

    history['train_loss'].append(avg_train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(avg_val_loss)
    history['val_acc'].append(val_acc)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
        f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%"
    )

# ==== Testing ====
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_acc = 100 * correct / total
print(f"\nTest set accuracy: {test_acc:.2f}%")

# ==== Visualize training history ====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss curves
ax1.plot(range(1, EPOCHS + 1), history['train_loss'], 'b-', label='Training loss', linewidth=2)
ax1.plot(range(1, EPOCHS + 1), history['val_loss'], 'r-', label='Validation loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and validation loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy curves
ax2.plot(range(1, EPOCHS + 1), history['train_acc'], 'b-', label='Training accuracy', linewidth=2)
ax2.plot(range(1, EPOCHS + 1), history['val_acc'], 'r-', label='Validation accuracy', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and validation accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("Training curves saved as training_history.png")

# Save training history to CSV
history_df = pd.DataFrame({
    'epoch': range(1, EPOCHS + 1),
    'train_loss': history['train_loss'],
    'train_acc': history['train_acc'],
    'val_loss': history['val_loss'],
    'val_acc': history['val_acc']
})
history_df.to_csv('training_history.csv', index=False)
print("Training history saved as training_history.csv")

# ==== Save model ====
torch.save(model.state_dict(), "spine_cnn_pytorch.pth")
print("Model saved as spine_cnn_pytorch.pth")
