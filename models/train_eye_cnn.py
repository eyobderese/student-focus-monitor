"""
EyeStateNet training script on the MRL Eye Dataset.

Run ONCE offline before the demo.
Output: models/saved/eye_cnn.pt

Expected directory layout:
    data/mrl_eyes/
        open/    ← images of open eyes
        closed/  ← images of closed eyes

If the MRL dataset uses a different layout, reorganise into these two
subfolders first. The script will print instructions if the folders are empty.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.attention_classifier import EyeStateNet

# ---------------------------------------------------------------------------
# Paths and hyper-parameters
# ---------------------------------------------------------------------------
DATA_DIR   = PROJECT_ROOT / "data" / "mrl_eyes"
SAVE_PATH  = PROJECT_ROOT / "models" / "saved" / "eye_cnn.pt"
EPOCHS     = 20
BATCH_SIZE = 64
LR         = 1e-3
SEED       = 42

torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def build_dataset() -> datasets.ImageFolder:
    if not (DATA_DIR / "open").exists() or not (DATA_DIR / "closed").exists():
        print(
            f"ERROR: Expected subfolders:\n"
            f"  {DATA_DIR}/open/\n"
            f"  {DATA_DIR}/closed/\n\n"
            "Download the MRL Eye Dataset (mrlEyes_2018_01 subset) from:\n"
            "  http://mrl.cs.vsb.cz/eyedataset\n"
            "Then reorganise images into open/ and closed/ subfolders."
        )
        sys.exit(1)

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])
    return datasets.ImageFolder(str(DATA_DIR), transform=transform)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train() -> None:
    dataset  = build_dataset()
    n        = len(dataset)
    n_train  = int(0.70 * n)
    n_val    = int(0.15 * n)
    n_test   = n - n_train - n_val

    generator = torch.Generator().manual_seed(SEED)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device    = torch.device("cpu")
    model     = EyeStateNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {n} images  ({n_train} train / {n_val} val / {n_test} test)")
    print(f"Classes: {dataset.classes}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_correct = train_total = 0
        total_loss    = 0.0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss    += loss.item() * len(y)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total   += len(y)

        scheduler.step()

        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_correct += (model(X).argmax(1) == y).sum().item()
                val_total   += len(y)

        train_acc = train_correct / train_total
        val_acc   = val_correct   / val_total

        print(f"Epoch {epoch:2d}/{EPOCHS}  "
              f"loss={total_loss/train_total:.4f}  "
              f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ✔ Saved best model  val_acc={val_acc:.4f}")

    print(f"\nBest val accuracy: {best_val_acc:.4f}")

    # Final test evaluation
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device, weights_only=True))
    model.eval()
    test_correct = test_total = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            test_correct += (model(X).argmax(1) == y).sum().item()
            test_total   += len(y)
    print(f"Test accuracy: {test_correct / test_total:.4f}")
    print(f"Weights saved → {SAVE_PATH}")


if __name__ == "__main__":
    train()
