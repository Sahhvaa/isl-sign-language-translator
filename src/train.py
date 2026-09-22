import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from model import SignClassifier

X = np.load("data/X.npy")
y = np.load("data/y.npy")
num_classes = len(np.unique(y))

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long()),
    batch_size=8, shuffle=True
)
X_val_t = torch.tensor(X_val).float()
y_val_t = torch.tensor(y_val).long()

model = SignClassifier(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

best_val_acc = 0.0

for epoch in range(50):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t).argmax(dim=1)
        val_acc = (val_preds == y_val_t).float().mean().item()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/sign_model.pt")
        marker = "  <- new best, saved"
    else:
        marker = ""

    print(f"Epoch {epoch+1}: loss={total_loss:.4f}  val_acc={val_acc:.2%}{marker}")

print(f"Best val accuracy: {best_val_acc:.2%}  (saved to models/sign_model.pt)")