import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import config
from model import SkeletonTransformer

def train():
    X_train = np.load(f"{config.PREPROCESSED_DATA_DIR}/X_train.npy")
    y_train = np.load(f"{config.PREPROCESSED_DATA_DIR}/y_train.npy")
    X_val = np.load(f"{config.PREPROCESSED_DATA_DIR}/X_test.npy")
    y_val = np.load(f"{config.PREPROCESSED_DATA_DIR}/y_test.npy")
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                            torch.tensor(y_train, dtype=torch.long)),
                              batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                          torch.tensor(y_val, dtype=torch.long)),
                            batch_size=config.BATCH_SIZE)
    
    model = SkeletonTransformer().to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    train_losses, val_accs = [], []
    best_acc = 0
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(config.DEVICE), y.to(config.DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))
        
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(config.DEVICE)
                pred = torch.argmax(model(X), dim=1).cpu().numpy()
                preds.extend(pred)
                trues.extend(y.numpy())
        acc = accuracy_score(trues, preds)
        val_accs.append(acc)
        print(f"Epoch {epoch+1}: Loss={train_losses[-1]:.4f}, Val Acc={acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"{config.CHECKPOINT_DIR}/best_model.pth")
    
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_accs, label='Val Accuracy')
    plt.legend()
    plt.savefig('training_curve.png')
    plt.show()
    print(f"最佳验证准确率: {best_acc:.4f}")

if __name__ == '__main__':
    train()