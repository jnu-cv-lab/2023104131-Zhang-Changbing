import torch
import numpy as np
import json
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import config
from model import SkeletonTransformer

def test():
    X_test = np.load(f"{config.PREPROCESSED_DATA_DIR}/X_test.npy")
    y_test = np.load(f"{config.PREPROCESSED_DATA_DIR}/y_test.npy")
    with open(f"{config.PREPROCESSED_DATA_DIR}/label_map.json", 'r') as f:
        label_map = json.load(f)
    inv_map = {v: k for k, v in label_map.items()}
    
    model = SkeletonTransformer().to(config.DEVICE)
    model.load_state_dict(torch.load(f"{config.CHECKPOINT_DIR}/best_model.pth", map_location=config.DEVICE))
    model.eval()
    
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), config.BATCH_SIZE):
            batch = torch.tensor(X_test[i:i+config.BATCH_SIZE], dtype=torch.float32).to(config.DEVICE)
            pred = torch.argmax(model(batch), dim=1).cpu().numpy()
            preds.extend(pred)
    
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_map.keys(), yticklabels=label_map.keys())
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()
    print(classification_report(y_test, preds, target_names=label_map.keys()))

if __name__ == '__main__':
    test()