import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def save_confusion_matrix(cm, classes, save_path):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(ax=ax, xticks_rotation=90, colorbar=False)

    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_loss_curve(losses, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8,5))
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(
        save_path, 
        dpi=300, 
        bbox_inches='tight'
    )
    plt.close()

def plot_accuracy_curve(accs, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8,5))
    plt.plot(accs)
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()