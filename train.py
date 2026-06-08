import os
import argparse
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

from src.config import *
from src.dataset import QuickDrawDataset
from src.model import QuickDrawCNN
from src.evaluate import evaluate
from src.visualize import save_confusion_matrix, plot_loss_curve, plot_accuracy_curve
from src.utils import save_metrics, seed_everything

def get_args():
    parser = argparse.ArgumentParser(
        description="Train a CNN model for QuickDraw sketch classification"
    )

    parser.add_argument("--data_path", type=str, default="data", help="Path to the directory containing .npy dataset files")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints", help="Directory to save model checkpoints (.pth files)")

    parser.add_argument("--samples_per_class", type=int, default=5000, help="Number of samples to load from each class")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")

    parser.add_argument("--train_ratio", type=float, default=0.56, help="Proportion of data used for training")
    parser.add_argument("--val_ratio", type=float, default=0.24, help="Proportion of data used for validation")

    parser.add_argument("--log_path", type=str, default="tensorboard_logs", help="Directory to store TensorBoard logs")
    return parser.parse_args()


def main():
    args = get_args()
    seed_everything(42)
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.makedirs(args.log_path)
    writer = SummaryWriter(args.log_path)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = QuickDrawDataset(data_path=args.data_path, samples_per_class=args.samples_per_class)
    labels = np.repeat(np.arange(len(CLASSES)), args.samples_per_class)
    indices = np.arange(len(dataset))

    train_idx, test_idx = train_test_split(
        indices,
        test_size= 0.2,
        random_state=42,
        stratify=labels
    )

    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.30,
        random_state=42,
        stratify=labels[train_idx]
    )

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


    model = QuickDrawCNN(len(CLASSES))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    # train

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        metrics = evaluate(model, val_loader, device)
        val_acc = metrics["accuracy"]
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        val_accuracies.append(val_acc)

        writer.add_scalar("Loss/Train", epoch_loss, epoch+1)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch+1)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_save_path = os.path.join(args.checkpoint_path, "best_model.pth")
            torch.save(model.state_dict(), model_save_path)
            
        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Loss: {epoch_loss:.4f} "
            f"Val Acc: {val_acc:.4f}"
        )
    writer.close()

    model_save_path = os.path.join(args.checkpoint_path, "best_model.pth")
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))

    test_metrics = evaluate(model, test_loader, device)
    save_metrics(test_metrics, "outputs/metrics.json")

    print("Final Test Metrics")
    for key, value in test_metrics.items():
        if key != "confusion_matrix":  
            print(f"Final Test {key.capitalize()}: {value:.4f}")



    # ve ma tran sai so
    save_confusion_matrix(
    test_metrics["confusion_matrix"],
    CLASSES,
    "outputs/confusion_matrix.png"
    )
    plot_loss_curve(train_losses, "outputs/loss_curve.png")
    plot_accuracy_curve(val_accuracies, "outputs/accuracy_curve.png")
    
if __name__ == '__main__':
    main()