import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(
                outputs,
                dim=1
            )

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average='macro'),
        "recall": recall_score(all_labels, all_preds, average='macro'),
        "f1": f1_score(all_labels, all_preds, average='macro'),
        "confusion_matrix": confusion_matrix(all_labels, all_preds)
    }
    return metrics