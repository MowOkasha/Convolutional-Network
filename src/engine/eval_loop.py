from __future__ import annotations

import torch


@torch.no_grad()
def evaluate(model, loader, criterion, device: str) -> dict:
    """
    Evaluate model on validation or test loader.

    Returns:
    - loss and accuracy
    - targets and predictions (useful for confusion matrix/report later)
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    all_targets: list[int] = []
    all_predictions: list[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_examples += batch_size
        total_loss += float(loss.item()) * batch_size

        predictions = logits.argmax(dim=1)
        total_correct += int((predictions == labels).sum().item())

        all_targets.extend(labels.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

    avg_loss = total_loss / max(total_examples, 1)
    accuracy = total_correct / max(total_examples, 1)

    return {
        "loss": avg_loss,
        "acc": accuracy,
        "targets": all_targets,
        "predictions": all_predictions,
    }
