from __future__ import annotations

import torch


def train_one_epoch(model, loader, criterion, optimizer, device: str) -> dict[str, float]:
    """
    Train model for one epoch.

    Returns averaged loss and accuracy so caller can log curves.
    """
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        # Move tensors to selected hardware.
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Standard forward -> loss -> backward -> optimizer step pipeline.
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_examples += batch_size
        total_loss += float(loss.item()) * batch_size

        predictions = logits.argmax(dim=1)
        total_correct += int((predictions == labels).sum().item())

    avg_loss = total_loss / max(total_examples, 1)
    accuracy = total_correct / max(total_examples, 1)

    return {"loss": avg_loss, "acc": accuracy}
