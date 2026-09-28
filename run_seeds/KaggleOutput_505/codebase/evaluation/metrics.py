import torch

def calculate_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculates the classification accuracy.
    
    PAPER-FACT: The target paper specifies cumulative classification accuracy over
    all classes seen so far. No other metrics are explicitly defined for FSCIL.
    """
    assert logits.dim() == 2, "Logits must be 2D (batch_size, num_classes)"
    assert targets.dim() == 1, "Targets must be 1D (batch_size)"
    
    if len(targets) == 0:
        return 0.0
        
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == targets).sum().item()
    return correct / len(targets)
