import torch
from typing import Dict, Any
from .metrics import calculate_accuracy

class FSCILEvaluator:
    """
    Evaluates a model on a given data loader without parameter updates.
    """
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        
    @torch.no_grad()
    def evaluate(self, data_loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """
        Runs evaluation over the provided data_loader.
        Ensures model is in eval mode.
        """
        self.model.eval()
        
        all_logits = []
        all_targets = []
        
        for images, targets in data_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # The model forward returns logits
            logits = self.model(images)
            
            all_logits.append(logits.detach())
            all_targets.append(targets.detach())
            
        if not all_logits:
            return {
                "accuracy": 0.0,
                "samples": 0
            }
            
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        accuracy = calculate_accuracy(all_logits, all_targets)
        
        return {
            "accuracy": accuracy,
            "samples": len(all_targets)
        }
