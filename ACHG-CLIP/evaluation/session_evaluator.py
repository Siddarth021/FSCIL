import torch
from typing import Dict, Any, List
from .evaluator import FSCILEvaluator
from data.session import FSCILDataManager

class FSCILSessionEvaluator:
    """
    Orchestrates the evaluation across all FSCIL sessions.
    """
    def __init__(self, model: torch.nn.Module, device: torch.device, data_manager: FSCILDataManager):
        self.model = model
        self.device = device
        self.data_manager = data_manager
        self.evaluator = FSCILEvaluator(model, device)
        
    def evaluate_session(self, session_id: int) -> Dict[str, Any]:
        """
        Evaluates a single session's cumulative test set.
        """
        session = self.data_manager.get_session(session_id)
        
        # Test loader inherently contains all classes seen up to session_id
        # (implemented per target paper's cumulative evaluation rule)
        metrics = self.evaluator.evaluate(session.test_loader)
        
        # Calculate classes seen
        classes_seen = self.data_manager.base_classes + (session_id * self.data_manager.classes_per_session)
        
        return {
            "session_id": session_id,
            "accuracy": metrics["accuracy"],
            "samples": metrics["samples"],
            "classes_seen": classes_seen
        }
        
    def evaluate_all_sessions(self, load_checkpoint_fn=None) -> List[Dict[str, Any]]:
        """
        Evaluates all sessions sequentially.
        If load_checkpoint_fn is provided, it will be called with the session_id
        to load the corresponding trained checkpoint before evaluation.
        """
        results = []
        for session_id in range(self.data_manager.num_sessions):
            if load_checkpoint_fn is not None:
                load_checkpoint_fn(session_id)
                
            session_metrics = self.evaluate_session(session_id)
            results.append(session_metrics)
            
        return results
