from .datasets import CIFAR100Dataset, MiniImageNetDataset, CUB200Dataset
from .transforms import get_transforms
from .session import FSCILDataManager

def get_data_manager(config: dict, data_root: str, synthetic: bool = False) -> FSCILDataManager:
    """
    Initializes the dataset based on the configuration and returns an FSCILDataManager.
    """
    dataset_name = config.get("dataset_name", {}).get("value")
    
    if dataset_name == "CIFAR-100":
        DatasetClass = CIFAR100Dataset
    elif dataset_name == "miniImageNet":
        DatasetClass = MiniImageNetDataset
    elif dataset_name == "CUB-200-2011":
        DatasetClass = CUB200Dataset
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")
        
    train_transforms = get_transforms(dataset_name, is_train=True)
    test_transforms = get_transforms(dataset_name, is_train=False)
    
    train_data = DatasetClass(root=data_root, name=dataset_name, train=True, transform=train_transforms, synthetic=synthetic).get_dataset()
    test_data = DatasetClass(root=data_root, name=dataset_name, train=False, transform=test_transforms, synthetic=synthetic).get_dataset()
    
    # Extract FSCIL parameters from config
    total_classes = config["total_classes"]["value"]
    base_classes = config["base_classes"]["value"]
    incremental_classes = config["incremental_classes_total"]["value"]
    classes_per_session = config["classes_per_incremental_session"]["value"]
    shots_per_class = config["shots_per_class"]["value"]
    
    base_batch_size = config["base_batch_size"]["value"]
    incremental_batch_size = config["incremental_batch_size"]["value"]
    
    # Use a deterministic seed for class ordering and few-shot sampling.
    seed_val = config.get("seed", 42)
    if isinstance(seed_val, dict):
        seed_val = seed_val.get("value", 42)
        
    manager = FSCILDataManager(
        train_dataset=train_data,
        test_dataset=test_data,
        total_classes=config.get("total_classes", {}).get("value", 100),
        base_classes=config.get("base_classes", {}).get("value", 60),
        incremental_classes=config.get("incremental_classes_total", {}).get("value", 40),
        classes_per_session=config.get("classes_per_incremental_session", {}).get("value", 5),
        shots_per_class=config.get("shots_per_class", {}).get("value", 5),
        base_batch_size=config.get("base_batch_size", {}).get("value", 4),
        incremental_batch_size=config.get("incremental_batch_size", {}).get("value", 4),
        seed=seed_val
    )
    
    return manager
