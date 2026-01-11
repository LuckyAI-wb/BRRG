import json
from typing import List, Optional, Dict, Any
from torch.utils.data import DataLoader, Dataset

DEFAULTS: Dict[str, Any] = {
    "image_path_1": "",
    "image_path_2": "",
    "report": "",
    "radgraph_entities": {},
    "radgraph_relations": {},
    "dynamic_kg": {},
}

class MIMICDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], keys: Optional[List[str]] = None):
        self.data = data
        # 默认把新字段也一起输出
        self.keys = keys or [
            "image_path_1",
            "image_path_2",
            "report",
            "radgraph_entities",
            "radgraph_relations",
            "dynamic_kg",
        ]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.data[index]
        out = {}
        for k in self.keys:
            # 为不同类型字段给出合理默认：路径/文本 -> 空串；字典型 -> {}
            out[k] = item.get(k, DEFAULTS.get(k, None))
        return out

    def __len__(self) -> int:
        return len(self.data)

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    # 简单聚合：同名键 -> 列表
    batched = {}
    for key in batch[0].keys():
        batched[key] = [sample.get(key) for sample in batch]
    return batched

def get_dataloader(
    split: str,
    batch_size: int = 4,
    dataset_path: str = r"E:\CKRRG\datasets\MIMC-CXR\dataset_with_dynamic_kg.json",
    num_workers: int = 0,
    shuffle: Optional[bool] = None,
    drop_last: bool = False,
    keys: Optional[List[str]] = None,
) -> DataLoader:
    """
    根据新的数据结构返回 DataLoader。
    - split: 'train' / 'val' / 'test'
    - keys: 想要从样本中取出的字段；默认会包含 image_path_1/2、report、radgraph_*、dynamic_kg
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if split not in data:
        raise KeyError(f"Split '{split}' 不存在。可用 split: {list(data.keys())}")

    ds = XRayDataset(data[split], keys=keys)

    if shuffle is None:
        shuffle = split.lower() == "train"

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
    return loader
