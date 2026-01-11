import os
from typing import Optional, Union, Dict
from PIL import Image
import torch
import torchvision.transforms as T

def preprocess_xray(
    frontal_path: str,
    lateral_path: Optional[str] = None,
    *,
    train: bool = False,
    image_size: int = 224,
    device: Optional[Union[str, torch.device]] = None,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    读取并预处理胸片：
      - 灰度->3通道
      - Resize + CenterCrop 到 image_size
      - train=True 时做轻量仿射增强（无水平翻转）
      - 使用 ImageNet 均值/方差做归一化
    返回：
      - 无侧位：Tensor [1,3,H,W]
      - 有侧位：{'frontal': Tensor[1,3,H,W], 'lateral': Tensor[1,3,H,W]}
    """
    assert os.path.isfile(frontal_path), f"Missing file: {frontal_path}"

    # 放在函数内部的 ImageNet 归一化参数
    IMAGENET_MEAN_STD = ([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])

    # 构建变换
    tfms = []
    if train:
        tfms += [
            T.Resize(int(image_size * 1.05)),
            T.CenterCrop(image_size),
            T.RandomApply([T.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.95, 1.05))], p=0.9),
        ]
    else:
        tfms += [T.Resize(image_size), T.CenterCrop(image_size)]
    tfms += [
        T.ToTensor(),
        # 灰度 -> 3 通道（复制）
        T.Lambda(lambda x: x.expand(3, *x.shape[1:]) if x.shape[0] == 1 else x),
        T.Normalize(mean=IMAGENET_MEAN_STD[0], std=IMAGENET_MEAN_STD[1]),
    ]
    tfm = T.Compose(tfms)

    def _load_one(p: str) -> torch.Tensor:
        img = Image.open(p).convert("L")
        ten = tfm(img).unsqueeze(0)  # [1,3,H,W]
        return ten.to(device) if device is not None else ten


    f = _load_one(frontal_path)

    if lateral_path and os.path.isfile(lateral_path):
        l = _load_one(lateral_path)
        return {"frontal": f, "lateral": l}

    return f






