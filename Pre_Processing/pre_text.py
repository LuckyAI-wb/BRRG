# pre_text.py
# -*- coding: utf-8 -*-
import re
from typing import List, Optional, Dict, Union

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

__all__ = ["preprocess_report"]

# 允许字符：字母数字 + 常用标点（按论文约束）
_ALLOWED_CHARS = r"[^a-z0-9\s\.\,\;\:\!\?\(\)\'\-\/]"
_space_re = re.compile(r"\s+")

def _clean_one(text: str, lowercase: bool = True) -> str:
    if text is None:
        return ""
    t = text.strip()
    if lowercase:
        t = t.lower()
    t = re.sub(_ALLOWED_CHARS, " ", t)     # 过滤到只剩允许字符
    t = _space_re.sub(" ", t).strip()      # 压缩空白
    return t

def preprocess_report(
    reports: List[str],
    *,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    bert_model_name_or_path: Optional[str] = None,
    max_len: int = 128,
) -> Dict[str, torch.Tensor]:
    assert isinstance(reports, list) and len(reports) > 0, "reports 需为非空列表"
    clean = [_clean_one(x, lowercase=True) for x in reports]

    # 分词器来源优先级：传入 tokenizer > 路径加载 > 抛错
    if tokenizer is None:
        assert bert_model_name_or_path is not None, "请传入 tokenizer 或 bert_model_name_or_path"
        tokenizer = AutoTokenizer.from_pretrained(bert_model_name_or_path, use_fast=True)

    batch = tokenizer(
        clean,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    # transformers 可能返回 token_type_ids 或不返回，保持原样即可
    # 统一为 LongTensor
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.long()
    return batch
