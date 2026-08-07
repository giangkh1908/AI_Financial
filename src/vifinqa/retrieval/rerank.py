"""rerank.py — LocalReranker: cross-encoder rerank cục bộ (transformers).

Hỗ trợ 2 loại model, auto-detect theo `architectures` trong config:
- BGE cross-encoder (vd `BAAI/bge-reranker-v2-m3`): `XLMRobertaForSequenceClassification`,
  input `tokenizer(query_với_instruction, document)`, score = logits (head đã sigmoid).
- Qwen3-Reranker (CausalLM): format yes/no + đọc logit ở vị trí cuối.

GPU GTX 1650 (4GB): dùng fp16 để đủ fit (~1.2GB). CPU: fp32.
"""

from __future__ import annotations

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

# Đa số cross-encoder VN (AITeamVN/Vietnamese_Reranker) KHÔNG cần instruction prefix —
# model card dùng `tokenizer(pairs)` trực tiếp. BGE gốc mới cần prefix; để rỗng mặc định.
_BGE_INSTRUCTION = ""

# Qwen3-Reranker yes/no format.
_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on "
    "the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
    "<|im_end|>\n<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n thinking\n\n response\n\n"
_DEFAULT_INSTRUCTION = (
    "The Document is a financial statement table of a Vietnamese listed company. "
    "Retrieve the table that contains the data needed to answer the financial Query."
)


class LocalReranker:
    """Cross-encoder rerank cục bộ — tự chọn BGE (SequenceClassification) hay Qwen (CausalLM)."""

    def __init__(self, model_dir: str, device: str = "cpu", instruction: str | None = None):
        self.device = device
        self.instruction = instruction or _DEFAULT_INSTRUCTION
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left")

        archs = (AutoConfig.from_pretrained(model_dir).architectures or [])
        is_seq = any("SequenceClassification" in a for a in archs)
        # fp16 trên CUDA (GTX 1650 4GB — fp32 model 2.4GB + activation dễ OOM); CPU fp32.
        dtype = torch.float16 if device.startswith("cuda") else torch.float32

        if is_seq:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_dir, dtype=dtype
            ).to(device).eval()
            self.mode = "bge"
            self.max_length = 512
            self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).to(device).eval()
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.true_id = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
            self.false_id = self.tokenizer("no", add_special_tokens=False).input_ids[0]
            self.mode = "qwen"
            self.max_length = 8192

    def _format_qwen(self, query: str, doc: str) -> str:
        return (
            f"{_PREFIX}<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}{_SUFFIX}"
        )

    @torch.no_grad()
    def score_pairs(self, pairs: list[tuple[str, str]], batch_size: int = 16) -> list[float]:
        """pairs = [(query, doc), ...] → list relevance score [0..1]."""
        scores: list[float] = []
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i : i + batch_size]
            if self.mode == "bge":
                enc = self.tokenizer(
                    [(_BGE_INSTRUCTION + q, d) for q, d in chunk],
                    padding=True, truncation=True, max_length=self.max_length, return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self.model(**enc).logits.view(-1)  # (batch,) — head đã sigmoid
                scores.extend(logits.float().tolist())
            else:
                texts = [self._format_qwen(q, d) for q, d in chunk]
                inputs = self.tokenizer(
                    texts, padding=False, truncation=True, return_attention_mask=False,
                    max_length=self.max_length,
                )
                inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logits = self.model(**inputs).logits[:, -1, :]
                true_v = logits[:, self.true_id]
                false_v = logits[:, self.false_id]
                stacked = torch.stack([false_v, true_v], dim=1)
                probs = torch.nn.functional.log_softmax(stacked, dim=1)
                scores.extend(probs[:, 1].exp().float().tolist())
        return scores