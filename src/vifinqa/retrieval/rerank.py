"""rerank.py — LocalReranker: Qwen3-Reranker-0.6B chạy local (transformers, CPU).

Cách dùng chuẩn theo model card: AutoModelForCausalLM + score token "yes"/"no"
(đọc logit ở vị trí cuối, log_softmax trên 2 token → score = P("yes")). Không dùng
AutoModelForSequenceClassification — head classification không nằm trong checkpoint
(là `1_LogitScore` của sentence-transformers).
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# bf16 weights trên CPU bị torch emulation → chậm ~3x; ép float32.
# transformers 5.x dùng `dtype` thay cho `torch_dtype` (deprecated).
_DTYPE_KWARG = {"dtype": torch.float32}

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
    """Cross-encoder rerank cục bộ (Qwen3-Reranker-0.6B)."""

    def __init__(self, model_dir: str, device: str = "cpu", instruction: str | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_dir).to(device).eval()
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.true_id = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        self.false_id = self.tokenizer("no", add_special_tokens=False).input_ids[0]
        self.instruction = instruction or _DEFAULT_INSTRUCTION
        self.device = device
        self.max_length = 8192

    def _format(self, query: str, doc: str) -> str:
        return (
            f"{_PREFIX}<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}{_SUFFIX}"
        )

    @torch.no_grad()
    def score_pairs(self, pairs: list[tuple[str, str]], batch_size: int = 16) -> list[float]:
        """pairs = [(query, doc), ...] → list relevance score [0..1]."""
        texts = [self._format(q, d) for q, d in pairs]
        scores: list[float] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            inputs = self.tokenizer(
                chunk, padding=False, truncation=True, return_attention_mask=False, max_length=self.max_length
            )
            inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            logits = self.model(**inputs).logits[:, -1, :]  # vị trí cuối
            true_v = logits[:, self.true_id]
            false_v = logits[:, self.false_id]
            stacked = torch.stack([false_v, true_v], dim=1)
            probs = torch.nn.functional.log_softmax(stacked, dim=1)
            scores.extend(probs[:, 1].exp().float().tolist())
        return scores
