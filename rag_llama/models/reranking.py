# Copyright (c) 2024 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.

from typing import List, Tuple
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification


class RerankingModel:
    'A wrapper around the sentence-transformers open source Cross-Encoders'

    def __init__(self, model_name: str = 'ms-marco-MiniLM-L-6-v2', device: str = 'cpu'):
        assert model_name is not None, model_name
        self.model_name = f'cross-encoder/{model_name}'
        self.device = device
        print(f'Loading {self.model_name} model and tokenizer from HuggingFace...')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device).eval()

    @torch.no_grad()
    def compute_rerank_scores(self, query: str, texts: List[str], normalize_score: bool = True) -> torch.Tensor:
        assert query is not None and len(query) > 0, query
        assert len(texts) > 0 and all([t is not None and len(t) > 0 for t in texts]), texts

        num_pairs = len(texts)

        # Tokenize sentences
        features = self.tokenizer([query] * num_pairs, texts, padding=True, truncation=True, return_tensors='pt').to(self.device)

        # Compute token embeddings
        scores = self.model(**features).logits

        if normalize_score:
            min_score = torch.min(scores)
            max_score = torch.max(scores)
            scores = (scores - min_score) / (max_score - min_score)
        return scores
