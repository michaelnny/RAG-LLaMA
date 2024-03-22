# Copyright (c) 2024 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.

from typing import List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def mean_pooling(token_embeddings, attention_mask):
    """Mean Pooling - Take attention mask into account for correct averaging"""
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)


class EmbeddingModel:
    'A wrapper around the sentence-transformers open source embedding model'

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = 'cpu'):
        assert model_name is not None, model_name
        self.model_name = f'sentence-transformers/{model_name}'
        self.device = device
        print(f'Loading {self.model_name} model and tokenizer from HuggingFace...')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

    @torch.no_grad()
    def compute_embeddings(self, texts: List[str], normalized_embed: bool = True) -> torch.Tensor:
        assert len(texts) > 0 and all([t is not None and len(t) > 0 for t in texts]), texts

        # Tokenize texts
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)

        # Compute token embeddings
        model_output = self.model(**encoded_input)

        # Perform pooling
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        sentence_embeddings = mean_pooling(token_embeddings, encoded_input['attention_mask'])

        # Normalize embeddings
        if normalized_embed:
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings
