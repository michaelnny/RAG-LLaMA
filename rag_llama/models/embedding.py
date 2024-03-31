# Copyright (c) 2024 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.

from typing import List
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel


def mean_pooling(embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean Pooling - Take attention mask into account for correct averaging"""
    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    return torch.sum(embeddings * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)


def post_embedding_processing(embeddings: torch.Tensor, attention_mask: torch.Tensor, normalized_embed: bool) -> torch.Tensor:
    # Perform pooling
    embeddings = mean_pooling(embeddings, attention_mask)

    # Normalize embeddings
    if normalized_embed:
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings


MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'


def create_embedding_tokenizer(ckpt_dir: str = None) -> BertTokenizer:

    if ckpt_dir and os.path.exists(ckpt_dir) and os.path.isdir(ckpt_dir):
        print(f'Loading tokenizer from {ckpt_dir}...')
        return AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True, do_basic_tokenize=True)

    print(f'Loading {MODEL_NAME} tokenizer from HuggingFace...')
    return AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, do_basic_tokenize=True)


def create_embedding_model(ckpt_dir: str = None) -> BertModel:
    if ckpt_dir and os.path.exists(ckpt_dir) and os.path.isdir(ckpt_dir):
        print(f'Loading model from {ckpt_dir}...')
        return AutoModel.from_pretrained(ckpt_dir)

    print(f'Loading {MODEL_NAME} model from HuggingFace...')
    return AutoModel.from_pretrained(MODEL_NAME)


class EmbeddingModel:
    'A wrapper around the sentence-transformers open source embedding model'

    def __init__(self, device: str = 'cpu', model_ckpt_dir: str = None, tokenizer_ckpt_dir: str = None):
        """
        Arguments:
            device (str): torch runtime device for the model, default 'cpu'.
            model_ckpt_dir (str): fine-tuned model checkpoint dir, default None.
            tokenizer_ckpt_dir (str): tokenizer checkpoint dir with possible custom tokens, default None.
        """
        self.device = device

        self.tokenizer = create_embedding_tokenizer(tokenizer_ckpt_dir)
        self.model = create_embedding_model(model_ckpt_dir).to(self.device).eval()

    @torch.no_grad()
    def compute_embeddings(self, texts: List[str], normalized_embed: bool = True) -> torch.Tensor:
        assert len(texts) > 0 and all([t is not None and len(t) > 0 for t in texts]), texts

        # Tokenize texts
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)

        # Compute token embeddings
        model_output = self.model(**encoded_input)

        # First element of model_output contains all token embeddings
        # embeddings = model_output[0]
        embeddings = model_output.last_hidden_state
        embeddings = post_embedding_processing(embeddings, encoded_input['attention_mask'], normalized_embed)

        return embeddings
