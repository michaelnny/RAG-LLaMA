# Copyright (c) 2024 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.

from typing import Mapping, List, Text, Any
import os
import pickle
import torch


from rag_llama.models.embedding import EmbeddingModel
from rag_llama.models.ranking import RankingModel


class NaiveRetriever:
    'A naive mechanism to load pre-computed embedding in cache and perform cosine similarity lookup'

    def __init__(self, embed_file: str, device: str = 'cpu'):
        """
        Arguments:
            embed_file (str): the `.pk` file contains pre-computed embedding as well as original text and other metadata.
            device (str): pytorch runtime device for the model.
        """
        assert os.path.exists(embed_file) and embed_file.endswith('.pk')

        self.device = device
        self.samples = pickle.load(open(embed_file, 'rb'))

        assert all(['embed' in d for d in self.samples])
        self.embed_matrix = torch.stack([torch.tensor(d['embed']) for d in self.samples]).to(self.device)

        self.embed_model = EmbeddingModel(device=self.device)

    def retrieve(self, query: str, top_k: int = 5, normalized_embed: bool = True) -> List[Mapping[Text, Any]]:
        """Retrieve top K items from the cache based on the cosine similarity scores.

        Arguments:
            query (str): the user query.
            top_k (int): number of items to retrieve during stage 1 naive embedding lookup
            normalized_embed (bool): normalize embedding

        Returns:
            a list of dict with key:value mapping with highest cosine similarity scores
        """
        assert query is not None and len(query) > 0, query
        assert top_k >= 1, top_k

        # Compute embedding for query
        curr_embed = self.embed_model.compute_embeddings([query], normalized_embed)  # [1, embed_size]

        # Compute cosine similarity between current embedding and all embeddings in cache
        similarities = torch.cosine_similarity(curr_embed, self.embed_matrix, dim=1)

        # Get top k similar indices
        top_k_indices = similarities.argsort(descending=True)[:top_k]

        # Retrieve top k similar items
        # similar_items = [self.samples[i] for i in top_k_indices]

        # Retrieve top k similar items with their scores
        similar_items = []
        for i in top_k_indices:
            item = self.samples[i]
            item['score'] = similarities[i].item()
            similar_items.append(item)

        return similar_items


class RerankRetriever(NaiveRetriever):
    'A simple implementation of two stage retrieval system'

    def __init__(self, embed_file: str, device: str = 'cpu'):
        """
        Arguments:
            embed_file (str): the `.pk` file contains pre-computed embedding as well as original text and other metadata.
            device (str): pytorch runtime device for the model.
        """

        NaiveRetriever.__init__(self, embed_file, device)

        self.rerank_model = RankingModel(device=self.device)

    def retrieve(self, query: str, top_k: int = 50, top_n: int = 5, normalize_score: bool = False) -> List[Mapping[Text, Any]]:
        """
        Do two staged retrieve:
            1. Retrieve top K items from the cache based on the cosine similarity scores
            2. Compute rerank score using reranking model based on the results of step 1, then get the top N items with hightest score

        Arguments:
            query (str): the user query.
            top_k (int): number of items to retrieve during stage 1 naive embedding lookup
            top_n (int): number of items to retrieve during stage 2 based on the rerank score
            normalize_score (bool): normalize rerank score

        Returns:
            a list of dict with key:value mapping with highest reranking scores
        """
        assert query is not None and len(query) > 0, query

        assert top_n >= 1, top_n
        assert top_k > top_n, top_k

        # Stage 1 - naive retrieval based on cosine similarity scores
        retrieved_items = NaiveRetriever.retrieve(self, query, top_k)

        # Stage 2 - compute rerank scores using another reranking model
        texts = [d['content'] for d in retrieved_items]
        scores = self.rerank_model.compute_rerank_scores(query, texts, normalize_score)
        scores = scores.squeeze()
        assert len(scores.shape) == 1

        # Get top n similar indices
        top_n_indices = scores.argsort(descending=True)[:top_n]

        # Retrieve top n items with their reranking scores
        similar_items = []
        for i in top_n_indices:
            item = retrieved_items[i]
            item['rerank_score'] = scores[i].item()
            similar_items.append(item)

        return similar_items
