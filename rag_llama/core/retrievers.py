# Copyright (c) 2024 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.

from typing import Tuple, Mapping, List, Text, Any
import os
import pickle
import torch
from rank_bm25 import BM25Okapi

from rag_llama.models.embedding import EmbeddingModel
from rag_llama.models.ranking import RankingModel


def reciprocal_rank_fusion(all_rankings: List[List[int]], k: int = 60) -> Tuple[List[int], List[float]]:
    """Takes in list of rankings produced by multiple retrieval algorithms,
    and returns newly of ranked and scored items.

    Arguments:
        all_rankings (List[List[int]]): a list of rankings (indices) produced by multiple retrieval algorithms.
        k (int): constant balance the rankings.

    Returns:
        tuple:
            new fused ranking
            new ranking scores
    """

    assert k > 1
    assert len(all_rankings) >= 2

    scores_dict = {}  # key is the index and value is the score of that index
    # 1. Take every retrieval algorithm ranking
    for algorithm_ranks in all_rankings:
        # 2. For each ranking, take the index and the ranked position
        for rank, idx in enumerate(algorithm_ranks):
            # 3. Calculate the score and add it to the index
            if idx in scores_dict:
                scores_dict[idx] += 1 / (k + rank)
            else:
                scores_dict[idx] = 1 / (k + rank)

    # 4. Sort the indices based on accumulated scores
    sorted_scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)

    rrf_ranks = list([item[0] for item in sorted_scores])
    rrf_scores = list([item[1] for item in sorted_scores])

    return rrf_ranks, rrf_scores


class StandardRetriever:
    """A naive mechanism to load pre-computed embedding in cache and perform cosine similarity lookup,
    with the option to use BM25 keyword-based search too."""

    def __init__(self, embed_file: str, device: str = 'cpu'):
        """
        Arguments:
            embed_file (str): the `.pk` file contains pre-computed embedding as well as original text and other metadata.
            device (str): pytorch runtime device for the model.
        """
        assert os.path.exists(embed_file) and embed_file.endswith('.pk')

        self.device = device
        self.samples = pickle.load(open(embed_file, 'rb'))

        assert all(['embed' in d and 'formatted_text' in d for d in self.samples])
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
        semantic_scores = torch.cosine_similarity(curr_embed, self.embed_matrix, dim=1)

        # Get top k similar indices
        top_k_indices = semantic_scores.argsort(descending=True)[:top_k]

        # Retrieve top k similar items
        # similar_items = [self.samples[i] for i in top_k_indices]

        # Retrieve top k similar items with their scores
        similar_items = []
        for i in top_k_indices:
            item = self.samples[i]
            item['semantic_score'] = semantic_scores[i].item()
            similar_items.append(item)

        return similar_items


class RerankRetriever(StandardRetriever):
    'A simple implementation of two stage retrieval system'

    def __init__(self, embed_file: str, device: str = 'cpu'):
        """
        Arguments:
            embed_file (str): the `.pk` file contains pre-computed embedding as well as original text and other metadata.
            device (str): pytorch runtime device for the model.
        """

        StandardRetriever.__init__(self, embed_file, device)
        self.rank_model = RankingModel(device=self.device)

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
        retrieved_items = StandardRetriever.retrieve(self, query, top_k)

        # Stage 2 - compute rerank scores using another reranking model
        texts = [d['content'] for d in retrieved_items]
        scores = self.rank_model.compute_rerank_scores(query, texts, normalize_score)
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


class HybridRetriever(StandardRetriever):
    """A naive mechanism to use hybrid solution for retrieval,
    where we combine embedding-based and BM25 keyword-based rankings by using reciprocal rank fusion (RRF)."""

    def __init__(self, embed_file: str, device: str = 'cpu'):
        """
        Arguments:
            embed_file (str): the `.pk` file contains pre-computed embedding as well as original text and other metadata.
            device (str): pytorch runtime device for the model.
        """
        StandardRetriever.__init__(self, embed_file, device)

        assert all(['formatted_text' in d for d in self.samples])

        # for BM25 keyword-based search
        def bm25_tokenizer(sentence):
            corpus = []
            newlines = sentence.split('\n')
            for line in newlines:
                for word in line.split(' '):
                    if len(word) > 2:  # Remove empty strings, assuming word must have at least 3 characters
                        corpus.append(word.lower())

            return corpus

        self.bm25_tokenizer = bm25_tokenizer
        self.document_texts = [bm25_tokenizer(doc['formatted_text']) for doc in self.samples]
        self.bm25 = BM25Okapi(self.document_texts)

    def get_bm25_scores(self, query: str) -> List[float]:
        """Returns BM25 keyword-based search scores"""
        tokenized_query = self.bm25_tokenizer(query)
        return self.bm25.get_scores(tokenized_query)

    def normalize_scores(self, scores: torch.Tensor) -> torch.Tensor:
        # Normalize into range [0, 1]
        min_val = torch.min(scores)
        max_val = torch.max(scores)
        if min_val == max_val:
            return scores

        normalized_tensor = (scores - min_val) / (max_val - min_val)
        return normalized_tensor

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

        bm25_scores = self.get_bm25_scores(query)

        bm25_scores = torch.tensor(bm25_scores)

        # Compute embedding for query
        curr_embed = self.embed_model.compute_embeddings([query], normalized_embed)  # [1, embed_size]
        # Compute cosine similarity between current embedding and all embeddings in cache
        semantic_scores = torch.cosine_similarity(curr_embed, self.embed_matrix, dim=1)

        # Normalize scores
        bm25_scores = self.normalize_scores(bm25_scores)

        # ranking the different scores
        semantic_ranking = semantic_scores.argsort(descending=True)
        bm25_ranking = bm25_scores.argsort(descending=True)

        rrf_ranks, rrf_scores = reciprocal_rank_fusion([semantic_ranking.tolist(), bm25_ranking.tolist()])
        top_k_indices = rrf_ranks[:top_k]

        # Retrieve top k similar items with their scores
        similar_items = []
        for i in top_k_indices:
            item = self.samples[i]
            item['bm25_scores'] = bm25_scores[i].item()
            item['semantic_score'] = semantic_scores[i].item()
            item['rrf_score'] = rrf_scores[i]
            similar_items.append(item)

        return similar_items
