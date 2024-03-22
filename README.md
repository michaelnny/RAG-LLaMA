# RAG-LLaMA

A clean and simple implementation of Retrieval Augmented Generation (RAG) to enhanced LLaMA chat model to answer questions based on Tesla user manuals. Open-source embedding and Cross-Encoders reranking models from Sentence Transformers are employed in this project.

# Disclaimer

**Project Purpose:** This project is for research and education only, focusing on the study of individual algorithms rather than the creation of a standard library. If you're looking for a ready-to-use library for production applications, this project may not be suitable for your needs.

**Bug Reporting and Contributions:** We run some testing upon working on the project, but we cannot guarantee it's bug-free. Bug reports and pull requests are highly encouraged and welcomed.

# Environment and Requirements

- Python 3.10.6
- PyTorch 2.2.1
- Tensorboard 2.13.0
- Transformers 4.39.0

# Code Structure

- `rag_llama` directory contains main source code for the project.

  - `cores` directory contains core modules like generation, retrieval module etc.
  - `models` contains the LLaMA model class and open-source embedding model (from Sentence Transformers).

    - `embedding.py` open-source embeddings model from Sentence Transformers, loaded from HuggingFace Hub.
    - `reranking.py` open-source Cross-Encoders model from Sentence Transformers, loaded from HuggingFace Hub.
    - `model.py` LLaMA 2 model.

  - `scripts` directory contains all source code for convert the model weights and build embedding for documents.

    - `build_embedding.py` extract text from Tesla manual (PDF) and compute embeddings (save to .pkl file).
    - `convert_meta_checkpoint.py` convert Meta's pre-trained LLaMA-2 weights to support our model in plain PyTorch code, so we can load it to start fine-tuning.

  - `chat_with_rag.ipynb` run chat completion with LLaMA 2 chat model using RAG (with rerank).
  - `naive_retrieval.ipynb` test naive retrieval.
  - `retrieval_with_rerank.ipynb` test of retrieval with reranking.

# Project Setup

```
python3 -m pip install --upgrade pip setuptools

python3 -m pip install -r requirements.txt
```

# Project Preparation

_Notice: The scripts in the project uses hard-coded file paths which may not exists in your environment. You should change these to suit your environment before you run any script_

## Download and prepare LLaMA chat model weights

1. **Download the fine-tuned chat model weights** please refer to https://github.com/facebookresearch/llama on how to download it.
2. **Convert Meta's fine-tuned chat model weights** using script `python3 -m rag_llama.scripts.convert_meta_checkpoint`, so it's compatible with our naming convention.

# Step 1 - Pre-compute Document Embeddings

Use the following script to extract sections from the Tesla manual pdf and pre-compute embeddings. Or you can use the `./data/Tesla_manual_embeddings.pk` directly.

```
python3 -m rag_llama.scripts.build_embedding --pdf_dir "./data/docs" --save_to "./data/Tesla_manual_embeddings.pk"
```

To test the retrieval systems and embeddings, open the `naive_retrieval.ipynb` or `retrieval_with_rerank.ipynb` to play with different retrieval systems.

# Step 2 - Run LLaMA Chat Completion with RAG

To play with LLaMA and RAG (with reranking), you can open the `chat_with_rag.ipynb`.

# License

This project is licensed under the MIT License (the "License")
see the LICENSE file for details

- The LLaMA2 model weights are licensed for both researchers and commercial entities. For details, visit: https://github.com/facebookresearch/llama#license
