# RAG-LLaMA

A clean and simple implementation of Retrieval Augmented Generation (RAG) to enhanced LLaMA chat model to answer questions from a private knowledge base. We use Tesla user manuals to build the knowledge base, and use open-source embedding and Cross-Encoders ranking models from Sentence Transformers in this project.

This entire project runs locally, no third-party APIs are needed (with the exception of downloading open-source model weights from the HuggingFace Hub).

# Disclaimer

**Project Purpose:** This is a toy project for research and education only, focusing on the study of individual algorithms rather than the creation of a standard library. If you're looking for a ready-to-use library for production applications, this project may not be suitable for your needs.

**Bug Reporting and Contributions:** We run some testing upon working on the project, but we cannot guarantee it's bug-free. Bug reports and pull requests are highly encouraged and welcomed.

# Environment and Requirements

- Python 3.10.6
- PyTorch 2.2.1
- Tensorboard 2.13.0
- Transformers 4.39.0

# Code Structure

- `rag_llama` directory contains main source code for the project.

  - `cores` directory contains core modules like retrieval, generation, and text extractions.
  - `models` contains the LLaMA model class and open-source embedding model (from Sentence Transformers).

    - `embedding.py` open-source embeddings model from Sentence Transformers, loaded from HuggingFace Hub.
    - `ranking.py` open-source Cross-Encoders ranking model from Sentence Transformers, loaded from HuggingFace Hub.
    - `model.py` LLaMA 2 model.

  - `scripts` directory contains all source code for convert the model weights and build embedding for documents.

    - `build_embedding.py` extract text from Tesla manual (PDF) and compute embeddings (save to .pkl file).
    - `convert_meta_checkpoint.py` convert Meta's pre-trained LLaMA-2 weights to support our model in plain PyTorch code, so we can load it to start fine-tuning.

- `play` directory contains notebooks to run tests.
  - `chatbot.ipynb` run Tesla customer support chatbot with LLaMA 2 chat model using RAG.
  - `standard_retriever.ipynb` test standard retrieval.
  - `retriever_with_rerank.ipynb` test retrieval with reranking.

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

To test the retrieval systems and embeddings, open the `standard_retriever.ipynb` or `retriever_with_rerank.ipynb` to play with different retrieval systems.

# Step 2 - Run RAG-based Chat Completion LLaMA

Once the document embeddings and retrieval systems have been tested. We can start integrate these modules into LLM.

Here's an overview of the steps involved when using RAG with reranking:

- Compute embedding for the user query using the same embedding model
- Looking for top K matches between user query embedding and the documents/sections embedding based on cosine similarity scores
- Compute relativity scores using a ranking model for each pair of `user query + single item in the top K matches`, and the select top N matches based on the scores
- Add selected top N documents/sections as part of user query and send to LLM

Here's an overview of the steps involved when using RAG with HyDE and reranking:

- Ask LLM to generate a passage based on the user
- Compute embedding for the passage from LLM using the same embedding model
- Looking for top K matches between passage embedding and the documents/sections embedding based on cosine similarity scores
- Compute relativity scores using a ranking model for each pair of `passage + single item in the top K matches`, and the select top N matches based on the scores
- Add selected top N documents/sections as part of user query and send to LLM

To play with RAG-based chatbot for Tesla customer support assistant, you can open the `chatbot.ipynb`.

**Note on the Chatbot:**

- This is a toy project and the performance of the chatbot might not be great.
- It requires at least 16GB of GPU VRAM if you want to run it on GPU.

# License

This project is licensed under the MIT License (the "License")
see the LICENSE file for details

- The LLaMA2 model weights are licensed for both researchers and commercial entities. For details, visit: https://github.com/facebookresearch/llama#license
