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
    - `tokenizer.py` tokenizer for LLaMA 2 model.

  - `convert_meta_checkpoint.py` convert Meta's pre-trained LLaMA-2 weights to support our model in plain PyTorch code, so we can load it to start fine-tuning.
  - `build_knowledge_base.py` extract text from Tesla manual (PDF) and compute embeddings (save to .pkl file).
  - `build_embed_finetune_dataset.py` extract text from Tesla manual (PDF) and build embedding model fine-tune dataset using alert codes (save to .pkl file).
  - `finetune_embedding.py` script to fine-tune the embedding model using alert codes dataset.

- `play` directory contains notebooks to run tests.
  - `chatbot.ipynb` run Tesla customer support chatbot with LLaMA 2 chat model using RAG.
  - `standard_retriever.ipynb` test standard retrieval.
  - `retriever_with_rerank.ipynb` test retrieval with reranking.
  - `hybrid_retriever.ipynb` test hybrid retrieval with semantic-search and keyword-search over Tesla car troubleshooting alert codes.
  - `retriever_alert_codes.ipynb` test the performance of fine-tuned embedding model over Tesla car troubleshooting alert codes.

# Project Setup

```
python3 -m pip install --upgrade pip setuptools

python3 -m pip install -r requirements.txt
```

# Project Preparation

_Notice: The scripts in the project uses hard-coded file paths which may not exists in your environment. You should change these to suit your environment before you run any script_

## Download and prepare LLaMA chat model weights

1. **Download the fine-tuned chat model weights** please refer to https://github.com/facebookresearch/llama on how to download it.
2. **Convert Meta's fine-tuned chat model weights** using script `python3 -m rag_llama.convert_meta_checkpoint`, so it's compatible with our naming convention.

# Step 1 - Build knowledge base by extract text and compute embeddings

Use the following script to extract sections from the Tesla manual pdf and pre-compute embeddings. Or you can use the `./data/Tesla_manual_embeddings.pk` directly.

```
python3 -m rag_llama.build_knowledge_base
```

To test the retrieval systems and embeddings, open the `standard_retriever.ipynb` or `retriever_with_rerank.ipynb` to play with different retrieval systems.

# Step 2 - Run RAG-based Chat Completion LLaMA

Once the document embeddings and retrieval systems have been tested. We can start integrate these modules into LLM.

Here's an overview of the steps involved when using RAG with reranking:

- Compute embedding for the user query using the same embedding model
- Looking for top K matches between user query embedding and the documents/sections embedding based on cosine similarity scores
- Compute relativity scores using a ranking model for each pair of `user query + single item in the top K matches`, and the select top N matches based on the scores
- Add selected top N documents/sections as part of user query and send to LLM

To play with RAG-based chatbot for Tesla customer support assistant, you can open the `chatbot.ipynb`.

**Note on the Chatbot:**

- This is a toy project and the performance of the chatbot might not be great.
- It requires at least 16GB of GPU VRAM if you want to run it on GPU.

# The alerts code problem

**Acknowledgement**: The Tesla alert code issue and fine-tuning of embedding model was first discussed in the blog post by Teemu Sormunen
https://medium.datadriveninvestor.com/improve-rag-performance-on-custom-vocabulary-e728b7a691e0

Similar to any software project, Tesla car user manual also comes with a list of troubleshooting alert codes to help user identify the problems with their cars. These alert codes often look like:

```
"APP_w224", "APP_w304", "BMS_a066", "BMS_a067", "CC_a001", "CC_a003", ...
```

However, these are not standard English words, and the pre-trained embedding model have no knowledge about these alert codes. In fact, the pre-trained BERT tokenizer will try to break these codes into separate parts, thus making them loss the original meaning.

```
question = "what does APP_w222 mean"
encode_output = tokenizer.tokenize(question)
decode_output = tokenizer.convert_tokens_to_string(encode_output)
print(encode_output)
['what', 'does', 'app', '_', 'w', '##22', '##2', 'mean']


print(decode_output)
what does app _ w222 mean
```

There are two approaches can we can use to solve this problem:

1. Use hybrid retrieval solution, where we mix semantic-search (embedding) with keyword-search (like MB25)
2. Fine-tune the embedding model to incorporate these alert codes

## Solution 1 - Hybrid retrieval

Hybrid retrieval seems to be a much easier solution, as it does not involve training the model. We only need to implement a keyword-search mechanism, and then performing an ranking operation based on the semantic-search and the keyword-search results.

We already implemented the hybrid retrieval component `HybridRetriever` inside `retrieval.py` module, it uses `BM25` (implemented using library `rank-bm25`) as the keyword-search algorithm.

You can open the `chatbot.ipynb` to play with it to see side-by-side the performance of the LLM agent with and without hybrid retrieval on the alert code problems.

## Solution 2 - Fine-tuning embedding model

To solve the alert code issue without using hybrid retrieval, we can fine-tune the embedding model. This is a much complex solution, but will generally yields better results than hybrid retrieval approach.

Generally speaking, to fine-tune the embedding model, it involves the following steps:

1. Prepare a fine-tune dataset
2. Fine-tune the embedding model by minimizing cosine embedding loss
3. Verify the fine-tune the embedding model

### 2.1 - Build fine-tune dataset

We first need to build the dataset contains the samples related to alert codes for fine-tuning the embedding model. In general, each sample should contain a query, in this case the query should also contain one alert code, and a positive passage related to the alert code, and a negative passage that's not related to the query.

Which can be done by running the following script.

```
python3 -m rag_llama.scripts.build_embed_finetune_dataset
```

We can monitoring the progress by using Tensorboard:

```
tensorboard --logdir=./logs
```

### 2.2 - Fine-tune embedding mode

The fine-tuning of the embedding model involves the following steps:

- Add alert codes as custom tokens to the pre-trained tokenizer
- Tokenize the sample texts in the datasets
- Set the `word_embeddings` layer trainable and frozen other layers in the model
- Start train the model over N epochs by minimizing the cosine embedding loss

We can run the following script to start the fine-tune process.

```
python3 -m rag_llama.finetune_embedding
```

### 2.3 - Verify the fine-tuned model

After the fine-tuning has finished, we can start verify the performance of the fine-tuned model by going through these steps:

- Build the knowledge base using the fine-tuned model by using `python3 -m rag_llama.scripts.build_knowledge_base`, while maintain the `model_ckpt_dir` and `tokenizer_ckpt_dir`
- Ues the new knowledge along with the fine-tuned model to build a retrieval component

You can open the `retriever_alert_codes.ipynb` to play with the fine-tuned model.

# License

This project is licensed under the MIT License (the "License")
see the LICENSE file for details

- The LLaMA2 model weights are licensed for both researchers and commercial entities. For details, visit: https://github.com/facebookresearch/llama#license

# Acknowledgement

The Tesla alert code issue and fine-tuning of embedding model was first discussed in the blog post by Teemu Sormunen
https://medium.datadriveninvestor.com/improve-rag-performance-on-custom-vocabulary-e728b7a691e0
