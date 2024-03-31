# Copyright (c) 2024 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.

"""Load text from Tesla PDF manuals and compute embedding vectors"""

import argparse
import os
import pickle
import time
import random
import torch

from rag_llama.models.embedding import EmbeddingModel
from rag_llama.core.extractions.tesla_manual import extract_tesla_manual_sections
from rag_llama.core.helper import find_certain_files_under_dir


def main():
    args = parser.parse_args()

    if os.path.exists(args.save_to):
        raise ValueError(f'Output file {args.save_to} already exits, aborting...')

    output_dir = os.path.dirname(args.save_to)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed_model = EmbeddingModel(device=device, model_ckpt_dir=args.model_ckpt_dir, tokenizer_ckpt_dir=args.tokenizer_ckpt_dir)

    # find all pdf files in target dir
    pdf_files = find_certain_files_under_dir(args.pdf_dir, '.pdf')
    num_files = len(pdf_files)

    if num_files == 0:
        raise RuntimeError(f'Found zero pdf files in {args.pdf_dir}, aborting...')

    # extract sections from PDF files
    t0 = time.time()
    extracted_sections = []
    for file in pdf_files:
        print(f'Extracting text from {file}, this may take a while...')
        extracted_sections.extend(extract_tesla_manual_sections(pdf_path=file, max_words=args.max_words, start_page=args.start_page, end_page=args.end_page))

    # compute embeddings in batches
    t1 = time.time()
    num_sections = len(extracted_sections)

    print(f'Finished extracting {num_sections} sections from {num_files} PDF files in {t1-t0:.4f} seconds')

    print(f'Computing embeddings for {num_sections} sections, this may take a while...')

    batch_size = args.batch_size

    for start_idx in range(0, num_sections, batch_size):
        end_idx = min(start_idx + batch_size, num_sections)
        batch_data = [data['formatted_text'] for data in extracted_sections[start_idx:end_idx]]
        embeddings = embed_model.compute_embeddings(batch_data)
        # Assign embeddings back to each item
        for i, embedding in enumerate(embeddings.tolist()):
            extracted_sections[start_idx + i]['embed'] = embedding

    t2 = time.time()

    assert all(['embed' in d for d in extracted_sections])

    print(f'Finished computing embeddings for {num_sections} sections from {num_files} PDF files in {t2-t1:.4f} seconds')

    # save embeddings
    print(f'Saving sections with embeddings to {args.save_to} ...')
    pickle.dump(extracted_sections, open(args.save_to, 'wb'))


if __name__ == '__main__':

    seed = 1
    torch.manual_seed(seed)
    random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf_dir', help='Source pdf files for Tesla user manual', type=str, default='./data/docs')
    parser.add_argument('--max_words', help='Maximum number of words per chunk', type=int, default=300)
    parser.add_argument('--start_page', help='Starts from page in the PDF', type=int, default=8)
    parser.add_argument('--end_page', help='Ends at page in the PDF', type=int, default=None)
    parser.add_argument('--batch_size', help='Batch size during compute embedding', type=int, default=128)
    parser.add_argument('--model_ckpt_dir', help='Fine-tuned embedding checkpoint dir, default none', type=str, default=None)
    parser.add_argument('--tokenizer_ckpt_dir', help='Tokenizer checkpoint dir, default none', type=str, default=None)
    parser.add_argument('--save_to', help='Save the embedding and text to .pk file', type=str, default='./data/Tesla_manual_embeddings.pk')

    # # Incase using fine-tuned embedding model
    # parser.add_argument('--model_ckpt_dir', help='Fine-tuned embedding checkpoint dir, default none', type=str, default='./checkpoints/finetune_embedding/epoch-50')
    # parser.add_argument('--tokenizer_ckpt_dir', help='Tokenizer checkpoint dir, default none', type=str, default='./checkpoints/finetune_embedding/tokenizer')
    # parser.add_argument('--save_to', help='Save the embedding and text to .pk file', type=str, default='./data/Tesla_manual_embeddings_finetuned.pk')

    main()
