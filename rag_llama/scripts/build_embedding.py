# Copyright (c) 2024 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.

"""Load text from Tesla PDF manuals and compute embedding vectors"""

from typing import List, Tuple, Mapping, Text, Any
import argparse
import os
import pickle
import time

from rag_llama.models.embedding import EmbeddingModel
from rag_llama.scripts.extract_pdf import extract_text_by_section_from_tesla_manual


def find_certain_files_under_dir(root_dir: str, file_type: str = '.txt') -> List[str]:
    """Given a root folder, find all files in this folder and it's sub folders that matching the given file type."""
    assert file_type in ['.txt', '.json', '.jsonl', '.parquet', '.zst', '.json.gz', '.jsonl.gz', '.pdf']

    files = []
    if os.path.exists(root_dir):
        for root, dirnames, filenames in os.walk(root_dir):
            for f in filenames:
                if f.endswith(file_type):
                    files.append(os.path.join(root, f))
    return files


def main():
    # Our sentences to encode
    args = parser.parse_args()

    if os.path.exists(args.save_to):
        raise ValueError(f'Output file {args.save_to} already exits, aborting...')

    output_dir = os.path.dirname(args.save_to)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    embed_model = EmbeddingModel(device=args.device)

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
        extracted_sections.extend(extract_text_by_section_from_tesla_manual(file, args.start_page, args.end_page))

    # compute embeddings in batches
    t1 = time.time()
    num_sections = len(extracted_sections)

    print(f'Finished extracting {num_sections} sections from {num_files} PDF files in {t1-t0:.4f} seconds')

    print(f'Computing embeddings for {num_sections} sections, this may take a while...')

    batch_size = args.batch_size

    for start_idx in range(0, num_sections, batch_size):
        end_idx = min(start_idx + batch_size, num_sections)
        # add more metadata to the embedding
        batch_data = [data['document'] + ' - ' + data['subject'] + ' - ' + data['section'] + ' - ' + data['content'] for data in extracted_sections[start_idx:end_idx]]
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf_dir', help='Source pdf files for Tesla user manual', type=str, required=True)
    parser.add_argument('--start_page', help='Starts from page in the PDF', type=int, default=8)
    parser.add_argument('--end_page', help='Ends at page in the PDF', type=int, default=None)
    parser.add_argument('--save_to', help='Save the embedding and text to .pk file', type=str, required=True)
    parser.add_argument('--device', help='Compute device during embedding model, default `cpu`', type=str, default='cpu')
    parser.add_argument('--batch_size', help='Batch size during compute embedding', type=int, default=128)

    main()
