# Copyright (c) 2024 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.

"""Load text from Tesla PDF manuals and build a dataset over alert codes to fine-tune embedding model"""

from typing import List, Tuple, Mapping, Text, Any
import argparse
import os
import random
import re
import pickle
import json
import time
import math
import torch


from rag_llama.core.extractions.tesla_manual import extract_tesla_manual_sections
from rag_llama.core.helper import find_certain_files_under_dir


# templates for user query with place holder for alert codes
alert_place_holder = '<ALERT_CODE>'

query_templates = [
    f'I see {alert_place_holder} on my car, what should I do',
    f'Why is there a {alert_place_holder} on my screen?',
    f'What does {alert_place_holder} mean? I see it on my screen.',
    f'There is a code {alert_place_holder} on the touchscreen of my car, what does that mean',
    f'Is code {alert_place_holder} serious? What should I do',
    f'How do I get rid of {alert_place_holder} on my Tesla car',
    f'Should I call service if I see {alert_place_holder} on my car',
    f'How to fix {alert_place_holder} error on my car',
    f'What does {alert_place_holder} mean?',
    f'My car is showing {alert_place_holder} error, what steps should I take',
    f'What are the potential causes of {alert_place_holder} appearing on my dashboard',
    f'How urgent is it to address {alert_place_holder} on my vehicle',
    f'Are there any temporary fixes for {alert_place_holder} before seeking professional help',
    f'Does {alert_place_holder} indicate a safety concern that needs immediate attention',
    f'Can {alert_place_holder} be resolved without visiting a mechanic',
    f'Are there any common triggers for {alert_place_holder} to appear',
    f'What are the consequences of ignoring {alert_place_holder}',
    f'Is {alert_place_holder} something I can troubleshoot on my own',
    f'What maintenance procedures might prevent {alert_place_holder} from occurring',
    f'Can you tell me how to fix error {alert_place_holder} on my car',
    f'I need some instructions on how to diagnose code {alert_place_holder} on a Tesla car',
]


ALERT_CODES_PREFIX = set(['APP', 'BMS', 'CC', 'CHG', 'CHGS', 'CP', 'DI', 'GTW', 'MCU', 'TAS', 'THC', 'UMC'])
ALERT_CODES_SUFFIX = set(['_a', '_f', '_u', '_w'])

ALERT_CODES = []

for alert_prefix in ALERT_CODES_PREFIX:
    for alert_suffix in ALERT_CODES_SUFFIX:
        alert_code = alert_prefix + alert_suffix
        if alert_code not in ALERT_CODES:
            ALERT_CODES.append(alert_code)

# Regular expression pattern to match alert codes
pattern = r'\b(?:' + '|'.join(ALERT_CODES) + r')\w*\b'

# print(pattern)


def detect_alert_codes(text):
    matches = re.findall(pattern, text)

    # Return True if any match is found, otherwise False
    if matches:
        return matches[0]
    return


def get_formatted_text(data) -> str:
    """Format the document to add more metadata"""
    if data is None:
        return ''

    formatted_text = f"Subject: {data['subject']} - {data['section']}\nContent: {data['content']}"
    return formatted_text


def insert_alert_code_randomly(text, code) -> str:
    """Inserts a alert code randomly into a text string, ensuring proper spacing and integrity."""

    words = text.split()  # Split text into words

    # Find a random insertion position
    positions = list(range(len(words)))
    insert_pos = random.choice(positions)

    words.insert(insert_pos, code)
    # Reconstruct the text with proper spacing
    modified_text = ' '.join(words)
    return modified_text


def generate_queries_based_on_alert_code(alert: str, num_samples_per_alert: int) -> str:
    assert alert and len(alert) > 3, alert
    assert 1 <= num_samples_per_alert < len(query_templates), num_samples_per_alert
    queries = random.sample(query_templates, num_samples_per_alert)
    # replace alert code place holder
    queries = [q.replace(alert_place_holder, alert) for q in queries]
    return queries


def generate_negative_matches(all_data: List[Any], alert_code: str, num_samples: int, random_alert_ratio: float = 0.7) -> List[str]:
    assert num_samples >= 1, num_samples
    assert random_alert_ratio >= 0.1, random_alert_ratio
    negative_samples = []

    num_without_alerts = 0
    while len(negative_samples) < num_samples:
        random_sample = random.choice(all_data)
        random_alert_code = detect_alert_codes(random_sample['section'])

        # never use the positive one for negative sample
        if random_alert_code == alert_code:
            continue

        # try to mix random alerts into negative samples
        elif random_alert_code is None:
            if num_without_alerts < math.ceil(num_samples * (1 - random_alert_ratio)):
                negative_samples.append(random_sample)
                num_without_alerts += 1
        else:
            negative_samples.append(random_sample)

    return [get_formatted_text(item) for item in negative_samples]


def main():
    args = parser.parse_args()

    assert 1 <= args.num_samples_per_alert <= len(query_templates), args.num_samples_per_alert

    if os.path.exists(args.save_to) and len(os.listdir(args.save_to)) > 0:
        raise ValueError(f'Output files already exits in {args.save_to}, aborting...')

    if not os.path.exists(args.save_to):
        os.makedirs(args.save_to, exist_ok=True)

    train_output_file = os.path.join(args.save_to, 'train.pk')
    val_output_file = os.path.join(args.save_to, 'validation.pk')
    meta_output_file = os.path.join(args.save_to, 'meta.json')

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

    # filter chunks with alert code
    alerts_map = {}
    for i, item in enumerate(extracted_sections):
        alert_code = detect_alert_codes(item['section'])
        if alert_code and alert_code not in alerts_map:  # avoid duplicates incase have multiple PDFs
            alerts_map[alert_code] = i

    # split alerts into train and validation subsets
    all_alert_codes = list(alerts_map.keys())
    all_alerts_indices = list(alerts_map.values())

    # start to build samples
    def build_subset_dataset(subset_indices) -> List[Mapping[Text, Any]]:
        dataset = []
        dataset_alert_codes = []
        for idx in subset_indices:
            item = extracted_sections[idx]
            alert_code = detect_alert_codes(item['section'])
            dataset_alert_codes.append(alert_code)
            queries = generate_queries_based_on_alert_code(alert_code, args.num_samples_per_alert)
            pos_matches = [get_formatted_text(item) for _ in range(args.num_samples_per_alert)]
            neg_matches = generate_negative_matches(extracted_sections, alert_code, args.num_samples_per_alert)

            # do more sanity check
            assert all([alert_code in pos for pos in pos_matches])
            assert all([alert_code in qz for qz in queries])
            assert all([alert_code not in neg for neg in neg_matches])
            for query, pos_match, neg_match in zip(queries, pos_matches, neg_matches):
                dataset.append({'query': query, 'positive_match': pos_match, 'negative_match': neg_match})

        return dataset, dataset_alert_codes

    random.shuffle(all_alerts_indices)
    num_train_alerts = int(len(all_alerts_indices) * 0.9)
    train_alerts_indices = all_alerts_indices[:num_train_alerts]
    val_alerts_indices = all_alerts_indices[num_train_alerts:]

    train_dataset, train_alert_codes = build_subset_dataset(train_alerts_indices)
    val_dataset, val_alert_codes = build_subset_dataset(val_alerts_indices)

    # check training dataset does not contain any alerts from validation dataset
    train_alerts_set = set(train_alert_codes)
    val_alerts_set = set(val_alert_codes)
    assert not val_alerts_set.issubset(train_alerts_set)

    metadata = {
        'num_train_samples': len(train_dataset),
        'num_validation_samples': len(val_dataset),
        'train_alert_codes': list(train_alerts_set),
        'validation_alert_codes': list(val_alerts_set),
        'alert_codes': list(train_alerts_set | val_alerts_set),  # later add as custom tokens to the pre-trained tokenizer
    }

    print(f'Saving datasets to {args.save_to} ...')
    pickle.dump(train_dataset, open(train_output_file, 'wb'))
    pickle.dump(val_dataset, open(val_output_file, 'wb'))
    with open(meta_output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


if __name__ == '__main__':

    seed = 1
    torch.manual_seed(seed)
    random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf_dir', help='Source pdf files for Tesla user manual', type=str, default='./data/docs')
    parser.add_argument('--max_words', help='Maximum number of words per chunk', type=int, default=250)
    parser.add_argument('--start_page', help='Starts from page in the PDF', type=int, default=8)
    parser.add_argument('--end_page', help='Ends at page in the PDF', type=int, default=None)
    parser.add_argument('--num_samples_per_alert', help='Generate number of random samples per alert code', type=int, default=20)
    parser.add_argument('--save_to', help='Directory to save the dataset to .pk file', type=str, default='./datasets/embed_alert_codes')

    main()
