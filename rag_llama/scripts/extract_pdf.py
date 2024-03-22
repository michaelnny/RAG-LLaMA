# Copyright (c) 2024 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.

"""Code to extract text from Tesla (PDF) manual on a section-by-section basis"""

import re
from typing import List, Tuple, Mapping, Text, Any
import fitz  # PyMuPDF


def estimate_word_count(text: str) -> int:
    """Estimates the number of words in a string by splitting on whitespace."""
    words = text.split()
    return len(words)


def is_within_bounds(bbox: Tuple, content_bounds: Tuple) -> bool:
    """Check whether a bounding box is inside another."""
    bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y = bbox
    data_min_x, data_min_y, data_max_x, data_max_y = content_bounds

    return (data_min_x <= bbox_min_x <= data_max_x) and (data_min_y <= bbox_min_y <= data_max_y) and (data_min_x <= bbox_max_x <= data_max_x) and (data_min_y <= bbox_max_y <= data_max_y)


# Help functions to check patterns in text
def is_starts_with_uppercase_keyword(text: str) -> bool:
    pattern = r'^[A-Z]+(:)'
    return bool(re.match(pattern, text))


def is_starts_with_bullet_point(text: str) -> bool:
    pattern = r'^(?:[0-9]+\.\s|•|◦\s)'
    return bool(re.match(pattern, text))


def is_starts_with_whitespace(text: str) -> bool:
    pattern = r'^\s'
    return bool(re.search(pattern, text))


def is_ends_with_whitespace(text: str) -> bool:
    pattern = r'\s$'
    return bool(re.search(pattern, text))


def is_starts_with_punctuation(text: str) -> bool:
    pattern = r'^[.,;:!?()]'
    return bool(re.match(pattern, text))


def is_ends_with_special_characters(text: str) -> bool:
    if not text:
        return False
    elif text.endswith('/'):  # ends with URL, got be a better way to do this
        return True
    elif text.endswith('-'):
        return True
    else:
        return False


def is_header_title(data: Mapping[Text, Any]) -> bool:
    if 'size' in data and float(data['size']) == 18.0:  # find section from header title
        return True
    return False


def is_section_title(data: Mapping[Text, Any]) -> bool:
    if 'size' in data and float(data['size']) == 14.0:
        return True
    return False


def merge_section_titles(objects: List[Mapping[Text, Any]], window_size: int = 3) -> List[Mapping[Text, Any]]:
    assert window_size >= 1
    results = []
    i = 0
    while i < len(objects):
        current_obj = objects[i]
        # Check if the text of the current object matches is section title
        if is_section_title(current_obj):
            # Look ahead within the defined window to see if subsequent objects can be merged to the current section title
            start_idx = i
            for j in range(1, window_size):
                if j >= len(objects):
                    break
                next_obj = objects[i + j]

                curr_text = current_obj['text']
                next_text = next_obj['text']

                if next_text != curr_text and is_section_title(next_obj):
                    # handle white space
                    if is_ends_with_whitespace(curr_text) or is_starts_with_whitespace(next_text) or is_starts_with_punctuation(next_text):
                        curr_text += next_text
                    else:
                        curr_text += ' ' + next_text
                    # Merge the text to current object
                    current_obj['text'] = curr_text
                    i = start_idx + j

            results.append(current_obj)
        else:
            # If the current object is not a section title, add it as it is
            results.append(current_obj)

        i += 1

    return results


def extract_text_by_section_from_tesla_manual(pdf_path: str, start_page: int = 1, end_page: int = None) -> Mapping[Text, Any]:
    """
    Extract text from Tesla (PDF) manual on a section-by-section basis.
    """
    assert start_page >= 1, start_page
    if end_page is not None:
        assert end_page >= start_page, end_page

    # bounding box for page content, needs to be turned for different document
    content_bounds = (34, 60, 600, 774)

    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    total_pages = len(pdf_document)

    # try to extract document title and other metadata from first page
    doc_title, doc_title_texts = extra_title_from_first_page(pdf_document.load_page(0))

    if start_page >= 1:
        start_page -= 1  # index starts from 0

    if end_page is None or end_page >= total_pages:
        end_page = total_pages
    else:
        end_page += 1  # include the end page

    all_spans = []
    page_title_dict = {}

    for page_number in range(start_page, end_page):
        page = pdf_document.load_page(page_number)

        # Extract the text and its bounding boxes
        page_dict = page.get_text('dict')

        for block in page_dict['blocks']:
            if 'lines' not in block:
                continue
            if 'spans' not in block['lines'][0]:
                continue

            for line in block['lines']:
                for span in line['spans']:
                    span['page'] = page_number + 1
                    curr_bbox = span['bbox']
                    curr_text = span['text']
                    curr_page = span['page']

                    # find page title, sometimes we don't have a section title in the content
                    if is_header_title(span) and curr_page not in page_title_dict:
                        page_title_dict[curr_page] = curr_text
                    elif not is_within_bounds(curr_bbox, content_bounds):
                        continue
                    else:
                        all_spans.append(span)

    # Close the PDF document
    pdf_document.close()

    # merge section title, some section title may be separated into multiple objects
    all_spans = merge_section_titles(all_spans)

    # construct the sections
    extracted_sections = []
    curr_sec_label = None
    curr_sec_content = None
    curr_sec_start_page = None
    last_span = None

    for span in all_spans:
        curr_text = span['text']
        curr_page = span['page']

        if is_header_title(span) or is_section_title(span):
            if curr_sec_content is not None:
                # save previous section
                extracted_sections.append(
                    {
                        'document': doc_title,
                        'subject': page_title_dict[curr_page],
                        'section': curr_sec_label if curr_sec_label is not None else 'Overview',
                        'content': curr_sec_content,
                        'start_page': curr_sec_start_page,
                        'end_page': curr_page,
                    }
                )
                curr_sec_content = None

            curr_sec_label = curr_text
        elif curr_sec_content is None:
            curr_sec_content = curr_text
            curr_sec_start_page = curr_page
        else:
            concat_str = ' '
            last_bbox = last_span['bbox'] if last_span is not None else None
            curr_bbox = span['bbox']

            if is_starts_with_uppercase_keyword(curr_text):  # handle 'WARNING', 'CAUTION' and other keywords
                concat_str = '\n'
            elif is_starts_with_bullet_point(curr_text):  # handle bullet points
                concat_str = '\n'
            elif last_bbox and (curr_bbox[1] - last_bbox[3]) > 10:  # there's a large gap on y axis between last element
                concat_str = '\n'
            elif is_ends_with_whitespace(curr_sec_content) or is_ends_with_special_characters(curr_sec_content) or is_starts_with_whitespace(curr_text) or is_starts_with_punctuation(curr_text):
                concat_str = ''

            curr_sec_content += concat_str + curr_text

        last_span = span

    return extracted_sections


def extra_title_from_first_page(page: fitz.Document) -> Tuple[str, List[str]]:
    texts = []
    page_dict = page.get_text('dict')

    for block in page_dict['blocks']:
        if 'lines' not in block:
            continue
        if 'spans' not in block['lines'][0]:
            continue

        for line in block['lines']:
            for span in line['spans']:
                texts.append(span['text'])

    combined_text = ' '.join(texts)
    # make title text nicer
    combined_text = transform_doc_title_text(combined_text)
    return combined_text, texts


def transform_doc_title_text(input_text):
    """Turn `MODEL S, 2021 +, OWNER'S MANUAL` into `MODEL S (2021 +) OWNER'S MANUAL`"""
    # Define a regular expression pattern to match the desired text pattern
    pattern_en = r"MODEL\s+(\w+) ?\s*(\d{4}\s*\+?)? ?\s*OWNER\'S\s+MANUAL"
    pattern_zh = r'MODEL\s+(\w+) ?\s*(\d{4}\s*\+?)? ?\s*车主手册'

    # Define the replacement pattern
    replacement_en = r"MODEL \1 (\2) OWNER'S MANUAL"
    replacement_zh = r'MODEL \1 (\2) 车主手册'

    # Use re.sub to perform the replacement
    output_text = re.sub(pattern_en, replacement_en, input_text)
    output_text = re.sub(pattern_zh, replacement_zh, output_text)

    return output_text


if __name__ == '__main__':

    # Sample documents (replace with your PDF filename)
    pdf_filename = './data/Tesla_ModelS_Owners_Manual.pdf'
    extracted_sections = extract_text_by_section_from_tesla_manual(pdf_filename, 8)

    for section in extracted_sections:
        print(f"Document: {section['document']}")
        print(f"Subject: {section['subject']}")
        print(f"Section: {section['section']}")
        print(f"Content: {section['content']}")
        print(f"Pages: [{section['start_page']} - {section['end_page']}]")
        print('\n')
