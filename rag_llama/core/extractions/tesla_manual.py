# Copyright (c) 2024 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.

"""Code to extract text from Tesla (PDF) manual"""
import os
import re
import copy
from typing import List, Tuple, Mapping, Text, Any
import fitz
import numpy as np


from rag_llama.core.chunks import split_text_into_chunks, estimate_word_count

Section = Mapping[Text, Any]


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
    pattern = r'^[.,;:!?()®]'
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


def is_valid_node(node: Mapping[Text, Any]) -> bool:
    if 'size' not in node or 'flags' not in node or 'text' not in node:
        return False
    return True


def is_page_title(node: Mapping[Text, Any]) -> bool:
    if not is_valid_node(node):
        return False
    elif float(node['size']) == 18.0:  # page title
        return True
    return False


def is_section_title(node: Mapping[Text, Any]) -> bool:
    if not is_valid_node(node):
        return False
    elif float(node['size']) == 12.0 or float(node['size']) == 14.0:
        return True
    elif float(node['size']) >= 11.6 and node['flags'] == 5 and node['text'] == '®':  # special copyright symbol in title
        return True
    return False


def merge_section_titles(objects: List[Section], window_size: int = 5) -> List[Section]:
    """Section title could be located in separate spans in case the title text is very long.
    We will try to do a look-ahead search and merge the titles.
    """
    assert window_size >= 1
    results = []
    i = 0
    while i < len(objects):
        current_obj = copy.deepcopy(objects[i])
        # Check if the text of the current object matches is section title
        if is_section_title(current_obj):
            # Look ahead within the defined window to see if subsequent objects can be merged to the current section title
            start_idx = copy.copy(i)
            for j in range(1, window_size):
                if j >= len(objects):
                    break

                next_idx = start_idx + j
                next_obj = objects[next_idx]

                if not is_section_title(next_obj):
                    break
                elif next_obj['page'] != current_obj['page']:
                    break

                next_text = next_obj['text']
                curr_text = current_obj['text']

                # handle white space
                if is_ends_with_whitespace(curr_text) or is_starts_with_whitespace(next_text) or is_starts_with_punctuation(next_text):
                    curr_text += next_text
                else:
                    curr_text += ' ' + next_text
                # Merge the text to current object
                current_obj['text'] = curr_text
                i = next_idx

            results.append(current_obj)
        else:
            # If the current object is not a section title, add it as it is
            results.append(current_obj)

        i += 1

    # for item in results:
    #     if 'APP_' in item['text'] or 'BMS_' in item['text'] or 'UMC_' in item['text']:
    #         print(item['text'])

    return results


def get_formatted_section_text(data: Section) -> str:
    """Format the document to add more metadata"""
    if data is None:
        return ''

    # page information may not accurate, since text have been splitted into smaller chunks
    formatted_text = f"Document: {data['document_title']}\nCar model: {data['car_model']}\nSubject: {data['subject']} - {data['section']}\nContent: {data['content']}\nPage: {data['page']}"
    return formatted_text


def split_sections_into_chunks(sections: List[Section], max_words: int) -> List[Section]:
    """Split the section content into smaller chunks."""

    assert max_words >= 10, max_words

    def get_section_metadata(item) -> Section:
        """Create a copy of the section metadata by exclude certain keys"""
        result = {}
        for k, v in item.items():
            if any([k == key for key in ('content', 'formatted_text', 'num_words')]):
                continue
            result[k] = v
        return result

    results = []
    for section in sections:
        chunks = split_text_into_chunks(section['content'], max_words)
        section_meta = get_section_metadata(section)
        if len(chunks) > 1:
            for i, chunk_text in enumerate(chunks):
                item = copy.deepcopy(section_meta)
                item['section'] += f' - part {i+1}'
                item['content'] = chunk_text
                results.append(item)
        else:
            results.append(section)

    return results


def format_sections_text(sections: List[Section]) -> List[Section]:
    """Add a 'formatted_text' proper to each section, where we mixing content along with some metadata together."""
    for section in sections:
        formatted_text = get_formatted_section_text(section)
        section['formatted_text'] = formatted_text
        section['num_words'] = estimate_word_count(section['content'])

    words = [item['num_words'] for item in sections]
    print(f'Mean number of words: {np.mean(words)}')
    print(f'Max number of words: {np.max(words)}')
    print(f'Min number of words: {np.min(words)}')


def extra_metadata_from_first_page(page: fitz.Document) -> Mapping[Text, Text]:
    """Extract document metadata from first page"""

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

    assert len(texts) == 5

    title = ' '.join(texts[:3])
    model = ' '.join(texts[:2])
    version = texts[3].split(':')[1]
    region = texts[-1]

    extracted = {
        'document_title': title.strip(),
        'car_model': model.strip(),
        'software_version': version.strip(),
        'region': region.strip(),
    }

    return extracted


def extract_spans_and_page_titles_from_pdf(pdf_document: Any, start_page: int = 1, end_page: int = None) -> Tuple[List[Any], Mapping[Text, Text]]:
    """Extract all spans within the main content, and the title for each page."""
    assert pdf_document, pdf_document

    # bounding box for page content, needs to be turned for different document
    content_bounds = (34, 60, 600, 774)

    total_pages = len(pdf_document)

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
                    if is_page_title(span) and curr_page not in page_title_dict:
                        page_title_dict[curr_page] = curr_text
                    elif not is_within_bounds(curr_bbox, content_bounds):
                        continue
                    else:
                        all_spans.append(span)

    return all_spans, page_title_dict


def extract_sections_from_spans(spans: List[Any], page_title_dict: Any, metadata: Any) -> List[Section]:
    """Concat all spans belong to the same section together"""
    # construct the sections
    extracted_sections = []
    curr_sec_label = None
    curr_sec_content = None
    curr_sec_start_page = None
    last_span = None

    for span in spans:
        curr_text = span['text']
        curr_page = span['page']

        if is_page_title(span) or is_section_title(span):
            if curr_sec_content is not None:
                # save previous section
                extracted_sections.append(
                    {
                        **metadata,
                        'subject': page_title_dict[curr_page],
                        'section': curr_sec_label if curr_sec_label is not None else 'Overview',
                        'content': curr_sec_content,
                        'page': f'{curr_sec_start_page} - {curr_page}' if curr_page > curr_sec_start_page else f'{curr_page}',
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


def extract_tesla_manual_sections(pdf_path: str, max_words: int, start_page: int = 1, end_page: int = None) -> Section:
    """
    Extract text from Tesla (PDF) manual on a section-by-section basis, with the option to split section content to maximum number of words.
    """
    if pdf_path is None or not os.path.exists(pdf_path):
        return
    assert max_words >= 50, max_words
    assert start_page >= 1, start_page
    if end_page is not None:
        assert end_page >= start_page, end_page

    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    # try to extract document title and other metadata from first page
    doc_metadata = extra_metadata_from_first_page(pdf_document.load_page(0))

    spans, page_titles = extract_spans_and_page_titles_from_pdf(pdf_document, start_page, end_page)

    # Close the PDF document
    pdf_document.close()

    # merge section title, some section title may be separated into multiple objects
    spans = merge_section_titles(spans)

    # extract sections from spans
    extracted_sections = extract_sections_from_spans(spans, page_titles, doc_metadata)

    # split section into smaller chunks
    section_chunks = split_sections_into_chunks(extracted_sections, max_words)

    # compute formatted text string for each extracted section
    format_sections_text(section_chunks)

    return section_chunks


if __name__ == '__main__':

    # Sample documents (replace with your PDF filename)
    pdf_filename = './data/docs/Tesla_ModelS_Owners_Manual_2021.pdf'
    extracted_sections = extract_tesla_manual_sections(pdf_filename, 380, 50, 999)

    # for section in extracted_sections:
    #     print(f"Document title: {section['document_title']}")
    #     print(f"Car model: {section['car_model']}")
    #     print(f"Software version: {section['software_version']}")
    #     print(f"Region: {section['region']}")
    #     print(f"Subject: {section['subject']}")
    #     print(f"Section: {section['section']}")
    #     print(f"Content: {section['content']}")
    #     print(f"Pages: [{section['start_page']} - {section['end_page']}]")
    #     print('\n')
