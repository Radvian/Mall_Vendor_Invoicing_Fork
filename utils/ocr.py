import os
from typing import List, Sequence

import fitz
import pandas as pd
from google.cloud import documentai, vision


def vision_ocr(content: bytes) -> vision.AnnotateImageResponse:
    image = vision.Image(content=content)
    vision_client = vision.ImageAnnotatorClient(client_options={"api_key": os.environ["GOOGLE_VISION_API_KEY"]})
    response = vision_client.text_detection(image=image)
    return response


def get_bytes(file_path: str) -> List[bytes]:
    """
    Get bytes content from file path
    """
    with open(file_path, "rb") as uploaded_file:
        if file_path.endswith(".pdf"):
            pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            file_bytes = [pdf_document.load_page(i).get_pixmap().tobytes() for i in range(len(pdf_document))]
        else:
            file_bytes = [uploaded_file.read()]
    return file_bytes


def online_process(content) -> documentai.Document:
    documentai_client = documentai.DocumentProcessorServiceClient()
    raw_document = documentai.RawDocument(content=content, mime_type="image/png")
    request = documentai.ProcessRequest(name=os.getenv("RESOURCE_NAME"), raw_document=raw_document)
    result = documentai_client.process_document(request=request)
    return result.document


def get_table_data(rows: Sequence[documentai.Document.Page.Table.TableRow], text: str) -> List[List[str]]:
    """
    Get Text data from table rows
    """
    all_values: List[List[str]] = []
    for row in rows:
        current_row_values: List[str] = []
        for cell in row.cells:
            current_row_values.append(text_anchor_to_text(cell.layout.text_anchor, text))
        all_values.append(current_row_values)
    return all_values


def text_anchor_to_text(text_anchor: documentai.Document.TextAnchor, text: str) -> str:
    """
    Document AI identifies table data by their offsets in the entirety of the
    document's text. This function converts offsets to a string.
    """
    response = ""
    # If a text segment spans several lines, it will
    # be stored in different text segments.
    for segment in text_anchor.text_segments:
        start_index = int(segment.start_index)
        end_index = int(segment.end_index)
        response += text[start_index:end_index]
    return response.strip().replace("\n", " ")


def extract_table(document: documentai.Document) -> List[pd.DataFrame]:
    header_row_values: List[List[str]] = []
    body_row_values: List[List[str]] = []

    # Input Filename without extension
    # output_file_prefix = splitext(FILE_PATH)[0]

    tables = []
    for page in document.pages:
        for index, table in enumerate(page.tables):
            header_row_values = get_table_data(table.header_rows, document.text)
            body_row_values = get_table_data(table.body_rows, document.text)

            # Create a Pandas Dataframe to print the values in tabular format.
            df = pd.DataFrame(
                data=body_row_values,
                columns=pd.MultiIndex.from_arrays(header_row_values),
            )

            # print(f"Page {page.page_number} - Table {index}")
            # print(df)
            tables.append(df)

            # Save each table as a CSV file
            # output_filename = f"{output_file_prefix}_pg{page.page_number}_tb{index}.csv"
            # df.to_csv(output_filename, index=False)
    return tables


def detect_text(content: bytes) -> str:
    document = online_process(content)
    tables = extract_table(document)
    table_text = "\n\n".join([table.to_markdown() for table in tables if not table.empty])
    return f"{document.text}\nbelow are the tables extracted from the document:\n\n{table_text}"
