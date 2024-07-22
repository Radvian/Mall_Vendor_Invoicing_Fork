import time

import fitz
import pandas as pd
import streamlit as st

# from dotenv import load_dotenv
from PIL import Image, ImageOps

from utils.extraction import main_extract
from utils.ocr import detect_text

# load_dotenv()


def main():
    st.title("Vendor Invoice Extraction")
    uploaded_file = st.file_uploader("Upload a .jpg or .pdf file", type=["jpg", "pdf"])

    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            file_bytes = [pdf_document.load_page(i).get_pixmap().tobytes() for i in range(len(pdf_document))]
        else:
            image = Image.open(uploaded_file)
            image = ImageOps.exif_transpose(image)
            file_bytes = [uploaded_file.getvalue()]

        # # Save the image to a temporary file
        # with open("temp.jpg", "wb") as f:
        #     f.write(uploaded_file.getbuffer())

        # Perform OCR on the image with loading spinner and timer
        with st.spinner("Performing OCR..."):
            start_time = time.time()
            ocr_results = [detect_text(file_byte) for file_byte in file_bytes]
            end_time = time.time()
            st.success(f"OCR completed in {end_time - start_time:.2f} seconds")

        # Extract structured information from OCR result with loading spinner and timer
        with st.spinner("Extracting information..."):
            start_time = time.time()
            invoice_outputs = [
                main_extract(ocr_result, selected_model_name="gpt-4o-mini") for ocr_result in ocr_results
            ]
            cost = sum([output["price"] for output in invoice_outputs])
            end_time = time.time()
            st.success(f"Information extraction completed in {end_time - start_time:.2f} seconds and cost ${cost}")

        # Display extracted invoice as table
        df = []
        for output in invoice_outputs:
            if "items" in output["response"]:
                data = pd.DataFrame([output["response"]]).explode("items").to_dict(orient="records")
            else:
                data = [output["response"]]
            df.extend(data)
        df = pd.json_normalize(df, sep="_")
        # df = pd.json_normalize([output["response"] for output in invoice_outputs])
        # df.columns = pd.MultiIndex.from_tuples([tuple(col.split(".")) for col in df.columns])
        # col2.header("Extracted Information")
        st.table(df)
        # insert_to_excel('vendor_invoice.xlsx', df)

        # Display OCR Result
        with st.expander("OCR Result"):
            for result in ocr_results:
                st.write(result)
                st.write("---")  # Add a separator for each result

        # Display the uploaded images side by side
        cols = st.columns(len(file_bytes))
        for col, file_byte in zip(cols, file_bytes):
            col.image(file_byte)


if __name__ == "__main__":
    main()
