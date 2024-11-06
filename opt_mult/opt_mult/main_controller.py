from typing import List

import pandas as pd
from opt_mult.compare.controller import CompareController
from opt_mult.detector.controller import DetectorController
from opt_mult.preprocessing.controller import PreprocessingController
from opt_mult.pdfparse.controller import PDFParseController
from opt_mult.interface import PDFParseOutput, PreprocessOutput, CompareDFOutput

from PIL import Image

import streamlit as st
from pathlib import Path

import fitz
from fitz import Document

from opt_mult.config import default_detector_model

compare_controller = CompareController()
detector_controller = DetectorController(default_detector_model)
preprocessing_controller = PreprocessingController()
pdfparse_controller = PDFParseController()

def full_pipeline(pdf_file: Document, ms_pdf_file: Document):
    st.title("Answer sheet process")
    pdfparse_output = pdfparse_controller.parse_pdf(pdf_file)
    print(len(pdfparse_output.pdf_images))
    visualise_pdfparse_output(pdfparse_output, "Answer sheet")
    preprocessing_output = preprocessing_controller.preprocess_images(pdfparse_output)
    visualise_lines_and_bboxes_output(preprocessing_output)
    detector_output_ans = detector_controller.infer_on_images(preprocessing_output)
    visualise_detector_output(detector_output_ans)

    st.title("Markscheme progress")
    pdfparse_output_ms = pdfparse_controller.parse_pdf(ms_pdf_file)
    visualise_pdfparse_output(pdfparse_output, "Markscheme sheet")
    preprocessing_output_ms = preprocessing_controller.preprocess_images(pdfparse_output_ms)
    visualise_lines_and_bboxes_output(preprocessing_output_ms)
    detector_output_ms = detector_controller.infer_on_images(preprocessing_output_ms)
    visualise_detector_output(detector_output_ms)

    st.title("Comparison and final mark")
    compare_df_output = compare_controller.compare_to_ms(detector_output_ms, detector_output_ans)
    print_results(compare_df_output)

# Title of the web app
st.title('File Upload Streamlit App')

# File uploader widget
markscheme_file = st.file_uploader("Choose a markscheme file to upload", type=["pdf"])

answer_sheet_file = st.file_uploader("Choose a answer file to upload", type=["pdf"])

def convert_from_bytes(uploaded_file) -> Document:
    if uploaded_file.type == "application/pdf":
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return pdf_document

def visualise_pdfparse_output(pdfparse_output:PDFParseOutput, title: str):
    with st.expander(title):
        for pdf_image in pdfparse_output.pdf_images:
            st.image(Image.fromarray(pdf_image))

def visualise_lines_and_bboxes_output(preprocessing_output_list: List[PreprocessOutput]):
    with st.expander("Gridlines detection"):
        for preprocess_output in preprocessing_output_list:
            st.image(preprocess_output.image_with_lines)
    with st.expander("Cropped areas"):
        for preprocess_output in preprocessing_output_list:
            st.image(preprocess_output.image_with_boxes)

def visualise_detector_output(detector_output: List[pd.DataFrame]):
    with st.expander("List of DataFrames"):
        for i, df in enumerate(detector_output, 1):
            st.subheader(f"DataFrame {i}")
            st.dataframe(df)

def print_results(compare_controller_output: CompareDFOutput):
    with st.expander("Results"):
        st.text(f"Correct answers: {compare_controller_output.correct}")
        st.text(f"Incorrect answers: {compare_controller_output.incorrect}")
        st.text(f"Unanswered questions: {compare_controller_output.unanswered}")
        st.text(f"Final mark: {compare_controller_output.calculate_final_mark():.2f}%")

# Check if the user has uploaded a file
if markscheme_file is not None and answer_sheet_file is not None:
    # Display the filename after upload
    st.write(f'answersheet file: {answer_sheet_file.name}')
    st.write(f'markscheme file: {markscheme_file.name}')

    full_pipeline(convert_from_bytes(answer_sheet_file), convert_from_bytes(markscheme_file))

else:
    st.write('Please upload a file to continue.')