from opt_mult.compare.controller import CompareController
from opt_mult.detector.controller import DetectorController
from opt_mult.preprocessing.controller import PreprocessingController
from opt_mult.pdfparse.controller import PDFParseController

import streamlit as st
from pathlib import Path

from .config import default_detector_model

compare_controller = CompareController()
detector_controller = DetectorController(default_detector_model)
preprocessing_controller = PreprocessingController()
pdfparse_controller = PDFParseController()

def full_pipeline(pdf_file_path: Path, ms_pdf_file_path: Path):
    pdfparse_output = pdfparse_controller.parse_pdf(pdf_file_path)
    preprocessing_output = preprocessing_controller.preprocess_images(pdfparse_output)
    detector_output_ans = detector_controller.infer_on_images(preprocessing_output)

    pdfparse_output_ms = pdfparse_controller.parse_pdf(ms_pdf_file_path)
    preprocessing_output_ms = preprocessing_controller.preprocess_images(pdfparse_output_ms)
    detector_output_ms = detector_controller.infer_on_images(preprocessing_output_ms)

    compare_controller.compare_to_ms(detector_output_ms, detector_output_ans)

# Title of the web app
st.title('File Upload Streamlit App')

# File uploader widget
markscheme_file = st.file_uploader("Choose a markscheme file to upload", type=["pdf"])

answer_sheet_file = st.file_uploader("Choose a answer file to upload", type=["pdf"])

# Check if the user has uploaded a file
if markscheme_file is not None and answer_sheet_file is not None:
    # Display the filename after upload
    st.write(f'answersheet file: {answer_sheet_file.name}')
    st.write(f'markscheme file: {markscheme_file.name}')

    full_pipeline(answer_sheet_file, markscheme_file)

else:
    st.write('Please upload a file to continue.')