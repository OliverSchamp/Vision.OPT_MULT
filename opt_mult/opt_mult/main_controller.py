
from opt_mult.compare.controller import CompareController
from opt_mult.detector.controller import DetectorController
from opt_mult.preprocessing.controller import PreprocessingController
from opt_mult.pdfparse.controller import PDFParseController
from opt_mult.interface import CompareDFOutput
from opt_mult.visualise import visualise_detector_output, visualise_lines_and_bboxes_output, visualise_pdfparse_output

import streamlit as st

import fitz
from fitz import Document

# use session state to initialise the classes only once

def initialise_heavy_classes():
    if "preprocessing_controller" not in st.session_state:
        st.session_state["preprocessing_controller"] = PreprocessingController(150) # threshold at 150 detections to make a line
    if "compare_controller" not in st.session_state:
        st.session_state["compare_controller"] = CompareController()
    if "detector_controller" not in st.session_state:
        st.session_state["detector_controller"] = DetectorController()
    if "pdfparse_controller" not in st.session_state:
        st.session_state["pdfparse_controller"] = PDFParseController()

def full_pipeline(pdf_file: Document, ms_pdf_file: Document):
    pdfparse_controller = st.session_state["pdfparse_controller"]
    preprocessing_controller = st.session_state["preprocessing_controller"]
    detector_controller = st.session_state["detector_controller"]
    compare_controller = st.session_state["compare_controller"]


    st.title("Markscheme progress")
    pdfparse_output_ms = pdfparse_controller.parse_pdf(ms_pdf_file)
    pdfparse_output_ms = pdfparse_output_ms.answer_sheets[0]
    visualise_pdfparse_output(pdfparse_output_ms, "Markscheme sheet")
    preprocessing_output_ms = preprocessing_controller.preprocess_images(pdfparse_output_ms)
    visualise_lines_and_bboxes_output(preprocessing_output_ms)
    detector_output_ms = detector_controller.infer_on_images(preprocessing_output_ms)
    visualise_detector_output(detector_output_ms)

    count = 0
    pdfparse_output = pdfparse_controller.parse_pdf(pdf_file)
    for single_paper_pdf in pdfparse_output.answer_sheets:
        count += 1
        try:
            st.title(f"Answer sheet process {count}")
            visualise_pdfparse_output(single_paper_pdf, "Answer sheet")
            preprocessing_output = preprocessing_controller.preprocess_images(single_paper_pdf)
            visualise_lines_and_bboxes_output(preprocessing_output)
            detector_output_ans = detector_controller.infer_on_images(preprocessing_output)
            visualise_detector_output(detector_output_ans, detector_output_ms)

            st.title("Comparison and final mark")
            compare_df_output = compare_controller.compare_to_ms(detector_output_ms.dataframes, detector_output_ans.dataframes)
            print_results(compare_df_output)
        except KeyError as e:
            st.text(f"KeyError: {e}")
        except Exception as e:
            error_str = str(e)
            st.text(f"ERROR: {error_str}")

# Title of the web app
st.title('Automatic Multiple Choice Marker')

st.write("Initialising... this can take a few minutes")
initialise_heavy_classes()
st.write("Init finished!")

# File uploader widget
markscheme_file = st.file_uploader("Choose a markscheme file to upload", type=["pdf"])

answer_sheet_file = st.file_uploader("Choose a answer file to upload", type=["pdf"])

def convert_from_bytes(uploaded_file) -> Document:
    if uploaded_file.type == "application/pdf":
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return pdf_document

def print_results(compare_controller_output: CompareDFOutput):
    with st.expander("Results"):
        st.text(f"Correct answers: {compare_controller_output.correct}")
        st.text(f"Incorrect answers: {compare_controller_output.incorrect}")
        st.text(f"Unanswered questions: {compare_controller_output.unanswered}")
        st.text(f"Final mark absolute: {compare_controller_output.correct - compare_controller_output.incorrect}")
        st.text(f"Final mark %: {compare_controller_output.calculate_final_mark():.2f}%")

# Check if the user has uploaded a file
if markscheme_file is not None and answer_sheet_file is not None:
    # Display the filename after upload
    st.write(f'answersheet file: {answer_sheet_file.name}')
    st.write(f'markscheme file: {markscheme_file.name}')

    full_pipeline(convert_from_bytes(answer_sheet_file), convert_from_bytes(markscheme_file))

else:
    st.write('Please upload a file to continue.')