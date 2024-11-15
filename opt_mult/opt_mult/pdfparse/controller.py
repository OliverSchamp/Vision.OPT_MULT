from pathlib import Path
from fitz import Document, Matrix
import numpy as np
from PIL import Image
import cv2
import streamlit as st

from ..interface import PDFParseOutput, ParsedSingleAnswerSheet

class PDFParseController:

    def __init__(self):
        self.test_save_path = Path("/home/oliver/Oliver.Mono/projects/Vision.OPT_MULT/data/testing")

    def binary_thresholding(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
        rgb_thresholded_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)

        return rgb_thresholded_image

    def parse_pdf(self, pdf_document: Document) -> PDFParseOutput:
        page_arrays = []
        for page_num in range(pdf_document.page_count):
            if page_num % 2 == 0:
                single_paper = ParsedSingleAnswerSheet(pdf_images=[])
            page = pdf_document[page_num]

            zoom_x, zoom_y = 6.0, 6.0
            mat = Matrix(zoom_x, zoom_y)
            pix = page.get_pixmap(matrix=mat)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

            save_pth = self.test_save_path / f"{page_num}.jpg"
            cv2.imwrite(str(save_pth), img_array)
            img_array = cv2.resize(img_array, (1760, 2560), interpolation=cv2.INTER_NEAREST)
            save_pth = self.test_save_path / f"{page_num}_resized_internearest.jpg"
            cv2.imwrite(str(save_pth), img_array)
            # img_array = self.binary_thresholding(img_array)
            single_paper.pdf_images.append(img_array)

            if page_num % 2 == 1:
                page_arrays.append(single_paper)
        
        pdf_document.close()
        output = PDFParseOutput(answer_sheets=page_arrays)
        return output