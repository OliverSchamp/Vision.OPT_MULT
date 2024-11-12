from pathlib import Path
from fitz import Document
import numpy as np
from PIL import Image

from ..interface import PDFParseOutput, ParsedSingleAnswerSheet

class PDFParseController:

    def __init__(self):
        pass

    def parse_pdf(self, pdf_document: Document) -> PDFParseOutput:
        page_arrays = []
        for page_num in range(pdf_document.page_count):
            if page_num % 2 == 0:
                single_paper = ParsedSingleAnswerSheet(pdf_images=[])
            page = pdf_document[page_num]
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_array = np.array(img)
            single_paper.pdf_images.append(img_array)

            if page_num % 2 == 1:
                page_arrays.append(single_paper)
        
        pdf_document.close()
        output = PDFParseOutput(answer_sheets=page_arrays)
        return output