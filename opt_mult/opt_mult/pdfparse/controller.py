from pathlib import Path
from fitz import Document
import numpy as np
from PIL import Image

from ..interface import PDFParseOutput

class PDFParseController:

    def __init__(self):
        pass

    def parse_pdf(self, pdf_document: Document) -> PDFParseOutput:
        page_arrays = []
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_array = np.array(img)
            page_arrays.append(img_array)
        
        pdf_document.close()
        output = PDFParseOutput(pdf_images=page_arrays)
        return output