from pathlib import Path
import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from ..interface import PDFParseOutput

pdf_path = Path("/home/oliver/Oliver.Mono/projects/Vision.OPT_MULT/data/full_images/curie.pdf")

class PDFParseController:

    def __init__(self):
        pass

    def parse_pdf(self, pdf_path: Path) -> PDFParseOutput:
        pdf_document = fitz.open(pdf_path)
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