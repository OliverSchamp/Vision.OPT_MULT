from pydantic import BaseModel
from typing import List, Dict
from dataclasses import dataclass
import numpy as np
from olv_primitives import Bbox

class MarkingResult(BaseModel):
    correct: int
    incorrect: int
    unanswered: int
    correct_idxs: List[int]
    incorrect_idxs: List[int]
    unanswered_idxs: List[int]

@dataclass
class PDFParseOutput:
    pdf_images: List[np.ndarray]

@dataclass
class PreprocessOutput:
    image: np.ndarray
    image_with_lines: np.ndarray
    image_with_boxes: np.ndarray
    crop_bboxes: List[Bbox]
    class_midpoints: Dict[int, List[float]]

class CompareDFOutput(BaseModel):
    correct: int
    incorrect: int
    unanswered: int

    def calculate_final_mark(self) -> float:
        return (self.correct-self.incorrect)/(self.correct+self.incorrect+self.unanswered) * 100