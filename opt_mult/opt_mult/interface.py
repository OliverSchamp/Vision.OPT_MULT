from pydantic import BaseModel
from typing import List, Dict
from dataclasses import dataclass
import numpy as np
from olv_primitives import Bbox
import pandas as pd

class MarkingResult(BaseModel):
    correct: int
    incorrect: int
    unanswered: int
    correct_idxs: List[int]
    incorrect_idxs: List[int]
    unanswered_idxs: List[int]

@dataclass
class ParsedSingleAnswerSheet:
    pdf_images: List[np.ndarray]

@dataclass
class PDFParseOutput:
    answer_sheets: List[ParsedSingleAnswerSheet]

@dataclass
class PreprocessOutput:
    image: np.ndarray
    image_with_lines: np.ndarray
    image_with_boxes: np.ndarray
    crop_bboxes: List[Bbox]
    class_midpoints: Dict[int, List[float]]
    num_vertical_lines: int

@dataclass
class DetectorOutput:
    dataframes: List[pd.DataFrame]
    images: List[np.ndarray]
    crop_bboxes_list: List[List[Bbox]]
    detections_per_bboxidx: List[Dict[int, List[Bbox]]]
    centres_for_answers_list: List[Dict[str, float]]

class CompareDFOutput(BaseModel):
    correct: int
    incorrect: int
    unanswered: int

    def calculate_final_mark(self) -> float:
        return (self.correct-self.incorrect)/(self.correct+self.incorrect+self.unanswered) * 100