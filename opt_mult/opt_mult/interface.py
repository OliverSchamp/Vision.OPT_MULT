from pydantic import BaseModel
from typing import List

class MarkingResult(BaseModel):
    correct: int
    incorrect: int
    unanswered: int
    correct_idxs: List[int]
    incorrect_idxs: List[int]
    unanswered_idxs: List[int]