# where the comparison between markscheme and 
import pandas as pd
from typing import Any, Dict, List
from ..interface import CompareDFOutput

class CompareController:

    def __init__(self):
        pass

    def compare_to_ms(self, ms_dfs: List[pd.DataFrame], ans_dfs: List[pd.DataFrame]) -> CompareDFOutput:
        result_list = []
        for ms_single, ans_single in zip(ms_dfs, ans_dfs):
            result_list.append(self.compare_answers(ans_single, ms_single))
        
        correct_answers = 0
        unanswered = 0
        wrong_answers = 0
        for result in result_list:
            correct_answers += result["consistent_rows"]
            unanswered += result["unanswered_rows"]
            wrong_answers += result["incorrect_rows"]
        
        return CompareDFOutput(correct=correct_answers, incorrect=wrong_answers, unanswered=unanswered)



    def compare_answers(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict[str, Any]:
        if df1.shape != df2.shape:
            raise ValueError(f"DataFrames must have the same shape {df1.shape} {df2.shape}")

        consistent_indices = []
        unanswered_indices = []
        incorrect_indices = []
        unanswered_count = 0
        incorrect_count = 0
        
        for idx, (row1, row2) in enumerate(zip(df1.values, df2.values)):
            if all(val1 == val2 for val1, val2 in zip(row1, row2)):
                consistent_indices.append(idx)
            else:
                if all(val == " " for val in row1):
                    unanswered_indices.append(idx)
                    unanswered_count += 1
                else:
                    incorrect_indices.append(idx)
                    incorrect_count += 1
        
        return {
            'consistent_rows': len(consistent_indices),
            'consistent_indices': consistent_indices,
            'unanswered_rows': unanswered_count,
            'unanswered_indices': unanswered_indices,
            'incorrect_rows': incorrect_count,
            'incorrect_indices': incorrect_indices
        }