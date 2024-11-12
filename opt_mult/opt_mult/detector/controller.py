from ..interface import PreprocessOutput
from ..config import default_detector_model, default_number_character_detector_model
from typing import List
import pandas as pd
import numpy as np

from olv_object_detection import load_object_detector

class DetectorController:

    def __init__(self):
        self.detector = load_object_detector(default_detector_model)
    
    def infer_on_images(self, preprocess_output_list: List[PreprocessOutput]) -> List[pd.DataFrame]:
        dataframe_list = []
        for preprocess_output in preprocess_output_list:
            dataframe_list.append(self.infer_on_image(preprocess_output))
        
        return dataframe_list
    
    def infer_on_image(self, preprocess_output: PreprocessOutput):
        df_cols = [chr(ord('A') + i) for i in range(preprocess_output.num_vertical_lines - 2)]
        df = pd.DataFrame(columns=df_cols)
        
        label_box = True
        for i, bbox in enumerate(preprocess_output.crop_bboxes):
            bbox_cropped_image = preprocess_output.image[int(bbox.ytl):int(bbox.ybr), int(bbox.xtl):int(bbox.xbr)]
            if label_box is True:
                label_box = False
                continue

            centres_for_answers = {mid:cls for mid, cls in zip(preprocess_output.class_midpoints[i], df_cols)}
            new_row = {col_value: " " for col_value in df_cols}
            detections = self.detector.infer_parsed(bbox_cropped_image)
            has_question_number = False
            for detection in detections:
                if detection.label == "question_number" and detection.mid.x < 200:
                    has_question_number = True
            
            if not has_question_number:
                continue

            for detection in detections:
                if detection.conf < 0.63 or detection.label != "answer":
                    continue

                centre_x = detection.mid.x
                diff = np.Inf
                for answer_x in centres_for_answers:
                    diff_ans = abs(centre_x-answer_x)
                    if diff_ans < diff:
                        answer = centres_for_answers[answer_x]
                        diff = diff_ans

                new_row[answer] = "X"

            df.loc[len(df)] = list(new_row.values())

        return df
