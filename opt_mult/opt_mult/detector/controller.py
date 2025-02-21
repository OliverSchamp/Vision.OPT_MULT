from ..interface import PreprocessOutput, DetectorOutput
from ..config import default_detector_model, conf_thresh
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

from olv_primitives import Bbox
from olv_object_detection import load_object_detector

class DetectorController:

    def __init__(self):
        self.detector = load_object_detector(default_detector_model)
    
    def infer_on_images(self, preprocess_output_list: List[PreprocessOutput]) -> DetectorOutput:
        detector_output = DetectorOutput(dataframes=[], detections_per_bboxidx=[], crop_bboxes_list=[], images=[], centres_for_answers_list=[])
        
        for preprocess_output in preprocess_output_list:
            detector_output.images.append(preprocess_output.image)
            detector_output.crop_bboxes_list.append(preprocess_output.crop_bboxes)
            df, dets_per_bb, centres_for_answers = self.infer_on_image(preprocess_output)
            detector_output.dataframes.append(df)
            detector_output.detections_per_bboxidx.append(dets_per_bb)
            detector_output.centres_for_answers_list.append(centres_for_answers)

        return detector_output
    
    def infer_on_image(self, preprocess_output: PreprocessOutput) -> Tuple[pd.DataFrame, Dict[int, List[Bbox]]]:
        df_cols = [chr(ord('A') + i) for i in range(preprocess_output.num_vertical_lines - 2)]
        df = pd.DataFrame(columns=df_cols)
        
        detections_per_bbox_crop: Dict[int, List[Bbox]] = {}
        label_box = True
        for i, bbox in enumerate(preprocess_output.crop_bboxes):
            bbox_cropped_image = preprocess_output.image[int(bbox.ytl):int(bbox.ybr), int(bbox.xtl):int(bbox.xbr)]
            if label_box is True:
                label_box = False
                continue

            centres_for_answers = {mid:cls for mid, cls in zip(preprocess_output.class_midpoints[i], df_cols)}
            new_row = {col_value: " " for col_value in df_cols}
            detections = self.detector.infer_parsed(bbox_cropped_image)
            qnum_detections = [detection for detection in detections if detection.conf >= 0.7 and detection.label == "question_number"]
            answer_detections = [detection for detection in detections if detection.conf >= conf_thresh and detection.label == "answer"]

            detections_per_bbox_crop[i] = qnum_detections + answer_detections
            has_question_number = False
            for detection in detections_per_bbox_crop[i]:
                if detection.label == "question_number":
                    has_question_number = True
            
            if not has_question_number:
                continue

            for detection in detections_per_bbox_crop[i]:
                if detection.conf < conf_thresh or detection.label != "answer":
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

        return df, detections_per_bbox_crop, centres_for_answers
