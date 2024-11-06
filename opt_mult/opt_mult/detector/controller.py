from ..interface import PreprocessOutput
from typing import List
import pandas as pd
from pathlib import Path
import numpy as np

from olv_object_detection import load_object_detector

class DetectorController:

    def __init__(self, detector_path: Path):
        self.detector = load_object_detector(detector_path)

        self.cropped_image_save_path = Path("/home/oliver/Oliver.Mono/projects/Vision.OPT_MULT/data/cropped_data/images")

        self.image_path = Path("/home/oliver/Oliver.Mono/projects/Vision.OPT_MULT/data/full_images/curie_2.jpg")
    
    def infer_on_images(self, preprocess_output_list: List[PreprocessOutput]) -> List[pd.DataFrame]:
        dataframe_list = []
        for preprocess_output in preprocess_output_list:
            dataframe_list.append(self.infer_on_image(preprocess_output))
        
        return dataframe_list
    
    def infer_on_image(self, preprocess_output: PreprocessOutput):
        df_cols = ['A', 'B', 'C', 'D']
        df = pd.DataFrame(columns=df_cols)
        
        label_box = True
        image_image_folder = self.cropped_image_save_path / self.image_path.stem
        image_image_folder.mkdir(parents=True, exist_ok=True)
        for i, bbox in enumerate(preprocess_output.crop_bboxes):
            bbox_cropped_image = preprocess_output.image[int(bbox.ytl):int(bbox.ybr), int(bbox.xtl):int(bbox.xbr)]
            if label_box is True:
                label_box = False
                continue

            centres_for_answers = {mid:cls for mid, cls in zip(preprocess_output.class_midpoints[i], df_cols)}
            new_row = {'A': " ", 'B': " ", 'C': " ", 'D': " "}
            detections = self.detector.infer_parsed(bbox_cropped_image)
            for detection in detections:
                if detection.conf < 0.7:
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
