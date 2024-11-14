import contextlib
import streamlit as st
from .interface import PreprocessOutput, PDFParseOutput, DetectorOutput
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw
from olv_draw import draw_bb, DrawParameters
from olv_draw.colors import red, green, blue
from olv_primitives import Bbox
import numpy as np
from .operations import xmymwh_xyxy
import statistics

def visualise_pdfparse_output(pdfparse_output:PDFParseOutput, title: str):
    with st.expander(title):
        for pdf_image in pdfparse_output.pdf_images:
            st.image(Image.fromarray(pdf_image))

def visualise_lines_and_bboxes_output(preprocessing_output_list: List[PreprocessOutput]):
    with st.expander("Gridlines detection"):
        for preprocess_output in preprocessing_output_list:
            st.image(preprocess_output.image_with_lines)
    with st.expander("Cropped areas"):
        for preprocess_output in preprocessing_output_list:
            st.image(preprocess_output.image_with_boxes)

def visualise_detector_output(ans_dets: DetectorOutput, ms_dets: Optional[DetectorOutput] = None):
    with st.expander("List of DataFrames"):
        for i, df in enumerate(ans_dets.dataframes, 1):
            st.subheader(f"DataFrame {i}")
            st.dataframe(df)
    
    # visualise the detections of the ms and the answers on the answer sheet
    ans_images_with_detections = []
    for idx, (image, crop_bboxes_list, detections_per_bboxidx, centres_for_answers) in enumerate(zip(ans_dets.images, ans_dets.crop_bboxes_list, ans_dets.detections_per_bboxidx, ans_dets.centres_for_answers_list)):
        h, w = image.shape[:2]
        
        image_pil = Image.fromarray(image)
        image_canvas = ImageDraw.Draw(image_pil)
        draw_parameters_ans = DrawParameters(fill_color=None, outline_color=red)
        draw_parameters_ms = DrawParameters(fill_color=None, outline_color=green)
        draw_parameters_qnum = DrawParameters(fill_color=None, outline_color=blue)

        ans_bboxes, crop_bboxes_with_qnum = return_bbox_detections_from_crops(crop_bboxes_list, detections_per_bboxidx, h, w)
        for ans_bbox in ans_bboxes:
            if ans_bbox.label == "answer":
                draw_bb(image_canvas, ans_bbox, draw_parameters_ans)
            else:
                draw_bb(image_canvas, ans_bbox, draw_parameters_qnum)

        if ms_dets is None:
            ans_images_with_detections.append(np.array(image_pil))
            continue
        ms_df = ms_dets.dataframes[idx]
        question_number_bboxes = [ans_bbox for ans_bbox in ans_bboxes if ans_bbox.label == "question_number"]
        median_bbox_mid = statistics.median([qb.mid.x for qb in question_number_bboxes])
        question_number_bboxes = [ans_bbox for ans_bbox in ans_bboxes if abs(ans_bbox.mid.x - median_bbox_mid) < 20]
        ms_bboxes: List[Bbox] = []
        # should be already sorted
        for i in range(len(question_number_bboxes)):
            with contextlib.suppress(KeyError): # allowed to fail as is just for graphic
                df_row = ms_df.loc[i]
                for entry, entry_x_coord in zip(df_row, centres_for_answers.keys()):
                    if entry == "X":
                        ms_bbox_ym = question_number_bboxes[i].ym
                        ms_bbox_xm = entry_x_coord + crop_bboxes_with_qnum[i].xtl
                        ms_bbox_height = 50
                        ms_bbox_width = 50

                        x1, y1, x2, y2 = xmymwh_xyxy(ms_bbox_xm, ms_bbox_ym, ms_bbox_width, ms_bbox_height)
                        ms_bb = Bbox.from_absolute(x1, y1, x2, y2, w, h)
                        ms_bb.label = "markscheme"
                        ms_bboxes.append(ms_bb)
        
        for ms_bbox in ms_bboxes:
            if ms_bbox.label == "markscheme":
                draw_bb(image_canvas, ms_bbox, draw_parameters_ms)
        
        ans_images_with_detections.append(np.array(image_pil))

    with st.expander("Marking Visualisation"):
        for image_with_dets in ans_images_with_detections:
            st.image(image_with_dets)
    
    

def return_bbox_detections_from_crops(crop_bboxes_list: List[Bbox], detections_per_bboxidx: Dict[int, List[Bbox]], h, w) -> Tuple[List[Bbox], List[Bbox]]:
    ans_bboxes = []
    crop_bboxes_with_qnum = []

    for ans_cropped_bboxidx, ans_cropped_bbox in enumerate(crop_bboxes_list):
        try:
            cropped_bbox_dets = detections_per_bboxidx[ans_cropped_bboxidx]
        except KeyError:
            continue
        
        is_question_box = False
        for cropped_box_det in cropped_bbox_dets:
            if cropped_box_det.label == "question_number" and cropped_box_det.mid.x < 200:
                is_question_box = True
        if is_question_box:
            crop_bboxes_with_qnum.append(ans_cropped_bbox)

        for bbox_detection_relative in cropped_bbox_dets:
            bbox_xtl = ans_cropped_bbox.xtl + bbox_detection_relative.xtl
            bbox_ytl = ans_cropped_bbox.ytl + bbox_detection_relative.ytl
            bbox_xbr = ans_cropped_bbox.xtl + bbox_detection_relative.xbr
            bbox_ybr = ans_cropped_bbox.ytl + bbox_detection_relative.ybr

            bbox_detection = Bbox.from_absolute(xtl=bbox_xtl, ytl=bbox_ytl, xbr=bbox_xbr, ybr=bbox_ybr, image_height=h, image_width=w)
            bbox_detection.label = bbox_detection_relative.label
            ans_bboxes.append(bbox_detection)
    
    return ans_bboxes, crop_bboxes_with_qnum
