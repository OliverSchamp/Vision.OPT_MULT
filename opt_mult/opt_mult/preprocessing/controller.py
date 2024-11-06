# where all the image preprocessing goes

from ..interface import PDFParseOutput, PreprocessOutput
import cv2
import numpy as np
import easyocr
from typing import List, Tuple, Dict
from ..do_lines_intersect import do_lines_intersect
from .intersection import find_intersections
from olv_draw import draw_bbs, DrawParameters
from PIL import Image, ImageDraw

class PreprocessingController:

    def preprocess_images(self, pdf_as_numpy: PDFParseOutput) -> List[PreprocessOutput]:
        preprocess_output = []
        for numpy_image in pdf_as_numpy.pdf_images:
            output = self.preprocess_image(numpy_image)
            preprocess_output.append(output)
        
        return preprocess_output

    def get_xcentres_for_each_answer(self, easyocr_detections: Tuple[List[int], str, float]) -> Dict[float, str]:
        output = {}

        for easyocr_detection in easyocr_detections:
            output[easyocr_detection[0][0][0] + easyocr_detection[0][2][0]] = easyocr_detection[1]
        
        return output

    def preprocess_image(self, numpy_image: np.ndarray, to_filter: bool = True):
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2GRAY)

        img_cropped = numpy_image[200:-200, 100:-100]

        reader = easyocr.Reader(['en'])  # 'en' is for English; you can add more languages

        results: List[Tuple[List[List[int]], str, float]] = reader.readtext(img_cropped)

        ocr_box_lines = []
        for result in results:
            for i, box_corner in enumerate(result[0]):
                ocr_box_lines.append([result[0][i%4], result[0][(i+1)%4]])
            if "No" in result[1]:
                no_box_y_coord = result[0][0][1]
        
        image = cv2.GaussianBlur(img_cropped, (5, 5), 0)
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges,1,np.pi/180,215, None, 0, 0)

        # remove all lines that cross through more than 4 ocr box lines
        h, w = image.shape[:2]

        intersection_counts_dict = {i:0 for i in range(len(lines))}
        for line_idx, line in enumerate(lines):
            rho,theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 2*w*(-b))
            y1 = int(y0 + 2*h*(a))
            x2 = int(x0 - 2*w*(-b))
            y2 = int(y0 - 2*h*(a))

            for ocr_box_line in ocr_box_lines:
                ocr_box_line_input = [ocr_box_line[0][0], ocr_box_line[0][1], ocr_box_line[1][0], ocr_box_line[1][1]]

                if do_lines_intersect(ocr_box_line_input, [x1, y1, x2, y2]):
                    intersection_counts_dict[line_idx] += 1

        indices_to_remove = []

        for line_idx in intersection_counts_dict:
            if intersection_counts_dict[line_idx] > 10:
                indices_to_remove.append(line_idx)

        lines = np.delete(lines, indices_to_remove, axis=0)

        if to_filter:
            rho_threshold = 40
            theta_threshold = 0.7

            similar_lines = {i : [] for i in range(len(lines))}
            for i in range(len(lines)):
                for j in range(len(lines)):
                    if i == j:
                        continue
                    
                    lines[i][0][0] = np.abs(lines[i][0][0])
                    lines[j][0][0] = np.abs(lines[j][0][0])

                    rho_i,theta_i = lines[i][0]
                    rho_j,theta_j = lines[j][0]

                    if abs(rho_i - rho_j) < rho_threshold and (abs(theta_i - theta_j) < theta_threshold or (np.pi - abs(theta_i - theta_j)) < theta_threshold):
                        similar_lines[i].append(j)

            indices = [i for i in range(len(lines))]
            indices.sort(key=lambda x : len(similar_lines[x]))

            line_flags = len(lines)*[True]
            for i in range(len(lines) - 1):
                if not line_flags[indices[i]]:
                    continue

                for j in range(i + 1, len(lines)): 
                    if not line_flags[indices[j]]:
                        continue

                    rho_i,theta_i = lines[indices[i]][0]
                    rho_j,theta_j = lines[indices[j]][0]
                    if abs(rho_i - rho_j) < rho_threshold and (abs(theta_i - theta_j) < theta_threshold or (np.pi - abs(theta_i - theta_j)) < theta_threshold):
                        line_flags[indices[j]] = False

        filtered_lines = []
        if to_filter:
            for i in range(len(lines)): # filtering
                if line_flags[i]:
                    filtered_lines.append(lines[i])
        else:
            filtered_lines = lines

        image_with_lines = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        filtered_lines_cartesian = []
        for line in filtered_lines:
            rho,theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 2000*(-b))
            y1 = int(y0 + 2000*(a))
            x2 = int(x0 - 2000*(-b))
            y2 = int(y0 - 2000*(a))

            cv2.line(image_with_lines,(x1,y1),(x2,y2),(0,0,255),2)

            filtered_lines_cartesian.append([x1, y1, x2, y2])
        
        # find each bounding box and crop the image

        # split into vertical and horizontal lines #####
        vertical_lines = []
        horizontal_lines = []

        for line in filtered_lines_cartesian:
            if abs(line[0]-line[2]) > abs(line[1]-line[3]):
                horizontal_lines.append(line)
            else:
                vertical_lines.append(line)
        # remove middle vertical lines
        vertical_lines = [vertical_line for vertical_line in vertical_lines if vertical_line[0] > 0]
        vertical_lines.sort(key=lambda x : x[0])
        # vertical_lines = [vertical_lines[0]] + [vertical_lines[-1]]

        # find the No. OCR coords
        horizontal_lines = [horizontal_line for horizontal_line in horizontal_lines if horizontal_line[1] > no_box_y_coord-70]

        crop_bboxes, class_midpoints = find_intersections(horizontal_lines, vertical_lines, h, w)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image_pil = Image.fromarray(image_rgb)
        image_canvas = ImageDraw.Draw(image_pil)
        draw_parameters = DrawParameters(fill_color=None)

        draw_bbs(image_canvas, crop_bboxes, draw_parameters)

        return PreprocessOutput(image=image_rgb, image_with_boxes=np.array(image_pil), image_with_lines=image_with_lines, crop_bboxes=crop_bboxes, class_midpoints=class_midpoints)


