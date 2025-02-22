# where all the image preprocessing goes

import math
import statistics
from ..interface import PDFParseOutput, PreprocessOutput
import cv2
import numpy as np
from typing import List, Tuple
from ..do_lines_intersect import do_lines_intersect
from .intersection import find_intersections
from olv_draw import draw_bbs, DrawParameters
from PIL import Image, ImageDraw
from ..config import default_number_character_detector_model
import streamlit as st

from olv_object_detection.load import load_object_detector

class PreprocessingController:

    def __init__(self, detection_threshold):
        self.detection_threshold = detection_threshold

        self.number_column_detector = load_object_detector(default_number_character_detector_model)

    def preprocess_images(self, pdf_as_numpy: PDFParseOutput) -> List[PreprocessOutput]:
        preprocess_output = []
        for numpy_image in pdf_as_numpy.pdf_images:
            output = self.preprocess_image(numpy_image)
            preprocess_output.append(output)
        
        return preprocess_output

    def avg_x(self, vertical_line: List[float]):
        return (vertical_line[0] + vertical_line[2])/2
    
    ###### ROTATING IMAGES: Not used in final ######
    
    def rotatedRectWithMaxArea(self, w, h, angle):
        """
        Given a rectangle of size wxh that has been rotated by 'angle' (in
        radians), computes the width and height of the largest possible
        axis-aligned rectangle (maximal area) within the rotated rectangle.
        """
        if w <= 0 or h <= 0:
            return 0,0

        width_is_longer = w >= h
        side_long, side_short = (w,h) if width_is_longer else (h,w)

        # since the solutions for angle, -angle and 180-angle are all the same,
        # if suffices to look at the first quadrant and the absolute values of sin,cos:
        sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
        if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
            # half constrained case: two crop corners touch the longer side,
            #   the other two corners are on the mid-line parallel to the longer line
            x = 0.5*side_short
            wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
        else:
            # fully constrained case: crop touches all 4 sides
            cos_2a = cos_a*cos_a - sin_a*sin_a
            wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

        return wr,hr

    def remove_black_outline(self, w_original, h_original, angle, img_rotated):
        wr, hr = self.rotatedRectWithMaxArea(w_original, h_original, angle*np.pi/180)
        h, w = img_rotated.shape[:2]

        x1 = int((w-wr)/2)
        x2 = int(x1 + wr)
        y1 = int((h-hr)/2)
        y2 = int(y1 + hr)
        return img_rotated[y1:y2, x1:x2]

    def rotate_image(self, image, angle):
        (height, width) = image.shape[:2]
        (cent_x, cent_y) = (width // 2, height // 2)

        mat = cv2.getRotationMatrix2D((cent_x, cent_y), -angle, 1.0)
        cos = np.abs(mat[0, 0])
        sin = np.abs(mat[0, 1])

        n_width = int((height * sin) + (width * cos))
        n_height = int((height * cos) + (width * sin))

        mat[0, 2] += (n_width / 2) - cent_x
        mat[1, 2] += (n_height / 2) - cent_y

        rotated_image = cv2.warpAffine(image, mat, (n_width, n_height))

        return self.remove_black_outline(width, height, angle, rotated_image)
    
    ##############################################################################

    def get_grid_lines(self, img_cropped: np.ndarray) -> Tuple[float, float, np.ndarray, List[List[List[float]]]]:
        number_detections = self.number_column_detector.infer_parsed(img_cropped, conf_thres=0.1)
        ocr_box_lines: List[List[List[float]]] = []
        for number_bbox in number_detections:
            bbox_as_array = number_bbox.as_array()
            for i in range(len(bbox_as_array)):
                ocr_box_lines.append([bbox_as_array[i%4], bbox_as_array[(i+1)%4]])
        
        image = cv2.GaussianBlur(img_cropped, (5, 5), 0)
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges,1,np.pi/180,self.detection_threshold, None, 0, 0)
        image = img_cropped # return to normal without blur

        vertical_thetas: List[float] = []
        horizontal_thetas: List[float] = []
        for i in range(lines.shape[0]):
            if lines[i][0][1] > np.pi/4 and lines[i][0][1] < (3*np.pi)/4:
                horizontal_thetas.append(lines[i][0][1])
            else:
                vertical_thetas.append(lines[i][0][1])

        # centre around 0 radians
        for i in range(len(vertical_thetas)):
            if vertical_thetas[i] > np.pi/2:
                vertical_thetas[i] -= np.pi

        # calculating the most likely angle for the horizontal grid lines
        horizontal_theta = statistics.median(horizontal_thetas)
        if round(statistics.mode(horizontal_thetas), 5) == round(np.pi / 2, 5) and statistics.median(horizontal_thetas) != statistics.mode(horizontal_thetas):
            st.write("Overriding with mode")
            horizontal_theta = np.pi / 2
        
        horizontal_theta = np.pi / 2 # NOTE: only for with scans. Here we assume a horizontal line angle of 90 degrees.
        vertical_theta = horizontal_theta - np.pi/2

        return horizontal_theta, vertical_theta, lines, ocr_box_lines

    def preprocess_image(self, numpy_image: np.ndarray, to_filter: bool = True) -> PreprocessOutput:
        """
        Function that filters out all the unneeded hough lines until we only have hough lines that we want to crop the image with, as input to the detector.
        """
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2GRAY)

        x_b = int((210/1760) * numpy_image.shape[1])
        y_b = int((100/2560) * numpy_image.shape[0])
        image = numpy_image[x_b:-x_b, y_b:-y_b]
        h, w = image.shape[:2]

        horizontal_theta, vertical_theta, lines, ocr_box_lines = self.get_grid_lines(image)

        if horizontal_theta != np.pi/2:
            angle_to_rotate = 90-(horizontal_theta*(180/np.pi))
            if abs(angle_to_rotate) > 1e-5:
                st.write(f"Rotating image by {angle_to_rotate} degrees")
                image = self.rotate_image(image, 90-(horizontal_theta*(180/np.pi)))
                horizontal_theta, vertical_theta, lines, ocr_box_lines = self.get_grid_lines(image)

        indices_to_delete = []
        for i in range(lines.shape[0]):
            if abs(lines[i][0][1] - horizontal_theta) > np.pi/360 and abs(lines[i][0][1] - vertical_theta) > np.pi/360:
                indices_to_delete.append(i)

        lines = np.delete(lines, indices_to_delete, axis=0)

        to_filter = True
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
            x1 = int(x0 + 2*w*(-b))
            y1 = int(y0 + 2*h*(a))
            x2 = int(x0 - 2*w*(-b))
            y2 = int(y0 - 2*h*(a))

            filtered_lines_cartesian.append([x1, y1, x2, y2])

        # split into vertical and horizontal lines #####
        vertical_lines = []
        horizontal_lines = []

        for line in filtered_lines_cartesian:
            if abs(line[0]-line[2]) > abs(line[1]-line[3]):
                horizontal_lines.append(line)
            else:
                vertical_lines.append(line)

        vertical_lines = [vertical_line for vertical_line in vertical_lines if vertical_line[0] > 0]
        vertical_lines.sort(key=lambda x : x[0])

        # remove ocr overlapping vertical lines
        intersection_counts_dict = {i:0 for i in range(len(lines))}
        for line_idx, line in enumerate(vertical_lines):
            for ocr_box_line in ocr_box_lines:
                ocr_box_line_input = [ocr_box_line[0][0], ocr_box_line[0][1], ocr_box_line[1][0], ocr_box_line[1][1]]

                if do_lines_intersect(ocr_box_line_input, line):
                    intersection_counts_dict[line_idx] += 1

        indices_to_remove = []

        for line_idx in intersection_counts_dict:
            if intersection_counts_dict[line_idx] > 0:
                indices_to_remove.append(line_idx)

        vertical_lines = [vertical_line for idx, vertical_line in enumerate(vertical_lines) if idx not in indices_to_remove] 

        # clean any vertical lines that dont match the grid
        vertical_line_spacing = [self.avg_x(vertical_lines[1:][i+1]) - self.avg_x(vertical_lines[1:][i]) for i in range(len(vertical_lines[1:])-1)]
        vertical_line_spacing.sort()
        vertical_line_max = max(vertical_line_spacing)
        
        new_vertical_lines = [vertical_lines[1]]
        for i in range(len(vertical_lines[1:])):
            predicted_xloc = new_vertical_lines[-1][0] + vertical_line_max
            for vertical_line in vertical_lines:
                if abs(self.avg_x(vertical_line) - predicted_xloc) < vertical_line_max/5:
                    new_vertical_lines.append(vertical_line)
                    break

        vertical_lines = [vertical_lines[0]] + new_vertical_lines

        num_vertical_lines = len(vertical_lines)

        horizontal_lines = [horizontal_line for horizontal_line in horizontal_lines]

        for line in vertical_lines + horizontal_lines:
            x1, y1, x2, y2 = line
            cv2.line(image_with_lines,(x1,y1),(x2,y2),(0,0,255),2)

        crop_bboxes, class_midpoints = find_intersections(horizontal_lines, vertical_lines, h, w)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image_pil = Image.fromarray(image_rgb)
        image_canvas = ImageDraw.Draw(image_pil)
        draw_parameters = DrawParameters(fill_color=None)

        draw_bbs(image_canvas, crop_bboxes, draw_parameters) # draws the bounding boxes that will be cropped out of the grid to go into the detector.

        return PreprocessOutput(image=image_rgb, image_with_boxes=np.array(image_pil), image_with_lines=image_with_lines, crop_bboxes=crop_bboxes, class_midpoints=class_midpoints, num_vertical_lines=num_vertical_lines)


