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

    def __init__(self, detection_threshold):
        self.detection_threshold = detection_threshold

    def preprocess_images(self, pdf_as_numpy: PDFParseOutput) -> List[PreprocessOutput]:
        preprocess_output = []
        for numpy_image in pdf_as_numpy.pdf_images:
            output = self.preprocess_image(numpy_image)
            preprocess_output.append(output)
        
        return preprocess_output

    def avg_x(self, vertical_line: List[float]):
        return (vertical_line[0] + vertical_line[2])/2

    def preprocess_image(self, numpy_image: np.ndarray, to_filter: bool = True):
        numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2GRAY)

        img_cropped = numpy_image[200:-200, 100:-100]

        reader = easyocr.Reader(['en'])  # 'en' is for English; you can add more languages

        results: List[Tuple[List[List[int]], str, float]] = reader.readtext(img_cropped)

        ocr_box_lines = []
        no_ocr_box_y_coords = []
        for result in results:
            for i, box_corner in enumerate(result[0]):
                ocr_box_lines.append([result[0][i%4], result[0][(i+1)%4]])
            if "No" in result[1]:
                no_ocr_box_y_coords.append(result[0][0][1])
                
        image = cv2.GaussianBlur(img_cropped, (5, 5), 0)
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges,1,np.pi/180,self.detection_threshold, None, 0, 0)

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

        vertical_thetas = []
        horizontal_thetas = []
        for i in range(lines.shape[0]):
            if lines[i][0][1] > np.pi/4 and lines[i][0][1] < (3*np.pi)/4:
                horizontal_thetas.append(lines[i][0][1])
            else:
                vertical_thetas.append(lines[i][0][1])

        for i in range(len(vertical_thetas)):
            if vertical_thetas[i] > np.pi/2:
                vertical_thetas[i] -= np.pi

        # find the median values and remove outliers
        horizontal_thetas.sort()
        horizontal_theta = horizontal_thetas[len(horizontal_thetas)//2]

        vertical_theta = horizontal_theta - np.pi/2

        indices_to_delete = []
        for i in range(lines.shape[0]):
            if abs(lines[i][0][1] - horizontal_theta) > np.pi/360 and abs(lines[i][0][1] - vertical_theta) > np.pi/360:
                indices_to_delete.append(i)

        lines = np.delete(lines, indices_to_delete, axis=0)

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
            x1 = int(x0 + 4000*(-b))
            y1 = int(y0 + 4000*(a))
            x2 = int(x0 - 4000*(-b))
            y2 = int(y0 - 4000*(a))

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

        vertical_lines = [vertical_line for vertical_line in vertical_lines if vertical_line[0] > 0]
        vertical_lines.sort(key=lambda x : x[0])

        # clean any vertical lines that dont match the grid
        vertical_line_spacing = [self.avg_x(vertical_lines[1:][i+1]) - self.avg_x(vertical_lines[1:][i]) for i in range(len(vertical_lines[1:])-1)]
        vertical_line_spacing.sort()
        vertical_line_median = vertical_line_spacing[len(vertical_lines[1:]) // 2]
        
        new_vertical_lines = [vertical_lines[1]]
        for i in range(len(vertical_lines[1:])):
            predicted_xloc = new_vertical_lines[-1][0] + vertical_line_median
            for vertical_line in vertical_lines:
                if abs(self.avg_x(vertical_line) - predicted_xloc) < vertical_line_median/4:
                    new_vertical_lines.append(vertical_line)
                    break

        vertical_lines = [vertical_lines[0]] + new_vertical_lines

        num_vertical_lines = len(vertical_lines)

        horizontal_lines = [horizontal_line for horizontal_line in horizontal_lines if horizontal_line[1] > min(no_ocr_box_y_coords)-70]

        for line in vertical_lines + horizontal_lines:
            x1, y1, x2, y2 = line
            cv2.line(image_with_lines,(x1,y1),(x2,y2),(0,0,255),2)

        crop_bboxes, class_midpoints = find_intersections(horizontal_lines, vertical_lines, h, w)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image_pil = Image.fromarray(image_rgb)
        image_canvas = ImageDraw.Draw(image_pil)
        draw_parameters = DrawParameters(fill_color=None)

        draw_bbs(image_canvas, crop_bboxes, draw_parameters)

        return PreprocessOutput(image=image_rgb, image_with_boxes=np.array(image_pil), image_with_lines=image_with_lines, crop_bboxes=crop_bboxes, class_midpoints=class_midpoints, num_vertical_lines=num_vertical_lines)


