import cv2
import numpy as np
import easyocr
from typing import List, Tuple, Dict
from do_lines_intersect import do_lines_intersect
from pathlib import Path
from olv_primitives import Bbox
from collections import defaultdict

full_images_folder_path = Path("/home/oliver/Oliver.Mono/projects/Vision.OPT_MULT/data/full_images")
training_data_folder_path = Path("/home/oliver/Oliver.Mono/projects/Vision.OPT_MULT/data/cropped_data/images")


def line_intersection(line1, line2):

    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    # Calculate the denominators and numerators for intersection formula
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0:
        return None  # Lines are parallel or collinear

    # Calculate the intersection point
    intersect_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    intersect_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
    
    # Check if the intersection point is within both line segments
    if (min(x1, x2) <= intersect_x <= max(x1, x2) and min(y1, y2) <= intersect_y <= max(y1, y2) and
        min(x3, x4) <= intersect_x <= max(x3, x4) and min(y3, y4) <= intersect_y <= max(y3, y4)):
        return (intersect_x, intersect_y)
    
    return None  # The intersection is outside the segment bounds

def find_intersections(horizontal_lines: List[List[int]], vertical_lines: List[List[int]], w, h) -> List[Bbox]:
    # sort both
    horizontal_lines.sort(key=lambda x: x[1])
    vertical_lines.sort(key= lambda x: x[0])

    intersection_points_by_idx = defaultdict(list)
    for vline_idx, vertical_line in enumerate(vertical_lines):
        for horizontal_line in horizontal_lines:
            intersection_point = line_intersection(horizontal_line, vertical_line)

            if intersection_point is not None:
                intersection_points_by_idx[vline_idx].append(intersection_point)
    
    for vline_idx in intersection_points_by_idx:
        assert len(intersection_points_by_idx[vline_idx]) == len(horizontal_lines)
    
    class_midpoints_by_horizontal_line = defaultdict(list)
    for i in range(len(horizontal_lines)):
        for vline_idx in intersection_points_by_idx:
            if vline_idx == 0 or vline_idx == len(intersection_points_by_idx) - 1:
                continue
            class_midpoints_by_horizontal_line[i].append((intersection_points_by_idx[vline_idx][i][0] + intersection_points_by_idx[vline_idx+1][i][0])/2 - intersection_points_by_idx[0][i][0])
    
    coords1 = intersection_points_by_idx[0]
    coords2 = intersection_points_by_idx[len(vertical_lines) - 1]
    assert len(coords1) == len(coords2)
    output = []
    for i in range(len(coords1)):
        try:
            output.append((coords1[i], coords2[i+1]))
        except IndexError:
            pass
    
    output_parsed = [Bbox.from_absolute(xtl, ytl, xbr, ybr, w, h) for ((xtl, ytl), (xbr, ybr)) in output]

    return output_parsed, class_midpoints_by_horizontal_line


def convert_full_image_to_data(full_image_path, training_data_folder_path):
    img = cv2.cvtColor(cv2.imread(str(full_image_path)), cv2.COLOR_BGR2GRAY)
    img_cropped = img[200:-200, 100:-100]
    reader = easyocr.Reader(['en'])
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
    lines = cv2.HoughLines(edges,1,np.pi/180,150, None, 0, 0)

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

    to_filter = True

    if to_filter:
        rho_threshold = 40
        theta_threshold = 1.5

        # how many lines are similar to a given one
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

        # ordering the INDECES of the lines by how many are similar to them
        indices = [i for i in range(len(lines))]
        indices.sort(key=lambda x : len(similar_lines[x]))

        # line flags is the base for the filtering
        line_flags = len(lines)*[True]
        for i in range(len(lines) - 1):
            if not line_flags[indices[i]]: # if we already disregarded the ith element in the ordered list then we don't care (we will not delete anything based on it and we will never reconsider using this line again)
                continue

            for j in range(i + 1, len(lines)): # we are only considering those elements that had less similar line
                if not line_flags[indices[j]]: # and only if we have not disregarded them already
                    continue

                rho_i,theta_i = lines[indices[i]][0]
                rho_j,theta_j = lines[indices[j]][0]
                if abs(rho_i - rho_j) < rho_threshold and (abs(theta_i - theta_j) < theta_threshold or (np.pi - abs(theta_i - theta_j)) < theta_threshold):
                    line_flags[indices[j]] = False # if it is similar and have not been disregarded yet then drop it now

    print('number of Hough lines:', len(lines))

    filtered_lines = []

    if to_filter:
        for i in range(len(lines)): # filtering
            if line_flags[i]:
                filtered_lines.append(lines[i])

        print('Number of filtered lines:', len(filtered_lines))
    else:
        filtered_lines = lines

    filtered_lines_cartesian = []
    for line in filtered_lines:
        rho,theta = line[0]
        print(theta)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 4000*(-b))
        y1 = int(y0 + 4000*(a))
        x2 = int(x0 - 4000*(-b))
        y2 = int(y0 - 4000*(a))

        filtered_lines_cartesian.append([x1, y1, x2, y2])

    vertical_lines = []
    horizontal_lines = []

    for line in filtered_lines_cartesian:
        if abs(line[0]-line[2]) > abs(line[1]-line[3]):
            horizontal_lines.append(line)
        else:
            vertical_lines.append(line)

    vertical_lines = [vertical_line for vertical_line in vertical_lines if vertical_line[0] > 0]
    vertical_lines.sort(key=lambda x : x[0])

    horizontal_lines = [horizontal_line for horizontal_line in horizontal_lines if horizontal_line[1] > min(no_ocr_box_y_coords)-70]

    crop_bboxes, _ = find_intersections(horizontal_lines, vertical_lines, w, h)


    label_box = True
    image_image_folder = training_data_folder_path / full_image_path.stem
    image_image_folder.mkdir(parents=True, exist_ok=True)
    for i, bbox in enumerate(crop_bboxes):
        bbox_cropped_image = image[int(bbox.ytl):int(bbox.ybr), int(bbox.xtl):int(bbox.xbr)]

        try:
            image_name = reader.readtext(bbox_cropped_image)[0][1]
            image_save_path = image_image_folder / f"{image_name}.jpg"

            cv2.imwrite(str(image_save_path), bbox_cropped_image)
        except:
            pass

for full_image_file in full_images_folder_path.glob("*.jpg"):
    convert_full_image_to_data(full_image_file, training_data_folder_path)