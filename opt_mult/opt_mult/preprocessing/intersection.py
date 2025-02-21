from collections import defaultdict
from typing import List, Optional, Tuple
from olv_primitives import Bbox

def line_intersection(line1: List[float], line2: List[float]) -> Optional[Tuple[float, float]]:

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

def find_intersections(horizontal_lines: List[List[int]], vertical_lines: List[List[int]], h:int, w:int) -> List[Bbox]:
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