def do_lines_intersect(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    # Helper function to calculate the orientation of the triplet (p, q, r)
    def orientation(px, py, qx, qy, rx, ry):
        # Calculate the determinant of the matrix formed by the three points
        val = (qy - py) * (rx - qx) - (qx - px) * (ry - qy)
        if val == 0:
            return 0  # Collinear
        elif val > 0:
            return 1  # Clockwise
        else:
            return 2  # Counterclockwise
    
    # Helper function to check if point q lies on line segment pr
    def on_segment(px, py, qx, qy, rx, ry):
        if min(px, rx) <= qx <= max(px, rx) and min(py, ry) <= qy <= max(py, ry):
            return True
        return False
    
    # Calculate the four orientations needed
    o1 = orientation(x1, y1, x2, y2, x3, y3)
    o2 = orientation(x1, y1, x2, y2, x4, y4)
    o3 = orientation(x3, y3, x4, y4, x1, y1)
    o4 = orientation(x3, y3, x4, y4, x2, y2)
    
    # General case: segments intersect if the orientations are different
    if o1 != o2 and o3 != o4:
        return True
    
    # Special cases: checking collinearity and overlap
    # x3 lies on segment x1-x2
    if o1 == 0 and on_segment(x1, y1, x3, y3, x2, y2):
        return True
    # x4 lies on segment x1-x2
    if o2 == 0 and on_segment(x1, y1, x4, y4, x2, y2):
        return True
    # x1 lies on segment x3-x4
    if o3 == 0 and on_segment(x3, y3, x1, y1, x4, y4):
        return True
    # x2 lies on segment x3-x4
    if o4 == 0 and on_segment(x3, y3, x2, y2, x4, y4):
        return True
    
    # If none of the cases apply, the segments do not intersect
    return False

if __name__ == "__main__":
    line1 = [0, 0, 5, 5]

    line2 = [0, 5, 4, 4]

    print(do_lines_intersect(line1, line2))