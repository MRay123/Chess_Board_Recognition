import cv2
import numpy as np

def detect_board_hough(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    
    edges = cv2.Canny(blur, 40, 120)
    
    # Hough lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                            minLineLength=100, maxLineGap=20)
    if lines is None:
        raise Exception("No Hough lines detected.")
    
    # Draw lines for debugging
    debug = img.copy()
    for l in lines:
        x1,y1,x2,y2 = l[0]
        cv2.line(debug,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.imwrite("debug_hough_lines.jpg", debug)

    # Compute intersections
    points = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            pt = intersect(lines[i][0], lines[j][0])
            if pt is not None:
                points.append(pt)
    
    points = np.array(points, dtype=np.float32)
    if len(points) < 4:
        raise Exception("Not enough intersection points.")
    
    # Use bounding rectangle of intersections
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.array(sorted_box(box), dtype=np.float32)
    
    # Warp
    dst = np.array([[0,0],[800,0],[800,800],[0,800]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(box, dst)
    board = cv2.warpPerspective(img, M, (800,800))
    
    cv2.imwrite("debug_board_rect.jpg", board)
    return board


def intersect(l1, l2):
    x1,y1,x2,y2 = l1
    x3,y3,x4,y4 = l2

    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if denom == 0:
        return None
    
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
    return (px,py)


def sorted_box(box):
    # Sort corners to TL, TR, BR, BL order
    s = box.sum(axis=1)
    diff = np.diff(box, axis=1)
    return [
        box[np.argmin(s)],
        box[np.argmin(diff)],
        box[np.argmax(s)],
        box[np.argmax(diff)]
    ]
