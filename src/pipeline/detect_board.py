import cv2
import numpy as np

# ------------------------------------------------------------
# Utility: Order points TL, TR, BR, BL
# ------------------------------------------------------------
def order_corners(pts):
    pts = np.array(pts)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")


# ------------------------------------------------------------
# Hough-line-based board detector
# ------------------------------------------------------------
def detect_using_lines(gray):
    edges = cv2.Canny(gray, 60, 180)
    
    # detect strong lines on the board edges
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120,
                            minLineLength=gray.shape[1]//4,
                            maxLineGap=40)
    if lines is None:
        return None
    
    lines = lines[:, 0]

    # endpoints of lines
    pts = []
    for x1, y1, x2, y2 in lines:
        pts.append([x1, y1])
        pts.append([x2, y2])
    
    pts = np.array(pts)

    # compute convex hull – should be the board
    hull = cv2.convexHull(pts)
    if len(hull) < 4:
        return None

    # simplify hull to 4 corners
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)
    if len(approx) != 4:
        return None

    return approx.reshape(4, 2)


# ------------------------------------------------------------
# Contour-based square-grid detection
# ------------------------------------------------------------
def detect_using_contours(gray):
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY,
                                   11, 3)

    cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST,
                               cv2.CHAIN_APPROX_SIMPLE)

    # find the largest quadrilateral
    max_area = 0
    best = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)

        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_area:
                max_area = area
                best = approx

    if best is None:
        return None

    return best.reshape(4, 2)


# ------------------------------------------------------------
# Fallback: OpenCV’s super robust chessboard detector
# ------------------------------------------------------------
def detect_using_chessboard(gray):
    pattern = (7, 7)  # inner corners on an 8x8 chessboard

    retval, corners = cv2.findChessboardCornersSB(gray, pattern,
                                                  flags=cv2.CALIB_CB_EXHAUSTIVE)
    if not retval:
        return None
    
    # estimate outer corners by extending inner corner grid
    corners = corners.reshape(-1, 2)

    # inner grid mapping
    tl = corners[0]
    tr = corners[6]
    bl = corners[-7]
    br = corners[-1]

    # extend outward by half a grid length
    dx = (tr - tl) / 7
    dy = (bl - tl) / 7

    outer = np.array([
        tl - dx - dy,
        tr + dx - dy,
        br + dx + dy,
        bl - dx + dy
    ])

    return outer.astype(np.float32)


# ------------------------------------------------------------
# MAIN: combine 3 methods
# ------------------------------------------------------------
def detect_board_corners(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) Try line-based
    c1 = detect_using_lines(gray)
    if c1 is not None:
        return order_corners(c1)

    # 2) Try contour-based
    c2 = detect_using_contours(gray)
    if c2 is not None:
        return order_corners(c2)

    # 3) Fallback: findChessboardCornersSB
    c3 = detect_using_chessboard(gray)
    if c3 is not None:
        return order_corners(c3)

    return None  # nothing found


# ------------------------------------------------------------
# Warp the board so it is flat top-down
# ------------------------------------------------------------
def warp_board(img, corners, out_size=800):
    dst = np.array([
        [0, 0],
        [out_size-1, 0],
        [out_size-1, out_size-1],
        [0, out_size-1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(corners.astype("float32"), dst)
    warped = cv2.warpPerspective(img, M, (out_size, out_size))
    return warped


# ------------------------------------------------------------
# Extract 64 squares
# ------------------------------------------------------------
def extract_squares(img_path):
    img = cv2.imread(img_path)
    corners = detect_board_corners(img)

    if corners is None:
        raise ValueError("Board not detected")

    warped = warp_board(img, corners, out_size=800)

    squares = []
    sq = 800 // 8
    for r in range(8):
        for c in range(8):
            crop = warped[r*sq:(r+1)*sq, c*sq:(c+1)*sq]
            squares.append(crop)

    return squares
