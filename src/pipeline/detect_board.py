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


# ============================================================
# Improved grid detection using Hough-lines on warped board
# ============================================================

def merge_close_lines(lines, thresh=8):
    """Merge detected lines that are close together."""
    if not lines:
        return []
    lines = sorted(lines)
    merged = [lines[0]]
    for x in lines[1:]:
        if abs(x - merged[-1]) > thresh:
            merged.append(x)
    return merged


def force_exact_lines(lines, expected, max_val):
    """
    If Hough gives too many/few lines, cluster or pad until exactly `expected` lines.
    """
    if len(lines) == 0:
        # fallback: evenly spaced
        return [int(i * max_val / (expected - 1)) for i in range(expected)]

    Z = np.array(lines, dtype=np.float32).reshape(-1, 1)

    # If too few lines, pad with uniform guesses.
    if len(Z) < expected:
        pad = np.linspace(0, max_val, expected).reshape(-1, 1)
        Z = np.vstack([Z, pad])

    # K-means cluster to exactly N lines
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
    _, _, centers = cv2.kmeans(Z, expected, None, criteria, 15, cv2.KMEANS_PP_CENTERS)

    centers = sorted(int(c[0]) for c in centers)
    return centers


def detect_grid_lines(warped):
    """
    Detects 9 vertical + 9 horizontal chessboard grid lines from the
    top-down warped board using HoughLinesP.
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 40, 170)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=80,
        minLineLength=warped.shape[1] // 5,
        maxLineGap=8
    )

    if lines is None:
        return None, None

    vertical = []
    horizontal = []

    for (x1, y1, x2, y2) in lines[:, 0]:
        if abs(x1 - x2) < 10:      # vertical-ish
            vertical.append(int((x1 + x2) // 2))
        elif abs(y1 - y2) < 10:    # horizontal-ish
            horizontal.append(int((y1 + y2) // 2))

    vertical = merge_close_lines(vertical)
    horizontal = merge_close_lines(horizontal)

    # Force exactly 9 boundaries (8 squares = 9 grid lines)
    vertical = force_exact_lines(vertical, expected=9, max_val=warped.shape[1])
    horizontal = force_exact_lines(horizontal, expected=9, max_val=warped.shape[0])

    return vertical, horizontal


def draw_grid_lines(warped, vertical, horizontal):
    """Returns a debug image with grid lines drawn."""
    dbg = warped.copy()
    H, W = warped.shape[:2]

    for x in vertical:
        cv2.line(dbg, (x, 0), (x, H), (0, 0, 255), 2)

    for y in horizontal:
        cv2.line(dbg, (0, y), (W, y), (0, 255, 0), 2)

    return dbg


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
