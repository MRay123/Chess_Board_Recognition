import cv2
import numpy as np


# ------------------------------------------------------------
# Utility: sort corners TL, TR, BR, BL
# ------------------------------------------------------------
def sort_corners(pts):
    pts = pts.reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


# ------------------------------------------------------------
# NEW: Expand SB hull outward so it matches the full board edge
# ------------------------------------------------------------
def expand_quad(quad, factor=1/7):
    """
    Expand a 4-corner quad outward.
    SB detects only inner 6×6 grid → expand outward to approximate full board.
    """
    center = np.mean(quad, axis=0)
    expanded = center + (quad - center) * (1 + factor * 2)
    return expanded.astype(np.float32)


# ============================================================
#  STAGE 1 — SB chessboard inner-corner detector
# ============================================================
def detect_with_sb(gray):
    pattern = (7, 7)

    found, corners = cv2.findChessboardCornersSB(
        gray, pattern,
        flags=cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if found:
        hull = cv2.convexHull(corners)
        return sort_corners(hull), "SB"

    return None, None


# ============================================================
#  STAGE 2 — Hough line detection of board grid
# ============================================================
def detect_with_hough(gray):
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=140)
    if lines is None:
        return None, None

    vertical = []
    horizontal = []

    for rho, theta in lines[:,0]:
        if abs(theta) < np.pi/6 or abs(theta - np.pi) < np.pi/6:
            vertical.append((rho, theta))
        elif abs(theta - np.pi/2) < np.pi/6:
            horizontal.append((rho, theta))

    if len(vertical) < 2 or len(horizontal) < 2:
        return None, None

    def line_intersection(l1, l2):
        rho1, th1 = l1
        rho2, th2 = l2

        A = np.array([
            [np.cos(th1), np.sin(th1)],
            [np.cos(th2), np.sin(th2)]
        ])
        b = np.array([rho1, rho2])
        return np.linalg.solve(A, b)

    v_sorted = sorted(vertical, key=lambda x: x[0])
    h_sorted = sorted(horizontal, key=lambda x: x[0])

    left, right = v_sorted[0], v_sorted[-1]
    top, bottom = h_sorted[0], h_sorted[-1]

    try:
        tl = line_intersection(left, top)
        tr = line_intersection(right, top)
        br = line_intersection(right, bottom)
        bl = line_intersection(left, bottom)

        quad = np.array([tl, tr, br, bl], dtype=np.float32)
        return quad, "HOUGH"
    except:
        return None, None


# ============================================================
#  STAGE 3 — Improved contour-based fallback
# ============================================================
def detect_with_contours(gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray2 = clahe.apply(gray)

    edges = cv2.Canny(gray2, 60, 180)
    edges = cv2.dilate(edges, np.ones((5,5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    best_area = 0
    best = None

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        area = cv2.contourArea(approx)
        if area < 5000:
            continue

        pts = approx.reshape(4,2)
        ok = True
        for i in range(4):
            a = pts[i]
            b = pts[(i+1)%4]
            c = pts[(i+2)%4]

            ba = a - b
            bc = c - b
            cosang = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc)+1e-10)
            ang = np.degrees(np.arccos(np.clip(cosang, -1,1)))

            if not (70 < ang < 110):
                ok = False
                break

        if ok and area > best_area:
            best_area = area
            best = approx

    if best is not None:
        return sort_corners(best), "CONTOUR"

    return None, None


# ============================================================
#  MASTER FUNCTION — Attempts all three detectors
# ============================================================
def robust_chessboard_detection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    quad, method = detect_with_sb(gray)
    if quad is not None:
        return quad, method

    quad, method = detect_with_hough(gray)
    if quad is not None:
        return quad, method

    quad, method = detect_with_contours(gray)
    if quad is not None:
        return quad, method

    return None, "NONE"


# ============================================================
#  DEMO USAGE
# ============================================================
if __name__ == "__main__":
    img = cv2.imread("data/data/6.jpg")

    quad, method = robust_chessboard_detection(img)
    print("Detection method:", method)

    if quad is not None:

        # ----------------------------------------------------
        # FIX: Expand SB hull so the box matches full board edge
        # ----------------------------------------------------
        if method == "SB":
            quad = expand_quad(quad)

        cv2.polylines(img, [quad.astype(int)], True, (0,255,0), 3)

    cv2.imwrite("result.jpg", img)
    print("Saved result.jpg")
