import cv2
import numpy as np

def detect_chessboard(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Try findChessboardCorners first
    pattern_size = (7, 7)  # inner corners of 8x8 squares
    found, corners = cv2.findChessboardCorners(gray, pattern_size)

    if not found:
        # Fallback: detect largest 4-sided contour
        return extract_via_contours(img)

    # Use corner points to estimate board location
    corners = corners.squeeze()

    # Fit bounding box around inner corners
    x_min, y_min = np.min(corners, axis=0)
    x_max, y_max = np.max(corners, axis=0)

    # Expand slightly since corners only mark inner grid
    pad_x = int((x_max - x_min) * 0.15)
    pad_y = int((y_max - y_min) * 0.15)

    x1 = max(0, int(x_min - pad_x))
    y1 = max(0, int(y_min - pad_y))
    x2 = min(img.shape[1], int(x_max + pad_x))
    y2 = min(img.shape[0], int(y_max + pad_y))

    board = img[y1:y2, x1:x2]
    board = cv2.resize(board, (800, 800))  # normalize size
    return board


def extract_via_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 120)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    debug = img.copy()

    biggest = None
    max_area = 0
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # draw every candidate contour
        cv2.drawContours(debug, [approx], -1, (0,255,0), 2)

        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > max_area and area > 0.1 * img.shape[0] * img.shape[1]:
                biggest = approx
                max_area = area

    # save debug file
    cv2.imwrite("debug_contours.jpg", debug)
    print("Saved debug_contours.jpg")

    if biggest is None:
        raise Exception("Cannot detect chessboard!")

    # draw selected contour
    debug2 = img.copy()
    cv2.drawContours(debug2, [biggest], -1, (0,0,255), 4)
    cv2.imwrite("selected_contour.jpg", debug2)
    print("Saved selected_contour.jpg")

    pts = biggest.reshape(4, 2).astype("float32")
    pts_sorted = sort_points(pts)
    board = warp_to_square(img, pts_sorted)
    return board



def sort_points(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)]
    ], dtype="float32")


def warp_to_square(img, pts):
    dst = np.array([[0,0], [800,0], [800,800], [0,800]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, M, (800, 800))
