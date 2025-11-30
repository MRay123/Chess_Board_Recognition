import cv2

def slice_squares(board_img):
    squares = []
    h, w, _ = board_img.shape
    sq_h, sq_w = h // 8, w // 8

    for r in range(8):
        for c in range(8):
            square = board_img[r*sq_h:(r+1)*sq_h, c*sq_w:(c+1)*sq_w]
            square = cv2.resize(square, (64, 64))
            squares.append(square)

    return squares
