import cv2
import sys

sys.path.append("./src")

from detect_board import detect_chessboard
from slice_squares import slice_squares
from detect_board_hough import detect_board_hough

IMAGE_PATH = "data/chess_board_1.jpg"  # Use raw string for Windows paths

def show_image(title, img):
    cv2.imwrite(f"{title}.jpg", img)
    print(f"Saved {title}.jpg")


def build_preview_grid(squares):
    rows = []
    for r in range(8):
        row_img = cv2.hconcat(squares[r*8:(r+1)*8])
        rows.append(row_img)
    return cv2.vconcat(rows)

def main():
    print("Loading image...")
    board = detect_board_hough(IMAGE_PATH)

    print("Showing detected chessboard...")
    show_image("Detected Board", board)

    print("Slicing into 64 squares...")
    squares = slice_squares(board)

    preview = build_preview_grid(squares)
    show_image("64 Squares Preview", preview)

if __name__ == "__main__":
    main()
