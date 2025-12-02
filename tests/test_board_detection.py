import cv2
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.detect_board import detect_board_corners, warp_board

# Path to your test images folder
TEST_IMAGES = "tests/images"   # create this folder & put chessboard photos inside


def show(title, img):
    os.makedirs("tests/output", exist_ok=True)
    safe = title.replace(" ", "_").replace(":", "_")
    path = f"tests/output/{safe}.jpg"
    cv2.imwrite(path, img)
    print(f"[saved] {path}")



def test_image(path):
    print(f"\n=== Testing {path} ===")

    img = cv2.imread(path)
    if img is None:
        print("Could not read image.")
        return

    # 1) Detect corners
    corners = detect_board_corners(img)
    if corners is None:
        print("❌ Board NOT detected.")
        return
    print("✔ Board detected!")

    # Draw detected corners
    dbg = img.copy()
    for (x, y) in corners:
        cv2.circle(dbg, (int(x), int(y)), 10, (0, 255, 0), -1)
    show("Detected Corners", dbg)

    # 2) Warp board
    warped = warp_board(img, corners, out_size=800)
    show("Warped Board", warped)

    # 3) Draw grid on warped
    grid = warped.copy()
    for i in range(1, 8):
        x = int(i * 800 / 8)
        cv2.line(grid, (x, 0), (x, 800), (0, 0, 255), 2)
        cv2.line(grid, (0, x), (800, x), (0, 0, 255), 2)

    show("Warped + Grid", grid)

    print("Done.\n")


def run_tests():
    print("=== Starting Board Detection Tests ===")

    for file in os.listdir(TEST_IMAGES):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            test_image(os.path.join(TEST_IMAGES, file))

    print("=== All tests finished ===")


if __name__ == "__main__":
    run_tests()
