from tensorflow import keras
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
from skimage.segmentation import clear_border
from Solve import *

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="Path to trained digit classifier")
ap.add_argument("-i", "--image", required=True,
	help="Path to input Sudoku puzzle image")
args = vars(ap.parse_args())

def find_puzzle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    puzzleCnt = None

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)

        if len(approx) == 4:
            puzzleCnt = approx
            break

    if puzzleCnt is None:
        raise Exception("Couls not find Sudoku puzzle outline.")
    
    puzzle = get_perspective(image, puzzleCnt.reshape(4,2))
    warped = get_perspective(gray, puzzleCnt.reshape(4,2))

    return (puzzle, warped, puzzleCnt)

def get_perspective(img, location, height=600, width=600):
    pts1 = np.float32([location[0], location[3], location[1], location[2]])
    pts2 = np.floar32([0,0], [width,0], [0,height], [width, height])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (width,height))
    return result

def get_InvPerspective(img, location, height=600, width=600):
    pts1 = np.float32([0,0], [width,0], [0,height], [width, height])
    pts2 = np.float32([location[0], location[3], location[1], location[2]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (img.shape[1],img.shape[0]))
    return result

def extract_digit(cell):
    thresh = cv2.threshold(cell, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)

    cnts = cv2.findContours(thresh.cpoy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) == 0:
        return None
    
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    (h,w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w*h)
    if percentFilled < 0.03:
        return None
    
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)

    return digit

print("[INFO] Processing image...")
image = cv2.imread(args["image"])
image = cv2.resize(image, (600,600))

print("[INFO] Loading digit classifier...")
model = load_model(args["model"])

(puzzleImage, warped, Location) = find_puzzle(image)

def getBoard(Image):
    board = np.zeros((9,9), dtype="int")

    stepX = Image.shape[1] // 9
    stepY = Image.shape[0] // 9

    cellLocs = []

    for y in range(0,9):
        row = []

        for x in range(0,9):
            startX = x*stepX
            startY = y*stepY
            endX = (x+1) * stepX
            endY = (y+1) * stepY

            row.append((startX,startY,endX,endY))

            cell = Image[startY:endY, startX:endX]
            digit = extract_digit(cell)

            if digit is not None:
                roi = cv2.resize(digit, (28,28))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                pred = model.predict(roi).argmax(axis=1)[0]
                board[y,x] = pred

        cellLocs.append(row)

    return board, cellLocs

def getSolution(cellLocs, solution):
    for(cellRow, boardRow) in zip(cellLocs, solution):
        for (box, digit) in zip(cellRow, boardRow):
            startX, startY, endX, endY = box

            textX = int((endX - startX)*0.33)
            textY = int((endY - startY)*-0.2)
            textX += startX
            textY += endY

            cv2.putText(puzzleImage, str(digit), (textX,textY),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
            
    return puzzleImage

f=4
count=0
while(f):
    Board, CellLoc = getBoard(warped)
    Board=Board.tolist()

    if(solve_sudoku(Board)):
        print("Solution")
        print_grid(Board)
        finimg = getSolution(CellLoc, Board)
        cv2.imshow("Result", finimg)
        cv2.waitKey(0)

        for y in range (0, count):
            finimg = cv2.rotate(finimg, cv2.ROTATE_90_COUNTERCLOCKWISE)
        fin = get_InvPerspective(finimg, Location)
        break
    else:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        puzzleImage = cv2.rotate(puzzleImage, cv2.ROTATE_90_CLOCKWISE)
        count += 1
    f -= 1