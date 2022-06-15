import cv2
import imutils
import pytesseract
import os
import numpy as np

from imutils.perspective import four_point_transform
from PIL import Image


class SudokuScanner():

    def __init__(self, tesseract_path) -> None:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    def scan_board(self, image_path: str) -> np.ndarray:
        """Scan Sudoku board from file to numpy array

        Args:
            image_path (str): path to the image

        Raises:
            FileNotFoundError
            TypeError: if img file is not readable

        Returns:
            np.ndarray: scanned board
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f'{image_path} not found!!!')

        image = cv2.imread(image_path)

        if image is None:
            raise TypeError(f'{image_path} is not valid image file')
        
        image_warped, image_warped_thresh = self._find_board(image)
        
        board = np.zeros((9, 9), dtype="int")

        step_x = image_warped.shape[1] // 9
        step_y = image_warped.shape[0] // 9
        
        # Loop over all cells in detected board
        for y in range(0, 9):
            for x in range(0, 9):
                
                # Find cell coordinates
                start_x = x * step_x
                start_y = y * step_y
                endX = (x + 1) * step_x
                endY = (y + 1) * step_y

                # get cell from image
                cell_thresh = image_warped_thresh[start_y:endY, start_x:endX]
                is_empty = SudokuScanner._check_cell(cell_thresh)

                # verify that the digit is not empty
                if not is_empty:
                    cell = image_warped[start_y:endY, start_x:endX]
                    pred = self._pred_number(cell)

                    if pred:
                        board[y, x] = int(pred[0])
        return board


    def _find_board(self, image):
        """Find and return Sudoku Board on image, removing perspective

        Args:
            image (_type_): input image

        Returns:
            (img, img): (board in gray, board thresholded)
        """
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Blurring to reduce noise
        blurred = cv2.GaussianBlur(img_gray, (3,3), 3)

        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Opening and closing to reduce redundant noise, and fill gaps in digits
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, element)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, element)

        # Find the biggest countour
        cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
       
        sudoku_board = None
        # Find the countour that is a board
        for c in cnts:
            
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            # If countour can be approximated by PolyDP with 4 points, 
            # it is our board
            if len(approx) == 4:
                sudoku_board = approx
                break
        
        # Remove perspective
        sudoku_warped = four_point_transform(img_gray, sudoku_board.reshape(4, 2))
        sudoku_warped_thresh = four_point_transform(thresh, sudoku_board.reshape(4, 2))
        
        return sudoku_warped, sudoku_warped_thresh
    
    def _pred_number(self, image) -> str:
        """Predict number in cell using Tesseract

        Args:
            image (img)

        Returns:
            str: predicted text
        """
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
        return pytesseract.image_to_string(img, 
            config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')

    @staticmethod
    def _zoom_image(img, zoom: float):
        """Get zoomed image

        Args:
            img : img to zoom into
            zoom (float): float from 0-1, zoom magnificator

        Returns:
            _type_: _description_
        """
        y, x = img.shape
        return img[0 + int(y*zoom): y - int(y*zoom),0 + int(x*zoom): x - int(x*zoom)]
    
    @staticmethod
    def _check_cell(cell, threshold = 0.08) -> bool:
        """Check if image contains a digit

        Args:
            cell (img): img to check

        Returns:
            bool
        """
        y, x = cell.shape
        percent_filled = cv2.countNonZero(SudokuScanner._zoom_image(cell, 0.2)) / float(y * x)
        return percent_filled < threshold
