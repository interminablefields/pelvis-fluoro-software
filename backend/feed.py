import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity as ssim

class Feed:
    """
    class to listen to video feed and detect new frames.
    """
    def __init__(self, vid_index: int):
        """
        initializes Feed class to handle video stream capturing from a given device.

        parameters:
        - vid_index (int): index of the capture device.
        """
        self.capture = cv.VideoCapture(vid_index)

        # TODO: maybe i don't need these . ?
        self.capture.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
        self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cur_frame = None
        self.left_crop = 0
        self.right_crop = 0
        self.top_crop = 0
        self.bottom_crop = 0

    def set_crop(self, left = None, right = None, top = None, bottom = None):
        """
        sets crop margins for frames captured from the video feed.

        parameters:
        - left (int, optional): # pixels to crop from the left edge.
        - right (int, optional): # pixels to crop from the right edge.
        - top (int, optional): # pixels to crop from the top edge.
        - bottom (int, optional): # pixels to crop from the bottom edge.
        """
        if(left):
            self.left_crop = left
        if(right):
            self.right_crop = right
        if(top):
            self.top_crop = top
        if(bottom):
            self.bottom_crop = bottom

    def get_frame(self):
        """
        captures and returns cropped frame from the video capture device.

        returns:
        - np.ndarray or None: The cropped frame as a numpy array if successful, None if unsuccessful.
        """
        ret, new_frame = self.capture.read()

        if not ret:
            return None
        h, w = new_frame.shape[:2]
        frow, fcol = self.top_crop, self.left_crop
        lrow, lcol = h - self.bottom_crop, w - self.right_crop

        if(frow < 0 or fcol < 0 or lrow < 0 or lcol < 0):
            return None
        
        if(frow > h or fcol > w or lrow > h or lcol > w):
            return None
        return new_frame[frow:lrow, fcol:lcol]

    def check_update_frame(self):
        """
        checks if the current frame from the video stream has changed significantly from the previous frame.

        returns:
        - np.ndarray or None: new frame if there is significant change, otherwise None.
        """
        new_frame = self.get_frame()
        if new_frame is None:
            return None

        if self.cur_frame is None:
            self.cur_frame = new_frame
            return new_frame
        
        if self.check_diff(self.cur_frame, new_frame):
            self.cur_frame = new_frame
            return new_frame
        
        return None

    def check_diff(self, frame1: np.ndarray, frame2: np.ndarray, thresh = 0.9):
        """
        checks if two frames are different based on set threshold using structural 
        similarity index from scikit-image.

        parameters:
        - frame1 (np.ndarray): first frame for comparison.
        - frame2 (np.ndarray): second frame for comparison.
        - thresh (float, optional): threshold for considering frames as different (default is 0.9).

        returns:
        - bool: True if the frames are considered different, False otherwise.
        """
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        sim = ssim(gray1, gray2, data_range=gray2.max() - gray2.min())
        return sim < thresh
    
    