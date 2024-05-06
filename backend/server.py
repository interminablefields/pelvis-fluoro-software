import socketio
import eventlet
from .analyzer import Analyzer
from .feed import Feed
import cv2 as cv
import numpy as np

class Server:
    """
    class to obtain new X-ray images from Feed instance, run through Analyzer instance,  
    and send to UI through websocket.
    """

    def __init__(self, port: int, model, weights_pth: str, set_feed=0, image_pths = None):
        """
        initializes local Analyzer and Feed instances and sets up websocket.
        parameters:
        - port (int): port number on which server will run.
        - model (torch.nn.Module): model to be used by Analyzer.
        - weights_pth (str): path to the weights for the model.
        - set_feed (int, optional): If non-zero, video feed will be initialized (default is 0).
        - image_pths (list, optional): Paths to sample images if video feed is not used (default is None).
        """
        self.port = port
        self.xray_analyzer = Analyzer(model= model, weights_pth= weights_pth)

        if(set_feed):
            self.feed_set = True
            self.feed = Feed(0)
            self.feed.set_crop(575, 580, 155, 175)
        else:
            self.feed_set = False
            self.imgs = image_pths
        self.sio = socketio.Server(cors_allowed_origins='*')
        self.app = socketio.WSGIApp(self.sio)

        @self.sio.event
        def connect(sid, environ):
            print('client connected ', sid)

        @self.sio.event
        def disconnect(sid):
            print('client disconnected ', sid)

    def start(self):
        """
        starts the server & spawns appropriate handlers based on whether 
        a video feed is set or static images are used.
        """
        if(self.feed_set):
            eventlet.spawn(self.send_feed_images)
        else:
            eventlet.spawn(self.send_sample_images)
        eventlet.wsgi.server(eventlet.listen(('', self.port)), self.app)
    
    def send_data_for_path(self, img_path: str):
        """
        processes an image at specified path & sends the data over the socket.
        Parameters:
        - img_path (str): Path to the image file.
        """
        im = cv.imread(img_path)
        json_data = self.xray_analyzer.get_xray_data(im)
        self.sio.emit("xray_data", json_data)

    def send_data_for_im(self, img: np.ndarray):
        """
        processes an image and sends the data over the socket.

        parameters:
        - img (np.ndarray): image to be processed.
        """
        json_data = self.xray_analyzer.get_xray_data(img)
        self.sio.emit("xray_data", json_data)
    
    def send_feed_images(self):
        """
        continuously captures images from the video feed, processes them, 
        and sends analysis to frontend if there are changes.
        """
        while True:
            update = self.feed.check_update_frame()
            if update is not None:
                self.send_data_for_im(update)
            eventlet.sleep(0.05)

    def send_sample_images(self):
        """
        continuously cycles through a list of images, processing and sending them 
        at intervals.
        """
        index = 0
        while True:
            self.send_data_for_path(self.imgs[index])
            index = (index + 1) % len(self.imgs)
            eventlet.sleep(5)
