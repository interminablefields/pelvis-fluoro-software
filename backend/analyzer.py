import torch
from torch.optim.adamw import AdamW
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import base64
import json
import copy
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import dependencies.geo_utils as geo_utils
import dependencies.heatmap_utils as heatmap_utils
import dependencies.line_cv as line_cv
from typing import List

class Analyzer:
    """
    class to perform analysis on x-ray images.
    """
    
    def __init__(self, model, weights_pth: str, dev = "cpu"):
        """
        initializes the Analyzer class with an existing model.

        parameters:
        - model (torch.nn.Module): specific model to use for inference.
        - weights_pth (str): path to the model weights.
        - dev (str, optional): device to use for computation (default is 'cpu').
        """
        self.model = model
        model.load_state_dict(torch.load(weights_pth, map_location=torch.device(dev)))

    def preprocess_im(self, img: np.ndarray):
        """
        preprocesses an image for model input.

        parameters:
        - img (np.ndarray): image to preprocess.

        returns:
        - np.ndarray: preprocessed image.
        """
        img = cv.resize(img, (224, 224)).astype(np.float32)
        image_min = np.percentile(img, 2)
        image_max = np.percentile(img, 98)
        img_normalized = (img - image_min) / (image_max - image_min)
        img_normalized = np.clip(img_normalized, 0, 1)

        return img_normalized

    def run_inference(self, im: np.ndarray, to_flip = False):
        """
        obtains model inference on an image.

        Parameters:
        - im (np.ndarray): image on which to run inference.
        - to_flip (bool, optional): whether to horizontally flip the image for inference (default is False).

        Returns:
        - dict: Logits from the model output.
        """
        img = self.preprocess_im(im)

        # forward pass
        logits = self.model.predict(img, flip=to_flip)
        return logits

    def analyze_logits(self, logits):
        """
        extracts relevant information from model-produced logits:
            * all wire line segment endpoints
            * all corridor line segment endpoints
            * all perimeter points for bones to create outlines

        parameters:
        - logits (dict): logits output from model.

        returns:
        - list: [corridor names, corridor endpoints, bone names, bone perimeter points, wire endpoints]
                where corridor and bone name/point lists are parallel.
        """

        valid_corridors = ['s1', 's2', 'ramus_left', 'ramus_right', 'teardrop_left', 'teardrop_right']
        valid_bones = ['hip_left', 'hip_right', 'femur_left', 'femur_right', 'sacrum']

        # create parallel lists of corridors | names , bones | names

        corridor_names = []
        corridor_maps = []
        bone_names = []
        bone_maps = []
        
        for name, heatmap in logits.items():
            if name.startswith("seg_"):
                seg_name = name[4:]
                if seg_name in valid_corridors:
                    corridor_maps.append(heatmap)
                    corridor_names.append(seg_name)
                elif seg_name in valid_bones:
                    bone_maps.append(heatmap)
                    bone_names.append(seg_name)
                elif seg_name == 'wire':
                    wire_map = heatmap
        
        wire_coords = self.get_all_wires(wire_map)

        # obtaining endpt coordinates for corridors
        corridor_coords = [None] * len(corridor_names)

        for i, corridor in enumerate(corridor_maps):
            corridor_pts = heatmap_utils.detect_corridor(corridor)
            if corridor_pts is not None:
                corridor_coords[i] = ( [int(x) for x in corridor_pts.p[:2].astype(int)],
                                    [int(x) for x in corridor_pts.q[:2].astype(int)] )
        
        # pull out defining points from bone contours
        bone_coords = [None] * len(bone_names)
        for i, bone in enumerate(bone_maps):
            bone_coords[i] = geo_utils.get_perim_pts(bone)

        return [corridor_names, corridor_coords, bone_names, bone_coords, wire_coords]
        
    def get_xray_data(self, img: np.ndarray):
        """
        for a given X-ray, returns json-format analysis in both supine and prone patient positions.

        parameters:
        - img (np.ndarray): The x-ray image to process.

        returns:
        - str: JSON string containing the processed x-ray data.

        json structure:
        {
            "img": {
                "enc": "base64_encoded_image_string",
                "width": 224,
                "height": 224
            },
            "coords": {
                "supine": {
                    "corridors": {
                        "<corridor_name>": {
                            "relevant_bones": ["<bone_name1>", "<bone_name2>"],
                            "coords": [[x1, y1], [x2, y2]]
                        },
                        ...
                    },
                    "bones": {
                        "<bone_name>": {
                            "coords": [[x1, y1], [x2, y2], ...]
                        },
                        ...
                    },
                    "wires": {
                        "wire<i>": {
                            "coords": [[x1, y1], [x2, y2]]
                        },
                        ...
                    }
                },
                "prone": {
                    ...
                }
            }
        }

        """
        logits_prone = self.run_inference(img)
        logits_supine = self.run_inference(img, to_flip=True)

        prone_data = self.analyze_logits(logits_prone)
        supine_data = self.analyze_logits(logits_supine)
        
        xray_data = [img, supine_data, prone_data]
        json_data = self.to_JSON(xray_data)

        return json_data
    
    def to_JSON(self, xray_data: List, img_size = 224):
        """
        helper to convert list-format x-ray analysis results to a JSON string.

        parameters:
        - xray_data (list): The x-ray data including image and pose information.
            xray_data[0]: input image
            xray_data[1]: output of analyze_logits on prone X-ray
            xray_data[2]: output of analyze_logits on supine (flipped) X-ray

        returns:
        - str: JSON serialized string of the x-ray data.
        """
        enc_img = self.encode_image(xray_data[0])

        supine_data = xray_data[1]
        prone_data = xray_data[2]

        data_dict = {}
        data_dict["img"] = {}
        data_dict["img"]["enc"] = enc_img
        data_dict["img"]["width"] = img_size
        data_dict["img"]["height"] = img_size
        supine_dict = self.turn_pose_to_dict(supine_data)
        prone_dict = self.turn_pose_to_dict(prone_data)
        data_dict["coords"] = {}
        data_dict["coords"]["supine"] = supine_dict
        data_dict["coords"]["prone"] = prone_dict

        return json.dumps(data_dict)

    def get_all_wires_liam(self, img_pth: str, logits: dict):
        """
        function using prior student code to extract wire and screw positions from logits.
        (currently doesn't work. potentially could be fixed)

        parameters:
        - img_pth (str): path to the image for preprocessing.
        - logits (dict): logits from model inference on a given X ray.

        returns:
        - list: Detected wire and screw segments.
        """
        img = self.preprocess_im(img_pth)
        wire_map = None

        for name, heatmap in logits.items():
            if name.startswith("seg_"):
                seg_name = name[4:]
                if seg_name == 'wire':
                    wire_map = heatmap
                elif seg_name == 'screw':
                    screw_map = heatmap
        wire_logits = {}
        wire_logits[0] = wire_map
        wire_logits[1] = screw_map
        output = line_cv.segments_from_logits_2(img, wire_logits)
        return output

    def get_all_wires(self, wire_map: np.ndarray, draw=False):
        """
        gets all wire endpoints from wire heatmap using cv hough line transform.

        parameters:
        - wire_map (np.array): heatmap of wires from model inference.
        - draw (bool, optional): whether to draw output (default is False).

        returns:
        - list: Coordinates of the detected wire endpoints.
        """
        if wire_map is None:
            return None
        norm = Normalize(vmin=wire_map.min(), vmax=wire_map.max())
        mapper = ScalarMappable(norm=norm)
        heatmap_im = mapper.to_rgba(wire_map, bytes=True)
        heatmap_im = np.uint8(heatmap_im)

        if(draw):
            cv.imshow('heatmap', heatmap_im)
            cv.waitKey(0)

        im_hsv = cv.cvtColor(heatmap_im, cv.COLOR_RGB2HSV)

        green_lower = np.array([30, 80, 80], dtype="uint8")
        green_upper = np.array([90, 255, 255], dtype="uint8")
        green_mask = cv.inRange(im_hsv, green_lower, green_upper)

        im_hsv[np.where(green_mask != 0)] = [255, 255, 255]
        im_hsv[np.where(green_mask == 0)] = [0, 0, 0]
        _, _, im_gray = cv.split(im_hsv)

        linesP = cv.HoughLinesP(im_gray, rho=1, theta=np.pi / 180, threshold=30, minLineLength=15, maxLineGap=6)
        line_endpts = []
        thresh = 4
        
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                sx, sy, ex, ey = int(l[0]), int(l[1]), int(l[2]), int(l[3])
                if sx == ex and sx < thresh:
                    continue
                if sy == ey and sy < thresh:
                    continue
                line_endpts.append( ([sx, sy], [ex, ey]) )
        line_endpts = self.nonmax_suppression_lines(line_endpts)

        if(draw):
            for i in range(0, len(line_endpts)):
                l = line_endpts[i]
                print(l)
                cv.line(im_hsv, (l[0][0], l[0][1]), (l[1][0], l[1][1]), (0,0,255), 1, cv.LINE_AA)
            cv.imshow('lines', im_hsv)
            cv.waitKey(0)

        return line_endpts
    
    def nonmax_suppression_lines(self, lines: List, dist_thresh = 5, angle_thresh = 10):
        """
        applies nms to refine a set of line segments based on proximity + angle.

        parameters:
        - lines (list): initial list of line segments.
        - dist_thresh (int, optional): dist threshold for considering lines as duplicates (default is 5).
        - angle_thresh (int, optional): degree threshold for considering lines as duplicates (default is 10).

        Returns:
        - list: The list of refined line segments.
        """
        final_lines = []
        remaining_lines = copy.deepcopy(lines)

        while remaining_lines:
            line1 = remaining_lines.pop(0)
            cur_matches = []
            removable_idxs = []

            for idx, line2 in enumerate(remaining_lines):
                p1, p2 = line1
                q1, q2 = line2
                dist = geo_utils.segment_to_segment_distance(line1, line2)
                angle_diff = abs(geo_utils.angle_between(p1, p2, q1, q2))
            
                if dist < dist_thresh and angle_diff < angle_thresh:
                    cur_matches.append(line2)
                    removable_idxs.append(idx)

            for idx in sorted(removable_idxs, reverse=True):
                remaining_lines.pop(idx)
        
            if len(cur_matches) > 0:
                endpts = [pt for match in cur_matches for pt in match]
                dists = [geo_utils.dist(pt, [0, 0]) for pt in endpts]
                endpt1 = endpts[dists.index(min(dists))]
                endpt2 = endpts[dists.index(max(dists))]
                remaining_lines.append([endpt1, endpt2])
            else:
                final_lines.append(line1)
        return final_lines
    
    def encode_image(self, img: np.ndarray):
        """
        encodes image into base64 string.

        parameters:
        - img (np.ndarray): image to encode.

        returns:
        - str: The encoded image as a base64 string.
        """
        _, buffer = cv.imencode('.png', img)
        enc_im = base64.b64encode(buffer).decode('utf-8')
        return enc_im

    def turn_pose_to_dict(self, data: List):
        """
        converts structured x-ray analysis for given pose into dictionary suitable for json serialization.

        parameters:
        - data (list):  pose data including corridors, bones, and wires.

        returns:
        - dict: pose data organized into a dictionary.
        """
        corridor_names, corridor_coords, bone_names, bone_coords, wire_coords = data

        relevant_bones = {"ramus_right": ["hip_right", "femur_right"], "ramus_left": ["hip_left", "femur_left"],
                            "teardrop_right": ["hip_right", "femur_right"], "teardrop_left": ["hip_left", "femur_left"],
                            "s1": ["sacrum"], "s2": ["sacrum"]}
        pose_dict = {}

        corr_dict = {}
        for i, name in enumerate(corridor_names):
            corr_dict[name] = {}
            corr_dict[name]["relevant_bones"] = relevant_bones[name]
            corr_dict[name]["coords"] = corridor_coords[i]

        bone_dict = {}
        for i, name in enumerate(bone_names):
            bone_dict[name] = {}
            bone_dict[name]["coords"] = bone_coords[i]

        wire_dict = {}
        for i, name in enumerate(wire_coords):
            wire_dict[f"wire{i}"] = {}
            wire_dict[f"wire{i}"]["coords"] = wire_coords[i]

        pose_dict["corridors"] = corr_dict
        pose_dict["bones"] = bone_dict
        pose_dict["wires"] = wire_dict

        return pose_dict