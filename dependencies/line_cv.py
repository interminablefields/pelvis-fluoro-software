import io
import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.morphology
from skimage.morphology import skeletonize
from skimage.transform import hough_line, hough_line_peaks

# from corridors.wiretwin.utils import crop_percent


# from tqdm.notebook import tqdm
# io utils
# datastructures
# 3D transformations functions
# rendering components
# from pytorch3d.transforms import Transform3d, Rotate, Translate
# from pytorch3d.transforms import *


def segments_from_logits(logits, border_width=20, debug=False):


    threshold = 0.3
    dilate_ratio = 10/1000
    near_cutoff = 5

    masks = []

    for k in logits:
        # seg_wire = logits[k]
        seg_wire = k
        # seg_wire_bin = crop_percent(seg_wire, crop_px)
        seg_wire_bin = seg_wire

        # plt.figure(figsize=(10, 10))
        # plt.imshow(seg_wire_bin)
        
        seg_wire_bin = (np.clip(seg_wire_bin, 0, 1)*255).astype(np.uint8)
        seg_wire_bin = cv2.threshold(seg_wire_bin, threshold*255, 255, cv2.THRESH_BINARY)[1]

        masks.append(seg_wire_bin.copy())

    # or together all masks
    seg_wire_bin = np.zeros_like(masks[0])
    for mask in masks:
        seg_wire_bin = np.logical_or(seg_wire_bin, mask)


    # plt.figure(figsize=(10, 10))
    # plt.imshow(seg_wire_bin)

    dilate_erode_radius = int(dilate_ratio * seg_wire_bin.shape[0])

    # skimage dilate and erode with round kernel
    seg_wire_bin = skimage.morphology.binary_dilation(seg_wire_bin, skimage.morphology.disk(dilate_erode_radius))
    seg_wire_bin = skimage.morphology.binary_erosion(seg_wire_bin, skimage.morphology.disk(dilate_erode_radius))

    # compute skeleton
    seg_wire_bin = skeletonize(seg_wire_bin)

    # set any pixels along the border to 0
    seg_wire_bin[:border_width, :] = 0
    seg_wire_bin[-border_width:, :] = 0
    seg_wire_bin[:, :border_width] = 0
    seg_wire_bin[:, -border_width:] = 0


    # plt.figure(figsize=(10, 10))
    # plt.imshow(seg_wire_bin)

    # skimage hough line transform
    h, theta, d = hough_line(seg_wire_bin)
    accums, angles, dists = hough_line_peaks(h, theta, d, min_distance=100, min_angle=10, threshold=0.4*np.max(h))
    if len(accums) == 0:
        return []

    # sort and get top 2
    accums, angles, dists = zip(*(sorted(zip(accums, angles, dists), key=lambda x: x[0], reverse=True)[:2]))

    # # show only detected lines
    # plt.figure(figsize=(2, 2))
    # plt.imshow(seg_wire_bin)
    # for accum, angle, dist in zip(accums, angles, dists):
    #     y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    #     y1 = (dist - seg_wire_bin.shape[1] * np.cos(angle)) / np.sin(angle)
    #     plt.plot((0, seg_wire_bin.shape[1]), (y0, y1), '-r')
    # # plt.xlim((0, seg_wire_bin.shape[1]))
    # # plt.ylim((seg_wire_bin.shape[0], 0))

    # find contours
    cv2_img = (np.clip(seg_wire_bin, 0, 1)*255).astype(np.uint8)
    # TODO: this is dumb, just sample points like the other version
    contours, hierarchy = cv2.findContours(cv2_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return []
    contour_points = np.vstack(contours).squeeze()

    # for each line, get the set of contour points within a 5 pixel radius
    near_contour_points = []
    for accum, angle, dist in zip(accums, angles, dists):
        if np.sin(angle) == 0:
            angle += 1e-6
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - seg_wire_bin.shape[1] * np.cos(angle)) / np.sin(angle)
        x0 = 0
        x1 = seg_wire_bin.shape[1]
        line = np.array([x0, y0, x1, y1])
        numerator = np.abs(np.cross(contour_points - line[:2], line[2:] - line[:2]))
        denominator = np.linalg.norm(line[2:] - line[:2])
        dists = numerator / denominator

        close_points = contour_points[dists < near_cutoff]
        if len(close_points) == 0:
            continue

        # get the distance along the line
        line_vec = line[2:] - line[:2]
        line_vec = line_vec / np.linalg.norm(line_vec)
        contour_points_vec = close_points - line[:2]
        dist_along_line = np.dot(contour_points_vec, line_vec)
        # get max and min dists
        max_dist = np.max(dist_along_line)
        min_dist = np.min(dist_along_line)

        # get locations on line
        max_loc = line[:2] + max_dist * line_vec
        min_loc = line[:2] + min_dist * line_vec

        # stack max and min
        max_min = np.vstack([max_loc, min_loc])

        near_contour_points.append((accum, max_min))
        # near_contour_points.append(contour_points[dists < near_cutoff])

    # plot the contour points as dots
    if debug:
        plt.figure(figsize=(2, 2))
        plt.imshow(seg_wire_bin)
        for accum, angle, dist, (accum, near_contour_point) in zip(accums, angles, dists, near_contour_points):
            print(accum, angle, dist, near_contour_point)
            plt.scatter(near_contour_point[:, 0], near_contour_point[:, 1], s=50)
        plt.xlim((0, seg_wire_bin.shape[1]))
        plt.ylim((seg_wire_bin.shape[0], 0))

    near_contour_points.sort(key=lambda x: x[0], reverse=True)

    return near_contour_points

# wire_segments = [segments_from_logits(logits, threshold, dilate_ratio, near_cutoff, debug=False) for logits in seg_wires]

# for img_idx, seg_col in enumerate(wire_segments):
#     seg_wire_bin = seg_wires[img_idx]
#     plt.figure(figsize=(3,2))
#     plt.imshow(seg_wire_bin, cmap="gray")
#     for accum, near_contour_point in seg_col:
#         plt.scatter(near_contour_point[:, 0], near_contour_point[:, 1], s=20)
#     plt.xlim((0, seg_wire_bin.shape[1]))
#     plt.ylim((seg_wire_bin.shape[0], 0))

def get_refined_mask_seg(seg_col, raw_img):
    # copy raw image and normalize by min/max
    raw_img = raw_img.copy()
    raw_img = (raw_img - raw_img.min()) / (raw_img.max() - raw_img.min())

    # with opencv make a mask with a white line for each contour line

    mask_line_width = 30
    masks = []
    for accum, near_contour_point in seg_col:
        mask = np.zeros_like(raw_img)
        mask = cv2.line(mask, tuple(near_contour_point[0].astype(np.int32)), tuple(near_contour_point[1].astype(np.int32)), 255, mask_line_width)
        masks.append(mask)

    # # show the masks
    # for mask in masks:
    #     plt.figure(figsize=(2, 2))
    #     plt.imshow(mask, cmap="gray")

    wire_segments_refined_view = []

    view_wire_masks = []
    for i, mask in enumerate(masks):
        # plt.figure(figsize=(2, 2))
        # plt.imshow(mask, cmap="gray")

        # cv inpaint
        inpaint_radius = 10
        inpaint_method = cv2.INPAINT_TELEA

        inpaint_mask = mask.astype(np.uint8)
        inpaint_mask = cv2.threshold(inpaint_mask, 0, 255, cv2.THRESH_BINARY)[1]

        inpaint_scalar = 1000 # for some reason inpaint doesn't work well with 0-1 float images
        inpaint_img = raw_img.copy() * inpaint_scalar
        inpaint_img = inpaint_img.astype(np.float32)
        assert inpaint_img.dtype == np.float32, inpaint_img.dtype
        inpaint_img = cv2.inpaint(inpaint_img, inpaint_mask, inpaint_radius, inpaint_method) / inpaint_scalar


        # plt.figure(figsize=(2, 2))
        # plt.imshow(raw_img[img_idx])
        # plt.colorbar()
        # plt.title("original")

        # plt.figure(figsize=(5,5))
        # plt.imshow(inpaint_img)
        # plt.colorbar()
        # plt.title("inpaint")

        # convert both to float and subtract
        subtracted = (inpaint_img.astype(np.float32) - raw_img.astype(np.float32))
        # clip below zero
        subtracted = np.clip(subtracted, 0, None)

        # plt.figure(figsize=(5, 5))
        # plt.imshow(subtracted)
        # plt.colorbar()
        # plt.title("subtracted")

        # convert to binary mask above threshold
        intensity_threshold = 0.2*subtracted.max()
        exact_mask = (subtracted > intensity_threshold).astype(np.uint8)
        
        # plt.figure(figsize=(2, 2))
        # plt.imshow(exact_mask, cmap="gray")
        # plt.colorbar()

        # # dialte with skimage
        # dilate_radius = 2
        # # exact_mask = skimage.morphology.binary_dilation(exact_mask, skimage.morphology.disk(dilate_radius))
        # # erode
        # exact_mask = skimage.morphology.binary_erosion(exact_mask, skimage.morphology.disk(dilate_radius))

        # # convert back to float
        # exact_mask = exact_mask.astype(np.float32)


        view_wire_masks.append(exact_mask)

        dilate_radius = 3
        dilated = skimage.morphology.binary_dilation(exact_mask, skimage.morphology.disk(dilate_radius))

        # skeletonize
        skeleton = skeletonize(dilated)

        # hough lines
        h, theta, d = hough_line(skeleton, theta = np.linspace(-np.pi / 2, np.pi / 2, 2*360, endpoint=False))
        accums, angles, dists = hough_line_peaks(h, theta, d, min_distance=1, min_angle=5, threshold=0.4*np.max(h))

        joined = list(zip(accums, angles, dists))
        joined.sort(key=lambda x: x[0], reverse=True)

        accum, angle, dist = joined[0]

        # plt.figure(figsize=(10,10))
        # plt.imshow(skeleton)
        # # plt.imshow(dilated)
        # # for accum, angle, dist in zip(accums, angles, dists):
        # y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        # y1 = (dist - exact_mask.shape[1] * np.cos(angle)) / np.sin(angle)
        # plt.plot((0, exact_mask.shape[1]), (y0, y1), '-r')
        # plt.xlim((0, exact_mask.shape[1]))
        # plt.ylim((exact_mask.shape[0], 0))

        if np.sin(angle) == 0:
            angle += 1e-6

        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - raw_img.shape[1] * np.cos(angle)) / np.sin(angle)
        x0 = 0
        x1 = raw_img.shape[1]
        line = np.array([x0, y0, x1, y1])

        accum, near_contour_point = seg_col[i]

        line_vec = line[2:] - line[:2]
        line_vec = line_vec / np.linalg.norm(line_vec)
        contour_points_vec = near_contour_point - line[:2]
        dist_along_line = np.dot(contour_points_vec, line_vec)

        max_dist = dist_along_line[0]
        min_dist = dist_along_line[1]

        max_loc = line[:2] + max_dist * line_vec
        min_loc = line[:2] + min_dist * line_vec

        wire_segments_refined_view.append((accum, np.vstack([max_loc, min_loc])))

    return wire_segments_refined_view

from skimage.morphology import medial_axis, skeletonize


def signed_angle_distance_half(a, b):
    a = a % (np.pi)
    b = b % (np.pi)
    return np.abs(a-b)
        

def segments_from_logits_2(raw_img, logits, debug=False):


    threshold = 0.1
    dilate_ratio = 10/1000
    near_cutoff = 5

    # masks = []
    

    accum_mask = np.zeros_like(logits[0])

    for k in logits:
        # seg_wire = logits[k]
        seg_wire_bin = logits[k]
        # seg_wire_bin = crop_percent(seg_wire, crop_px)

        # clip 0,1
        seg_wire_bin = np.clip(seg_wire_bin, 0, 1)
        
        # blur
        seg_wire_bin = cv2.GaussianBlur(seg_wire_bin, (5,5), 0)

        # plt.figure(figsize=(10,10))
        # plt.imshow(seg_wire_bin)
        # plt.colorbar()
        
        # seg_wire_bin = (np.clip(seg_wire_bin, 0, 1)*255).astype(np.uint8)
        # seg_wire_bin = cv2.threshold(seg_wire_bin, threshold*255, 255, cv2.THRESH_BINARY)[1]

        # masks.append(seg_wire_bin.copy())

        accum_mask += seg_wire_bin

    # or together all masks
    # seg_wire_bin = np.zeros_like(masks[0])
    # for mask in masks:
    #     seg_wire_bin = np.logical_or(seg_wire_bin, mask)


    seg_wire_bin = accum_mask
    # seg_wire_bin_low = seg_wire_bin.copy()

    # plt.figure(figsize=(5,5))
    # plt.imshow(seg_wire_bin)
    # plt.colorbar()
    # plt.show()

    # TODO: don't clip logits, this is bad
    seg_wire_bin = (np.clip(seg_wire_bin, 0, 1)*255).astype(np.uint8)
    seg_wire_bin = cv2.threshold(seg_wire_bin, threshold*255, 255, cv2.THRESH_BINARY)[1]

    seg_wire_bin_low = (np.clip(seg_wire_bin, 0, 1)*255).astype(np.uint8)
    seg_wire_bin_low = cv2.threshold(seg_wire_bin_low, threshold/2*255, 255, cv2.THRESH_BINARY)[1] # TODO: not needed

    # plt.figure(figsize=(5, 5))
    # plt.imshow(seg_wire_bin)
    # plt.show()
    

    # clip 0,1

    if debug:
        plt.figure(figsize=(5, 5))
        plt.imshow(seg_wire_bin)
        plt.colorbar()

    dilate_erode_radius = int(dilate_ratio * seg_wire_bin.shape[0])

    # skimage dilate and erode with round kernel
    seg_wire_bin = skimage.morphology.binary_dilation(seg_wire_bin, skimage.morphology.disk(dilate_erode_radius))
    seg_wire_bin = skimage.morphology.binary_erosion(seg_wire_bin, skimage.morphology.disk(dilate_erode_radius))


    # plt.figure(figsize=(2, 2))
    # plt.imshow(mask, cmap="gray")

    # cv inpaint
    inpaint_radius = 10
    inpaint_method = cv2.INPAINT_TELEA

    inpaint_mask = seg_wire_bin.astype(np.uint8)
    inpaint_mask = cv2.threshold(inpaint_mask, 0, 255, cv2.THRESH_BINARY)[1]

    inpaint_scalar = 1000 # for some reason inpaint doesn't work well with 0-1 float images
    # assert raw_img is between 0 and 1
    assert raw_img.dtype == np.float32, raw_img.dtype
    assert raw_img.min() >= 0 and raw_img.max() <= 1
    raw_img_01 = raw_img.copy().astype(np.float32)
    inpaint_img = raw_img_01 * inpaint_scalar
    inpaint_img = inpaint_img.astype(np.float32)
    assert inpaint_img.dtype == np.float32, inpaint_img.dtype

    inpaint_img = cv2.cvtColor(inpaint_img, cv2.COLOR_BGR2GRAY)
    raw_img_01 = cv2.cvtColor(raw_img_01, cv2.COLOR_BGR2GRAY)

    # TODO: try scipy inpaint 
    inpaint_img = cv2.inpaint(inpaint_img, inpaint_mask, inpaint_radius, inpaint_method) / inpaint_scalar

    if debug:
        plt.figure(figsize=(5, 5))
        plt.imshow(raw_img_01)
        plt.colorbar()
        plt.title("original")

    if debug:
        plt.figure(figsize=(5, 5))
        plt.imshow(inpaint_img)
        plt.colorbar()
        plt.title("inpaint")

    # convert both to float and subtract
    subtracted = (inpaint_img.astype(np.float32) - raw_img_01.astype(np.float32))
    subtracted = -subtracted
    # clip below zero
    subtracted = np.clip(subtracted, 0, None)

    if debug:
        plt.figure(figsize=(5, 5))
        plt.imshow(subtracted)
        plt.colorbar()
        plt.title("subtracted")

    # convert to binary mask above threshold
    intensity_threshold = 0.2*subtracted.max()
    exact_mask = (subtracted > intensity_threshold).astype(np.uint8)

    
    if debug:
        plt.figure(figsize=(5, 5))
        plt.imshow(exact_mask)

    global skeleton_mask
    skeleton_mask = skeletonize(exact_mask)

    skel, distance = medial_axis(exact_mask, return_distance=True)

    if debug:
        plt.figure(figsize=(5, 5))
        plt.imshow(skeleton_mask)

        bytesio = io.BytesIO()
        plt.savefig(bytesio, format='png')
        bytesio.seek(0)
        debug_images.append(bytesio)
        plt.close()

    # global dist_on_skel
    dist_on_skel = distance * skeleton_mask

    if debug:
        plt.figure(figsize=(5, 5))
        plt.imshow(dist_on_skel)
    
    nograd_mask = np.zeros_like(dist_on_skel)
    
    dist = 5
    for i in range(dist_on_skel.shape[0]):
        for j in range(dist_on_skel.shape[1]):
            if dist_on_skel[i, j] == 0:
                continue
    
            neighborhood_mask = np.zeros((dist*2+1, dist*2+1))
            get_connected_neighborhood(dist_on_skel, neighborhood_mask, i, j, dist, dist, dist)
    
            neighborhood_vals = []
            for u in range(dist*2+1):
                for v in range(dist*2+1):
                    if neighborhood_mask[u, v] == 1:
                        neighborhood_vals.append(dist_on_skel[i+u-dist, j+v-dist])
            if np.max(neighborhood_vals) - np.min(neighborhood_vals) < 2:
                nograd_mask[i, j] = 1
    
    if debug:
        plt.figure(figsize=(5, 5))
        plt.imshow(nograd_mask)

        bytesio = io.BytesIO()
        plt.savefig(bytesio, format='png')
        bytesio.seek(0)
        debug_images.append(bytesio)
        plt.close()

    h, theta, d = hough_line(nograd_mask, theta=np.linspace(-np.pi / 2, np.pi / 2, 360))
    accums, angles, dists = hough_line_peaks(h, theta, d, min_distance=10, min_angle=10, threshold=20)
    if len(accums) == 0:
        return []

    big_lines = (sorted(zip(accums, angles, dists), key=lambda x: x[0], reverse=True)[:10])

    if debug:
        plt.figure(figsize=(5,5))
        # plt.imshow(seg_wire_bin_og)
        plt.imshow(nograd_mask, cmap="gray")
        plt.colorbar()
        for accum, angle, dist in big_lines:
            y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - seg_wire_bin.shape[1] * np.cos(angle)) / np.sin(angle)
            plt.plot((0, seg_wire_bin.shape[1]), (y0, y1), '-r', linewidth=0.5)
        plt.xlim((0, seg_wire_bin.shape[1]))
        plt.ylim((seg_wire_bin.shape[0], 0))

        bytesio = io.BytesIO()
        plt.savefig(bytesio, format='png')
        bytesio.seek(0)
        debug_images.append(bytesio)
        plt.close()

    angle_mask = np.zeros_like(skeleton_mask, dtype=np.float32)
    
    window_size = 10
    for i in range(skeleton_mask.shape[0]):
        for j in range(skeleton_mask.shape[1]):
            if skeleton_mask[i, j] == 0:
                continue
    
    
            neighborhood_mask = np.zeros((window_size*2+1, window_size*2+1))
            get_connected_neighborhood(skeleton_mask, neighborhood_mask, i, j, window_size, window_size, window_size)
    
            h, theta, d = hough_line(neighborhood_mask, theta=np.linspace(-np.pi / 2, np.pi / 2, 90))
            accums, angles, dists = hough_line_peaks(h, theta, d, min_distance=1, min_angle=50, threshold=1)
            if len(accums) == 0:
                continue
            
            lines = (sorted(zip(accums, angles, dists), key=lambda x: x[0], reverse=True)[:10])
    
            # get list of lines that intersect with middle pixel
            intersecting_lines = []
            px_dist_thresh = 1
            thresh_ada = px_dist_thresh
            while len(intersecting_lines) == 0 and px_dist_thresh < 10:
                for accum, angle, dist in lines:
                    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
                    y1 = (dist - neighborhood_mask.shape[1] * np.cos(angle)) / np.sin(angle)
                    x0 = 0
                    x1 = neighborhood_mask.shape[1]
                    middle_pixel_coords = (window_size, window_size)
                    middle_pixel_dist_to_line = np.abs((y1-y0)*middle_pixel_coords[1] - (x1-x0)*middle_pixel_coords[0] + x1*y0 - y1*x0) / np.sqrt((y1-y0)**2 + (x1-x0)**2)
                    if middle_pixel_dist_to_line < thresh_ada:
                        intersecting_lines.append((accum, angle, dist))
                thresh_ada += 1
    
    
            if len(intersecting_lines) == 0:
                # angle_mask[i, j] = 0
                if len(lines) > 0:
                    angle_mask[i, j] = lines[0][1]
                    continue
                angle_mask[i, j] = np.inf
                continue
    
            if len(intersecting_lines) == 1:
                angle_mask[i, j] = intersecting_lines[0][1]
                continue
    
    
            # first two intersecting_lines
            first_line = intersecting_lines[0]
            second_line = intersecting_lines[1]
    
            # get angle between lines
            angle_between_lines = np.abs(first_line[1] - second_line[1])
            # if angle is small
            if angle_between_lines < np.pi/4:
                angle_mask[i, j] = intersecting_lines[0][1]
                continue
    
            # second to first ratio
            second_to_first_ratio = second_line[0] / first_line[0]
    
            if second_to_first_ratio < 0.6:
                angle_mask[i, j] = intersecting_lines[0][1]
                continue
    
            # if angle is large
            # angle_mask[i, j] = 0
            # angle_mask[i, j] = np.inf
            angle_mask[i, j] = intersecting_lines[0][1]


    near_contour_points = []
    for accum, angle, dist in big_lines:
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - seg_wire_bin.shape[1] * np.cos(angle)) / np.sin(angle)
        x0 = 0
        x1 = seg_wire_bin.shape[1]
        
        line_points = np.zeros_like(dist_on_skel).astype(np.uint8)
        for i in range(dist_on_skel.shape[0]):
            for j in range(dist_on_skel.shape[1]):
                if dist_on_skel[i, j] == 0:
                    continue
                dist_to_line_thresh = 2


                dist_to_line = np.abs((y1-y0)*j - (x1-x0)*i + x1*y0 - y1*x0) / np.sqrt((y1-y0)**2 + (x1-x0)**2)
                
                # if dist_to_line < dist_to_line_thresh:
                if dist_to_line < dist_to_line_thresh and abs(signed_angle_distance_half(angle_mask[i, j], angle)) < np.deg2rad(5):
                    line_points[i, j] = 1

        # plt.figure(figsize=(5, 5))
        # plt.imshow(dist_on_skel)
        # plt.plot((x0, x1), (y0, y1), '-r', linewidth=0.5)
        # plt.colorbar()
        # plt.xlim((0, seg_wire_bin.shape[1]))
        # plt.ylim((seg_wire_bin.shape[0], 0))

        line_points = line_points*255
        nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(line_points, connectivity=8)
        sizes = stats[:, -1]
        sizes = sizes[1:]
        nb_blobs -= 1
        # min_size = 5
        min_size = 10


        filtered_blobs = []
        for blob in range(nb_blobs):
            if sizes[blob] >= min_size:
                filtered_blobs.append(blob)

        if len(filtered_blobs) == 0:
            continue

        if np.sin(angle) == 0:
            angle += 1e-6
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - seg_wire_bin.shape[1] * np.cos(angle)) / np.sin(angle)
        x0 = 0
        x1 = seg_wire_bin.shape[1]
        line = np.array([x0, y0, x1, y1])
        
        # # plt.figure(figsize=(5, 5))
        # # plt.imshow(im_result)

        # im_result = np.zeros_like(im_with_separated_blobs)
        # for blob in filtered_blobs:
        #     im_result[im_with_separated_blobs == blob + 1] = 255

        # close_points = np.argwhere(im_result > 0)[:, ::-1]
        # if len(close_points) == 0:
        #     continue

        blob_endpoints = []

        line_vec = line[2:] - line[:2]
        line_len = np.linalg.norm(line_vec)
        line_vec = line_vec / np.linalg.norm(line_vec)

        for blob in filtered_blobs:
            close_points = np.argwhere(im_with_separated_blobs == blob + 1)[:, ::-1]

            # get the distance along the line

            contour_points_vec = close_points - line[:2]
            dist_along_line = np.dot(contour_points_vec, line_vec)
            # get max and min dists
            max_dist = np.max(dist_along_line)
            min_dist = np.min(dist_along_line)

            max_loc = line[:2] + max_dist * line_vec
            min_loc = line[:2] + min_dist * line_vec

            blob_endpoints.append((min_dist, max_dist, blob, min_loc, max_loc))

            # # get locations on line
            # max_loc = line[:2] + max_dist * line_vec
            # min_loc = line[:2] + min_dist * line_vec

            # # stack max and min
            # # max_min = np.vstack([max_loc, min_loc])

            # blob_endpoints.append((max_loc, min_loc))

        # sort blob_endpoints by dist along line
        blob_endpoints.sort(key=lambda x: x[0], reverse=False)


        joined_blobs = [[blob_endpoints[0]]]
        
        exact_mask_bools = exact_mask > 0
        seg_wire_bin_low_bools = seg_wire_bin_low > 0
        ored_points = np.logical_or(exact_mask_bools, seg_wire_bin_low_bools)

        # plt.figure(figsize=(5, 5))
        # plt.imshow(ored_points)
        # plt.colorbar()
        # plt.title("ored_points")

        exact_mask_points = np.argwhere(ored_points)[:, ::-1] # TODO: add this
        # exact_mask_points = np.argwhere(exact_mask > 0)[:, ::-1]
        
        dist_to_line = np.abs(np.cross(exact_mask_points - line[:2], line_vec))
        dist_thresh = 3
        
        exact_mask_points = exact_mask_points[dist_to_line < dist_thresh]
        
        contour_points_vec = exact_mask_points - line[:2]
        dist_along_line = np.dot(contour_points_vec, line_vec)

        step_size = 1
        for blob_idx in range(len(blob_endpoints)-1):

            curr_end = blob_endpoints[blob_idx][1]
            next_beg = blob_endpoints[blob_idx+1][0]

            march_loc = curr_end
            connected = False

            # if they are closer than 3 pixels, join them
            if next_beg - curr_end < 3:
                connected = True

            if not connected:
                faults_in_a_row_thresh = 40 # TODO resolution dependent
                faults_in_a_row = 0
                while True:
                    # if there is not a point between far_loc and far_loc + step_size, break
                    if march_loc > next_beg:
                        connected = True
                        break
                    if np.sum((dist_along_line > march_loc) & (dist_along_line < march_loc + step_size)) == 0:
                        faults_in_a_row += 1
                    else:
                        faults_in_a_row = 0

                    if faults_in_a_row > faults_in_a_row_thresh:
                        break
                    march_loc += step_size

            if connected:
                # joins.append((blob_idx, blob_idx+1))
                joined_blobs[-1].append(blob_endpoints[blob_idx+1])
            else:
                joined_blobs.append([blob_endpoints[blob_idx+1]])


        # plt.figure(figsize=(5, 5))
        # plt.imshow(im_with_separated_blobs)
        # plt.title("im_with_separated_blobs")

        for joined_blob in joined_blobs:
            im_result = np.zeros_like(im_with_separated_blobs)
            for blob in joined_blob:
                im_result[im_with_separated_blobs == blob[2] + 1] = 255


            h, theta, d = hough_line(im_result, theta=np.linspace(-np.pi / 2, np.pi / 2, 180))
            accums, angles, dists = hough_line_peaks(h, theta, d, min_distance=1, min_angle=1, threshold=1)
            if len(accums) == 0:
                continue
            
            lines = (sorted(zip(accums, angles, dists), key=lambda x: x[0], reverse=True)[:10])

            if len(lines) == 0:
                continue

            accum, angle, dist = lines[0]


            if np.sin(angle) == 0:
                angle += 1e-6
            y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
            y1 = (dist - seg_wire_bin.shape[1] * np.cos(angle)) / np.sin(angle)
            x0 = 0
            x1 = seg_wire_bin.shape[1]
            line = np.array([x0, y0, x1, y1])
            
            line_vec = line[2:] - line[:2]
            line_len = np.linalg.norm(line_vec)
            line_vec = line_vec / np.linalg.norm(line_vec)

            line_points = np.zeros_like(dist_on_skel).astype(np.uint8)
            for i in range(dist_on_skel.shape[0]):
                for j in range(dist_on_skel.shape[1]):
                    if dist_on_skel[i, j] == 0:
                        continue
                    dist_to_line_thresh = 2
            
            
                    dist_to_line = np.abs((y1-y0)*j - (x1-x0)*i + x1*y0 - y1*x0) / np.sqrt((y1-y0)**2 + (x1-x0)**2)
                    
                    # if dist_to_line < dist_to_line_thresh:
                    if dist_to_line < dist_to_line_thresh and abs(signed_angle_distance_half(angle_mask[i, j], angle)) < np.deg2rad(5):
                        line_points[i, j] = 1

            # close_points = np.argwhere(line_points > 0)[:, ::-1]

            # contour_points_vec = close_points - line[:2]
            # dist_along_line = np.dot(contour_points_vec, line_vec)
            prev_min_vec = joined_blob[0][3] - line[:2]
            prev_max_vec = joined_blob[-1][4] - line[:2]
            prev_min_dist = np.dot(prev_min_vec, line_vec)
            prev_max_dist = np.dot(prev_max_vec, line_vec)

            # # get max and min dists
            # prev_max_dist = np.max(dist_along_line)
            # prev_min_dist = np.min(dist_along_line)

            exact_mask_points = np.argwhere(line_points > 0)[:, ::-1]
            
            if len(exact_mask_points) == 0:
                continue
            
            dist_to_line = np.abs(np.cross(exact_mask_points - line[:2], line_vec))
            dist_thresh = 3
            
            exact_mask_points = exact_mask_points[dist_to_line < dist_thresh]
            
            contour_points_vec = exact_mask_points - line[:2]
            dist_along_line = np.dot(contour_points_vec, line_vec)

            max_dist = np.max(dist_along_line)
            min_dist = np.min(dist_along_line)
            
            far_loc = prev_max_dist
            
            while True:
                if far_loc > max_dist:
                    break
                # if there is not a point between far_loc and far_loc + step_size, break
                if np.sum((dist_along_line > far_loc) & (dist_along_line < far_loc + step_size)) == 0:
                    break
                far_loc += step_size
            
            near_loc = prev_min_dist
            
            while True:
                if near_loc < min_dist:
                    break
                if np.sum((dist_along_line < near_loc) & (dist_along_line > near_loc - step_size)) == 0:
                    break
                near_loc -= step_size

            # get locations on line
            max_loc = line[:2] + far_loc * line_vec
            min_loc = line[:2] + near_loc * line_vec

            # plt.figure(figsize=(5,5))
            # plt.imshow(im_result, cmap="gray")
            # plt.colorbar()
            # y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
            # y1 = (dist - seg_wire_bin.shape[1] * np.cos(angle)) / np.sin(angle)
            # plt.plot((0, seg_wire_bin.shape[1]), (y0, y1), '-r', linewidth=0.5)
            # plt.xlim((0, seg_wire_bin.shape[1]))
            # plt.ylim((seg_wire_bin.shape[0], 0))
            # plt.scatter(max_loc[0], max_loc[1], s=50, marker='o')
            # plt.scatter(min_loc[0], min_loc[1], s=50, marker='o')
            
            # stack max and min
            # max_min = np.vstack([max_loc, min_loc])
            # min_max = (min_loc, max_loc)
            # max_min = (max_loc, min_loc)
            
            # blob_endpoints.append((max_loc, min_loc))



            # plt.figure(figsize=(5, 5))
            # plt.imshow(im_result)
            # plt.title("im_result")

            def is_on_edge(point, cropped_width, cropped_height, border_tol_percent):
                border_tol = border_tol_percent * min(cropped_width, cropped_height)
                return point[0] < border_tol or point[0] > cropped_width - border_tol or point[1] < border_tol or point[1] > cropped_height - border_tol
            
            border_tol_percent = 0.05 # for real
            # border_tol_percent = 0.01 # for debug
            cropped_width = seg_wire_bin.shape[1]
            cropped_height = seg_wire_bin.shape[0]

            mid_loc = (max_loc + min_loc) / 2

            max_on_edge = is_on_edge(max_loc, cropped_width, cropped_height, border_tol_percent)
            min_on_edge = is_on_edge(min_loc, cropped_width, cropped_height, border_tol_percent)
            mid_on_edge = is_on_edge(mid_loc, cropped_width, cropped_height, border_tol_percent)

            if max_on_edge and min_on_edge and mid_on_edge:
                continue
            

            if debug:
                
                plt.figure(figsize=(5, 5))
                plt.imshow(im_result)
                # plt.scatter(max_min[:, 0], max_min[:, 1], s=50)
                plt.scatter(max_loc[0], max_loc[1], s=50, marker='o')
                plt.scatter(min_loc[0], min_loc[1], s=50, marker='o')
                # plt.plot((x0, x1), (y0, y1), '-r', linewidth=0.5)

            near_contour_points.append((accum, (min_loc, max_loc)))
        # near_contour_points.append(contour_points[dists < near_cutoff])


    exact_mask_bools = exact_mask > 0
    seg_wire_bin_low_bools = seg_wire_bin_low > 0
    ored_points = np.logical_or(exact_mask_bools, seg_wire_bin_low_bools)
    

    extended_points = []
    for seg_points in near_contour_points:
        step_size = 1
        # extend the endpoint as long as it is on the exact_mask

        line_vec = seg_points[1][1] - seg_points[1][0]
        line_len = np.linalg.norm(line_vec)
        line_vec = line_vec / line_len

        exact_mask_points = np.argwhere(ored_points)[:, ::-1]

        dist_to_line = np.abs(np.cross(exact_mask_points - seg_points[1][0], line_vec))
        dist_thresh = 3

        exact_mask_points = exact_mask_points[dist_to_line < dist_thresh]

        contour_points_vec = exact_mask_points - seg_points[1][0]
        dist_along_line = np.dot(contour_points_vec, line_vec)

        far_loc = line_len

        while True:
            # if there is not a point between far_loc and far_loc + step_size, break
            if np.sum((dist_along_line > far_loc) & (dist_along_line < far_loc + step_size)) == 0:
                break
            far_loc += step_size

        near_loc = 0

        while True:
            if np.sum((dist_along_line < near_loc) & (dist_along_line > near_loc - step_size)) == 0:
                break
            near_loc -= step_size

        # get locations on line
        far_loc = seg_points[1][0] + far_loc * line_vec
        near_loc = seg_points[1][0] + near_loc * line_vec

        def is_on_edge(point, cropped_width, cropped_height, border_tol_percent):
            border_tol = border_tol_percent * min(cropped_width, cropped_height)
            return point[0] < border_tol or point[0] > cropped_width - border_tol or point[1] < border_tol or point[1] > cropped_height - border_tol
        
        border_tol_percent = 0.05 # for real
        # border_tol_percent = 0.01 # for debug
        cropped_width = seg_wire_bin.shape[1]
        cropped_height = seg_wire_bin.shape[0]
        if is_on_edge(far_loc, cropped_width, cropped_height, border_tol_percent):
            far_loc = None

        if is_on_edge(near_loc, cropped_width, cropped_height, border_tol_percent):
            near_loc = None

        # extended_points.append((seg_points[0], np.vstack([near_loc, far_loc])))
        # extended_points.append((seg_points[0], (near_loc, far_loc)))
        extended_points.append((seg_points[1][0], seg_points[1][1], near_loc, far_loc))

        # print(extended_points[-1])



    # plot the contour points as dots
    # if debug:
    #     plt.figure(figsize=(5,5))
    #     plt.imshow(raw_img)

    #     debug_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
    #     # for accum, angle, dist, (accum, near_contour_point) in zip(accums, angles, dists, near_contour_points):
    #     #     print(accum, angle, dist, near_contour_point)
    #     #     plt.scatter(near_contour_point[:, 0], near_contour_point[:, 1], s=50)
    #     for i, (accum, near_contour_point) in enumerate(near_contour_points):
    #         # plt.scatter(near_contour_point[:, 0], near_contour_point[:, 1], s=50)
    #         plt.scatter(near_contour_point[0][0], near_contour_point[0][1], s=50, marker='o', c=debug_colors[i%len(debug_colors)], alpha=0.5)
    #         plt.scatter(near_contour_point[1][0], near_contour_point[1][1], s=50, marker='o', c=debug_colors[i%len(debug_colors)], alpha=0.5)
    #     for i, (accum, near_contour_point) in enumerate(extended_points):
    #         # plt.scatter(near_contour_point[:, 0], near_contour_point[:, 1], s=100, marker='*')
    #         if near_contour_point[0] is not None:
    #             plt.scatter(near_contour_point[0][0], near_contour_point[0][1], s=150, marker='*', c=debug_colors[i%len(debug_colors)], alpha=0.5)
    #         if near_contour_point[1] is not None:
    #             plt.scatter(near_contour_point[1][0], near_contour_point[1][1], s=150, marker='*', c=debug_colors[i%len(debug_colors)], alpha=0.5)

    #     plt.xlim((0, seg_wire_bin.shape[1]))
    #     plt.ylim((seg_wire_bin.shape[0], 0))

    # near_contour_points.sort(key=lambda x: x[0], reverse=True)
    # print(len(extended_points))
    return extended_points

def get_connected_neighborhood(dist_on_skel, neighborhood_mask, i, j, u, v, dist):
    neighborhood_mask[u, v] = 1

    neighbor_coords = [(u+1, v), (u-1, v), (u, v+1), (u, v-1), (u+1, v+1), (u-1, v+1), (u+1, v-1), (u-1, v-1)]

    for x,y in neighbor_coords:
        in_bounds = (i+x-dist >= 0 and i+x-dist < dist_on_skel.shape[0] and j+y-dist >= 0 and j+y-dist < dist_on_skel.shape[1])
        in_neighbor_bounds = (x >= 0 and x < dist*2+1 and y >= 0 and y < dist*2+1)
        if in_bounds and in_neighbor_bounds and neighborhood_mask[x, y] == 0 and dist_on_skel[i+x-dist, j+y-dist] > 0:
            get_connected_neighborhood(dist_on_skel, neighborhood_mask, i, j, x, y, dist)


def skeleton_metrics():
    pass
    # sk_orients = np.zeros((skeleton_mask.shape[0], skeleton_mask.shape[1], 2)) - 1

    # dist = 20
    # for i in range(skeleton_mask.shape[0]):
    #     for j in range(skeleton_mask.shape[1]):
    #         if dist_on_skel[i, j] == 0:
    #             continue
            
    #         neighborhood_mask = np.zeros((dist*2+1, dist*2+1))
    #         get_connected_neighborhood(dist_on_skel, neighborhood_mask, i, j, dist, dist, dist)

    #         # fit ellipse to neighborhood
    #         neighborhood_coords = np.argwhere(neighborhood_mask > 0)
    #         ellipse = cv2.fitEllipse(neighborhood_coords)

    #         # get major axis to minor axis ratio
    #         major_axis = ellipse[1][0]
    #         minor_axis = ellipse[1][1]

    #         sk_orients[i,j,0] = major_axis / minor_axis

    #         # angle
    #         sk_orients[i,j,1] = ellipse[2]

    # plt.figure(figsize=(5, 5))
    # plt.imshow(sk_orients[:, :, 0])
    # plt.colorbar()

    # plt.figure(figsize=(5, 5))
    # plt.imshow(sk_orients[:, :, 1])
    # plt.colorbar()
            