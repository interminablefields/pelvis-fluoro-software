"""Utils for handling heatmaps/logits."""
import numpy as np
import torch
from scipy import ndimage
from typing import Optional, Union, overload
import logging
import killeengeo as kg
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor

log = logging.getLogger(__name__)


@overload
def get_heatmap_threshold(h: np.ndarray, fraction: float) -> float:
    ...


@overload
def get_heatmap_threshold(h: torch.Tensor, fraction: float) -> torch.Tensor:
    ...


def get_heatmap_threshold(h: np.ndarray, fraction=0.5) -> float:
    """Get the threshold for a heatmap.

    Args:
        h (np.ndarray): A 2D array
        fraction (float): Fraction of the heatmap range to set the threshold at. Higher values keeps fewer pixels.

    """
    hmin = h.min()
    hmax = h.max()
    return hmin + (hmax - hmin) * fraction


def detect_landmark(
    heatmap: np.ndarray,
    rel_threshold: float = 0.5,
) -> np.ndarray:
    """Detect a landmark in the (single) heatmap.

    Args:
        heatmap (np.ndarray): A 2D array
        threshold (float, optional): The threshold to use. Defaults to None.

    Returns:
        np.ndarray: The xy coordinates of the landmark, or [-1, -1] if no landmark was found.

    """
    if len(heatmap.shape) != 2:
        raise ValueError("Expected a 2D heatmap. Got shape: {}".format(heatmap.shape))

    threshold = get_heatmap_threshold(
        heatmap,
        fraction=rel_threshold,
    )
    indices = heatmap > threshold
    if not indices.any():
        return np.array([-1, -1], dtype=np.float32)

    heatmap_over_threshold = np.where(indices, heatmap, 0)
    labels, num_objects = ndimage.label(heatmap_over_threshold)
    centers = ndimage.center_of_mass(
        heatmap_over_threshold, labels=labels, index=range(1, num_objects + 1)
    )
    if isinstance(centers, tuple):
        return np.array([centers[1], centers[0]], dtype=np.float32)

    elif isinstance(centers, list) and len(centers) == 1:
        c = centers[0]
        return np.array([c[1], c[0]], dtype=np.float32)

    elif isinstance(centers, list) and len(centers) > 0:
        if len(centers) > 30:
            log.debug("too many landmarks (model likely not trained)")
            return np.array([-1, -1], dtype=np.float32)
        log.debug(f"Found multiple landmarks: {centers}")
        cs = np.array(centers)
        max_c = None
        max_value = None
        for c in cs:
            i, j = c.astype(int)
            if (0 <= i < heatmap.shape[0]) and (0 <= j < heatmap.shape[1]):
                value = heatmap[i, j]
                if max_value is None or value > max_value:
                    max_value = value
                    max_c = c
        return np.array([max_c[1], max_c[0]], dtype=np.float32)
    else:
        return np.array([-1, -1], dtype=np.float32)


def detect_landmarks(
    heatmaps: np.ndarray,
    order: str = "xy",
    rel_threshold: float = 0.5,
) -> np.ndarray:
    """Detect landmarks in heatmaps.

    Args:
        heatmaps (np.ndarray): A 3D array of shape `[N, H, W]`.
        order (str, optional): The format of the output. Can be "ij" or "xy". Defaults to "ij".

    Returns:
        np.ndarray: An array of shape `[N, 2]` with the coordinates of the landmarks or -1 if not found.
    """

    num_keypoints, H, W = heatmaps.shape
    keypoint_preds = -1 * np.ones((num_keypoints, 2), dtype=np.float32)
    for k in range(num_keypoints):
        heatmap = heatmaps[k]
        kpt = detect_landmark(heatmap, rel_threshold=rel_threshold)
        if order == "xy":
            keypoint_preds[k] = kpt
        elif order == "ij":
            keypoint_preds[k] = np.array([kpt[1], kpt[0]])
        else:
            raise ValueError(f"Invalid order: {order}")

    return keypoint_preds


def detect_line(
    heatmap: np.ndarray,
    threshold: Optional[float] = None,
) -> Optional[kg.Line2D]:
    """Fit a 2D line to the heatmap.

    Args:
        heatmap: The heatmap to fit the line to.

    Returns:


    """
    if threshold is None:
        threshold = get_heatmap_threshold(heatmap, fraction=0.5)
    rr, cc = np.where(heatmap > threshold)
    if len(rr) < 10:
        return None

    points = np.stack([cc, rr], axis=1)

    # TODO: potential problem when line is vertical.
    model = RANSACRegressor(LinearRegression())
    try:
        model.fit(
            points[:, 0, np.newaxis], points[:, 1], np.clip(heatmap[rr, cc], 0, None)
        )
    except ValueError:
        return None
    except ZeroDivisionError:
        return None

    # y = m x + b -> -m x + y - b = 0
    m = model.estimator_.coef_[0]
    b = model.estimator_.intercept_
    l = kg.line(-m, 1, -b)

    return l


def detect_corridor(
    heatmap: np.ndarray,
    threshold: Optional[float] = None,
    step: float = 1,
) -> Optional[kg.Segment2D]:
    """Detect a corridor in the heatmap.

    Args:
        heatmap (np.ndarray): A 2D array of shape `[H, W]`.
        order (str, optional): The format of the output. Can be "ij" or "xy". Defaults to "ij".

    Returns:
        np.ndarray: An array of shape `[N, 2]` with the coordinates of the landmarks or -1 if not found.
    """

    threshold = get_heatmap_threshold(heatmap, fraction=0.5)
    l = detect_line(heatmap, threshold=threshold)
    if l is None:
        return None

    seg = heatmap > threshold

    # Get the largest contiguous section of seg
    labels, num_objects = ndimage.label(seg)
    if num_objects == 0:
        return None
    if num_objects == 1:
        label = 1
    else:
        label = np.bincount(labels.flat)[1:].argmax() + 1

    seg = labels == label

    h, w = heatmap.shape
    # Get the segments of the edge of the image
    left = kg.segment(0, 0, 0, h)
    right = kg.segment(w, 0, w, h)
    top = kg.segment(0, 0, w, 0)
    bottom = kg.segment(0, h, w, h)

    # Get the intersection of the line with the edge of the image
    def _meet(l_: kg.Line2D, s_: kg.Segment2D) -> kg.Point2D | None:
        try:
            return l_.meet(s_)
        except kg.MeetError:
            return None

    points = [_meet(l, left), _meet(l, right), _meet(l, top), _meet(l, bottom)]
    points = [p for p in points if p is not None]
    if len(points) != 2:
        return None

    a, b = points
    v = (b - a).hat()

    # Walk from a to b along the line and find the first point that is in the seg.
    for t in np.arange(0, (a - b).norm(), step):
        p = a + v * float(t)
        x, y = int(p.x), int(p.y)
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        if seg[y, x]:
            break
    else:
        return None

    p1 = p

    # Walk from b to a along the line and find the first point that is in the seg.
    for t in np.arange(0, (b - a).norm(), step):
        p = b - v * float(t)
        x, y = int(p.x), int(p.y)
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        if seg[y, x]:
            break
    else:
        return None

    p2 = p

    return kg.segment(p1, p2)

    # TODO: get start and endpoints by sampling along the line in the image
    # and then getting the first and last point that are on the line.
    raise NotImplementedError

    return None
