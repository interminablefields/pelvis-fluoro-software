import numpy as np
import killeengeo as kg
from typing import TypeVar, Optional
from perphix.data import PerphixBase
import heatmap_utils


def detect_landmarks(
    logits, rel_threshold: float = 0.5
) -> tuple[np.ndarray, list[str]]:
    """Pelvis-specific version of landmark detection.

    Performs checks after doing regular detection that each landmark is on a valid segmentation.

    Args:
        logits (dict[str, np.ndarray]): The logits. Logits containing "heatmap_" will be used for landmark detection.
            "seg_" logits will be used to check that the landmarks are on a valid segmentation. Primarily "hip_left" and "hip_right".

    Returns:
        tuple[np.ndarray, list[str]]: The landmarks and their names.

    """

    keypoint_names: list[str] = []
    heatmaps = []
    for name, heatmap in logits.items():
        if name.startswith("heatmap_"):
            keypoint_names.append(name[len("heatmap_") :])
            heatmaps.append(heatmap)

    heatmaps = np.stack(heatmaps, axis=0)
    keypoints = heatmap_utils.detect_landmarks(heatmaps, rel_threshold=rel_threshold)

    h, w = heatmaps.shape[1:]
    out_keypoints = []
    out_names = []
    for kpt_name, (x, y) in zip(keypoint_names, keypoints):
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        seg_name = "seg_hip_" + ("left" if kpt_name.lower().startswith("l_") else "right")
        seg = logits[seg_name]
        if seg[int(y), int(x)] < 0.5:
            continue
        out_keypoints.append([x, y])
        out_names.append(kpt_name)

    out_keypoints = np.array(out_keypoints)

    return out_keypoints, out_names


def get_landmarks(logits) -> dict[str, kg.Point2D]:
    """Get the landmarks from the logits.

    Args:
        logits (dict[str, np.ndarray]): The logits. Logits containing "heatmap_" will be used for landmark detection.
            "seg_" logits will be used to check that the landmarks are on a valid segmentation. Primarily "hip_left" and "hip_right".

    Returns:
        dict[str, kg.Point2D]: The landmarks.

    """
    keypoints, keypoint_names = detect_landmarks(logits)
    return {name: kg.p(kpt) for name, kpt in zip(keypoint_names, keypoints)}


S = TypeVar("S", bound=kg.Segment)


def _maybe_flip(segment: S, landmarks: dict[str, kg.Point2D], priority: list[str]) -> S:
    """Get the endpoint of segment that is closest to the first landmark in `priority`.

    If none of the landmarks in priority are not present in `landmarks`, the segment is returned unchanged.
    """

    for name in priority:
        try:
            p = kg.p(landmarks[name])
        except KeyError:
            continue
        if (segment.p - p).norm() < (segment.q - p).norm():
            return segment
        else:
            return kg.segment(segment.q, segment.p)
    return segment


def detect_corridor(
    logits: dict[str, np.ndarray], corridor_type: str, landmarks: dict[str, kg.Point2D]
) -> Optional[kg.Segment2D]:
    """Detect the corridor from the logits."""
    heatmap = logits[f"seg_{corridor_type}"]
    s = heatmap_utils.detect_corridor(heatmap)
    if s is None:
        return None

    # TODO: figure out whether to flip the corridor's endpoints, which are currently arbitrary. This
    # should be consistent for each corridor. For example, the ramus corridors should always have
    # the pubic point first, and the teardrops should have the anterior point first, etc. Easy to do
    # based on the keypoints, if they are available.

    # This is only really effective when taking a nearly perpendicular shot.
    # Otherwise, it cannot be relied on. BUT it will be necessary when we triangulate the corridor
    # from multiple shots, because then they can be compared in 3D to the atlas points.
    match str(corridor_type).lower():
        case "ramus_left":
            s = _maybe_flip(s, landmarks, ["l_sps", "l_ips", "l_mof", "r_sps", "r_ips", "r_mof"])
        case "ramus_right":
            s = _maybe_flip(s, landmarks, ["r_sps", "r_ips", "r_mof", "l_sps", "l_ips", "l_mof"])
        case "teardrop_left":
            s = _maybe_flip(s, landmarks, ["l_asis"])
        case "teardrop_right":
            s = _maybe_flip(s, landmarks, ["r_asis"])
        case "s1_left" | "s1" | "s2":
            s = _maybe_flip(s, landmarks, ["l_gsn"])
        case "s1_right":
            s = _maybe_flip(s, landmarks, ["r_gsn"])
        case _:
            pass

    return s


def get_corridors_and_landmarks(
    logits: dict[str, np.ndarray]
) -> tuple[dict[str, kg.Segment2D], dict[str, kg.Point2D]]:
    """Get the corridors from the logits."""
    landmarks = get_landmarks(logits)
    corridors = {
        corridor_type: detect_corridor(logits, corridor_type, landmarks)
        for corridor_type in PerphixBase.corridor_names
    }
    return corridors, landmarks
