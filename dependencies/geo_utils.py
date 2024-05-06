import math
import numpy as np
import cv2

def dist(point1, point2):
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def angle_between(p1, p2, q1, q2):
        P1 = np.array(p1)
        Q1 = np.array(q1)
        P2 = np.array(p2)
        Q2 = np.array(q2)

        v1 = P2 - P1
        v2 = Q2 - Q1
        dprod = np.dot(v1, v2)

        m1 = np.linalg.norm(v1)
        m2 = np.linalg.norm(v2)

        angle_cos = abs(dprod / (m1 * m2))
        angle = np.degrees(np.arccos(np.clip(angle_cos, -1.0, 1.0)))

        return angle if angle <= 180 else 360 - angle

def get_perim_pts(bone_heatmap, threshold = .5, epsilon=0.005):
        perim_pts = []
        bool_mask = bone_heatmap > threshold

        contours, _ = cv2.findContours(
                bool_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
                approx = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)

                for point in approx:
                        perim_pts.append([int(point[0][0]), int(point[0][1])])
        
        return perim_pts

def check_intersection(segment1, segment2):

        p1 = segment1[0]
        q1 = segment1[1]
        p2 = segment2[0]
        q2 = segment2[1]

        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)

        if (o1 != o2 and o3 != o4):
                return True

        if (o1 == 0 and on_segment(p1, p2, q1)):
                return True

        if (o2 == 0 and on_segment(p1, q2, q1)):
                return True

        if (o3 == 0 and on_segment(p2, p1, q2)):
                return True

        if (o4 == 0 and on_segment(p2, q1, q2)):
                return True
        return False

def on_segment(p, q, r):
        if (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])):
                return True
        return False


def distance_point_to_segment(px, py, ax, ay, bx, by):
        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        ab_ap = abx * apx + aby * apy
        ab_ab = abx * abx + aby * aby

        if ab_ab == 0:
                return dist([px, py], [ax, ay])

        t = ab_ap / ab_ab
        t = max(0, min(1, t))

        projx = ax + t * abx
        projy = ay + t * aby

        return dist([px, py], [projx, projy])

def segment_to_segment_distance(s1, s2):
        x1, y1 = s1[0]
        x2, y2 = s1[1]
        x3, y3 = s2[0]
        x4, y4 = s2[1]

        if check_intersection(s1, s2):
                return 0

        dist1 = distance_point_to_segment(x1, y1, x3, y3, x4, y4)
        dist2 = distance_point_to_segment(x2, y2, x3, y3, x4, y4)
        dist3 = distance_point_to_segment(x3, y3, x1, y1, x2, y2)
        dist4 = distance_point_to_segment(x4, y4, x1, y1, x2, y2)

        return min(dist1, dist2, dist3, dist4)

def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
                return 0
        elif val > 0:
                return 1
        else:
                return 2
