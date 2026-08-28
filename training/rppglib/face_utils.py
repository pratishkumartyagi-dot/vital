import numpy as np
import cv2
from scipy.spatial import ConvexHull
import face_alignment

NUM_LANDMARKS = 68

class FaceAligmentExtractor:
    def __init__(self, device='cpu'):
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device)
        self.prev_preds = np.zeros((68, 2), dtype='int16')
        
    def process_frame(self, frame):
        preds = self.fa.get_landmarks(frame)
        if preds is None:
            preds = self.prev_preds
        else:
            preds = preds[0].astype('int16')
        self.prev_preds = preds
        return preds

def detect_landmarks(video, device='cpu'):
    lm_detector = FaceAligmentExtractor(device=device)
    points = list()
    for frame in video:
        lms = lm_detector.process_frame(frame)
        points.append(lms)
    points = np.array(points)
    return points

def find_bbox(landmarks):
    x_min = landmarks[:, :, 0].min()
    x_max = landmarks[:, :, 0].max()
    y_min = landmarks[:, :, 1].min()
    y_max = landmarks[:, :, 1].max()
    return x_min, x_max, y_min, y_max

def crop_video(video, bbox):
    x_min, x_max, y_min, y_max = bbox
    video = video[:, y_min: y_max, x_min: x_max]
    return video

def crop_landmarks(landmarks, bbox):
    x_min, x_max, y_min, y_max = bbox
    landmarks[:, :, 0] -= x_min
    landmarks[:, :, 1] -= y_min
    return landmarks

def get_convex_points(points):
    hull = ConvexHull(points)
    convex_points = points[hull.vertices].astype('int32')
    return convex_points

def get_mask(frame, poly_points):
    H, W = frame.shape[:2]
    mask = np.zeros((H, W), dtype='uint8')
    polys = [poly_points, ]
    cv2.fillPoly(mask, polys, 255)
    mask = mask > 125
    return mask

def process_video(video, device='cpu'):
    video = video.copy()
    landmarks = detect_landmarks(video, device=device)
    bbox = find_bbox(landmarks)
    video = crop_video(video, bbox)
    landmarks = crop_landmarks(landmarks, bbox)

    for i in range(video.shape[0]):
        points = landmarks[i]
        frame = video[i]
        convex = get_convex_points(points)
        mask = get_mask(frame, convex)
        video[i] *= mask[:, :, None]
    
    return video, landmarks

def crop_face_video(video):
    gray = np.mean(video, axis=3)
    mask = gray > 0
    y_idx = np.where(mask.any(axis=1))[0]
    x_idx = np.where(mask.any(axis=0))[0]
    if len(y_idx) == 0 or len(x_idx) == 0:
        return video
    y_min, y_max = y_idx[0], y_idx[-1]
    x_min, x_max = x_idx[0], x_idx[-1]
    return video[:, y_min:y_max+1, x_min:x_max+1]
