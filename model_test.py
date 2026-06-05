import os
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
import re
import time

from core.log import get_logger, configure_root

logger = get_logger(__name__)

# ================= Parameter Configuration =================
model_path = r'C:/Users/tommy/OneDrive - Michigan State University/data/system_integration/training_data/2025_9_30/model/rg-nir1/grading/weights/best.engine'
image_folder = r"C:/Yuyuan/program_test/rg-nir-G1/"
img_size = [768, 1024]
conf_thres = 0.5
iou_thres = 0.3
max_missing = 15


# ================= Helper Functions =================
def compute_iou_matrix(boxesA, boxesB):
    """Compute IOU matrix"""
    iou_matrix = np.zeros((len(boxesA), len(boxesB)), dtype=np.float32)
    for i, boxA in enumerate(boxesA):
        for j, boxB in enumerate(boxesB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            if interArea == 0:
                iou = 0.0
            else:
                boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
                boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
                iou = interArea / float(boxAArea + boxBArea - interArea)
            iou_matrix[i, j] = iou
    return iou_matrix


def natural_sort_key(s):
    """Sort files in natural numeric order"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


# ================= Single Object Tracker =================
class TrackedObject:
    def __init__(self, object_id, bbox, center, prev_center=None, dt=1.0):
        self.id = object_id
        self.bbox = bbox
        self.center = center
        self.missing_frames = 0
        self.age = 0
        self.confirmed = False
        self.active = True  # whether still in frame
        self.trajectory = [center]

        # === Kalman Filter ===
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.05

        if prev_center is not None:
            vx = (center[0] - prev_center[0]) / dt
            vy = (center[1] - prev_center[1]) / dt
        else:
            vx, vy = 0, 0
        self.kf.statePost = np.array([center[0], center[1], vx, vy], dtype=np.float32)

    def predict(self):
        pred = self.kf.predict()
        return (int(pred[0]), int(pred[1]))

    def update(self, center, max_speed=60):
        pred = self.predict()
        dist = np.linalg.norm(np.array(center) - np.array(pred))
        if dist > max_speed:
            center = (0.7 * np.array(center) + 0.3 * np.array(pred)).tolist()

        measurement = np.array(center, dtype=np.float32)
        self.kf.correct(measurement)

        self.center = center
        self.missing_frames = 0
        self.age += 1
        if self.age >= 3:
            self.confirmed = True

        smooth_center = (
            int(0.6 * self.center[0] + 0.4 * pred[0]),
            int(0.6 * self.center[1] + 0.4 * pred[1])
        )
        self.trajectory.append(smooth_center)


# ================= Multi-Object Manager =================
class TrackerManager:
    def __init__(self, img_width, max_missing=8, iou_threshold=0.5):
        self.trackers = []
        self.next_id = 0
        self.max_missing = max_missing
        self.iou_threshold = iou_threshold
        self.img_width = img_width

    def update(self, detections):
        if len(detections) == 0:
            for t in self.trackers:
                t.missing_frames += 1
            self.trackers = [t for t in self.trackers if t.missing_frames < self.max_missing]
            return

        detection_centers = [((x1 + x2)/2, (y1 + y2)/2) for (x1, y1, x2, y2) in detections]

        # === No existing trackers: create new ===
        if len(self.trackers) == 0:
            for i, c in enumerate(detection_centers):
                # Only track objects entering from the left
                if c[0] < self.img_width * 0.15:
                    self.trackers.append(TrackedObject(self.next_id, detections[i], c))
                    self.next_id += 1
            return

        # === Build cost matrix (IOU + center distance) ===
        cost_matrix = np.zeros((len(self.trackers), len(detections)))
        for i, tracker in enumerate(self.trackers):
            pred_center = tracker.predict()
            for j, det in enumerate(detections):
                det_center = detection_centers[j]
                iou = compute_iou_matrix([tracker.bbox], [det])[0, 0]
                dist = np.linalg.norm(np.array(pred_center) - np.array(det_center))
                dist_norm = np.clip(dist / 150, 0, 1)
                combined_score = 0.85 * iou + 0.15 * (1 - dist_norm)
                cost_matrix[i, j] = 1 - combined_score

        matched_row, matched_col = linear_sum_assignment(cost_matrix)
        matched_indices = []
        matched_tracker_indices = set()

        for i, j in zip(matched_row, matched_col):
            if cost_matrix[i, j] < (1 - self.iou_threshold):
                self.trackers[i].update(detection_centers[j])
                self.trackers[i].bbox = detections[j]
                matched_indices.append(j)
                matched_tracker_indices.add(i)
            else:
                self.trackers[i].missing_frames += 1

        # === Unmatched trackers ===
        unmatched_trackers = set(range(len(self.trackers))) - matched_tracker_indices
        for i in unmatched_trackers:
            self.trackers[i].missing_frames += 1

        # === New objects only enter from left ===
        unmatched_detections = set(range(len(detections))) - set(matched_indices)
        for j in unmatched_detections:
            c = detection_centers[j]
            if c[0] < self.img_width * 0.15:  # enter from left
                self.trackers.append(TrackedObject(self.next_id, detections[j], c))
                self.next_id += 1

        # === Remove objects that exited right edge ===
        for t in self.trackers:
            if t.center[0] > self.img_width * 0.99:  # beyond right boundary
                t.active = False
        self.trackers = [t for t in self.trackers if t.missing_frames < self.max_missing and t.active]


# ================= Main Program =================
if __name__ == '__main__':
    configure_root()

    logger.info("Loading YOLO model...")
    model = YOLO(model_path, task="segment")

    tracker = TrackerManager(
        img_width=img_size[1],
        max_missing=max_missing,
        iou_threshold=iou_thres
    )

    exts = (".bmp", ".jpg", ".jpeg", ".png", ".tif")
    image_files = sorted(
        [f for f in os.listdir(image_folder) if f.lower().endswith(exts)],
        key=natural_sort_key
    )

    if not image_files:
        raise FileNotFoundError(f"No images found in {image_folder}")

    logger.info(f"Found {len(image_files)} images to process.")
    start = time.time()

    for idx, filename in enumerate(image_files, 1):
        image_path = os.path.join(image_folder, filename)
        frame = cv2.imread(image_path)
        img = cv2.resize(frame, (1024, 768))
        if img is None:
            logger.warning(f"Cannot load image: {filename}")
            continue

        results = model(
            source=img,
            imgsz=img_size,
            conf=conf_thres,
            iou=0.7,
            verbose=False,
            save=False,
            show=False,
            device=0
        )

        detections = []
        masks_bin = []
        for r in results:
            masks = getattr(r, "masks", None)
            mask_data = masks.data.cpu().numpy() if masks is not None else []
            for i, box in enumerate(r.boxes):
                if box.conf < conf_thres:
                    continue
                xmin, ymin, xmax, ymax = map(int, box.xyxy.cpu().numpy()[0])
                conf = float(box.conf.cpu().numpy())
                cls = int(box.cls.cpu().numpy())
                detections.append([xmin, ymin, xmax, ymax, conf, cls])
                # masks_bin.append((mask * 255).astype("uint8"))
                if masks is not None and i < len(mask_data):
                    mask_bin = (mask_data[i] > 0.5).astype("uint8")  # binarize
                    masks_bin.append(mask_bin)
                else:
                    masks_bin.append(None)

        tracker.update([d[:4] for d in detections])

        # annotated = img.copy()
        for obj, det, mask in zip(tracker.trackers, detections, masks_bin):
            if not obj.confirmed:
                continue
            x1, y1, x2, y2 = map(int, obj.bbox)
            conf = float(det[4])
            cls = int(det[5])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'ID:{obj.id} Class:{cls}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # if mask is not None:
            #     color = (0, 255, 0)
            #     mask_colored = np.zeros_like(img, dtype=np.uint8)
            #     mask_colored[:, :, 1] = mask * 255  # G channel
            #     img = cv2.addWeighted(img, 1.0, mask_colored, 0.5, 0)

        cv2.imshow("YOLO + Directional Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    total_time = time.time() - start
    cv2.destroyAllWindows()
    logger.info(f"Tracking finished! Total time: {total_time:.3f}s")
