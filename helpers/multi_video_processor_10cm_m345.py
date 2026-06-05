"""
Multispectral video processing pipeline for chestnut grading (Experiment 3).
Fuses RGB, Source 1 (NIR1), and Source 2 (NIR2) video streams into 
the 4-channel input [R, NIR1, NIR2, DIFF] expected by the model.
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
from collections import defaultdict
import math
import argparse
import sys

from core.log import get_logger, configure_root
logger = get_logger(__name__)

# Add models directory to path for relative imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'models'))
try:
    from custom_yolo import ChestnutYOLO
except ImportError:
    ChestnutYOLO = None


class MultiSourceVideoProcessor:
    """Process synchronized video streams with multispectral fusion and YOLO tracking"""
    
    def __init__(self, model_path, output_dir, tracker_config="botsort.yaml", ar_alpha=None,
                 enable_spd=True, enable_nir_fusion=True, enable_chestnut_head=True):
        # Load model with architecture flags matching the training config
        if ChestnutYOLO is not None:
            logger.info(f"Loading ChestnutYOLO (SPD={enable_spd}, NIR={enable_nir_fusion}, Head={enable_chestnut_head})")
            model_obj = ChestnutYOLO(
                model_base=model_path,
                enable_spd=enable_spd,
                enable_nir_fusion=enable_nir_fusion,
                enable_chestnut_head=enable_chestnut_head,
                alpha_init=ar_alpha if ar_alpha is not None else 0.1
            )
            self.model_obj = model_obj  # Keep alive so hook closures stay valid
            self.model = model_obj.model
            # Move NIR fusion module to GPU (it lives on the wrapper, not inside YOLO)
            if hasattr(model_obj, 'nir_fusion'):
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model_obj.nir_fusion.to(device)
                logger.info(f"  [OK] NIRDiffFusion moved to {device}")

        else:
            self.model = YOLO(model_path)
            
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.tracker_config = tracker_config
        
        # Get class names
        class_names = self.model.names
        self.defective_class_id = None
        for k, v in class_names.items():
            if 'defective' in v.lower():
                self.defective_class_id = k
                break
        if self.defective_class_id is None:
            self.defective_class_id = 0
            
        logger.info(f"Model loaded. Defective class ID: {self.defective_class_id}")

    def get_next_experiment_folder(self, base_name="multi_video_grading"):
        existing_folders = list(self.output_dir.glob(f"{base_name}_*"))
        existing_numbers = []
        for folder in existing_folders:
            try:
                num = int(folder.name.split('_')[-1])
                existing_numbers.append(num)
            except ValueError:
                continue
        next_num = max(existing_numbers) + 1 if existing_numbers else 1
        exp_folder = self.output_dir / f"{base_name}_{next_num:02d}"
        exp_folder.mkdir(exist_ok=True, parents=True)
        return exp_folder

    def calculate_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def get_global_percentiles_efficient(self, path, samples=150):
        """Pre-scan video to find intensity bounds for consistent normalization"""
        logger.info(f"Analyzing global contrast for {Path(path).name}...")
        cap = cv2.VideoCapture(str(path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - 1, samples, dtype=int)
        
        global_hist = np.zeros(256, dtype=np.float64)
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            global_hist += hist.flatten()
        
        cap.release()
        cdf = np.cumsum(global_hist)
        total_pixels = cdf[-1]
        p_low = np.searchsorted(cdf, 0.005 * total_pixels)
        p_high = np.searchsorted(cdf, 0.995 * total_pixels)
        logger.info(f"   Done. Global bounds: low={p_low}, high={p_high}")
        return p_low, p_high

    def process_multispectral_video(self, 
                                   rgb_path, s1_path, s2_path,
                                   confidence_threshold=0.5,
                                   image_size=640,
                                   defect_weight=1.5,
                                   line_y=None,
                                   line_offset=30,
                                   min_count_dist=100,
                                   count_memory_frames=15,
                                   max_merge_distance=50,
                                   max_merge_frames=10,
                                   hit_threshold=32,
                                   defect_ratio_threshold=0.58):
        
        rgb_path = Path(rgb_path)
        s1_path = Path(s1_path)
        s2_path = Path(s2_path)
        
        logger.info(f"{'='*70}")
        logger.info(f"Fusing Source Streams:")
        logger.info(f"  RGB: {rgb_path.name}")
        logger.info(f"  S1:  {s1_path.name}")
        logger.info(f"  S2:  {s2_path.name}")
        logger.info(f"{'='*70}")
        
        exp_folder = self.get_next_experiment_folder()
        logger.info(f"Output folder: {exp_folder.name}")
        
        # --- GLOBAL PRE-SCAN ---
        # Get stable bounds for NIR2 Source 2 to prevent flicker/bad starts
        p_low, p_high = self.get_global_percentiles_efficient(s2_path)
        scale = 255.0 / (p_high - p_low + 1e-6)
        
        # Open video streams
        cap_rgb = cv2.VideoCapture(str(rgb_path))
        cap_s1 = cv2.VideoCapture(str(s1_path))
        cap_s2 = cv2.VideoCapture(str(s2_path))
        
        width = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap_rgb.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if line_y is None:
            line_y = height // 2
        
        # Output video
        output_video_path = exp_folder / f"{rgb_path.stem}_multispectral_tracked.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        # Tracking & Voting state
        track_history = defaultdict(lambda: {
            'sum_conf_defective': 0.0,
            'sum_conf_normal': 0.0,
            'max_conf_defective': 0.0,
            'hit_count_defective': 0,
            'frames_seen': 0,
            'last_position': (0, 0),
            'last_seen_frame': 0
        })
        lost_tracks_buffer = {}
        tracker_id_to_seq_id = {}
        recently_counted_buffer = []
        current_count = 0
        final_sample_results = {}
        frame_number = 0
        
        while cap_rgb.isOpened():
            ret_rgb, frame_rgb = cap_rgb.read()
            ret_s1, frame_s1 = cap_s1.read()
            ret_s2, frame_s2 = cap_s2.read()
            
            if not ret_rgb or not ret_s1 or not ret_s2:
                break
                
            frame_number += 1
            
            # --- MULTISPECTRAL FUSION ---
            # 1. Extract Red channel (OpenCV BGR -> R is index 2)
            r_ch = frame_rgb[:, :, 2]
            
            # 2. Extract NIR channels (assume grayscale/single channel used)
            nir1 = cv2.cvtColor(frame_s1, cv2.COLOR_BGR2GRAY) if frame_s1.ndim == 3 else frame_s1
            nir2 = cv2.cvtColor(frame_s2, cv2.COLOR_BGR2GRAY) if frame_s2.ndim == 3 else frame_s2
            
            # 3. Enhanced: Guarded Min-Max Stretching (Matches Training Exactly)
            # This restores the 90%+ logic while preventing noise in empty frames.
            # Only stretch if there is actual content (intensity range > 10).
            min_val, max_val = np.min(nir2), np.max(nir2)
            if (max_val - min_val) > 10:
                nir2_stretched = cv2.normalize(nir2, None, 0, 255, cv2.NORM_MINMAX)
            else:
                # Frame is too flat/dark; just multiply by a safe factor or leave it
                nir2_stretched = (nir2.astype(np.float32) * (255.0 / 64.0)).clip(0, 255).astype(np.uint8)
            
            # 4. Calculate DIFF: clip(NIR1 - NIR2, 0, 255)
            diff = np.clip(nir1.astype(np.int16) - nir2_stretched.astype(np.int16), 0, 255).astype(np.uint8)
            
            # 5. Stack 4 channels: [R, NIR1, NIR2, DIFF]
            stack = np.dstack([r_ch, nir1, nir2_stretched, diff])
            
            # --- YOLO TRACKING ---
            # Inference on the 4-channel stack
            results = self.model.track(
                stack, 
                tracker=self.tracker_config,
                conf=confidence_threshold,
                imgsz=image_size,
                persist=True,
                verbose=False,
                retina_masks=True
            )
            
            annotated_frame = frame_rgb.copy()
            
            # Counting lines
            cv2.line(annotated_frame, (0, line_y - line_offset), (width, line_y - line_offset), (255, 255, 0), 1)
            cv2.line(annotated_frame, (0, line_y + line_offset), (width, line_y + line_offset), (255, 255, 0), 1)
            cv2.line(annotated_frame, (0, line_y), (width, line_y), (255, 0, 0), 2)
            
            current_frame_ids = []
            current_frame_detections = []
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.cpu().numpy()
                masks = results[0].masks
                mask_points_list = masks.xy if masks is not None else [None] * len(boxes)
                
                for b_id, b_cls, b_xyxy, b_conf, mask_poly in zip(
                    boxes.id, boxes.cls, boxes.xyxy, boxes.conf, mask_points_list
                ):
                    track_id = int(b_id)
                    class_id = int(b_cls)
                    conf = float(b_conf)
                    x1, y1, x2, y2 = map(int, b_xyxy)
                    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    
                    current_frame_ids.append(track_id)
                    current_frame_detections.append({
                        'id': track_id, 'cls': class_id, 'conf': conf,
                        'box': (x1, y1, x2, y2), 'center': center, 'mask': mask_poly
                    })

            # ID Recovery (simplified from video_processor.py)
            new_tracks = [d for d in current_frame_detections if track_history[d['id']]['frames_seen'] == 0]
            for new_track in new_tracks:
                new_id, new_pos = new_track['id'], new_track['center']
                best_match_id = None
                min_dist = float('inf')
                for lost_id, lost_data in list(lost_tracks_buffer.items()):
                    if frame_number - lost_data['last_seen_frame'] > max_merge_frames:
                        del lost_tracks_buffer[lost_id]
                        continue
                    dist = self.calculate_distance(new_pos, lost_data['last_position'])
                    if dist < max_merge_distance and dist < min_dist:
                        min_dist, best_match_id = dist, lost_id
                if best_match_id is not None:
                    track_history[new_id] = lost_tracks_buffer[best_match_id].copy()
                    if best_match_id in tracker_id_to_seq_id:
                        tracker_id_to_seq_id[new_id] = tracker_id_to_seq_id[best_match_id]
                    del lost_tracks_buffer[best_match_id]

            # Process Detections
            for data in current_frame_detections:
                track_id, class_id, conf, center = data['id'], data['cls'], data['conf'], data['center']
                x1, y1, x2, y2 = data['box']
                history = track_history[track_id]
                history['last_position'], history['last_seen_frame'] = center, frame_number
                history['frames_seen'] += 1
                
                if class_id == self.defective_class_id:
                    history['sum_conf_defective'] += (conf * defect_weight)
                    history['max_conf_defective'] = max(history['max_conf_defective'], conf)
                    if conf > 0.6: # High confidence hit
                        history['hit_count_defective'] += 1
                else:
                    history['sum_conf_normal'] += conf
                
                # Advanced Voting: Base Ratio + Temporal Consistency
                # A nut is defective if:
                # 1. The weighted average favors defective (defect_ratio > threshold)
                # 2. OR sustained "High Confidence" evidence (hit_count > threshold)
                #    (Default 32 frames = ~25% of sequence, balances recall vs glare)
                sum_conf = history['sum_conf_defective'] + history['sum_conf_normal']
                defect_ratio = history['sum_conf_defective'] / sum_conf if sum_conf > 0 else 0.0
                
                is_defective = (defect_ratio > defect_ratio_threshold) or \
                               (history['hit_count_defective'] >= hit_threshold)

                # Counting Logic
                if (line_y - line_offset) < center[1] < (line_y + line_offset):
                    if track_id not in tracker_id_to_seq_id:
                        is_duplicate = False
                        matched_seq_id = None
                        for recent in recently_counted_buffer:
                            if self.calculate_distance(center, recent['pos']) < min_count_dist:
                                is_duplicate, matched_seq_id = True, recent['seq_id']
                                break
                        if is_duplicate:
                            tracker_id_to_seq_id[track_id] = matched_seq_id
                        else:
                            current_count += 1
                            tracker_id_to_seq_id[track_id] = current_count
                            recently_counted_buffer.append({'pos': center, 'frame': frame_number, 'seq_id': current_count})
                            cv2.line(annotated_frame, (0, line_y), (width, line_y), (0, 255, 0), 4)

                # Grading
                display_id = tracker_id_to_seq_id.get(track_id, "?")
                color = (0, 0, 255) if is_defective else (0, 255, 0)
                label = f"#{display_id} {'DEFECTIVE' if is_defective else 'Normal'} ({max(defect_ratio, 1-defect_ratio):.0%})"
                
                if track_id in tracker_id_to_seq_id:
                    final_sample_results[tracker_id_to_seq_id[track_id]] = {
                        'prediction': 'defective' if is_defective else 'normal',
                        'confidence': history['max_conf_defective'] if is_defective else (1.0 - defect_ratio),
                        'num_detections': history['frames_seen']
                    }
                
                # Drawing
                if data['mask'] is not None:
                    cnt = data['mask'].astype(np.int32)
                    overlay = annotated_frame.copy()
                    cv2.fillPoly(overlay, [cnt], color)
                    cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
                    cv2.polylines(annotated_frame, [cnt], True, color, 1)
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Clean memory
            recently_counted_buffer = [x for x in recently_counted_buffer if (frame_number - x['frame']) < count_memory_frames]
            
            cv2.putText(annotated_frame, f"Count: {current_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            out.write(annotated_frame)
            if frame_number % 20 == 0:
                logger.info(f"  Frame {frame_number}/{total_frames} | Count: {current_count}")
        
        cap_rgb.release()
        cap_s1.release()
        cap_s2.release()
        out.release()
        
        # Save CSV
        csv_path = exp_folder / f"{rgb_path.stem}_multispectral_predictions.csv"
        csv_data = []
        for seq_id in sorted(final_sample_results.keys()):
            res = final_sample_results[seq_id]
            csv_data.append({'nut_id': seq_id, 'prediction': res['prediction'], 'conf': f"{res['confidence']:.4f}", 'frames': res['num_detections']})
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)
        
        logger.info(f"[OK] Finished! Results saved to: {exp_folder}")
        return output_video_path, csv_path


def main():
    configure_root()

    parser = argparse.ArgumentParser(description='Multispectral Video Processor (Ablation Study)')
    parser.add_argument('--rgb', required=True, help='Path to RGB video')
    parser.add_argument('--s1', required=True, help='Path to Source 1 (NIR1) video')
    parser.add_argument('--s2', required=True, help='Path to Source 2 (NIR2) video')
    parser.add_argument('--model', required=True, help='Path to model (.pt)')
    parser.add_argument('--output', default='results/multi_video_processing', help='Output base directory')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--defect-weight', type=float, default=1.5, help='Weight for defective class')
    parser.add_argument('--line-y', type=int, default=None, help='Y line for counting')
    parser.add_argument('--hit-threshold', type=int, default=32, help='Frames required for hit-based defective classification')
    parser.add_argument('--ratio-threshold', type=float, default=0.58, help='Defect ratio threshold for voting')
    # Architecture flags (MUST match training config)
    parser.add_argument('--enable-spd', action='store_true', help='Enable SPD-Conv (Model 4, 5)')
    parser.add_argument('--enable-nir-fusion', action='store_true', help='Enable NIR-Diff Fusion (Model 3, 4, 5)')
    parser.add_argument('--enable-chestnut-head', action='store_true', help='Enable ChestnutHead AR prior (Model 5 only)')
    
    args = parser.parse_args()
    
    processor = MultiSourceVideoProcessor(
        args.model, args.output,
        enable_spd=args.enable_spd,
        enable_nir_fusion=args.enable_nir_fusion,
        enable_chestnut_head=args.enable_chestnut_head
    )
    processor.process_multispectral_video(
        args.rgb, args.s1, args.s2,
        confidence_threshold=args.conf,
        defect_weight=args.defect_weight,
        line_y=args.line_y,
        hit_threshold=args.hit_threshold,
        defect_ratio_threshold=args.ratio_threshold
    )

if __name__ == '__main__':
    main()
