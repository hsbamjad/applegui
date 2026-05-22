import cv2
import os
import re
import argparse
import sys
import csv

def create_video_from_frames(input_folder, output_filename, fps=60, lossless=False):
    """
    Converts a folder of image frames into a video.
    
    Args:
        input_folder (str): Path to folder containing image files.
        output_filename (str): Path where the video will be saved.
        fps (int): Frames per second for the video (default: 60).
        lossless (bool): If True, writes MJPEG .avi instead of MP4.
                         Recommended for NIR channels — mp4v is lossy.
    """
    # 1. Get all image files
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    images = [img for img in os.listdir(input_folder) if img.lower().endswith(valid_extensions)]
    
    if not images:
        print(f"Error: No images found in '{input_folder}'")
        return

    # Natural Sorting (e.g., frame_2.png before frame_10.png)
    images.sort(key=lambda f: int(re.sub('\\D', '', f)))

    print(f"Found {len(images)} images in '{input_folder}'.")
    print(f"First 3 files: {images[:3]}")

    # 2. Read the first frame to get dimensions
    frame_path = os.path.join(input_folder, images[0])
    frame = cv2.imread(frame_path)
    
    if frame is None:
        print(f"Error: Could not read first frame at {frame_path}. Check if file is valid.")
        return
        
    height, width, _ = frame.shape
    size = (width, height)

    # 3. Initialize VideoWriter
    if lossless:
        if not output_filename.lower().endswith(".avi"):
            output_filename = os.path.splitext(output_filename)[0] + ".avi"
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        print("Codec: MJPEG (.avi) — lossless mode")
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print("Codec: mp4v (.mp4)")

    out = cv2.VideoWriter(output_filename, fourcc, fps, size)

    # 4. Write frames
    print(f"Writing video to: {output_filename}")
    for i, image_name in enumerate(images):
        img_path = os.path.join(input_folder, image_name)
        frame = cv2.imread(img_path)
        
        if frame is None:
            print(f"Warning: Skipping invalid frame '{image_name}'")
            continue
            
        out.write(frame)
        
        # Progress update every 500 frames
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(images)} frames...")

    out.release()
    print(f"Success! Video saved as: {output_filename}")

    # Write sequence log CSV
    csv_path = os.path.splitext(output_filename)[0] + "_sequence.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "filename"])
        for idx, name in enumerate(images, start=1):
            writer.writerow([idx, name])
    print(f"Sequence log saved as: {csv_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert a folder of image frames into a video.")
    
    parser.add_argument("--input", "-i", type=str, 
                        help="Path to the folder containing image frames.")
    parser.add_argument("--output", "-o", type=str, 
                        help="Path and name for the output video file.")
    parser.add_argument("--fps", "-f", type=int, default=60, 
                        help="Playback frame rate (default: 60).")
    parser.add_argument("--lossless", "-l", action="store_true",
                        help="Write MJPEG .avi instead of MP4. Recommended for NIR channels.")

    args = parser.parse_args()

    # Default paths — update to point at your apple image sequence folders
    default_input  = r"S:\MSU_Research\ASABE AIM26\data\apple_frames\CH1_Color"
    default_output = r"S:\MSU_Research\ASABE AIM26\data\apple_videos\CH1_Color.mp4"

    input_path  = args.input  or default_input
    output_path = args.output or default_output

    if not os.path.exists(input_path):
        print(f"Error: Input directory '{input_path}' does not exist.")
        sys.exit(1)

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    create_video_from_frames(input_path, output_path, fps=args.fps, lossless=args.lossless)

if __name__ == "__main__":
    main()
