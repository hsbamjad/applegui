import cv2
import os
import re
import argparse
import sys


def create_video_from_frames(input_folder, output_filename, fps=60, lossless=False):
    """
    Converts a folder of image frames into a video file.

    Args:
        input_folder  (str):  Path to folder containing image files.
        output_filename (str): Path where the video will be saved.
        fps (int):  Frames per second for the video (default: 60).
        lossless (bool): If True, writes MJPEG .avi instead of MP4.
                         Recommended for NIR channels — MP4/mp4v is lossy
                         and will alter pixel values used by the YOLO model.
    """
    # 1. Collect image files
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    images = [img for img in os.listdir(input_folder)
              if img.lower().endswith(valid_extensions)]

    if not images:
        print(f"Error: No images found in '{input_folder}'")
        return

    # Natural sort — frame_2.png before frame_10.png
    images.sort(key=lambda f: int(re.sub(r'\D', '', f) or 0))

    print(f"Found {len(images)} images in '{input_folder}'.")
    print(f"First 3 files: {images[:3]}")

    # 2. Read first frame to get dimensions
    frame_path = os.path.join(input_folder, images[0])
    frame = cv2.imread(frame_path)

    if frame is None:
        print(f"Error: Could not read first frame at '{frame_path}'. Check the file.")
        return

    # Handle grayscale NIR frames — OpenCV reads as 2D, VideoWriter needs 3-channel
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    height, width = frame.shape[:2]
    size = (width, height)

    # 3. Choose codec
    if lossless:
        # MJPEG in AVI container — near-lossless, preserves NIR pixel fidelity
        if not output_filename.lower().endswith(".avi"):
            output_filename = os.path.splitext(output_filename)[0] + ".avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        print("Codec: MJPEG (.avi) — lossless mode")
    else:
        # mp4v in MP4 container — smaller file, lossy (fine for Color/RGB preview)
        if not output_filename.lower().endswith(".mp4"):
            output_filename = os.path.splitext(output_filename)[0] + ".mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        print("Codec: mp4v (.mp4) — lossy mode (use --lossless for NIR channels)")

    # 4. Initialize VideoWriter
    out = cv2.VideoWriter(output_filename, fourcc, fps, size)

    if not out.isOpened():
        print(f"Error: VideoWriter failed to open '{output_filename}'. Check codec support.")
        return

    # 5. Write frames
    print(f"Writing {fps} FPS video to: {output_filename}")
    for i, image_name in enumerate(images):
        img_path = os.path.join(input_folder, image_name)
        frame = cv2.imread(img_path)

        if frame is None:
            print(f"Warning: Skipping unreadable frame '{image_name}'")
            continue

        # Convert grayscale to BGR if needed
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        out.write(frame)

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(images)} frames written...")

    out.release()
    print(f"\nDone. Video saved: {output_filename}")
    print(f"Total frames: {len(images)}  |  Duration: {len(images)/fps:.1f}s at {fps} FPS")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a folder of image frames into a video.\n"
                    "Supports CH1 (Color), CH2 (NIR1), CH3 (NIR2) image sequences.\n"
                    "Use --lossless for NIR channels to avoid codec-induced pixel distortion.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--input",  "-i", type=str,
                        help="Path to folder containing image frames.")
    parser.add_argument("--output", "-o", type=str,
                        help="Output video file path (.mp4 or .avi).")
    parser.add_argument("--fps",    "-f", type=int, default=60,
                        help="Playback frame rate (default: 60).")
    parser.add_argument("--lossless", "-l", action="store_true",
                        help="Write MJPEG .avi instead of MP4. "
                             "Recommended for NIR channels (CH2, CH3).")

    args = parser.parse_args()

    # Default paths — update these to point at your apple image sequence folders
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
