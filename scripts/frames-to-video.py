import cv2
import os
import re
import argparse
import sys


def create_video_from_frames(input_folder, output_filename, fps=60, lossless=False):
    """
    Converts a folder of image frames into a video.

    Args:
        input_folder (str): Path to folder containing image files.
        output_filename (str): Path where the video will be saved.
        fps (int): Frames per second for the video (default: 60).
        lossless (bool): If True, writes MJPEG .avi instead of MP4.
                         Recommended for NIR channels -- mp4v is lossy.
    """
    # 1. Get all image files
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    images = [img for img in os.listdir(input_folder) if img.lower().endswith(valid_extensions)]

    if not images:
        print(f"  Warning: No images found in '{input_folder}', skipping.")
        return

    # Natural Sorting (e.g., frame_2.png before frame_10.png)
    images.sort(key=lambda f: int(re.sub('\\D', '', f)))

    print(f"  Found {len(images)} images.")

    # 2. Read the first frame to get dimensions
    frame_path = os.path.join(input_folder, images[0])
    frame = cv2.imread(frame_path)

    if frame is None:
        print(f"  Error: Could not read first frame at {frame_path}. Skipping.")
        return

    height, width, _ = frame.shape
    size = (width, height)

    # 3. Initialize VideoWriter
    if lossless:
        if not output_filename.lower().endswith(".avi"):
            output_filename = os.path.splitext(output_filename)[0] + ".avi"
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_filename, fourcc, fps, size)

    # 4. Write frames
    for i, image_name in enumerate(images):
        img_path = os.path.join(input_folder, image_name)
        frame = cv2.imread(img_path)

        if frame is None:
            print(f"  Warning: Skipping invalid frame '{image_name}'")
            continue

        out.write(frame)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(images)} frames...")

    out.release()
    print(f"  Saved: {output_filename}")


def batch_process(source_root, output_root, fps=60, lossless=False):
    """
    Loops through all subfolders (G1, G2 ... Gn) in source_root,
    creates a video for each, and saves to output_root/GroupName/GroupName.mp4
    """
    ext = ".avi" if lossless else ".mp4"

    # Find all subfolders, sorted naturally
    groups = sorted(
        [d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))],
        key=lambda f: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', f)]
    )

    if not groups:
        print(f"Error: No subfolders found in '{source_root}'")
        sys.exit(1)

    print(f"Found {len(groups)} groups: {groups}\n")

    for group in groups:
        input_folder  = os.path.join(source_root, group)
        output_folder = os.path.join(output_root, group)
        os.makedirs(output_folder, exist_ok=True)
        output_file   = os.path.join(output_folder, group + ext)

        print(f"[{group}]  {input_folder}")
        create_video_from_frames(input_folder, output_file, fps=fps, lossless=lossless)

    print(f"\nAll done. Videos saved under: {output_root}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert image frames to video. Single folder or batch (all subgroups)."
    )

    parser.add_argument("--input",  "-i", type=str, required=True,
                        help="Input folder. In batch mode, this is the root containing G1/G2... subfolders.")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output folder. In batch mode, group subfolders are created here automatically.")
    parser.add_argument("--fps",    "-f", type=int, default=60,
                        help="Frame rate (default: 60).")
    parser.add_argument("--lossless", "-l", action="store_true",
                        help="Write MJPEG .avi instead of MP4. Recommended for NIR channels.")
    parser.add_argument("--batch", "-b", action="store_true",
                        help="Batch mode: loop through all subfolders of --input and create one video each.")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input path '{args.input}' does not exist.")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    if args.batch:
        batch_process(args.input, args.output, fps=args.fps, lossless=args.lossless)
    else:
        ext = ".avi" if args.lossless else ".mp4"
        folder_name   = os.path.basename(args.input.rstrip("/\\"))
        output_file   = os.path.join(args.output, folder_name + ext)
        create_video_from_frames(args.input, output_file, fps=args.fps, lossless=args.lossless)


if __name__ == "__main__":
    main()
