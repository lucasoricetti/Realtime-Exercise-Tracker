import cv2
import argparse
import os
from datetime import datetime
from AIGym_Modified import AIGym_Modified

def main():
    parser = argparse.ArgumentParser(description="Pose-based workout tracker")

    parser.add_argument(
        "--model", choices=["s", "m", "l"], default="m",
        help="Choose model: 's' = fast, 'm' = balanced, 'l' = more accurate (default: m)"
    )
    parser.add_argument(
        "--input", choices=["webcam", "video"], default="webcam",
        help="Input source: 'webcam' for real-time or 'video' for file (default: webcam)"
    )
    parser.add_argument(
        "--video_path", type=str, default=None,
        help="Path to video file (required if --input is 'video')"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Use this flag if you want to save the output video"
    )

    args = parser.parse_args()

    # Map input model to model file
    model_map = {
        "s": "yolo11s-pose.pt",
        "m": "yolo11m-pose.pt",
        "l": "yolo11l-pose.pt"
    }
    model_path = model_map[args.model]

    # Select video input
    if args.input == "webcam":
        cap = cv2.VideoCapture(0)
        video_name = f"webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
        fps = 20  # Use realistic FPS for webcam recording
    else:
        if not args.video_path:
            raise ValueError("You must specify --video_path when using input=video")
        if not os.path.isfile(args.video_path):
            raise FileNotFoundError(f"The file {args.video_path} does not exist.")
        cap = cv2.VideoCapture(args.video_path)
        base_video_name = os.path.splitext(os.path.basename(args.video_path))[0]
        video_name = f"output_{base_video_name}.avi"
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # fallback in case of bad FPS

    assert cap.isOpened(), "Error opening video source."

    # Output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Get frame size
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Set up writer
    if args.save:
        output_path = os.path.join(output_dir, video_name)
        video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    else:
        video_writer = None

    # Initialize AIGym_Modified
    gym = AIGym_Modified(
        show=True,
        model=model_path
    )

    print("Processing started... Press 'q' in the window or Ctrl+C in terminal to stop.")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Empty frame or video ended.")
                break

            results = gym.process(frame)

            if video_writer:
                video_writer.write(results.plot_im)

            # Display and wait for 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Recording stopped by user with 'q'.")
                break

    except KeyboardInterrupt:
        print("\nRecording interrupted with Ctrl+C.")

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    print("Processing completed.")

if __name__ == "__main__":
    main()