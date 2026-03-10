# Real-Time-Vehicle-Tracking-and-Counting

## Project Overview

This project implements a real-time vehicle tracking and counting system utilizing YOLO (You Only Look Once) for object detection and ByteTrack for multi-object tracking. The system processes video streams to identify various vehicle types, track their movement, and accurately count them as they cross a predefined virtual line. The output is an annotated video showcasing bounding boxes, unique tracking IDs, and real-time vehicle counts.

## Features

*   **Real-time Object Detection:** Employs a YOLO model (specifically `yolo11m.pt`, likely a custom or slightly older version of YOLOv8m) for efficient and accurate vehicle detection.
*   **Multi-Object Tracking:** Integrates ByteTrack for robust and persistent tracking of individual vehicles across video frames.
*   **Vehicle Classification:** Identifies and categorizes detected objects into common vehicle types: bicycles, cars, motorcycles, buses, and trucks.
*   **Line-Crossing Counting:** Implements a mechanism to count vehicles as they cross a user-defined virtual line, providing directional or total traffic flow data.
*   **Annotated Video Output:** Generates an output video with visual overlays including bounding boxes, tracking IDs, vehicle class labels, and real-time count displays.
*   **Configurable Parameters:** Allows customization of detection confidence, Intersection Over Union (IOU) thresholds, image size, and the position of the counting line.

## Technical Stack

*   **Python:** The primary programming language for the project.
*   **Ultralytics YOLO:** Framework for state-of-the-art object detection models.
*   **OpenCV (`cv2`):** Used for video stream processing, drawing annotations, and video output.
*   **PyYAML:** For handling configuration files, specifically for the ByteTrack parameters.
*   **Argparse:** For command-line argument parsing in the standalone script.
*   **FFmpeg:** (Used in the Jupyter Notebook) For post-processing video output to ensure broader compatibility and efficient playback.

## Getting Started

### Prerequisites

Before running the project, ensure you have Python 3.x installed. The required Python libraries can be installed using `pip`.

### Installation

1.  **Clone the repository (or extract the provided zip file):**

    ```bash
    # If it were a git repository
    # git clone <repository_url>
    # cd RealTimeVehicleTrackingandCounting
    ```

2.  **Install dependencies:**

    ```bash
    pip install ultralytics opencv-python pyyaml
    ```

    *Note: The Jupyter Notebook also suggests `pip -q install -U ultralytics opencv-python pyyaml`.*

3.  **Download Model Weights:**

    The project uses `yolo11m.pt` as the model weights. Ensure this file is present in the project directory. If not, you might need to download a compatible YOLOv8m model (e.g., `yolov8m.pt`) from the Ultralytics GitHub or website and rename it, or adjust the `run_vehicle_tracking.py` script to point to the correct weight file.

## Usage

### Running the Standalone Script

The `run_vehicle_tracking.py` script can be executed from the command line with various arguments:

```bash
python run_vehicle_tracking.py \
    --source <video_source> \
    --weights <model_weights_path> \
    --conf <confidence_threshold> \
    --iou <iou_threshold> \
    --imgsz <image_size> \
    --line-ratio <line_position_ratio> \
    --output <output_video_path>
```

**Arguments:**

*   `--source`: Path to the input video file (e.g., `input.mp4`) or `0` for webcam feed. Default is `0`.
*   `--weights`: Path to the YOLO model weights file (e.g., `yolo11m.pt`). Default is `yolo11m.pt`.
*   `--conf`: Confidence threshold for object detection (e.g., `0.25`). Default is `0.10`.
*   `--iou`: Intersection Over Union (IOU) threshold for non-maximum suppression (e.g., `0.7`). Default is `0.50`.
*   `--imgsz`: Image size for inference (e.g., `640`). Default is `1280`.
*   `--line-ratio`: Vertical position of the counting line as a ratio of the video height (0.0 to 1.0). Default is `0.62`.
*   `--output`: Path to save the output video. If not specified, it defaults to `vehicle_tracking_output.mp4` for webcam or `<input_base_name>_tracked.mp4` for video files.

**Example:**

```bash
python run_vehicle_tracking.py --source "path/to/your/video.mp4" --output "output_tracked_video.mp4"
```

### Using the Jupyter Notebook

The `real-time-vehicle-tracking-and-counting-yolo11 (1).ipynb` notebook provides a detailed, interactive walkthrough of the project. It includes:

*   Setup and configuration steps.
*   Code cells for executing the tracking and counting logic.
*   Visualization of the results.
*   Post-processing steps using `ffmpeg` to convert the output video to a browser-compatible H.264 MP4 format.

To run the notebook:

1.  Ensure you have Jupyter Notebook or JupyterLab installed (`pip install notebook` or `pip install jupyterlab`).
2.  Navigate to the project directory in your terminal.
3.  Start Jupyter:
    ```bash
    jupyter notebook
    ```
4.  Open the `real-time-vehicle-tracking-and-counting-yolo11 (1).ipynb` file and run the cells sequentially.

## Project Structure

```
.  
├── bytetrack_low.yaml             # ByteTrack configuration (generated dynamically)
├── kaggle dataset link            # Link to the dataset used (if any)
├── real-time-vehicle-tracking-and-counting-yolo11 (1).ipynb # Jupyter Notebook for interactive execution
├── run_vehicle_tracking.py        # Main Python script for vehicle tracking and counting
├── vehicle_tracking_output.mp4    # Example output video (or placeholder)
└── yolo11m.pt                     # YOLO model weights file
```

## License

[Specify your project's license here, e.g., MIT, Apache 2.0, etc.]

## Acknowledgements

*   Ultralytics for the YOLO framework.
*   OpenCV for computer vision functionalities.
*   The authors of ByteTrack for the tracking algorithm.
*   Kaggle for providing a platform and datasets (as indicated by the notebook).

---
