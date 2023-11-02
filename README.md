# CCTV People Counter

CCTV People Counter is a Python application that uses YOLO (You Only Look Once) object detection and tracking to count the number of people in a video and display the results in a user-friendly way.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Demo](#demo)


## Features

- Upload a video file for processing.
- Utilizes YOLO object detection and tracking to count the number of people.
- Converts the output video to H.264 format for compatibility.
- Displays the processed video and average count in the web interface.
- Includes a progress bar and spinner loader to provide visual feedback during processing.

## Requirements

Ensure you have the following prerequisites installed:

- Python 3.x
- Streamlit (can be installed via pip)
- OpenCV (can be installed via pip)
- FFmpeg for video conversion (installation may vary by platform)
- The YOLO model (yolov8m.pt or any other compatible model)

You can create a virtual environment and install the required packages using the provided `requirements.txt`:

```
pip install requirements.txt
```


## Installation

1. Clone the repository:

```
git clone https://github.com/QuantuM410/cctv-human-count-detection.git
``` 


2. Change to the project directory:

```
cd cctv-human-count-detection
```


3. Install the required dependencies as mentioned in the [Requirements](#requirements) section.

4. Place the YOLO model file (e.g., yolov8m.pt) in the project directory.

## Usage

Run the Streamlit app with:

```
streamlit run app.py
```


1. Upload a video file for processing.
2. The app will display the processed video and average count.
3. You can monitor the progress of video processing via the progress bar or spinner loader.

## Demo
