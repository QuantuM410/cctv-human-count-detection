from ultralytics import YOLO
import numpy as np
from supervision.utils.video import get_video_frames_generator, VideoInfo, VideoSink
from supervision.utils.notebook import plot_image
from supervision.detection.core import Detections #gives detection object with xyxy, confidence, class_id attributes for each detection in the frame'
from supervision.annotators.core import BoundingBoxAnnotator 
from supervision.tracker.byte_tracker.core import ByteTrack
from supervision.geometry.core import Point
from supervision.draw.color import Color
from supervision.detection.line_counter import LineZone, LineZoneAnnotator

model = YOLO("yolov8m.pt")
model.fuse()

SOURCE_VIDEO_PATH = "people-walking.mp4"
TARGET_VIDEO_PATH = "people-walking-result.mp4"
video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH) #returns video informations such as its size, total frames, fps

LINE_START = Point(0, video_info.height // 2) #pierces through the center of the window
LINE_END = Point(video_info.width, video_info.height // 2)

generator = get_video_frames_generator(SOURCE_VIDEO_PATH)

line_counter = LineZone(start=LINE_START, end=LINE_END)

box_annotator = BoundingBoxAnnotator(thickness=4)   

line_annotator = LineZoneAnnotator(text_scale=2, thickness=4)

CLASS_NAMES_DICT = model.model.names

CLASS_ID = [0]

with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    for frame in generator: 
        '''
        tqdm is a progress bar library with good support for nested loops and Jupyter/IPython notebooks.
        frame gives an ndarray numpy array of frame in the give video
        '''

        results = model(frame)[0]

        detections = Detections( #supervision uses Detection type objects 
            xyxy=results.boxes.xyxy.cpu().numpy(),
            confidence=results.boxes.conf.cpu().numpy(),
            class_id=results.boxes.cls.cpu().numpy().astype(int)
        )
        
        line_counter.trigger(detections=detections) #updates the line counter with the detections in the frame
        # detections = detections[(detections.class_id == CLASS_ID[0]) & (detections.confidence > 0.5)]

        frame = box_annotator.annotate(scene=frame, detections=detections)

        line_annotator.annotate(frame=frame, line_counter=line_counter)

        sink.write_frame(frame) #write each predicted frame to the target video path


