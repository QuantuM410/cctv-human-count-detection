from ultralytics import YOLO
from supervision.utils.video import get_video_frames_generator, VideoInfo, VideoSink
from supervision.utils.notebook import plot_image
from supervision.detection.core import Detections #gives detection object with xyxy, confidence, class_id attributes for each detection in the frame'
from supervision.annotators.core import BoundingBoxAnnotator 
from supervision.tracker.byte_tracker.core import ByteTrack
from supervision.geometry.core import Point
from supervision.annotators.core import LabelAnnotator
from supervision.detection.line_counter import LineZone, LineZoneAnnotator

model = YOLO("yolov8m.pt")
model.fuse()

SOURCE_VIDEO_PATH = "people-walking.mp4"
TARGET_VIDEO_PATH = "people-walking-result.mp4"
video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH) #returns video informations such as its size, total frames, fps

LINE_START = Point(0, video_info.height // 2) #pierces through the center of the window
LINE_END = Point(video_info.width, video_info.height // 2)

tracker = ByteTrack()

generator = get_video_frames_generator(SOURCE_VIDEO_PATH)

line_counter = LineZone(start=LINE_START, end=LINE_END)

box_annotator = BoundingBoxAnnotator(thickness=4)   

label_annotator = LabelAnnotator()

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

        detections = Detections.from_ultralytics(results)#supervision uses Detection type objects
        detections = detections[(detections.class_id == 0) & (detections.confidence > 0.5)]
        detections = tracker.update_with_detections(detections)

        labels = [
            f"#{tracker_id} {results.names[class_id]}"
            for class_id, tracker_id
            in zip(detections.class_id, detections.tracker_id)
        ]
        
        line_counter.trigger(detections=detections) #updates the line counter with the detections in the frame

        frame = box_annotator.annotate(scene=frame, detections=detections)

        line_annotator.annotate(frame=frame, line_counter=line_counter)

        annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
        label_annotator.annotate(annotated_frame, detections=detections, labels=labels)

        sink.write_frame(annotated_frame) #write each predicted frame to the target video path


