from ultralytics import YOLO
import supervision as sv
import numpy as np


model = YOLO("yolov8m.pt") 
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[(detections.class_id == 0) & (detections.confidence > 0.5)]
    detections = tracker.update_with_detections(detections)

    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ]

    annotated_frame = box_annotator.annotate(
        frame.copy(), detections=detections)
    return label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels)

video_info = sv.VideoInfo.from_video_path('people-walking.mp4')
frames_generator = sv.get_video_frames_generator('people-walking.mp4')
sv.process_video(
    source_path="people-walking.mp4",
    target_path="result.mp4",
    callback=callback,
)