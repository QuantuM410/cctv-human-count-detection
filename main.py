from ultralytics import YOLO
from supervision.utils.video import get_video_frames_generator, VideoInfo, VideoSink
from supervision.detection.core import Detections
from supervision.tracker.byte_tracker.core import ByteTrack
from supervision.geometry.core import Point
from supervision.annotators.core import BoundingBoxAnnotator, LabelAnnotator
from supervision.detection.line_counter import LineZone, LineZoneAnnotator

def process_video(input_video_path, output_video_path, yolo_model_path):
    # Load YOLO model and fuse it
    yolo_model = YOLO(yolo_model_path)
    yolo_model.fuse()

    # Get video information
    video_info = VideoInfo.from_video_path(input_video_path)

    # Define the line parameters
    LINE_START = Point(0, video_info.height // 2)
    LINE_END = Point(video_info.width, video_info.height // 2)

    # Initialize the tracker
    tracker = ByteTrack()

    # Get video frame generator
    frame_generator = get_video_frames_generator(input_video_path)

    # Initialize the line counter
    line_counter = LineZone(start=LINE_START, end=LINE_END)

    # Initialize annotators
    box_annotator = BoundingBoxAnnotator(thickness=4)
    label_annotator = LabelAnnotator()
    line_annotator = LineZoneAnnotator(text_scale=2, thickness=4)

    with VideoSink(output_video_path, video_info) as sink:
        for frame in frame_generator:
            results = yolo_model(frame)[0]

            detections = Detections.from_ultralytics(results)
            detections = detections[(detections.class_id == 0) & (detections.confidence > 0.5)]
            detections = tracker.update_with_detections(detections)

            labels = [f"#{tracker_id} {results.names[class_id]}" for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)]
            
            # Update the line counter
            line_counter.trigger(detections=detections)

            # Annotate the frame
            annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
            label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
            line_annotator.annotate(annotated_frame, line_counter=line_counter)

            # Write the annotated frame to the output video
            sink.write_frame(annotated_frame)

if __name__ == "__main__":
    SOURCE_VIDEO_PATH = "people-walking.mp4"
    TARGET_VIDEO_PATH = "people-walking-result.mp4"
    YOLO_MODEL_PATH = "yolov8m.pt"

    process_video(SOURCE_VIDEO_PATH, TARGET_VIDEO_PATH, YOLO_MODEL_PATH)
