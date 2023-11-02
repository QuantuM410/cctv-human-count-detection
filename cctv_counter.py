from ultralytics import YOLO
import math
import numpy as np  
from supervision.utils.video import get_video_frames_generator, VideoInfo, VideoSink
from supervision.detection.core import Detections
from supervision.tracker.byte_tracker.core import ByteTrack
from supervision.geometry.core import Point
from supervision.annotators.core import BoundingBoxAnnotator, LabelAnnotator
from supervision.detection.tools.polygon_zone import PolygonZone, PolygonZoneAnnotator
from supervision.draw.color import Color

class CCTVCounter():

    def process_video(input_video_path, output_video_path, yolo_model_path):
        # Loads YOLO model and fuse it
        yolo_model = YOLO(yolo_model_path)
        yolo_model.fuse()

        # Gets video information
        video_info = VideoInfo.from_video_path(input_video_path)

        polygon_map = np.array([[0,video_info.height],[video_info.width,video_info.height],[video_info.width,0],[0,0]])

        # Defining the polygon parameters
        zone = PolygonZone(polygon=polygon_map, frame_resolution_wh=video_info.resolution_wh)

        # Initialize the tracker
        tracker = ByteTrack()

        # Gets video frame generator
        frame_generator = get_video_frames_generator(input_video_path)

        # Initializes annotators
        box_annotator = BoundingBoxAnnotator(thickness=4)
        zone_annotator = PolygonZoneAnnotator(zone=zone, color=Color.white(), thickness=4, text_scale=4, text_thickness=4)
        label_annotator = LabelAnnotator()

        total_count = 0  # Variable to keep track of the total count

        with VideoSink(output_video_path, video_info) as sink:
            for frame in frame_generator:
                results = yolo_model(frame)[0]

                detections = Detections.from_ultralytics(results)
                detections = detections[(detections.class_id == 0) & (detections.confidence > 0.5)]

                #triggers zone on each frame and display the objects inside the zone
                zone.trigger(detections=detections)

                #updates the tracker for each object in the zone
                detections = tracker.update_with_detections(detections)

                labels = [f"#{tracker_id} {results.names[class_id]}" for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)]
                
                # Annotates the frame
                frame = box_annotator.annotate(scene=frame, detections=detections)
                frame = zone_annotator.annotate(scene=frame)
                label_annotator.annotate(scene=frame, detections=detections, labels=labels)

                # Counting the number of objects in the zone
                total_count += len(detections)

                # Writes the annotated frame to the output video
                sink.write_frame(frame)
        
        average_count = total_count // video_info.total_frames
        
        return math.ceil(average_count)

# for testing the inference
if __name__ == "__main__":
    SOURCE_VIDEO_PATH = "people-walking.mp4"
    TARGET_VIDEO_PATH = "people-walking-zone-result.mp4"
    YOLO_MODEL_PATH = "yolov8m.pt"

    CCTVCounter.process_video(SOURCE_VIDEO_PATH, TARGET_VIDEO_PATH, YOLO_MODEL_PATH)
