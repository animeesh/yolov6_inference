import time
import cv2
import numpy as np
import onnxruntime

from utils import xywh2xyxy, nms, draw_detections


class YOLOv6:

    def __init__(self, path, conf_thres=0.9, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(path,
                                                    providers=['CUDAExecutionProvider',
                                                               'CPUExecutionProvider'])
        # Get model info
        self.get_input_details()
        self.get_output_details()

    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        output = self.inference(input_tensor)

        # Process output data
        self.boxes, self.scores, self.class_ids = self.process_output(output)

        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor


    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})[0]

        # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_output(self, output):
        predictions = np.squeeze(output)

        # Filter out object confidence scores below threshold
        obj_conf = predictions[:, 4]
        predictions = predictions[obj_conf > self.conf_threshold]
        obj_conf = obj_conf[obj_conf > self.conf_threshold]

        # Multiply class confidence with bounding box confidence
        predictions[:, 5:] *= obj_conf[:, np.newaxis]

        # Get the scores
        scores = np.max(predictions[:, 5:], axis=1)

        # Filter out the objects with a low score
        predictions = predictions[obj_conf > self.conf_threshold]
        scores = scores[scores > self.conf_threshold]

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 5:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes /= np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids)

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]


if __name__ == '__main__':

    import cv2

    # from YOLOv6 import YOLOv6

    # Initialize the webcam
    #cap = cv2.VideoCapture("/Users/animeshkumarnayak/Downloads/street.mp4")
    cap = cv2.VideoCapture(0)
    # used to record the time when we processed last frame
    vid_writer=None
    prev_frame_time = 0

    # used to record the time at which we processed current frame
    new_frame_time = 0

    # Initialize YOLOv6 object detector
    model_path = "yolov6n.onnx"
    yolov6_detector = YOLOv6(model_path, conf_thres=0.4, iou_thres=0.5)

    cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
    while cap.isOpened():

        # Read frame from the video
        ret, frame = cap.read()

        if not ret:
            break

        # Update object localizer
        boxes, scores, class_ids = yolov6_detector(frame)

        combined_img = yolov6_detector.draw_detections(frame)
        font = cv2.FONT_HERSHEY_SIMPLEX
        new_frame_time = time.time()

        # Calculating the fps

        # fps will be number of frame processed in given time frame
        # since their will be most of time error of 0.001 second
        # we will be subtracting it to get more accurate result
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # converting the fps into integer
        fps = int(fps)

        #cv2.line(combined_img, (20, 25), (127, 25), [85, 45, 255], 30)
        cv2.putText(combined_img, f'FPS: {fps}', (11, 35), 0, 1, [225, 255, 255], thickness=2)#, lineType=cv2.LINE_AA)
        # putting the FPS count on the frame
        #cv2.putText(combined_img, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
        
        #for saving video
        video_path="videos/output.avi"
        if isinstance(vid_writer, cv2.VideoWriter):
            print(vid_writer)
            vid_writer.release()
            #fourcc = 'mp4v'  # output video codec

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #print(w,h)
            vid_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
            vid_writer.write(combined_img)
        
        cv2.imshow("Detected Objects", combined_img)

        # Press key q to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
