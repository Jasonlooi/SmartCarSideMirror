from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
from time import monotonic
from shapely.geometry import Point, Polygon

nnPath    = str((Path(__file__).parent / Path('./models/OpenVINO_2021_2/vehicle-detection-adas-0002.blob')).resolve().absolute())
videoPath = str((Path(__file__).parent / Path('./dataset/rgb_pasar.mp4')).resolve().absolute())

labelMap = ["background", "vehicle","Pedestrian","Bike"]

#Read datasets reference
raw = []
ref_pixels = {}
with open('datasets.csv','r')as f:
    data = f.readlines()
for i in data:
    raw.append(i.split(','))
for i in raw:
    ref_pixels[int(float(i[0]))] = float(i[1].strip())

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a neural network that will make predictions based on the source frames
# DetectionNetwork class produces ImgDetections message that carries parsed
# detection results.
nn = pipeline.createMobileNetDetectionNetwork()
nn.setBlobPath(nnPath)

nn.setConfidenceThreshold(0.4)
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)

#Define source
# Create XLinkIn object as conduit for sending input video file frames
# to the neural network
xinFrame = pipeline.createXLinkIn()
xinFrame.setStreamName("inFrame")
# Connect (link) the video stream from the input queue to the
# neural network input
xinFrame.out.link(nn.input)

# Create neural network output (inference) stream
nnOut = pipeline.createXLinkOut()
nnOut.setStreamName("nn")
nn.out.link(nnOut.input)

# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    out = cv2.VideoWriter('rgb.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (1920, 1080))
    # Start pipeline
    device.startPipeline()

    # Define queues for image frames
    # Input queue for sending video frames to device
    qIn_Frame = device.getInputQueue(name="inFrame", maxSize=8, blocking=False)

    qDet = device.getOutputQueue(name="nn", maxSize=8, blocking=False)

    cap = cv2.VideoCapture(videoPath)

    startTime = time.monotonic()
    counter = 0
    ini_dist = 0
    detections = []
    frame = None

    # nn data (bounding box locations) are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


    def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
        return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()


    def displayFrame(name, frame):

        # Draw the Polygon
        coord = [(450, 1080), (450, 653), (550, 653), (1920, 1080)]
        polyRoad = Polygon(coord)

        pts = np.array([(450, 1080), (450, 653), (550, 653), (1920, 1080)])
        cv2.polylines(frame, [pts], True, (0, 255, 255))

        # Zero all the related list and dictionary for getting updated data
        dict_box = {}
        bbox_list = []
        car_num = 0

        for detection in detections:
            bbox_data = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

            # For intersection
            x1, y1, x2, y2 = bbox_data[0], bbox_data[1], bbox_data[2], bbox_data[3]
            P1, P2, P3, P4 = map(Point, [(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
            centroid = Point((x1 + x2) / 2, y1)

            # Check whether the bounding box of car is within the box or not
            if polyRoad.contains(centroid) or polyRoad.contains(P1) or polyRoad.contains(P2) or polyRoad.contains(
                    P3) or polyRoad.contains(P4):


                # Find the closest car by finding the largest pixel value
                dict_box[car_num] = bbox_data
                car_num += 1
                bbox_list.append(bbox_data[3])
                index = np.argmax(bbox_list)
                bbox = dict_box[index]
                cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_DUPLEX,
                            1, (255, 255, 0), 2)

                if bbox[3] < 1041 and bbox[3] > 652: # Distance in between 3m and 50m
                    result_dist = float(ref_pixels[bbox[3]])

                    if bbox[3] > 652 and bbox[3] <691: # If distance less than or equal to 20m. Bounding box become green
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
                        cv2.putText(frame, "{:.2f}m".format(result_dist), (bbox[0] + 10, bbox[1] + 50),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 3)

                    elif bbox[3] > 690: # If distance less than or equal to 20m. Bounding box become yellow
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 3)
                        cv2.putText(frame, "{:.2f}m".format(result_dist), (bbox[0] + 10, bbox[1] + 50),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 3)

                elif bbox[3] > 1040: # If distance less than or equal to 20m. Bounding box become red
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
                    cv2.putText(frame, "...CAUTION...", (bbox[0] + 10, bbox[1] + 50),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 3)

        # Reduce the size of preview
        scale = 70
        width = int(frame.shape[1] * scale / 100)
        height = int(frame.shape[0] * scale / 100)
        dim = (width, height)
        out.write(frame)
        M = cv2.resize(frame,dim)

        # SHow the result
        cv2.imshow(name, M)

    while True:
        # Get image frames from camera or video file
        read_correctly, frame = cap.read()

        # Prepare image frame from video for sending to device
        img = dai.ImgFrame()
        img.setData(to_planar(frame, (672,384)))
        img.setTimestamp(monotonic())
        img.setWidth(672)
        img.setHeight(384)
        # Use input queue to send video frame to device
        qIn_Frame.send(img)
        inDet = qDet.tryGet()

        if inDet is not None:
            detections = inDet.detections
            counter += 1

        # if the frame is available, render detection data on frame and display.
        if frame is not None:

            #Draw a Polygon on the frame
            cv2.line(frame, (450, 1080), (450, 653), (255, 255, 0), 3)
            cv2.line(frame, (450, 653), (550, 653), (255, 255, 0), 3)
            cv2.line(frame, (550, 653), (1920, 1080), (255, 255, 0), 3)
            displayFrame("frame test", frame)

        # key for action
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
