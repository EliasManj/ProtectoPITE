from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

class VideoModel():

    def __init__(self):
        self.ap = argparse.ArgumentParser()
        self.ap.add_argument("-d", "--detector", help="path to OpenCV's deep learning face detector", default='face_detection_model')
        self.ap.add_argument("-m", "--embedding-model", help="path to OpenCV's deep learning face embedding model", default='openface_nn4.small2.v1.t7')
        self.ap.add_argument("-r", "--recognizer", help="path to model trained to recognize faces", default='output/recognizer.pickle')
        self.ap.add_argument("-l", "--le", help="path to label encoder", default='output/le.pickle')
        self.ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
        self.args = vars(self.ap.parse_args())

    # load our serialized face detector from disk
    def open_models(self):
        print("[INFO] loading face detector...")
        self.protoPath = os.path.sep.join([self.args["detector"], "deploy.prototxt"])
        self.modelPath = os.path.sep.join([self.args["detector"],	"res10_300x300_ssd_iter_140000.caffemodel"])
        self.protoPath = os.path.join(os.getcwd(), self.protoPath)
        self.modelPath = os.path.join(os.getcwd(), self.modelPath)
        self.detector = cv2.dnn.readNetFromCaffe(self.protoPath, self.modelPath)
        print("[INFO] loading face recognizer...")
        self.embedder = cv2.dnn.readNetFromTorch(self.args["embedding_model"])
        # load the actual face recognition model along with the label encoder
        self.recognizer = pickle.loads(open(self.args["recognizer"], "rb").read())
        self.le = pickle.loads(open(self.args["le"], "rb").read())

    def start_stream(self):
        # initialize the video stream, then allow the camera sensor to warm up
        print("[INFO] starting video stream...")
        self.vs = VideoStream(src=1).start()
        self.time.sleep(2.0)
        # start the FPS throughput estimator
        self.fps = FPS().start()	

    def video_loop(self):
        while True:
            frame = self.vs.read()
            frame = imutils.resize(frame, width=600)
            (h, w) = frame.shape[:2]
            imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
            # detect face
            self.detector.setInput(imageBlob)
            detections = self.detector.forward()
            # loop over detections
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > self.args["confidence"]:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    face = frame[startY:endY, startX:endX]
                    (fH, fW) = face.shape[:2]
                    if fW < 20 or fH < 20:
                        continue
                    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                    embedder.setInput(faceBlob)
                    vec = embedder.forward()
                    preds = recognizer.predict_proba(vec)[0]
                    j = np.argmax(preds)
                    proba = preds[j]
                    name = le.classes_[j]
                    text = "{}: {:.2f}%".format(name, proba * 100)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                    cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            fps.update()
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        fps.stop()
        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.stop()

            

if __name__ == "__main__":
    video_model = VideoModel()
    video_model.open_models()
    video_model.start_stream()
    video_model.video_loop()

