from imutils.video import VideoStream
from imutils.video import FPS
import PIL.Image, PIL.ImageTk
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import tkinter

class VideoModel():

    def __init__(self, video_source = 0):
        self.video_source = video_source
        self.ap = argparse.ArgumentParser()
        self.ap.add_argument("-d", "--detector", help="path to OpenCV's deep learning face detector", default='face_detection_model')
        self.ap.add_argument("-m", "--embedding-model", help="path to OpenCV's deep learning face embedding model", default='openface_nn4.small2.v1.t7')
        self.ap.add_argument("-r", "--recognizer", help="path to model trained to recognize faces", default='output/recognizer.pickle')
        self.ap.add_argument("-l", "--le", help="path to label encoder", default='output/le.pickle')
        self.ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
        self.args = vars(self.ap.parse_args())
        self.protoPath = None
        self.modelPath = None
        self.detector = None
        self.recognizer = None        
        self.le = None
        self.embedder = None
        # get video source width and heigh

    def initialize(self):
        self.open_models()
        self.start_stream()

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
        self.vid = VideoStream(src=self.video_source).start()
        self.width = self.vid.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        time.sleep(2.0)
        # start the FPS throughput estimator
        self.fps = FPS().start()	

    def get_frame(self):
        if self.vid.stream.isOpened():
            ret, frame = self.vid.stream.read()
            frame = imutils.resize(frame, width=600)
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    def detect_face(self, frame):
        frame = imutils.resize(frame, width=600)

    def video_loop(self):
        while True:
            frame = self.vid.read()
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
                    self.embedder.setInput(faceBlob)
                    vec = self.embedder.forward()
                    preds = self.recognizer.predict_proba(vec)[0]
                    j = np.argmax(preds)
                    proba = preds[j]
                    name = self.le.classes_[j]
                    text = "{}: {:.2f}%".format(name, proba * 100)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                    cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            self.fps.update()
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        self.fps.stop()
        # do a bit of cleanup
        cv2.destroyAllWindows()
        self.vid.stop()

    def __del__(self):
        self.fps.stop()
        # do a bit of cleanup
        cv2.destroyAllWindows()
        self.vid.stop()

class App:

    def __init__(self, window, title, video_source=0):
        self.window = window
        self.window.title = title
        self.video_source = video_source
        # open video source
        self.video_model = VideoModel(video_source)
        self.video_model.initialize()
        self.vid = self.video_model.vid.stream
        # create a canvas
        self.canvas = tkinter.Canvas(window, width = self.video_model.width, height = self.video_model.height)
        self.canvas.pack()
        # loop
        self.delay = 15
        self.update()
        self.window.mainloop()

    def update(self):
        ret, frame = self.video_model.get_frame()
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
        self.window.after(self.delay, self.update)


    

if __name__ == "__main__":
    app = App(window = tkinter.Tk(), title = "Test", video_source=1)


