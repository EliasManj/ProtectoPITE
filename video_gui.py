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

class PredictedFace():

    photo_buffer = []
    
    @staticmethod
    def add_faces(detected_faces):
        names = list(map( lambda x: x.name, detected_faces))
        buffer_names = list(map( lambda x: x.name, PredictedFace.photo_buffer))
        dictfilt = lambda x, y: [ i for i in x if i.name in set(y) ]
        for predicted_face in detected_faces:    
            if predicted_face.name in buffer_names:
                pass
            else:
                PredictedFace.photo_buffer.append(predicted_face)
        to_delete = [ face for face in PredictedFace.photo_buffer if face.name not in set(names) ]
        PredictedFace.photo_buffer = [ face for face in PredictedFace.photo_buffer if face.name in set(names) ]
        for obj in to_delete:
            del obj
        return

    def __init__(self, name, proba):
        self.name = name
        self.proba = proba
        self.desc = ''
        self.photo = None
        self.get_info()

    def get_info(self):
        self.get_desc()
        self.get_profile_pic()

    def get_desc(self):
        try:
            path = os.path.join('dataset', self.name, '{0}.txt'.format(self.name))
            file = open(path, "r")
            self.desc = file.read()
            if not self.desc:
                self.desc = 'no description'
        except OSError:
            self.desc = 'no description'

    def get_profile_pic(self):
        files  = [os.path.join('dataset', self.name, self.name + s) for s in ['.jpg', '.png', '.jpeg']]
        x = list(filter( lambda x : os.path.isfile(x), files ))
        if len(x) == 0:
            return
        path = x[0]
        try:
            img = PIL.Image.open(path)
            img = img.resize((200,200), PIL.Image.ANTIALIAS)
            if img:
                self.photo = PIL.ImageTk.PhotoImage(img)
        except Exception as ex:
            print(str(ex))
            pass

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
        # get video source width and heighs

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
        self.width = 600
        self.height = self.vid.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        time.sleep(2.0)
        # start the FPS throughput estimator
        self.fps = FPS().start()	

    def get_frame(self):
        if self.vid.stream.isOpened():
            ret, frame = self.vid.stream.read()
            if ret:
                frame, faces = self.detect_faces_in_frame(frame)
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    def get_frame_and_info(self):
        try:
            if self.vid.stream.isOpened():
                ret, frame = self.vid.stream.read()
                if ret:
                    frame = self.detect_faces_in_frame(frame)
                    return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    return (ret, None)
            else:
                return (ret, None)
        except:
            return (False, None)

    # this method will modify the frame so that it will show detected faces
    def detect_faces_in_frame(self, frame):
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
        # detect face
        self.detector.setInput(imageBlob)
        detections = self.detector.forward()
        # loop over detections
        detected_faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.args["confidence"]:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                self.embedder.setInput(faceBlob)
                vec = self.embedder.forward()
                preds = self.recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = self.le.classes_[j]
                text = "{}: {:.2f}%".format(name, proba * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
                self.fps.update()
                face_info = PredictedFace(name, proba*100)
                detected_faces.append(face_info)
        PredictedFace.add_faces(detected_faces)
        return frame

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
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            self.fps.update()
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        self.__del__()

    def __del__(self):
        self.fps.stop()
        # do a bit of cleanup
        cv2.destroyAllWindows()
        self.vid.stop()

class DetectionPanel:

    detection_panels = []

    def __init__(self, parent, face):
        self.parent = parent
        self.face = face
        self.face_name = tkinter.StringVar()
        self.face_desc = tkinter.StringVar()
        self.face_proba = tkinter.StringVar()

    def update(self):
        self.face_name.set(self.face.name)
        self.face_desc.set(self.face.desc)
        self.face_proba.set('proba: {0}'.format(self.face.proba))
        self.image_label.configure(image = self.face.photo)
        
    def construct_panel(self):
        self.frame = tkinter.Frame(self.parent, width = 300, height = 300)
        self.frame.grid(row = 0, column = len(DetectionPanel.detection_panels))
        self.name_label = tkinter.Label(self.frame, textvariable = self.face_name)
        self.name_label.grid(row = 0)
        self.image_label = tkinter.Label(self.frame, image = self.face.photo)
        self.image_label.grid(row = 1)
        self.desc_label = tkinter.Label(self.frame, textvariable = self.face_desc)
        self.desc_label.grid(row = 2)
        self.proba_label = tkinter.Label(self.frame, textvariable = self.face_proba)
        self.proba_label.grid(row = 3)
        
    def recycle_panel(self, face):
        self.face = face
        self.update()
        
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
        self.canvas.grid(row=0)
        # panel
        self.frame = tkinter.Frame(window)
        self.label = tkinter.Label(window, text="Detections")
        self.label.grid(row=1)
        self.frame.grid(row=2)
        # loop
        self.delay = 15
        self.start_loop()

    def start_loop(self):
        self.update()
        self.window.mainloop()

    def update(self):
        ret, frame = self.video_model.get_frame_and_info()
        faces = PredictedFace.photo_buffer
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
            if(faces):
                self.check_faces(faces)
        self.window.after(self.delay, self.update)

    def check_faces(self, faces):
        detection_names = [face.name for face in faces]
        buffer_names = [panel.face.name for panel in DetectionPanel.detection_panels]
        new_faces = [face for face in faces if face.name not in buffer_names]
        faces_to_remove = [panel for panel in DetectionPanel.detection_panels if panel.face.name not in detection_names]
        if new_faces:
            if faces_to_remove:
                self.update_panels_recycle(faces_to_remove, new_faces)
                if len(new_faces) > len(faces_to_remove):
                    unadded = new_faces[len(faces_to_remove):]
                    self.add_new_panels(unadded)
            else:
                self.add_new_panels(new_faces)
        self.remove_old(detection_names)
            
    def remove_old(self, detection_names):
        to_destroy = [ panel for panel in DetectionPanel.detection_panels if panel.face.name not in detection_names ]
        DetectionPanel.detection_panels = [ panel for panel in DetectionPanel.detection_panels if panel.face.name in detection_names ]
        for obj in to_destroy:
            obj.frame.grid_forget()
            del obj

    def update_panels_recycle(self, recycle_panels, new_faces):
        for face, panel in zip(new_faces, recycle_panels):
            panel.recycle_panel(face)
    
    def add_new_panels(self, new_faces):
        for face in new_faces:
            new_panel = DetectionPanel(self.frame, face)
            new_panel.construct_panel()
            new_panel.update()
            DetectionPanel.detection_panels.append(new_panel)               

if __name__ == "__main__":
    app = App(window = tkinter.Tk(screenName="tkinter window"), title = "Test", video_source=0)


