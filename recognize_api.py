import os
import numpy as np
import argparse
import imutils
import pickle
import cv2

class Prediction:

	def __init__(self, name, probability):
		self.name = name
		self.probability = probability
	
	def __repr__(self):
		return "name: {}; probability: {}".format(self.name, self.probability)

class Face_reader:

	detector = "face_detection_model"
	embedding_model = "openface_nn4.small2.v1.t7" 
	recognizer = os.path.join("output","recognizer.pickle") 
	le = os.path.join("output","le.pickle")

	protoPath = os.path.join(detector, "deploy.prototxt")
	modelPath = os.path.join(detector, "res10_300x300_ssd_iter_140000.caffemodel")

	detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
	embedder = cv2.dnn.readNetFromTorch(embedding_model)

	recognizer = pickle.loads(open(recognizer, "rb").read())
	le = pickle.loads(open(le, "rb").read())

	confidence = 0.5

	def classify_image(self, image_path):
		image = cv2.imread(image_path)
		if image is None:
			return None 
		image = imutils.resize(image, width=600)
		(h, w) = image.shape[:2]

		# construct a blob from the image
		imageBlob = cv2.dnn.blobFromImage(
			cv2.resize(image, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)
		
		# apply OpenCV's deep learning-based face detector to localize
		# faces in the input image
		self.detector.setInput(imageBlob)
		detections = self.detector.forward()

		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with the
			# prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections
			if confidence > self.confidence:
				# compute the (x, y)-coordinates of the bounding box for the
				# face
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# extract the face ROI
				face = image[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]

				# ensure the face width and height are sufficiently large
				if fW < 20 or fH < 20:
					continue
				
				# construct a blob for the face ROI, then pass the blob
				# through our face embedding model to obtain the 128-d
				# quantification of the face
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
					(0, 0, 0), swapRB=True, crop=False)
				self.embedder.setInput(faceBlob)
				vec = self.embedder.forward()

				# perform classification to recognize the face
				preds = self.recognizer.predict_proba(vec)[0]
				j = np.argmax(preds)
				proba = preds[j]
				name = self.le.classes_[j]

				return Prediction(name, proba*100)