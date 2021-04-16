import cv2
import warnings
import numpy as np
from PIL import Image
from numpy import load
from numpy import asarray
from numpy import expand_dims
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")
box_faces = []

def get_embedding(model, face_pixels):
	face_pixels = face_pixels.astype('float32')
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	samples = expand_dims(face_pixels, axis=0)
	yhat = model.predict(samples)
	return yhat[0]

def extract_face(filename, required_size=(160, 160)):
	image = Image.open(filename)
	image = image.convert('RGB')
	pixels = asarray(image)
	detector = MTCNN()
	results = detector.detect_faces(pixels)
	face_array1 = []
	for a in range(len(results)):
		x1, y1, width, height = results[a]['box']
		x1, y1 = abs(x1), abs(y1)
		x2, y2 = x1 + width, y1 + height
		face = pixels[y1:y2, x1:x2]
		box_faces.append([x1, y1, width, height])
		image = Image.fromarray(face)
		image = image.resize(required_size)
		face_array1.append(asarray(image))
	return face_array1


data = load(r'C:\\Users\\gnith\\Desktop\\MTCNN1\\data.npz')
trainX, trainy = data['arr_0'], data['arr_1']

in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)

out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(trainX, trainy)

model_face = load_model(r'C:\\Users\\gnith\\Desktop\\MTCNN1\\facenet_keras.h5')

image_path = 'a2.jpg'

for a in extract_face(r'C:\\Users\\gnith\\Desktop\\MTCNN1\\'+image_path):
	test_image = get_embedding(model_face,a)
	test_image = np.array([test_image])
	test_image = in_encoder.transform(test_image)

	yhat_class = model.predict(test_image)
	yhat_prob = model.predict_proba(test_image)

	class_index = yhat_class[0]
	class_probability = yhat_prob[0,class_index] * 100
	predict_names = out_encoder.inverse_transform(yhat_class)
	print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))

image = cv2.imread(image_path)

for a in box_faces:
	x, y, w, h = a
	cv2.rectangle(image, (x, y), (x + w, y + h), (255,0,0), 2)
	text = "%s (%.3f)" % (predict_names[0], class_probability)
	cv2.putText(image, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255,0,0), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)