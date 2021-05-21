"""Sample prediction script for TensorFlow 2"""
import argparse
import tensorflow as tf
import numpy as np
import PIL.Image, PIL.ImageDraw
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from cv2 import *


MODEL_FILENAME = 'model.pb'
LABELS_FILENAME = 'labels.txt'
IMAGE_FILENAME = 'Tea-Leaves-Growing.jpg'
WEBCAM_IMAGE_FILENAME = 'webcam.jpg'

class ObjectDetection:
    INPUT_TENSOR_NAME = 'image_tensor:0'
    OUTPUT_TENSOR_NAMES = ['detected_boxes:0', 'detected_scores:0', 'detected_classes:0']

    def __init__(self, model_filename):
        graph_def = tf.compat.v1.GraphDef()
        with open(model_filename, 'rb') as f:
            graph_def.ParseFromString(f.read())

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        # Get input shape
        with tf.compat.v1.Session(graph=self.graph) as sess:
            self.input_shape = sess.graph.get_tensor_by_name(self.INPUT_TENSOR_NAME).shape.as_list()[1:3]

    def predict_image(self, image):
        image = image.convert('RGB') if image.mode != 'RGB' else image
        image = image.resize(self.input_shape)

        inputs = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
        with tf.compat.v1.Session(graph=self.graph) as sess:
            output_tensors = [sess.graph.get_tensor_by_name(n) for n in self.OUTPUT_TENSOR_NAMES]
            outputs = sess.run(output_tensors, {self.INPUT_TENSOR_NAME: inputs})
            return outputs


def convert_to_opencv(image):
    # RGB -> BGR conversion is performed as well.
    image = image.convert('RGB')
    r,g,b = np.array(image).T
    opencv_image = np.array([b,g,r]).transpose()
    return opencv_image


def take_photo_from_webcam():
    cam = cv2.VideoCapture(0)   # 0 -> index of camera
    s, img = cam.read()
    file_name = WEBCAM_IMAGE_FILENAME
    if s:    # frame captured without any errors
        cv2.namedWindow("cam-test",cv2.WINDOW_AUTOSIZE)
        cv2.imshow("cam-test",img)
        waitKey(0)
        cv2.destroyWindow("cam-test")
        cv2.imwrite(file_name,img) #save image
        return img


def predict(model_filename, image_filename):
    od_model = ObjectDetection(model_filename)

    image = PIL.Image.open(image_filename)
    return od_model.predict_image(image)


def ShowImageWithBestPrediction(imagefilename, predictions):
    
    print("Show Image With Best Prediction")
    image = PIL.Image.open(imagefilename)
    draw = PIL.ImageDraw.Draw(image)
    w, h = image.size

    pred = predictions[0]   
    

    x_min, y_min, x_max, y_max  = pred[0]

    x_min = x_min * w
    y_min = y_min * h
    x_max = x_max * w
    y_max = y_max * h

    draw.rectangle([(x_min,y_min ), (x_max, y_max)], fill=None, width=3, outline ="red")

    image.show()
    
def ShowImageWithBoundingBoxes(imagefilename, predictions):
    
    print("Show Image With Bounding Boxes")
    image = PIL.Image.open(imagefilename)
    draw = PIL.ImageDraw.Draw(image)
    w, h = image.size

       

    for pred in zip(*predictions):
           
        if(pred[1]>0.3):
            x_min, y_min, x_max, y_max  = pred[0]

            x_min = x_min * w
            y_min = y_min * h
            x_max = x_max * w
            y_max = y_max * h
            draw.rectangle([(x_min,y_min ), (x_max, y_max)], fill=None, width=3, outline ="red")

    image.show()


def PrintPredictions(predictions, labels_filename):
    with open(labels_filename) as f:
        labels = [l.strip() for l in f.readlines()]

    for pred in zip(*predictions):
        print(f"Class: {labels[pred[2]]}, Probability: {pred[1]}, Bounding box: {pred[0]}")


def main():
    parser = argparse.ArgumentParser('Object Detection for Custom Vision TensorFlow model')
    parser.add_argument('--image_filename', type=str, default= IMAGE_FILENAME, help='Filename for the input image')
    parser.add_argument('--model_filename', type=str, default=MODEL_FILENAME, help='Filename for the tensorflow model')
    parser.add_argument('--labels_filename', type=str, default=LABELS_FILENAME, help='Filename for the labels file')
    args = parser.parse_args()
    
    #webcam_image_filename = take_photo_from_webcam()
    predictions = predict(args.model_filename, args.image_filename)

    PrintPredictions(predictions, args.labels_filename)    

    ShowImageWithBoundingBoxes(args.image_filename, predictions)

if __name__ == '__main__':
    main()