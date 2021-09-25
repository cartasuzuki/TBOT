"""Sample prediction script for TensorFlow 2"""
import argparse
from pandas.core.series import Series
import tensorflow as tf
import numpy as np
import PIL.Image, PIL.ImageDraw
import tbotcam
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from cv2 import *


MODEL_FILENAME = 'model.pb'
LABELS_FILENAME = 'labels.txt'
IMAGE_FILENAME = 'Tea-Leaves-Growing2.jpg'
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

def CreatePredictionDataFrame(predictions, imagefilename):
    predictionsDF = pd.DataFrame(columns = ["x", "y", "X", "Y"])
    image = PIL.Image.open(imagefilename)
    w, h = image.size
    for pred in zip(*predictions):
        if(pred[1]>0.3):
            predSeries = pd.Series(pred[0], index=predictionsDF.columns)
            predSeries['x'] = predSeries['x'] * w
            predSeries['X'] = predSeries['X'] * w
            predSeries['y'] = predSeries['y'] * h
            predSeries['Y'] = predSeries['Y'] * h
            print("PredSERIES")
            print(predSeries)
            predictionsDF = predictionsDF.append(predSeries,ignore_index=True)
    return predictionsDF

def ConvertPredictionsToRealWorldCoordinates(predictionsDF):
    
    
    cam = tbotcam.cam
    cam.LoadFromFile(cam)
    data = []
    for i in range(len(predictionsDF)) :
        pred = predictionsDF.iloc[i]
        print(pred)
        xyz = cam.calculate_XYZ(cam, pred['x'],pred['y'] )
        XYZ = cam.calculate_XYZ(cam, pred['X'],pred['Y'] )
        item = np.append(xyz,XYZ)
        data.append(item)

    predictionsRealWorldCoordinatesDF = pd.DataFrame(data, columns = ["x", "y", "z", "X", "Y", "Z"])    

    return predictionsRealWorldCoordinatesDF


def test_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 2)
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser('Object Detection for Custom Vision TensorFlow model')
    parser.add_argument('--image_filename', type=str, default= WEBCAM_IMAGE_FILENAME, help='Filename for the input image')
    parser.add_argument('--model_filename', type=str, default=MODEL_FILENAME, help='Filename for the tensorflow model')
    parser.add_argument('--labels_filename', type=str, default=LABELS_FILENAME, help='Filename for the labels file')
    args = parser.parse_args()
    
  #  test_webcam()

    #webcam_image_filename = take_photo_from_webcam()
    predictions = predict(args.model_filename, args.image_filename)
    
    PrintPredictions(predictions, args.labels_filename)    

    ShowImageWithBoundingBoxes(args.image_filename, predictions)

    predictionsDF = CreatePredictionDataFrame(predictions, WEBCAM_IMAGE_FILENAME)
    realworld = ConvertPredictionsToRealWorldCoordinates(predictionsDF)
    print("Real World Coordinates")
    print(realworld)
    print(predictionsDF)
    print("Finished...")

if __name__ == '__main__':
    main()