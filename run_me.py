from Detector import *

modelURL="http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"
classFile="coco.names"
imagePath="t1.jpg"
videoPath="video1.mp4"
threshold=0.5
detector=Detector()
detector.readClasses(classFile)
#detector.downloadModel(modelURL)
detector.loadModel()
detector.predictImage(imagePath,threshold)
#detector.predictvideo(videoPath,threshold)