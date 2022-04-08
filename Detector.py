import cv2 ,time, os , tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file
np.random.seed(20)
class Detector:
    def __init__(self):
        pass
    
    def readClasses(self, classesFilePath):
        with open(classesFilePath,'r') as f:
            self.classesList=f.read().splitlines()
        
        #color list
        modelURL="http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"

        self.colorList=np.random.uniform(low=0, high=255,size=(len(self.classesList),3))
        fileName=os.path.basename(modelURL)
        self.modelName=fileName[:fileName.index('.')]
        self.cacheDir="./pretrained_models"

    def downloadModel(self,modelURL):
        fileName=os.path.basename(modelURL)
        self.modelName=fileName[:fileName.index('.')]
        self.cacheDir="./pretrained_models"
        os.makedirs(self.cacheDir, exist_ok=True)
        get_file(fname=fileName,
                 origin=modelURL,cache_dir=self.cacheDir,cache_subdir="checkpoints",extract=True)
    def loadModel(self):
        print("loading Model"+self.modelName)
        tf.keras.backend.clear_session()
        self.model=tf.saved_model.load(os.path.join(self.cacheDir,"checkpoints",self.modelName, "saved_model"))
        print("Model "+self.modelName+" loaded successfully..") 
    
    def createBoundingBox(self, image, threshold=0.5):
        inputTensor=cv2.cvtColor(image.copy(),cv2.COLOR_BGR2RGB)
        inputTensor=tf.convert_to_tensor(inputTensor,dtype=tf.uint8)
        inputTensor=inputTensor[tf.newaxis,...]
        
        detections=self.model(inputTensor)
        bboxs=detections['detection_boxes'][0].numpy()
        classIndexes=detections['detection_classes'][0].numpy().astype(np.int32)
        classScores=detections['detection_scores'][0].numpy()
        imH, imW, imC=image.shape
        bboxIdx=tf.image.non_max_suppression(bboxs,classScores,max_output_size=50,iou_threshold=threshold,score_threshold=threshold)
        
        
        if len(bboxIdx)!=0 :
            for i in bboxIdx :
                bbox=tuple(bboxs[i].tolist())
                classConfidence= round(classScores[i]*100)
                classIndex=classIndexes[i]
                classLabelText=self.classesList[classIndex]
                classcolor=self.colorList[classIndex]
                displayText='{}: {}%'.format(classLabelText,classConfidence)
                ymin,xmin,ymax,xmax=   bbox
                xmin,xmax,ymin,ymax=(xmin*imW,xmax*imW,ymin*imH,ymax*imH)
                xmin,xmax,ymin,ymax=int (xmin), int(xmax),int(ymin),int(ymax)
                cv2.rectangle(image,(xmin,ymin),(xmax,ymax),color=classcolor,thickness=1)
                cv2.putText(image,displayText,(xmin,ymin-10),cv2.FONT_HERSHEY_PLAIN,1,classcolor,2)
        return image     
                
                
        
    def predictImage(self,imagePath,threshold=0.5):
        image = cv2.imread(imagePath)
        image= cv2.resize(image,(500,500))
        bboxImage=self.createBoundingBox(image,threshold)
        cv2.imwrite(self.modelName+".jpg",bboxImage)
        cv2.imshow("window",image)
        cv2.waitKey(0)
        cv2.destroyAllWindows
    def predictvideo(self,videoPath,threshold=0.5):
        cap=cv2.VideoCapture(videoPath)
        if(cap.isOpened()==False):
            print("error opening file")
            return
        (success, image)=cap.read()
        
        startTime=0
        
        while success:
            currentTime=time.time()
            fps=1/(currentTime-startTime)
            startTime=currentTime
            bboxImage=self.createBoundingBox(image, threshold)
            cv2.putText(bboxImage,"FPS: "+str(int(30)),(20,70),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
            cv2.imshow("result",bboxImage)
            key=cv2.waitKey(1) & 0xFF
            if key==ord("q"):
                break
            (success,image)=cap.read()
        cv2.destroyAllWindows()
        