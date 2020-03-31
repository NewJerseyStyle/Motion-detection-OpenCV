import cv2 as cv
import numpy as np
from datetime import datetime
import time

class MotionDetectorInstantaneous():
    
    def onChange(self, val): #callback when the user change the detection threshold
        self.threshold = val
    
    def __init__(self,threshold=8, doRecord=True, showWindows=True):
        self.writer = None
        self.font = None
        self.doRecord=doRecord #Either or not record the moving object
        self.show = showWindows #Either or not show the 2 windows
        self.frame = None
    
        self.capture=cv.VideoCapture(0)
        self.frame = self.capture.read()[1] #Take a frame to init recorder
        if doRecord:
            self.initRecorder()
        
        self.frame1gray = cv.cvtColor(self.frame, cv.COLOR_BGR2GRAY).astype(np.uint8) #Convert to gray otherwise can't do threshold
        
        #Will hold the thresholded result
        self.res = np.zeros(self.frame.shape[1::-1]).astype(np.uint8)
        
        self.frame2gray = np.zeros(self.frame.shape[1::-1]).astype(np.uint8) #Gray frame at t
        
        self.width = self.frame.shape[0]
        self.height = self.frame.shape[1]
        self.nb_pixels = self.width * self.height
        self.threshold = threshold
        self.isRecording = False
        self.trigger_time = 0 #Hold timestamp of the last detection

        self.kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
        
        if showWindows:
            cv.namedWindow("Image")
            cv.createTrackbar("Detection treshold: ", "Image", self.threshold, 100, self.onChange)
        
    def initRecorder(self): #Create the recorder
        codec = cv.VideoWriter_fourcc('M', 'J', 'P', 'G') #('W', 'M', 'V', '2')
        self.writer=cv.VideoWriter(datetime.now().strftime("%b-%d_%H_%M_%S")+".wmv", codec, 5, self.frame.shape[1::-1], 1)
        #FPS set to 5 because it seems to be the fps of my cam but should be ajusted to your needs
        self.font = cv.FONT_HERSHEY_SIMPLEX #Creates a font

    def run(self):
        started = time.time()
        while True:
            
            curframe = self.capture.read()[1]
            instant = time.time() #Get timestamp o the frame
            
            self.processImage(curframe) #Process the image
            
            if not self.isRecording:
                if self.somethingHasMoved():
                    self.trigger_time = instant #Update the trigger_time
                    if instant > started +5:#Wait 5 second after the webcam start for luminosity adjusting etc..
                        print(datetime.now().strftime("%b %d, %H:%M:%S"), "Something is moving !")
                        if self.doRecord: #set isRecording=True only if we record a video
                            self.isRecording = True
            else:
                if instant >= self.trigger_time +10: #Record during 10 seconds
                    print(datetime.now().strftime("%b %d, %H:%M:%S"), "Stop recording")
                    self.isRecording = False
                else:
                    cv.putText(curframe,datetime.now().strftime("%b %d, %H:%M:%S"), (25,30),self.font, 1, (255, 0, 0), 2, cv.LINE_AA) #Put date on the frame
                    self.writer.write(curframe) #Write the frame
            
            if self.show:
                cv.imshow("Image", curframe)
                cv.imshow("Res", self.res)
                
            self.frame2gray = self.frame1gray.copy()
            c=cv.waitKey(1) % 0x100
            if c==27 or c == 10: #Break if user enters 'Esc'.
                break            
    
    def processImage(self, frame):
        self.frame2gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #Convert to gray otherwise can't do threshold
        
        #Absdiff to get the difference between to the frames
        self.res = cv.absdiff(self.frame1gray, self.frame2gray)
        
        #Remove the noise and do the threshold
        self.res = cv.GaussianBlur(self.res, (21,21), 0) #Remove false positives
        self.res = cv.morphologyEx(self.res, cv.MORPH_OPEN, self.kernel)
        self.res = cv.morphologyEx(self.res, cv.MORPH_CLOSE, self.kernel)
        self.res = cv.threshold(self.res, 10, 255, cv.THRESH_BINARY_INV)[1]

    def somethingHasMoved(self):
        nb=0 #Will hold the number of black pixels
        min_threshold = (self.nb_pixels/100) * self.threshold #Number of pixels for current threshold
        nb = self.nb_pixels - cv.countNonZero(self.res)
        if (nb) > min_threshold:
           return True
        else:
           return False
        
if __name__=="__main__":
    detect = MotionDetectorInstantaneous(doRecord=True)
    detect.run()
