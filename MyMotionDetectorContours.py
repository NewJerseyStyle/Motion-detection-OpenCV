import cv2 as cv
import numpy as np
from datetime import datetime
import time

class MotionDetectorAdaptative():
    
    def onChange(self, val): #callback when the user change the detection threshold
        self.threshold = val
    
    def __init__(self,threshold=25, doRecord=True, showWindows=True):
        self.writer = None
        self.font = None
        self.doRecord=doRecord #Either or not record the moving object
        self.show = showWindows #Either or not show the 2 windows
        self.frame = None
    
        self.capture=cv.VideoCapture(0)
        self.frame = self.capture.read()[1] #Take a frame to init recorder
        if doRecord:
            self.initRecorder()
        
        self.absdiff_frame = None
        self.previous_frame = None
        
        self.surface = self.frame.shape[0] * self.frame.shape[1]
        self.currentsurface = 0
        self.currentcontours = None
        self.threshold = threshold
        self.isRecording = False
        self.trigger_time = 0 #Hold timestamp of the last detection
        self.es = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9,4))
        
        if showWindows:
            cv.namedWindow("Image")
            cv.createTrackbar("Detection treshold: ", "Image", self.threshold, 100, self.onChange)
        
    def initRecorder(self): #Create the recorder
        codec = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.writer=cv.VideoWriter(datetime.now().strftime("%b-%d_%H_%M_%S")+".wmv", codec, 5, self.frame.shape[1::-1], 1)
        #FPS set to 5 because it seems to be the fps of my cam but should be ajusted to your needs
        self.font = cv.FONT_HERSHEY_SIMPLEX #Creates a font

    def run(self):
        started = time.time()
        while True:
            
            currentframe = self.capture.read()[1]
            instant = time.time() #Get timestamp o the frame
            
            self.processImage(currentframe) #Process the image
            
            if not self.isRecording:
                if self.somethingHasMoved():
                    self.trigger_time = instant #Update the trigger_time
                    if instant > started +10:#Wait 5 second after the webcam start for luminosity adjusting etc..
                        print("Something is moving !")
                        if self.doRecord: #set isRecording=True only if we record a video
                            self.isRecording = True
                currentframe = cv.drawContours(currentframe, self.currentcontours, -1, (0, 255, 0), cv.FILLED)
            else:
                if instant >= self.trigger_time +10 and not self.somethingHasMoved(): #Record during 10 seconds
                    print("Stop recording")
                    self.isRecording = False
                else:
                    cv.putText(currentframe,datetime.now().strftime("%b %d, %H:%M:%S"), (25,30),self.font, 1, (255, 0, 0), 2, cv.LINE_AA) #Put date on the frame
                    self.writer.write(currentframe) #Write the frame
            
            if self.show:
                cv.imshow("Image", currentframe)
                
            c=cv.waitKey(1) % 0x100
            if c==27 or c == 10: #Break if user enters 'Esc'.
                break            
    
    def processImage(self, curframe):
            curframe = cv.GaussianBlur(curframe, (21,21), 0) #Remove false positives
            
            if self.absdiff_frame is None: #For the first time put values in difference, temp and moving_average
                self.absdiff_frame = curframe.copy()
                self.previous_frame = curframe.copy()
                self.average_frame = np.float32(curframe) #Should convert because after runningavg take 32F pictures
            else:
                cv.accumulateWeighted(curframe, self.average_frame, 0.05) #Compute the average
            
            self.previous_frame = self.average_frame.astype(np.uint8) #Convert back to 8U frame
            
            self.absdiff_frame = cv.absdiff(curframe, self.previous_frame) # moving_average - curframe
            
            self.gray_frame = cv.cvtColor(self.absdiff_frame, cv.COLOR_BGR2GRAY) #Convert to gray otherwise can't do threshold
            self.gray_frame = cv.threshold(self.gray_frame, 5, 255, cv.THRESH_BINARY)[1]

            self.gray_frame = cv.dilate(self.gray_frame, self.es) #to get object blobs
            # cv.Erode(self.gray_frame, self.gray_frame, None, 10)

            
    def somethingHasMoved(self):
        
        # Find contours
        contours = cv.findContours(self.gray_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[1]

        self.currentcontours = contours #Save contours
        
        self.currentsurface = sum([cv.contourArea(c) for c in contours]) #For all contours compute the area
        
        avg = (self.currentsurface*100)/self.surface #Calculate the average of contour area on the total size
        self.currentsurface = 0 #Put back the current surface to 0
        
        if avg > self.threshold:
            return True
        else:
            return False

        
if __name__=="__main__":
    detect = MotionDetectorAdaptative(threshold=5, doRecord=True)
    detect.run()
