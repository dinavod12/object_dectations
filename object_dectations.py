import numpy as np
import cv2

class Object_det:
     
    def __init__(self,accumulated_weight,ROI_top
                 ,ROI_bottom,ROI_right,ROI_left):
        self._background = None
        self.accumulated_weight = accumulated_weight
        self.ROI_top = ROI_top
        self.ROI_bottom = ROI_bottom
        self.ROI_right = ROI_right
        self.ROI_left = ROI_left
        
    #accumulated_weight
    def cal_accum_avg(self,frame, accumulated_weight):
        #global background
        if self._background is None:
            self._background = frame.copy().astype("float")
            return None
        cv2.accumulateWeighted(frame, self._background, accumulated_weight)
    
    #segment_hand
    def segment_hand(self,frame, threshold=25):
        #global background
        diff = cv2.absdiff(self._background.astype("uint8"), frame)
        _ , thresholded = cv2.threshold(diff, threshold,255,cv2.THRESH_BINARY)
        # Grab the external contours for the image
        contours, hierarchy = cv2.findContours(thresholded.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        else:
            hand_segment_max_cont = max(contours, key=cv2.contourArea)
            return (thresholded, hand_segment_max_cont)
    
    #main
    @property
    def main(self):
        cam = cv2.VideoCapture(0)
        num_frames = 0
        element = 10
        num_imgs_taken = 0
        while True:
            ret, frame = cam.read()
            frame = cv2.flip(frame, 1)
            frame_copy = frame.copy()
            roi = frame[self.ROI_top:self.ROI_bottom,self.ROI_right:self.ROI_left]
            gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

            if num_frames < 60:
                self.cal_accum_avg(gray_frame,self.accumulated_weight)
                if num_frames <= 59:
                    cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT",(80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            
            elif num_frames <= 300: 
                hand = self.segment_hand(gray_frame)
                cv2.putText(frame_copy, "Adjust hand...Gesture for" + str(element), (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)

                if hand is not None:
                    thresholded, hand_segment = hand
                    cv2.drawContours(frame_copy, [hand_segment + (self.ROI_right,self.ROI_top)], -1, (255, 0, 0),1)
                    cv2.putText(frame_copy, str(num_frames)+"For" + str(element),(70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    cv2.imshow("Thresholded Hand Image", thresholded)
            else: 
                hand = self.segment_hand(gray_frame)
                if hand is not None:
                    thresholded, hand_segment = hand
                    cv2.drawContours(frame_copy, [hand_segment + (self.ROI_right,self.ROI_top)], -1, (255, 0, 0),1)
                    cv2.putText(frame_copy, str(num_frames), (70, 45),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    cv2.putText(frame_copy, str(num_imgs_taken) + 'images' +"For"+ str(element), (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 2)
                    cv2.imshow("Thresholded Hand Image", thresholded)
                    if num_imgs_taken <= 300:
                        cv2.imwrite(r"C:\Users\Rudra\Desktop\Data_made\train\\"+str(element)+"\\" +str(num_imgs_taken+300) + '.jpg', thresholded)
                    else:
                        break
                    num_imgs_taken +=1
                else:
                    cv2.putText(frame_copy, 'No hand detected...', (200, 400),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            cv2.rectangle(frame_copy, (self.ROI_left, self.ROI_top), (self.ROI_right,self.ROI_bottom), (255,128,0), 3)
            
            cv2.putText(frame_copy, "DataFlair hand sign recognition_ _ _", (10, 20), cv2.FONT_ITALIC, 0.5, (51,255,51), 1)
    
            num_frames += 1

            cv2.imshow("Sign Detection", frame_copy)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
        cv2.destroyAllWindows()
        cam.release()
        
#The parameter should be provided by you ..according to your needs
    
#Object_det(0.5,50,300,50,500).main
        
#Single image
class Object_det_single:
    def __init__(self,accumulated_weight,ROI_top
                 ,ROI_bottom,ROI_right,ROI_left):
        self._background = None
        self.accumulated_weight = accumulated_weight
        self.ROI_top = ROI_top
        self.ROI_bottom = ROI_bottom
        self.ROI_right = ROI_right
        self.ROI_left = ROI_left
        
    #accumulated_weight
    def cal_accum_avg(self,frame, accumulated_weight):
        #global background
        if self._background is None:
            self._background = frame.copy().astype("float")
            return None
        cv2.accumulateWeighted(frame, self._background, accumulated_weight)
    
    #segment_hand
    def segment_hand(self,frame, threshold=25):
        #global background
        diff = cv2.absdiff(self._background.astype("uint8"), frame)
        _ , thresholded = cv2.threshold(diff, threshold,255,cv2.THRESH_BINARY)
        # Grab the external contours for the image
        contours, hierarchy = cv2.findContours(thresholded.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        else:
            hand_segment_max_cont = max(contours, key=cv2.contourArea)
            return (thresholded, hand_segment_max_cont)
    
    #main
    @property
    def main(self):
        cam = cv2.VideoCapture(0)
        num_frames = 0
        element = 10
        num_imgs_taken = 0
        while True:
            ret, frame = cam.read()
            frame = cv2.flip(frame, 1)
            frame_copy = frame.copy()
            roi = frame[self.ROI_top:self.ROI_bottom,self.ROI_right:self.ROI_left]
            gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

            if num_frames < 60:
                self.cal_accum_avg(gray_frame,self.accumulated_weight)
                if num_frames <= 59:
                    cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT",(80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            
            elif num_frames <= 300: 
                hand = self.segment_hand(gray_frame)
                cv2.putText(frame_copy, "Adjust hand...Gesture for" + str(element), (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)

                if hand is not None:
                    thresholded, hand_segment = hand
                    cv2.drawContours(frame_copy, [hand_segment + (self.ROI_right,self.ROI_top)], -1, (255, 0, 0),1)
                    cv2.putText(frame_copy, str(num_frames)+"For" + str(element),(70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    cv2.imshow("Thresholded Hand Image", thresholded)
            else: 
                hand = self.segment_hand(gray_frame)
                if hand is not None:
                    thresholded, hand_segment = hand
                    cv2.drawContours(frame_copy, [hand_segment + (self.ROI_right,self.ROI_top)], -1, (255, 0, 0),1)
                    cv2.putText(frame_copy, str(num_frames), (70, 45),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    cv2.putText(frame_copy, str(num_imgs_taken) + 'images' +"For"+ str(element), (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 2)
                    cv2.imshow("Thresholded Hand Image", thresholded)
                    if num_imgs_taken <= 0:
                        cv2.imwrite(r"C:\Users\Rudra\Desktop\Data_made\train\\"+str(element)+"\\" +str(num_imgs_taken+300) + '.jpg', thresholded)
                    else:
                        break
                    num_imgs_taken +=1
                else:
                    cv2.putText(frame_copy, 'No hand detected...', (200, 400),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            cv2.rectangle(frame_copy, (self.ROI_left, self.ROI_top), (self.ROI_right,self.ROI_bottom), (255,128,0), 3)
            
            cv2.putText(frame_copy, "DataFlair hand sign recognition_ _ _", (10, 20), cv2.FONT_ITALIC, 0.5, (51,255,51), 1)
    
            num_frames += 1

            cv2.imshow("Sign Detection", frame_copy)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
        cv2.destroyAllWindows()
        cam.release()
        
#Object_det_single(0.5,50,300,50,500).main
   