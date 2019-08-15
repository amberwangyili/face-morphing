import cv2
import dlib
import numpy as np

class Marker():
    def __init__(self,path1,path2):
        
        self.source = cv2.imread(path1,cv2.IMREAD_COLOR)
        self.target = cv2.imread(path2,cv2.IMREAD_COLOR)
        self.landmark1 = []
        self.landmark2 = []
        self.name1  = 'target,s:save'
        self.name2  = 'source,s:save'

    def draw_mark1(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.landmark1.append([x,y])
            cv2.circle(self.source, (x,y),3,(0, 0, 255),-1)

    def draw_mark2(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.landmark2.append([x,y])
            cv2.circle(self.target, (x,y),3,(0, 0, 255),-1)
    

    def gen_mark1(self):
        cv2.namedWindow(self.name1)
        cv2.setMouseCallback(self.name1,self.draw_mark1)

        while(1):
            cv2.imshow(self.name1,self.source)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("s"):
                cv2.destroyAllWindows()
                break
        return np.asarray(self.landmark1)

    def gen_mark2(self):
        cv2.namedWindow(self.name2)
        cv2.setMouseCallback(self.name2,self.draw_mark2)

        while(1):
            cv2.imshow(self.name2,self.target)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("s"):
                cv2.destroyAllWindows()
                break

        return np.asarray(self.landmark2)

    def auto_gen_mark(self):
        predictor_path = "./shape_predictor_68_face_landmarks.dat"
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
        face1 = detector(self.source, 1)
        face2 = detector(self.target,1)
        if len(face1) != 1 or len(face2)!= 1:
            print("Number of faces detected is not 1! ")
        shape1 = predictor(self.source, face1[0])
        shape2 = predictor(self.target, face2[0])

        for i in range(0,shape1.num_parts):
            self.landmark1.append([shape1.part(i).x, shape1.part(i).y])
        for j in range(0,shape2.num_parts):
            self.landmark2.append([shape2.part(j).x, shape2.part(j).y])
        

        width = self.source.shape[1]-1
        height = self.source.shape[0]-1
        
        self.landmark1.append((0, 0))
        self.landmark1.append((0, 0))
        
        self.landmark1.append((0, height))
        self.landmark1.append((0, height))
        
        self.landmark1.append((width, 0))
        self.landmark1.append((width, height))
        
        self.landmark1.append((0,height//2))
        self.landmark1.append((width//2,0))
        self.landmark1.append((width,height//2))
        self.landmark1.append((width//2,height))


        self.landmark2.append((0, 0))
        self.landmark2.append((0, 0))
        
        self.landmark2.append((0, height))
        self.landmark2.append((0, height))
        
        self.landmark2.append((width, 0))
        self.landmark2.append((width, height))
        
        self.landmark2.append((0,height//2))
        self.landmark2.append((width//2,0))
        self.landmark2.append((width,height//2))
        self.landmark2.append((width//2,height))
    
        return np.asarray(self.landmark1),np.asarray(self.landmark2)

