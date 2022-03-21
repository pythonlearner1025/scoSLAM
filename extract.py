import math
import numpy as np 
import cv2 as cv
from matplotlib import pyplot as plt

class Extractor(object):
    def __init__(self, frame):
        self.frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    def extractAll(self):
        kp = np.concatenate((self.extractGoodFeatures(),self.extractORBLines()),axis=0)
        return np.int0(kp)

    def extractGoodFeatures(self):
        kp = cv.goodFeaturesToTrack(self.frame,1000, 0.02,10,
                useHarrisDetector=True, k=0.04)
        return np.int0(kp)
    
    # use this
    def extractORB(self):
        orb = cv.ORB_create()
        kp = orb.detect(self.frame,None)
        kp, des = orb.compute(self.frame, kp)
        return kp, des
    
    def extractORBLines(self):
        kp = self.extractORB()
        det, lines = extractLines(self.frame)
        if det:
            pts = extractPointsFromLines(lines)
            pts = np.expand_dims(pts,1)
            # combine
            kp = np.concatenate((kp,np.int0(pts)),axis=0)
            return kp
        else:
            return kp

##################################################
# Helper Functions
##################################################

# ToDo:
# apply mask so only bottom half 
# lines are detected
def extractLines(img):

    dst = cv.Canny(img, 50, 200, None, 3)
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    # tweak with this to reduce noise as much as possible
    lines = cv.HoughLinesP(dst, 1, np.pi / 180, 200, None, 80, 8)
    linePts = []
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            linePts.append([[l[0],l[1]],[l[2], l[3]]])
    if not linePts:
        return False, linePts
    return True, linePts

def dist2p(x1,y1,x2,y2):
    return ((x1-x2)**2+(y1-y2)**2)**0.5 

def getdx(slope,c):
    return (-slope+((slope)**2 + 4*c**2)**0.5)/2

def largerX(p1, p2):
    p1x, p2x = p1[0], p2[0] 
    if max(p1x,p2x) == p1[0]:
        return p1
    else:
        return p2

# works, but is it efficient enough
def extractPointsFromLines(lines):
    c = 15
    extra_points = [] 
    for line in lines:
        p1, p2 = line[0], line[1]
        if largerX(p1,p2) == p2:
            pass
        else:
            p1, p2 = line[1], line[0]
        diff = math.ceil(dist2p(p1[0],p1[1],p2[0],p2[1]))
        dx,dy,slope = 0,0,0
        # if diff of x is 0, line goes straight up
        if (p2[0] - p1[0]) == 0:
            dx = 0
            dy = c
        else:
            slope = (p2[1]-p1[1]) / (p2[0] - p1[0])
            dx = getdx(slope,c)
            dy = dx*slope
        # intervals of 3
        #print(f'slope: {slope}')
        #print(f'dx: {dx}, dy: {dy}')
        #print('-'*50)
        for _ in range(0,diff+c,c):
            p1[0] += dx
            p1[1] += dy
            #print(f'x: {p1[0]}, y: {p1[1]}')
            extra_points.append([p1[0], p1[1]])
    #print(extra_points)
    return extra_points
            

            

        



