import cv2 as cv
import numpy as np
import os
import imutils
from extract import Extractor
from match import match
from draw import draw
from utils import mergeNumpyList

def main_loop():
    # TODO

    # now that matching complete, use matched points
    # to initialize the 3D map via triangulation, and 
    # then proceed from there 
    BASE = "unlabeled/"
    VIDS = list(range(5,10))
    test = os.path.join(BASE,str(VIDS[0])+'.mp4')

    cap = cv.VideoCapture(test)

    prev = None
    use = False
    c = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret and use:
            print('here')
            currExt = Extractor(frame)
            # kps, des
            currPts, currDes = currExt.extractORB()
            
            prevExt = Extractor(prev)
            # kps, des
            prevPts, prevDes = prevExt.extractORB()
            
            ################################################## 
            # find fundemental matrix using prev, curr pts
            curr_matches,prev_matches,total_matches = match(currPts,currDes,prevPts, prevDes)
            
            cm = np.int32(curr_matches)
            pm = np.int32(prev_matches)
            F,mask = cv.findFundamentalMat(cm,pm,cv.FM_LMEDS)
            
            print(F)
            # select only inliner points
            cm = cm[mask.ravel()==1]
            pm = pm[mask.ravel()==1]

            print(cm.shape, pm.shape)

            ##################################################  
            # for converting back to matrix
            #pts = cv.KeyPoint_convert(kp)

            # match
            
            img_matches = np.empty((max(frame.shape[0], prev.shape[0]), 
                frame.shape[1]+prev.shape[1], 3), dtype=np.uint8)

            cv.drawMatches(frame,currPts, prev, prevPts, total_matches, img_matches,
                    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            # hide single points for now
            #draw(frame, currPts)

            cv.imshow('Matches',img_matches)
            cv.waitKey(1)
            use = False
        c+=1
        if c % 5 == 0:
            prev = frame
            use = True

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main_loop()



