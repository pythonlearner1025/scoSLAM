import cv2 as cv

#green
COLOR = (0,255,0)

def draw(img, pts):
    
    for pt in pts:
        cx,cy = int(pt[0][0]), int(pt[0][1])
        cv.circle(img, (cx,cy), 1, COLOR, -1)

