import numpy as np 
import cv2 as cv

# a,b  are descriptors
def match(a_pts,a_des,b_pts,b_des):
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
    search_params = dict(checks=75)

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(a_des,b_des,2)
    
    a_match = []
    b_match = []
    aggregate = []
    t = 0.7
    for bundle in matches:
        # order m,n where m.d < n.d 
        # if dist of m is greater than 70 percent
        # of n's distance, m,n are too similiar and 
        # thus must be discarded
        if len(bundle) > 1:
            m = bundle[0]
            n = bundle[1]
            if m.distance < t * n.distance:
                aggregate.append(m)
                a_match.append(a_pts[m.queryIdx].pt)
                b_match.append(b_pts[m.trainIdx].pt)

    return a_match, b_match, aggregate
