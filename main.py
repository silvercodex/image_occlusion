import os
import cv2
import numpy as np



lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
path = "images"
def load_images(dir = ""):
    files = os.listdir(dir)
    images = []
    for f in files:
        images.append(cv2.imread(dir + "/" + f))
    return images

def get_edges(images):
    return [cv2.Canny(image,100,200) for image in images]


if __name__ == '__main__':
    images = load_images(path)

    edges = get_edges(images)
    
    pts = np.argwhere(edges[0]==255).reshape(-1,1,2).astype(np.float32)
    gray0 = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY)
    #pts = cv2.goodFeaturesToTrack(edges[0], mask = None, **feature_params)
    
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray0, gray1, pts, None, **lk_params)


    p0 = cv2.goodFeaturesToTrack(gray0, mask = None, **feature_params)
    p0 = pts
    color = np.random.randint(0,255,(len(p1),3)) 
    good_new = p1[st==1]
    good_old = p0[st==1]
    frame = images[1]
    mask = np.zeros_like(frame)
    
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)
    cv2.imwrite("img.png",img)