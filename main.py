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
    files.sort()
    for f in files:
        print(f)
        images.append(cv2.imread(dir + "/" + f))
    return images

def get_edges(images):
    return [cv2.Canny(image,100,200) for image in images]

def to_warp(w):
    w_ = np.zeros(shape[0]*shape[1]).astype(np.float16)
    x = int(w[0])
    y = int(w[1])
    if x >=0 and y >=0 and x <=shape[1]-1 and y <=shape[0]-1:
        w_[shape[1]*y+x] = (1-(w[0]-x))*(1-(w[1]-y))
        if x <shape[1]-1:
            w_[shape[1]*y+x+1] = ((w[0]-x))*(1-(w[1]-y))
        if y <shape[0]-1:
            w_[shape[1]*(y+1)+x] = (1-(w[0]-x))*((w[1]-y))
        if x <shape[1]-1 and y <shape[0]-1:
            w_[shape[1]*(y+1)+x+1] = ((w[0]-x))*((w[1]-y))
    
    return w_

def estimate_warp(edges,image0,image1):
    pts = np.argwhere(edges>0)
    pts[:,[0,1]] = pts[:,[1,0]]
    pts = pts.reshape(-1,1,2).astype(np.float32)
    gray0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray0, gray1, pts, None, **lk_params)
    color = np.random.randint(0,255,(len(p1),3)) 
    h, mask = cv2.findHomography(pts, p1, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    old_points = pts.reshape(-1,2).T
    old_points = np.append(old_points,np.ones((1,old_points.shape[1])), 0)
    new_points = np.dot(h,old_points)
    new_points/=new_points[2,:]
    new_points = new_points[:2,:].T
    d = (p1.reshape(-1,2)-new_points)

    d = np.sqrt(d[:,0]**2 + d[:,1]**2)

    points = np.argwhere(d>2*d.std()).reshape(-1)
    p0_back = pts[points]
    p1_back = p1[points]

    h_back, mask_back = cv2.findHomography(p0_back, p1_back, method=cv2.RANSAC, ransacReprojThreshold=5.0)

    points = np.argwhere(d<=2*d.std()).reshape(-1)
    p0_occ = pts[points]
    p1_occ = p1[points]

    h_occ, mask_occ = cv2.findHomography(p0_occ, p1_occ, method=cv2.RANSAC, ransacReprojThreshold=5.0)


    if np.linalg.norm(h_back[:2,2])>np.linalg.norm(h_occ[:2,2]):
        h_back,h_occ = h_occ,h_back

    coor = np.array([(x,y,1) for y in range(edges.shape[0]) for x in range(edges.shape[1])]).T

    coor_trans = np.dot(np.linalg.inv(h_back),coor)
    coor_trans/=coor_trans[2,:]
    coor_trans = coor_trans[:2,:].T
    warp_back = np.apply_along_axis(to_warp,1,coor_trans.astype(np.float16))


    coor = np.array([(x,y,1) for x in range(edges.shape[1]) for y in range(edges.shape[0])]).T

    coor_trans = np.dot(np.linalg.inv(h_occ),coor)
    coor_trans/=coor_trans[2,:]
    coor_trans = coor_trans[:2,:].T
    warp_occ = np.apply_along_axis(to_warp,1,coor_trans.astype(np.float16))


    return warp_back,warp_occ, h_back, h_occ






if __name__ == '__main__':
    
    images = load_images(path)
    shape = images[0].shape
    edges = get_edges(images)
    warp_backs = []
    warp_occs = []
    h_backs = []
    h_occs = []
    for i in range(1,len(images)):
        print(i)
        warp_back,warp_occ, h_back,h_occ = estimate_warp(edges[0],images[0],images[i])
        warp_backs.append(warp_back)
        warp_occs.append(warp_occ)
        h_backs.append(h_back)
        h_occs.append(h_occ)

    


    
    pts = np.argwhere(edges[0]>0)
    pts[:,[0,1]] = pts[:,[1,0]]
    pts = pts.reshape(-1,1,2).astype(np.float32)
    gray0 = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY)
    #pts = cv2.goodFeaturesToTrack(edges[0], mask = None, **feature_params)
    
    p1, st, err = cv2.calcOpticalFlowPyrLK(gray0, gray1, pts, None, **lk_params)


    #p0 = cv2.goodFeaturesToTrack(gray0, mask = None, **feature_params)
    p0 = pts
    color = np.random.randint(0,255,(len(p1),3)) 
    #good_new = p1[st==1]
    #good_old = p0[st==1]
    
    h, mask = cv2.findHomography(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    frame = images[1]
    mask = np.zeros_like(frame)
    old_points = p0.reshape(-1,2).T
    old_points = np.append(old_points,np.ones((1,old_points.shape[1])), 0)

    new_points = np.dot(h,old_points)
    new_points/=new_points[2,:]
    new_points = new_points[:2,:].T

    d = (p1.reshape(-1,2)-new_points)

    d = np.sqrt(d[:,0]**2 + d[:,1]**2)

    points = np.argwhere(d>2*d.std()).reshape(-1)
    p0_back = p0[points]
    p1_back = p1[points]

    h_back, mask_back = cv2.findHomography(p0_back, p1_back, method=cv2.RANSAC, ransacReprojThreshold=5.0)

    points = np.argwhere(d<=2*d.std()).reshape(-1)
    p0_occ = p0[points]
    p1_occ = p1[points]

    h_occ, mask_occ = cv2.findHomography(p0_occ, p1_occ, method=cv2.RANSAC, ransacReprojThreshold=5.0)

    h_back[:2,2]

    if np.linalg.norm(h_back[:2,2])>np.linalg.norm(h_occ[:2,2]):
        h_back,h_occ = h_occ,h_back
    
    coor = np.array([(x,y,1) for y in range(edges[0].shape[0]) for x in range(edges[0].shape[1])]).T

    coor_trans = np.dot(np.linalg.inv(h_back),coor)
    coor_trans/=coor_trans[2,:]
    coor_trans = coor_trans[:2,:].T

    for i,(new,old) in enumerate(zip(p0, p1)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        #frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)
    cv2.imwrite("img.png",img)