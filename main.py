import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

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

def estimate_warp(edges,image0,image1, k = 5):
    feat_detector = cv2.ORB_create(nfeatures=500)
    image_1_kp, image_1_desc = feat_detector.detectAndCompute(image0, None)
    image_2_kp, image_2_desc = feat_detector.detectAndCompute(image1, None)
    bfm = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bfm.match(image_1_desc, image_2_desc),
                     key=lambda x: x.distance)[:500]

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

    points = np.argwhere(d>1*d.std()).reshape(-1)
    p0_back = pts[points]
    p1_back = p1[points]
    print(len(points), len(d))
    h_back, mask_back = cv2.findHomography(p0_back, p1_back, method=cv2.RANSAC, ransacReprojThreshold=5.0)

    points = np.argwhere(d<=1*d.std()).reshape(-1)
    p0_occ = pts[points]
    p1_occ = p1[points]

    h_occ, mask_occ = cv2.findHomography(p0_occ, p1_occ, method=cv2.RANSAC, ransacReprojThreshold=5.0)


    if np.linalg.norm(h_back[:2,2])<np.linalg.norm(h_occ[:2,2]):
        h_back,h_occ = h_occ,h_back


    coor = np.array([(x,y) for y in range(edges.shape[0]) for x in range(edges.shape[1])])
    #import pdb; pdb.set_trace()
    x = p0_back.reshape(-1,2)[:,0]
    y = p0_back.reshape(-1,2)[:,1]

    x1 = p1_back.reshape(-1,2)[:,0]
    y1 = p1_back.reshape(-1,2)[:,1]

    x2 = coor[:,0]
    y2 = coor[:,1]

    #f_x = interp2d(x, y, x1, kind='linear')
    #f_y = interp2d(x, y, y1, kind='linear')
    knn = KNeighborsRegressor(k)
    knn.fit(p0_back.reshape(-1,2),p1_back.reshape(-1,2)-p0_back.reshape(-1,2))
    V_back = knn.predict(coor) + coor
    #X = np.array([f_x(x_i,y_i) for x_i,y_i in zip(x2,y2)])
    #Y = np.array([f_y(x_i,y_i) for x_i,y_i in zip(x2,y2)])
    #V_back = np.concatenate([X.reshape(-1,1),Y.reshape(-1,1)],axis = 1)



    x = p0_occ.reshape(-1,2)[:,0]
    y = p0_occ.reshape(-1,2)[:,1]

    x1 = p1_occ.reshape(-1,2)[:,0]
    y1 = p1_occ.reshape(-1,2)[:,1]

    x2 = coor[:,0]
    y2 = coor[:,1]

    knn = KNeighborsRegressor(k)
    knn.fit(p0_occ.reshape(-1,2),p1_occ.reshape(-1,2)-p0_occ.reshape(-1,2))
    V_occ = knn.predict(coor) + coor
    #f_x = interp2d(x, y, x1, kind='linear')
    #f_y = interp2d(x, y, y1, kind='linear')

    #X = np.array([f_x(x_i,y_i) for x_i,y_i in zip(x2,y2)])
    #Y = np.array([f_y(x_i,y_i) for x_i,y_i in zip(x2,y2)])
    #V_occ = np.concatenate([X.reshape(-1,1),Y.reshape(-1,1)],axis = 1)
    if np.linalg.norm(h_back[:2,2])>np.linalg.norm(h_occ[:2,2]):
        return V_occ,V_back, h_back, h_occ
    else:
        return V_back,V_occ, h_back, h_occ

    coor = np.array([(x,y,1) for y in range(edges.shape[0]) for x in range(edges.shape[1])]).T

    coor_trans = np.dot(np.linalg.inv(h_back),coor)
    coor_trans/=coor_trans[2,:]
    coor_trans = coor_trans[:2,:].T
    V_back = coor_trans
    #warp_back = np.apply_along_axis(to_warp,1,coor_trans.astype(np.float16))


    coor = np.array([(x,y,1) for y in range(edges.shape[0]) for x in range(edges.shape[1])]).T

    coor_trans = np.dot(np.linalg.inv(h_occ),coor)
    coor_trans/=coor_trans[2,:]
    coor_trans = coor_trans[:2,:].T
    V_occ = coor_trans
    #warp_occ = np.apply_along_axis(to_warp,1,coor_trans.astype(np.float16))


    return V_back,V_occ, h_back, h_occ


def background_estimate(image, h):
    return cv2.warpPerspective(image,np.linalg.inv(h),(image.shape[1],image.shape[0]))


def transform_from_motion(image, v):
    image_new = np.zeros(image.shape)
    
    def process_pixel(i):
        image_new[i] = im(v[i])#np.dot(get_w(v[i]),im(v[i])).reshape(-1)

    process_pixel_v = np.vectorize(process_pixel)
    #def w(v_0):
    #    x = int(v_0[0])
    #    y = int(v_0[1])
    #   return np.array([(1-(v_0[0]-x))*(1-(v_0[1]-y)),((v_0[0]-x))*(1-(v_0[1]-y)),(1-(v_0[0]-x))*((v_0[1]-y)),((v_0[0]-x))*((v_0[1]-y))]).reshape(1,-1)
    def im(v_0):
        x = int(v_0[0])
        y = int(v_0[1])
        x = int(np.clip(x,0,shape[1]-1))
        y = int(np.clip(y,0,shape[0]-1))

        #def check(dex):
        #    if dex < image.shape[0] and dex >=0:
        #        return image[dex]
        #    else:
        #        return np.array([0.0,0.0,0.0])
        return image[shape[1]*y+x]#np.array([check(shape[1]*y+x),check(shape[1]*y+x+1),check(shape[1]*(y+1)+x),check(shape[1]*(y+1)+x+1)])

    process_pixel_v(list(range(image.shape[0])))
    return image_new

def get_im(v_0,image):
    x = int(v_0[0])
    y = int(v_0[1])
    x = int(np.clip(x,0,shape[1]-1))
    y = int(np.clip(y,0,shape[0]-1))

    #def check(dex):
    #    if dex < image.shape[0] and dex >=0:
    #        return image[dex]
    #    else:
    #        return np.array([0.0,0.0,0.0])
    return image[shape[1]*y+x]#np.array([check(shape[1]*y+x),check(shape[1]*y+x+1),check(shape[1]*(y+1)+x),check(shape[1]*(y+1)+x+1)])

def get_w(v_0):
    x = int(v_0[0])
    y = int(v_0[1])
    return np.array([(1-(v_0[0]-x))*(1-(v_0[1]-y)),((v_0[0]-x))*(1-(v_0[1]-y)),(1-(v_0[0]-x))*((v_0[1]-y)),((v_0[0]-x))*((v_0[1]-y))]).reshape(1,-1)

def get_dex(x,y):
    return [shape[1]*x + y, shape[1]*y+x+1,shape[1]*(y+1)+x,shape[1]*(y+1)+x+1]

def optimize_images(lambda1 = 1, lambda2 = .1, lambda3 = 3000, lambda4 = .5,alpha = .01, beta = .001):
    global A
    global background
    global occlusion
    grad_A = np.zeros(A.shape)
    grad_background = np.zeros(background.shape)
    grad_occlusion = np.zeros(occlusion.shape)

    loss = []
    for i in range(len(V_backs)):
        loss.append(images[i+1]-transform_from_motion(occlusion,V_occs[i])-transform_from_motion(A,V_occs[i])*transform_from_motion(background,V_backs[i]))
    Im_A_G = grad_spacial(A.astype(np.float32))
    Im_O_G = grad_spacial(occlusion)
    Im_B_G = grad_spacial(background)
    def calc_gradA(V_b,V_o):
        def check(dex,im):
            if dex < im.shape[0] and dex >=0:
                return True
            else:
                return False

        def process_pixel(dex,v_o,v_b):
            grad_A[dex] += -get_im(v_b,background).reshape(-1)*np.sign(l[dex]) + 2*lambda1*Im_A_G[dex]#-get_w(v_o)[0][j]*np.dot(get_w(v_b),get_im(v_b,background)).reshape(-1)*np.sign(l[dex]) + 2*lambda1*Im_A_G[dex]#-get_w(v_o)[0][j]*get_im(v_b,background).reshape(-1)*np.sign(l[dex]) + 2*lambda1*Im_A_G[dex]#-get_w(v_o)[0][j]*np.dot(get_w(v_b),get_im(v_b,background)).reshape(-1)*np.sign(l[dex]) + 2*lambda1*Im_A_G[dex]

        #for i in range(len(V_b)):
        def process_warp(i):
            x_o = int(V_o[i][0])
            y_o = int(V_o[i][1])
            x_o = int(np.clip(x_o,0,shape[1]-1))
            y_o = int(np.clip(y_o,0,shape[0]-1))
            dex = shape[1]*y_o+x_o
            #dexes = get_dex(x_o,y_o)
            #for j,dex in enumerate(dexes):
             #   if check(dex,background):
            process_pixel(dex,V_o[i],V_b[i])
        
        
        process_warp_v = np.vectorize(process_warp)

        process_warp_v(list(range(len(V_b))))

    def calc_gradO(V_b,V_o):
        def check(dex,im):
            if dex < im.shape[0] and dex >=0:
                return True
            else:
                return False
        def process_pixel(dex,v_o,v_b):
            grad_occlusion[dex] += -np.sign(l[dex]) + lambda2*np.sign(Im_O_G[dex]) + 2*lambda3*Im_O_G[dex]*(Im_B_G[dex]**2)#-get_w(v_o)[0][j]*np.sign(l[dex]) + lambda2*np.sign(Im_O_G[dex]) + 2*lambda3*Im_O_G[dex]*(Im_B_G[dex]**2)
        
        
        def process_warp(i):
            x_o = int(V_o[i][0])
            y_o = int(V_o[i][1])
            x_o = int(np.clip(x_o,0,shape[1]-1))
            y_o = int(np.clip(y_o,0,shape[0]-1))
            dex = shape[1]*y_o+x_o
            #dexes = get_dex(x_o,y_o)
            #for j,dex in enumerate(dexes):
             #   if check(dex,background):
            process_pixel(dex,V_o[i],V_b[i])
        process_warp_v = np.vectorize(process_warp)

        process_warp_v(list(range(len(V_b))))
    
    def calc_gradB(V_b,V_o):
        def check(dex,im):
            if dex < im.shape[0] and dex >=0:
                return True
            else:
                return False
        def process_pixel(dex,v_o,v_b):
            grad_background[dex] += -get_im(v_o,A).reshape(-1)*np.sign(l[dex]) + lambda2*np.sign(Im_B_G[dex]) + 2*lambda3*Im_B_G[dex]*(Im_O_G[dex]**2)#-get_w(v_b)[0][j]*np.dot(get_w(v_o),get_im(v_o,A)).reshape(-1)*np.sign(l[dex]) + lambda2*np.sign(Im_B_G[dex]) + 2*lambda3*Im_B_G[dex]*(Im_O_G[dex]**2)
        
        def process_warp(i):
            x_o = int(V_o[i][0])
            y_o = int(V_o[i][1])
            x_o = int(np.clip(x_o,0,shape[1]-1))
            y_o = int(np.clip(y_o,0,shape[0]-1))
            dex = shape[1]*y_o+x_o
            #dexes = get_dex(x_o,y_o)
            #for j,dex in enumerate(dexes):
             #   if check(dex,background):
            process_pixel(dex,V_o[i],V_b[i])
        process_warp_v = np.vectorize(process_warp)

        process_warp_v(list(range(len(V_b))))


    total_loss1 = np.sum([np.abs(l) for l in loss])
    total_loss1 += (Im_A_G**2).sum()*lambda1 + lambda2*(np.abs(Im_B_G).sum()+np.abs(Im_O_G).sum())
    total_loss1 += lambda3*(Im_O_G**2 * Im_B_G**2).sum()
    print(total_loss1)
    #l = loss[0]
    for i in range(len(loss)):
        l = loss[i]
        calc_gradA(V_backs[i],V_occs[i])
        calc_gradO(V_backs[i],V_occs[i])
        calc_gradB(V_backs[i],V_occs[i])
    grad_background = np.clip(grad_background,-.1,.1)
    grad_occlusion = np.clip(grad_occlusion,-.1,.1)
    grad_A = np.clip(grad_A,-.1,.1)

    background -= beta*grad_background
    occlusion -= beta*grad_occlusion
    A -= alpha*grad_A
    occlusion = np.clip(occlusion,0,1)
    background = np.clip(background,0,1)
    A = np.clip(A,0,1)
    
    Im_A_G = grad_spacial(A.astype(np.float32))
    Im_O_G = grad_spacial(occlusion)
    Im_B_G = grad_spacial(background)
    loss = []
    for i in range(len(V_backs)):
        loss.append(images[i+1]-transform_from_motion(occlusion,V_occs[i])-transform_from_motion(A,V_occs[i])*transform_from_motion(background,V_backs[i]))

    total_loss = np.sum([np.abs(l) for l in loss])
    total_loss += (Im_A_G**2).sum()*lambda1 + lambda2*(np.abs(Im_B_G).sum()+np.abs(Im_O_G).sum())
    total_loss += lambda3*(Im_O_G**2 * Im_B_G**2).sum()
    print(total_loss)
    #print(grad_A)
    return total_loss1, total_loss



def optimize_warps(lambda1 = 1, lambda2 = .1, lambda3 = 3000, lambda4 = .5, alpha = .01, beta = .1):
    global V_backs
    global V_occs
    global A
    global background
    global occlusion
    G_backs = [np.zeros(v.shape) for v in V_backs]
    G_occs = [np.zeros(v.shape) for v in V_occs]
    coor = np.array([(x,y) for y in range(shape[0]) for x in range(shape[1])])
    loss = []
    for i in range(len(V_backs)):
        loss.append(images[i+1]-transform_from_motion(occlusion,V_occs[i])-transform_from_motion(A,V_occs[i])*transform_from_motion(background,V_backs[i]))
    

    def calc_gradOcc(V_b,V_o,index):
        def check(dex,im):
            if dex < im.shape[0] and dex >=0:
                return True
            else:
                return False

        def process_pixel(dex,v_o,v_b):
            #w_grad = np.zeros(4)
            #x_o = int(v_o[0])
            #y_o = int(v_o[1])
  
            #dexes = get_dex(x_o,y_o)
            #for i,d in enumerate(dexes):
            #    if check(d,occlusion):
            #        w_grad[i] += (-(occlusion[dexes[i]] + A[dexes[i]]*np.dot(get_w(v_b),get_im(v_b,background)))*np.sign(l[dex])).mean()
            #w = get_w(v_b)-alpha*w_grad
            #w = w.reshape(-1)
            #xd = w[3]/(w[2]+w[3])
            #yd = w[3]/(w[1]+w[3])
            G_occs[index][dex] += (V_O_grad[dex]*l[dex].mean()  + (A_grad[dex]*background[dex]*np.sign(l[dex])).mean())   #np.array([xd - (v_o[0]-x_o),yd - (v_o[1]-y_o)]) + lambda4*np.sign(V_O_grad[dex])

        V_O_grad = grad_spacial(V_o-coor)
        A_grad = grad_spacial(A)


        #for i in range(len(V_b)):
        def process_warp(i):
            #x_o = int(V_o[i][0])
            #y_o = int(V_o[i][1])
            #dexes = get_dex(x_o,y_o)
            #for j,dex in enumerate(dexes):
            process_pixel(i,V_o[i],V_b[i])
        
        
        process_warp_v = np.vectorize(process_warp)

        process_warp_v(list(range(len(V_b))))
    

    def calc_gradBack(V_b,V_o,index):
        def check(dex,im):
            if dex < im.shape[0] and dex >=0:
                return True
            else:
                return False

        def process_pixel(dex,v_o,v_b):
            #w_grad = np.zeros(4)
            #x_o = int(v_o[0])
            #y_o = int(v_o[1])
            #dexes = get_dex(x_o,y_o)
            #for i,d in enumerate(dexes):
            #    if check(d,occlusion):
            #        w_grad[i] += (-background[dexes[i]]*np.dot(get_w(v_o),get_im(v_o,A))*np.sign(l[dex])).mean()
            #w = get_w(v_b)-alpha*np.clip(w_grad,-.1,.1)
            #w = w.reshape(-1)
            #xd = w[3]/(w[2]+w[3])
            #yd = w[3]/(w[1]+w[3])
            G_backs[index][dex] += V_B_grad[dex]*(A[dex]*l[dex]).mean()#np.array([xd - (v_o[0]-x_o),yd - (v_o[1]-y_o)]) + lambda4*np.sign(V_B_grad[dex])

        

        V_B_grad = grad_spacial(V_b-coor)

        #for i in range(len(V_b)):
        def process_warp(i):
            #x_o = int(V_o[i][0])
            #y_o = int(V_o[i][1])
            #dexes = get_dex(x_o,y_o)
            #for j,dex in enumerate(dexes):
            process_pixel(i,V_o[i],V_b[i])
        
        
        process_warp_v = np.vectorize(process_warp)

        process_warp_v(list(range(len(V_b))))
    total_loss1 = np.sum([np.abs(l) for l in loss])
    print(total_loss1)

    for i in range(len(V_backs)):
        l = loss[i]
        calc_gradOcc(V_backs[i],V_occs[i],i)
        calc_gradBack(V_backs[i],V_occs[i],i)
    G_backs = [np.clip(g,-.1,.1) for g in G_backs]
    G_occs = [np.clip(g,-.1,.1) for g in G_occs]
    for i in range(len(V_backs)):
        V_backs[i]-=beta*G_backs[i]
        V_occs[i]-=beta*G_occs[i]
    
    loss = []
    for i in range(len(V_backs)):
        loss.append(images[i+1]-transform_from_motion(occlusion,V_occs[i])-transform_from_motion(A,V_occs[i])*transform_from_motion(background,V_backs[i]))


    total_loss = np.sum([np.abs(l) for l in loss])
    print(total_loss)
    return total_loss1, total_loss

    

            
        


def grad_spacial(im):
    s = im.shape[-1]
    temp = im.reshape(shape[0],shape[1],-1)
    x = cv2.Sobel(temp,cv2.CV_64F,1,0,ksize=3)
    y = cv2.Sobel(temp,cv2.CV_64F,0,1,ksize=3)
    return (x + y).reshape(-1,s)

def downsample_images(images,d):
    x= int(images[0].shape[1]/(2**d))
    y= int(images[0].shape[0]/(2**d))
    return [cv2.resize(image,(x,y)) for image in images]


if __name__ == '__main__':
    
    images_original = load_images(path)
    l = len(images_original)//2
    images_original = [images_original[l]] + images_original[:l] + images_original[l+1:]  
    min_size = min(images_original[0].shape[:2])
    depth = int(np.log2(min_size)) - 3
    images = downsample_images(images_original, depth)
    shape = images[0].shape
    edges = get_edges(images)
    coor = np.array([(x,y) for y in range(shape[0]) for x in range(shape[1])])
    V_backs = []
    V_occs = []
    h_backs = []
    h_occs = []
    background= []
    target = []
    A = []
    for i in range(1,len(images)):
        print(i)
        V_back,V_occ,h_back,h_occ = estimate_warp(edges[0],images[0],images[i], 10)
        background.append(transform_from_motion(images[i].reshape(-1,3), V_back).reshape(shape[0],shape[1],3))
        target.append(transform_from_motion(images[i].reshape(-1,3), V_occ).reshape(shape[0],shape[1],3))
        V_backs.append(2*coor - V_back)
        V_occs.append(2*coor - V_occ)
        h_backs.append(h_back)
        h_occs.append(h_occ)


    for i,b in enumerate(background):
        cv2.imwrite("background_t_" + str(i) + ".png",b.reshape(shape[0],shape[1],3))

    background_t = np.array(background).mean(axis = 0)
    for i in range(len(images)-1):
        A.append(np.abs(background[i]-background_t))
    background = background_t
    #del background_t
    A = np.array(A).mean(axis = 0)
    A = 1-(A>=.1)
    background = background.reshape(-1,3)/255.
    A = A.reshape(-1,3).astype(np.float32)

    occlusion = []
    for i in range(len(images)-1):
        occlusion.append(target[i].reshape(-1,3)-A*transform_from_motion(background,V_backs[i]))
    occlusion = np.array(occlusion).mean(axis = 0)/255.
    images = [image.reshape(-1,3)/255. for image in images]

    occlusion = np.clip(occlusion,0,1)
    background = np.clip(background,0,1)
    cv2.imwrite("background1.png",background.reshape(shape[0],shape[1],3)*255)
    cv2.imwrite("occlusion1.png",occlusion.reshape(shape[0],shape[1],3)*255)
    #optimize_images(alpha = .001, beta = .000000000001)


    for i in range(len(V_backs)):
        cv2.imwrite("occlusion_" + str(i) + ".png",transform_from_motion(occlusion,V_occs[i]).reshape(shape[0],shape[1],3)*255)
        cv2.imwrite("occlusion_h" + str(i) + ".png",background_estimate(occlusion.reshape(shape[0],shape[1],3), np.linalg.inv(h_occs[i]))*255)
        cv2.imwrite("background_" + str(i) + ".png",transform_from_motion(background,V_backs[i]).reshape(shape[0],shape[1],3)*255)
        cv2.imwrite("background_h" + str(i) + ".png",background_estimate(background.reshape(shape[0],shape[1],3), np.linalg.inv(h_backs[i]))*255)

    #optimize_warps(alpha = .0001, beta = .0001)
    last_loss = 1000000
    last_loss2 = 1000000
    beta =.1
    beta2 = .01
    counter1 = 0
    counter2 = 0
    background_t = background_t.reshape(-1,3)/255.
    for i in range(1000):
        print(i)
        #if i == 18:
        #    import pdb; pdb.set_trace()
        background_0 = background.copy()
        occlusion_0 = occlusion.copy()
        A_0 = A.copy()
        V_backs_0 = [V.copy() for V in V_backs]
        V_occs_0 = [V.copy() for V in V_occs]
        last_loss,loss = optimize_images(0,0,0,0,alpha = .01, beta = beta)
        
        if loss >= last_loss:
            counter1 +=1
            background = background_0.copy()
            occlusion = occlusion_0.copy()
            A = A_0.copy()
            if counter1 > 3 and beta > .0001:
                beta *=.1
            if counter1 > 5:
                background+=np.random.normal(scale = .001, size = background.shape)
                background = background.clip(0,1)
                occlusion+=np.random.normal(scale = .001, size = occlusion.shape)
                occlusion = occlusion.clip(0,1)
                A +=np.random.normal(scale = .001, size = A.shape)
                A = A.clip(0,1)
        else:
            counter1 = 0
        if loss < last_loss:
            last_loss = loss

        last_loss2, loss = optimize_warps(0,0,0,0,alpha = .01, beta = beta2)
        #print(last_loss2,loss)
        if loss >= last_loss2:
            V_backs = V_backs_0.copy()
            V_occs = V_occs_0.copy()
            counter2 +=1
            if counter2 > 3 and beta > .0001:
                beta2 *=.1
            if counter2 > 5:
                V_backs = [V + np.random.normal(scale = .0000001, size = V.shape) for V in V_backs]
                print(((V_backs[0]- V_backs_0[0])**2).mean())
                V_occs = [V + np.random.normal(scale = .0000001, size = V.shape) for V in V_backs]
        else:
            counter2 = 0
        if loss < last_loss2:
            last_loss2 = loss
        print(counter1,counter2)
        print(np.abs(background- background_t/255.).mean())
        cv2.imwrite("background2.png",background.reshape(shape[0],shape[1],3)*255)
        cv2.imwrite("occlusion2.png",occlusion.reshape(shape[0],shape[1],3)*255)