from landmarker import Marker
import numpy as np
import math
import cv2
import imageio


def morph_correspondence(landmark_img1,landmark_img2,alpha):

    landmark_morphed = np.asarray([[math.floor(alpha*point1[0]+(1-alpha)*point2[0]),math.floor(alpha*point1[1]+(1-alpha)*point2[1])]\
                        for point1, point2 in zip(landmark_img1,landmark_img2)])
    return landmark_morphed

def get_coordinate(landmark,tri):
    return landmark[tri]



def check_circle(tri,coor,point):
    pts = np.asarray([coor[v] for v in tri])
    M = np.linalg.det(np.hstack((pts,np.asarray([[1],[1],[1]]))))

    x_col = pts[:,0]
    y_col = pts[:,1]
    xy_col = x_col**2+y_col**2
    Mx = np.linalg.det(np.vstack((xy_col,y_col,[1,1,1])).T)
    My = np.linalg.det(np.vstack((xy_col,x_col,[1,1,1])).T)

    x0 = 1/2*Mx/M
    y0 = -1/2*My/M
    center = np.asarray([x0,y0])           

    return np.sum(np.square(point-center)) <= np.sum(np.square(pts[0] - center))

def update(tri,coor,point):
    
    p = np.asarray(point)
    index = len(coor)
    coor.append(p)
    
    #check for the triangles whose circumcircle contains p
    cavity_set = []
    cavity_edge = []
    
    for t in tri:
        if check_circle(t,coor,p):
            cavity_set.append(t)
    T = cavity_set[0]
    e = 0
    
    while(1):
        
        #T = (a,b,c), tri[T][0] is the neibor who shares edge bc
        tri_share = tri[T][e]
        p1,p2 = T[(e+1)%3], T[(e-1)%3]
        
        if tri_share in cavity_set:
            e = (tri[tri_share].index(T)+1)%3
            T = tri_share
        else:
            cavity_edge.append((p1,p2,tri_share))
            e = (e+1)%3
            if cavity_edge[0][0] == cavity_edge[-1][1]: break
        
    for t in cavity_set:
        del tri[t]
        
    #star shape new triangle set
    update_tri = []
    
    for (p0,p1,tri_share) in cavity_edge:
        
        new_tri = (index,p0,p1)
        update_tri.append(new_tri)
        
        tri[new_tri] = [tri_share,None,None]
        if tri_share:
            for idx,neighbor in enumerate(tri[tri_share]):
                if neighbor:
                    if p0 in neighbor and p1 in neighbor:
                        tri[tri_share][idx] = new_tri
        
        
    update_num = len(update_tri)

    for idx, t in enumerate(update_tri):
        tri[t][1] = update_tri[(idx+1)%update_num]
        tri[t][2] = update_tri[(idx-1)%update_num]

def get_delaunay(landmarks1,landmarks2):
    point_coor = []
    tri_neigbour = {}

    #{ trianlge index : neigbor1 index, neigbor 2 index, neighbor3 index}

    point_coor.append(np.asarray([-10000,-10000]))
    point_coor.append(np.asarray([10000,-10000]))
    point_coor.append(np.asarray([10000,10000]))
    point_coor.append(np.asarray([-10000,10000]))

    tri_neigbour[(0,1,3)] = [(2,3,1),None,None] #CCW
    tri_neigbour[(2,3,1)] = [(0,1,3),None,None] 
    points = (landmarks1+landmarks2)//2
    for p in points:
        update(tri_neigbour,point_coor,p)
    tri = np.asarray([(i-4, j-4, k-4) for (i, j, k) in tri_neigbour if i > 3 and j > 3 and k > 3])
    return tri


def apply_bilinear_interpolation(img, point):
    x = point[0]
    y = point[1] 
    if (x < 0 or y < 0
       or x > img.shape[0] - 1 or y > img.shape[1] - 1):
        if x<0:
            x = 0
        if y<0:
            y = 0
        if x > img.shape[0] - 1:
            x = img.shape[0]-1
        if y > img.shape[1] - 1:
            y = img.shape[1]-1
    delta_x =  abs(np.round(x) - x)
    delta_y =  abs(np.round(y) - y)
    x1 = int(np.round(x))
    y1 = int(np.round(y))
    I1 = img[x1][y1] * (1 - delta_x) * (1 - delta_y)
    I2 = img[int(x)][y1] * (delta_x) * (1 - delta_y)
    I3 = img[x1][int(y)] * (1 - delta_x) * (delta_y)
    I4 = img[int(x)][int(y)] * (delta_x) * (delta_y)
    return I1 + I2 + I3 + I4  #resulting intensity

def getwarped(img1Cropped,warp,r2):
    homograph = np.vstack((warp,[0,0,1]))
    inv_homo = np.linalg.inv(homograph)
    wH = r2[0]
    wW =r2[1]
    warped = np.zeros((r2[1],r2[0],3))
    for x in range(wH):
        for y in range(wW):
            pixel = np.asarray([[x],[y],[1]])
            ref_coor = np.dot(inv_homo,pixel)
            point = np.asarray([ref_coor[1][0],ref_coor[0][0]])

            intensity = [int(apply_bilinear_interpolation(img1Cropped[:,:,i], point)) for i in range(3)]

            warped[y,x,0],warped[y,x,1],warped[y,x,2] = intensity
    warped = warped.astype(np.int)  
    return warped

def get_morph_image(img1,img2,tri1,tri2):
    
	crop_tri1 = []
	crop_tri2 = []
	r1 = cv2.boundingRect(tri1)
	r2 = cv2.boundingRect(tri2)

	for i in range(0, 3):
	    crop_tri1.append(((tri1[i][0] - r1[0]),(tri1[i][1] - r1[1])))
	    crop_tri2.append(((tri2[i][0] - r2[0]),(tri2[i][1] - r2[1])))

	img1Cropped = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

	p = np.asarray(crop_tri1)
	q = np.asarray(crop_tri2)
	source_tri = np.hstack((q,[[1],[1],[1]])).T
	target_tri = np.hstack((p,[[1],[1],[1]])).T


	# warp = target_tri.dot(np.linalg.inv(source_tri)).T[:2,:].astype(np.float64)
	M = find_homo(np.float32(crop_tri1),np.float32(crop_tri2))
	M = np.asarray(M).astype(np.float64)
	warp = M[:,3:].T
	img2Cropped = getwarped( img1Cropped, warp, [r2[2], r2[3]])
	mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
	cv2.fillConvexPoly(mask, np.int32(crop_tri2), (1.0, 1.0, 1.0), 16, 0);
	img2Cropped = img2Cropped * mask


	img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
	img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Cropped

def find_homo(fp, tp):
    q = fp
    p = tp
    dim = len(q[0])
    c = [[0.0 for a in range(dim)] for i in range(dim+1)]
    for j in range(dim):
        for k in range(dim+1):
            for i in range(len(q)):
                qt = list(q[i]) + [1]  # extend a colum with value 1
                c[k][j] += qt[k] * p[i][j]
    Q = [[0.0 for a in range(dim)] + [0] for i in range(dim+1)]
    for qi in q:
        qt = list(qi) + [1]
        for i in range(dim+1):
            for j in range(dim+1):
                Q[i][j] += qt[i] * qt[j]

    m = [Q[i] + c[i] for i in range(dim+1)]

    (h, w) = (len(m), len(m[0]))
    for y in range(0, h):
        maxrow = y
        for y2 in range(y+1, h):
            if abs(m[y2][y]) > abs(m[maxrow][y]):
                maxrow = y2
        (m[y], m[maxrow]) = (m[maxrow], m[y])
        for y2 in range(y+1, h):
            c = m[y2][y] / m[y][y]
            for x in range(y, w):
                m[y2][x] -= m[y][x] * c
    for y in range(h-1, 0-1, -1):
        c = m[y][y]
        for y2 in range(0, y):
            for x in range(w-1, y-1, -1):
                m[y2][x] -= m[y][x] * m[y2][y] / c
        m[y][y] /= c
        for x in range(h, w):
            m[y][x] /= c
     

    return m



def get_img_morph(alpha,img1,img2,tri,tri1_coor,tri2_coor,tri_morph_coor):

	imgMorph1 = np.full_like(fill_value=255,a=img1,dtype=img1.dtype)
	imgMorph2 = imgMorph1.copy()
	for i in range(len(tri_morph_coor)):
	    tri = tri_morph_coor[i]
	    tri1 = tri1_coor[i]
	    tri2 = tri2_coor[i]
	    get_morph_image(img1,imgMorph1,tri1,tri)
	    get_morph_image(img2,imgMorph2,tri2,tri)

	img = alpha*imgMorph1+(1.0-alpha)*imgMorph2
	return img

def gen_gif(path1,path2):
	img1 = cv2.imread(path1)
	img2 = cv2.imread(path2)
	marker = Marker(path1,path2)
	landmark1,landmark2 = marker.auto_gen_mark()
	landmarks_morph = morph_correspondence(landmark1,landmark2,0.5)

	tri = get_delaunay(landmark1,landmark2)
	tri1_coor = get_coordinate(landmark1,tri)
	tri2_coor = get_coordinate(landmark2,tri)
	tri_morph_coor = get_coordinate(landmarks_morph,tri)

	imgmorphlist = []
	imgmorphlist.append(img2)
	frame = 60
	for i in range(1,frame):
	    alpha = float(i)/(frame)
	    if i%5 == 0:
	        print(str(i)+'/'+str(frame)+'completed')
	    img = get_img_morph(alpha,img1,img2,tri,tri1_coor,tri2_coor,tri_morph_coor)
	    img = cv2.cvtColor(np.uint8(img[:,:,::-1]),cv2.COLOR_BGR2RGB)
	    imgmorphlist.append(img)
	imgmorphlist.append(img1)


	gif_name = './res/'+file_path[0][6:-4]+'_'+file_path[1][6:-4]+'.gif'
	imageio.mimwrite(gif_name, np.asarray([i[:,:,::-1] for i in imgmorphlist]).astype(np.uint8), duration=0.04)

