import numpy as np
from numpy.linalg import inv


def affine(current_frame, H, s):
    result = np.zeros(s, np.uint8)
    R = np.zeros((2,2))
    T = np.zeros((1,2))
    np.copyto(R, H[0:2, 0:2])
    np.copyto(T, H[:,2])
    cx = int(s[0]/2)
    cy = int(s[1]/2)
    r= inv(R)
    print('',T)
    w=0
    while w<s[0]:
        h=0
        while h<s[1]:
            p = np.transpose(np.dot(r, [(w-cx),(h-cy)]))+T+[cx, cy]
            p = p.astype(int)
            #print('', p[0,0])
            #print('', [w, h])
            if p[0,0]<s[0] and p[0,0]>0 and p[0,1]< s[1] and p[0,1]>0:
                result[w,h] = current_frame[p[0,0],p[0,1]]
                #print('',p[0])
            h+=1
        w+=1

    return result