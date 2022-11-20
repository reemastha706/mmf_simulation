#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv,kv, kn, jn_zeros
import math
import pandas as pd
from colorsys import hls_to_rgb
from scipy.linalg import expm


# In[2]:


get_ipython().run_line_magic('matplotlib', 'qt')
#parameters defined

lambda_ = 1.064  #1064*10(-9)
k = np.round(2*np.pi/lambda_,2)
a = 40   #in microns  #core radius
b =  2.4*a  #maximum radius outside
NA = 0.095  #0.095
nc = 1.48   #index core
n = 1.44  #index cladding
delta = (nc - n)/n 
Vmax = a*NA*k
N = 500
modes = Vmax**2/2    #number of modes 
V = 2*Vmax
modes


# In[3]:


#defining meshgrid
uu = np.linspace(0,V,N)
ww = uu
u,w = np.meshgrid(uu,ww)


# In[4]:


##finding the intersection of u-w branch
def find_intersections(A, B):
    #this function copied from https://stackoverflow.com/questions/3252194/numpy-and-line-intersections#answer-9110966
    # min, max and all for arrays
    amin = lambda x1, x2: np.where(x1<x2, x1, x2)
    amax = lambda x1, x2: np.where(x1>x2, x1, x2)
    aall = lambda abools: np.dstack(abools).all(axis=2)
    slope = lambda line: (lambda d: d[:,1]/d[:,0])(np.diff(line, axis=0))

    x11, x21 = np.meshgrid(A[:-1, 0], B[:-1, 0])
    x12, x22 = np.meshgrid(A[1:, 0], B[1:, 0])
    y11, y21 = np.meshgrid(A[:-1, 1], B[:-1, 1])
    y12, y22 = np.meshgrid(A[1:, 1], B[1:, 1])

    m1, m2 = np.meshgrid(slope(A), slope(B))
    m1inv, m2inv = 1/m1, 1/m2

    yi = (m1*(x21-x11-m2inv*y21) + y11)/(1 - m1*m2inv)
    xi = (yi - y21)*m2inv + x21

    xconds = (amin(x11, x12) < xi, xi <= amax(x11, x12), 
              amin(x21, x22) < xi, xi <= amax(x21, x22) )
    yconds = (amin(y11, y12) < yi, yi <= amax(y11, y12),
              amin(y21, y22) < yi, yi <= amax(y21, y22) )

    return xi[aall(xconds)], yi[aall(yconds)]


# In[5]:


#to draw a circle around the u-w branch
def circle():
    return np.power(u,2) + np.power(w,2) - np.power(Vmax,2)

#to store the bessel function solutions
def zeros(l, N):
    arr = jn_zeros(l,N)
    arr = np.delete(arr, np.argwhere( (arr >= Vmax) ))
    return arr


# In[6]:



def diff(l,u,w):
    if l == 0:
        LHS = u*jv(l+1,u)/(jv(l,u))
        RHS = w*kn(l+1,w)/(kn(l,w))
        diff_ = LHS - RHS
    else:
        LHS = u*jv(l-1,u)/(jv(l,u))
        RHS = w*kn(l-1,w)/(kn(l,w))
        diff_ = LHS + RHS
    zero = zeros(l,N)
    return zero, diff_


# In[7]:


#finding intersection point around the circle
def besselmode(l):
    zero, value = diff(l,u,w)
    radius = circle()
    v = plt.contour(u,w, value,[0], colors = 'k')
    r = plt.contour(u,w,radius, [0], colors = 'b')
    xi = np.array([])
    yi = np.array([])
    output = np.array([])
    for linecol in v.collections:
        for path in linecol.get_paths():
            for linecol2 in r.collections:
                for path2 in linecol2.get_paths():
                    xinter, yinter = find_intersections(path.vertices, path2.vertices)
                    xi = np.append(xi, xinter)
                    yi = np.append(yi, yinter)
    tol = 0.1
    output = zip(xi,yi)
    for k in zero:
        output = [(i,j) for i,j in output if np.abs(i-k) > tol]
    x, y = zip(*output)
    plt.scatter(x,y, s=20)
#     plt.title('uw branch')
#     plt.xlabel('u')
#     plt.ylabel('w')
    plt.close()
    return x,y
#


# In[8]:


#phase of the fiber modes
def phase(ind, phi, deg):
    if deg  == 'sin':
        phase = np.cos(l*phi[ind])
    else:
        phase = np.cos(l*phi[ind]) + 1j*np.sin(l*phi[ind])
    return phase
        


# In[9]:


#calculating electric fields
def fieldcalc(u,w):
    n = 50
    x = np.linspace(-b/2,b/2,n)
    y = x

    xx, yy = np.meshgrid(x,y)
    gammax = np.diag(xx.flatten())
    gammay = np.diag(yy.flatten())
    r = np.sqrt(xx**2 + yy**2)
    rr = np.reshape(r, n*n)
    phi = np.reshape( np.arctan2(yy,xx), len(x)*len(y))
    E = np.zeros(len(rr)).astype("complex_")
    J1 = jv(l,u[m])
    K1 = kv(l,w[m])
    rr[rr<np.finfo(np.float32).eps] = np.finfo(np.float32).eps
    ind1 = rr <= a
    J2 = jv(l,u[m]*rr[ind1]/a)
    ind2 = rr > a
    K2 = kv(l,w[m]*rr[ind2]/a)
    E[ind1] = (J2/J1)*phase(ind1,phi, 'exp')
    E[ind2] = (K2/K1)*phase(ind2, phi, 'exp')
    E = np.reshape(E, [n,n]).astype(np.complex64)
    E = E / np.sqrt(np.sum(np.abs(E)**2))
    return gammax, gammay,x,y,E


# In[10]:


#finding propagation constants
def betas(x,y):    
    factor = [i / j for i, j in zip(np.power(x,2), (np.power(x,2)+np.power(y,2)))]
    for idx in factor:
        beta.append(n*k*(1+delta-delta*idx))
    return beta


# In[11]:


#color plot
#Cite: 10.5281/zenodo.1419005. 
def colorize(z, theme = 'dark', saturation = 1., beta = 1.4, transparent = False, alpha = 1.):
    r = np.abs(z)
    r /= np.max(np.abs(r))
    arg = np.angle(z) 

    h = (arg + np.pi)  / (2 * np.pi) + 0.5
    l = 1./(1. + r**beta) if theme == 'white' else 1.- 1./(1. + r**beta)
    s = saturation

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0,2) 
    if transparent:
        a = 1.-np.sum(c**2, axis = -1)/3
        alpha_channel = a[...,None]**alpha
        return np.concatenate([c,alpha_channel], axis = -1)
    else:
        return c


# In[12]:


#transmission matrix 
def transmissionmatrix(B, L):
    return expm(1j*B*L)
#propagation constants for bending fiber 
def bendingfiber(curve_x, curve_y, CM, gx, gy):
    B_curved = B - nc*k*((1./curve_x*( CM.transpose().conjugate().dot(gx).dot(CM)))+(1./curve_y*( CM.transpose().conjugate().dot(gy).dot(CM))))
    return np.array(B_curved)
#Converted Transmission matrix as output
def convertedTM(CM, TM):
    return CM.dot(TM).dot(np.conj(CM).T)


# In[13]:


#print out all the modes
num = 18 #for the given parameters there are 18 l-modes
l_ = []
m_ = []
beta = []
for l in range(-num, num):
    u_,w_ = besselmode(l)
    beta_ = betas(u_,w_)
    M = len(u_)
    for m in range(M):
        gx, gy, x,y,E = fieldcalc(u_,w_)
        m_.append(E)
    l_.append(m_)
    m_ = []


# In[14]:


df = pd.DataFrame(data = l_, index = None, columns = None)  #converting lists of modes into dataframe for better understanding
row = len(df.iloc[0,:])
col = len(df[int(row/2)])
fig, axs = plt.subplots(row,col, figsize = (6,6))
fig.subplots_adjust(hspace=0, wspace=0)
for i in range(0, row):
    for j in range(len(df[i])):
        if isinstance(df[i][j], np.ndarray):
            im = axs[i,j].imshow(colorize(df[i][j], 'turbo'), aspect = 'equal')
            axs[i,j].axis('off')
        else:
            axs[i,j].lines: axs[i,j].set_visible(False) 

        fig.subplots_adjust(right=0.8)
        plt.show()


# In[15]:


#arranging all modes in a matrix [based on the paper Seeing through chaos, fig 1 c]
mat = []
r, c = np.shape(df)
for i in range(r):
    for j in range(c):
        mat.append(df.iloc[i][j])
#removing all none type objects
res = list(filter(lambda item: item is not None, mat))  
#2D matrix to 1 D array
for i in range(len(res)):
    res[i] = res[i].flatten()
#conversion matrix
CM = np.array(res)
CM = CM.T  #transposing the conversion matrix
plt.imshow(colorize(CM, 'turbo'), aspect = 'auto')
plt.ylabel('no of modes')
plt.xlabel('pixel')
plt.title('Conversion Matrix') 


# In[16]:



curve_x = 1*10**5 #bending in x and y direction
curve_y = 1*10**5
distance = 1000000   #length of a fiber

B = np.diag(beta)   #construction a matrix for propagation constants
beta_curved = bendingfiber(curve_x, curve_y, CM, gx,gy) #propagation constant for bending fiber


# In[20]:


#calculating transmission matrices for straight and bending fibers
TM = transmissionmatrix(B, distance)
TM1 = transmissionmatrix(beta_curved, distance)

plt.figure(1)
plt.imshow(colorize(TM))
plt.title('TM for straight fibers')
plt.figure(2)
plt.imshow(colorize(TM1))
plt.title('TM for bent fibers')


# In[18]:


#converted matrices for straight and bending fibers

TM_ = convertedTM(CM,TM)
TM__ = convertedTM(CM,TM1)
plt.figure(1)
plt.imshow(np.abs(TM_))
plt.title('Converted Matrix for straight fibers')
plt.figure(2)
plt.imshow(np.abs(TM__))
plt.title('Converted Matrix for bent fibers')


# In[19]:


#testing a single pixel along the fiber 
z = np.zeros(2500)
z[1264]=1

plt.figure(1)
plt.imshow(np.abs((TM_.dot(z)).reshape((50,50)))**2)
plt.title('straight fiber')
plt.colorbar()
plt.figure(2)
plt.imshow(np.abs((TM__.dot(z)).reshape((50,50)))**2)
plt.title('bent fiber')
plt.colorbar()


# In[ ]:




