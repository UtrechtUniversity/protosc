#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 19:05:02 2018

@author: sjoerdstuit
"""
#import sklearn

def fft_imrebuild(amps,phases):
    """fft_imrebuild(amps,phases) rebuilds an image from provided amp-spectrum
    and provided phase spectrum"""
    import numpy as np
    im = np.real(np.fft.ifft2(amps*np.exp(1j*phases)))
    return im


def im_scale(im):
    """im_scale(im), scales matrix between 0-1"""
    im = im-im.min()
    im = im/im.max()
    return im

#diameter = [10]
def im_anglemap(diameter):
    """im_anglemap(diameter)"""
    import numpy as np
    if np.size(np.shape(diameter))==1:
        nX = float(diameter[0]);
        nY = float(diameter[0]);
    else:
        nX = float(diameter[1]);
        nY = float(diameter[0]);
    [X,Y] = np.meshgrid(np.arange(1.,nX+1),np.arange(1.,nY+1));
    X = X - ((nX+1)/2); 
    Y = Y - ((nY+1)/2);
    t = np.arctan(Y/X)
    t[np.isnan(t)]=0;
    M = im_scale(t)*180;
    M[:,np.arange(0,np.shape(M)[1]/2)] = M[:,np.arange(0,np.shape(M)[1]/2)]+180;
    return M

def im_radimap(diameter):
    import numpy as np
    if np.size(np.shape(diameter))==1:
        size_1val = float(diameter[0]);
        radius = (size_1val-1)/2;
        [X, Y] = np.meshgrid(np.arange(-radius,radius+1),np.arange(-radius,radius+1));
        M = np.sqrt(X**2+Y**2);
    else:
        size_1val = float(np.max(diameter))
        radius = (size_1val-1)/2;
        [X, Y] = np.meshgrid(np.arange(-radius,radius+1),np.arange(-radius,radius+1));
        M1 = np.sqrt(X**2+Y**2);
        if np.shape(M1)[0]-float(diameter[0])>0:
            v1 = np.floor((np.shape(M1)[0]-float(diameter[0]))/2)
            v2 = np.shape(M1)[0] - np.ceil((np.shape(M1)[0]-float(diameter[0]))/2)
            M = M1[np.arange(int(v1),int(v2)),:]
        if (np.shape(M1)[1]-diameter[1])>0:
            v1 = np.floor((np.shape(M1)[1]-float(diameter[1]))/2)
            v2 = np.shape(M1)[1] - np.ceil((np.shape(M1)[1]-float(diameter[1]))/2)
            M = M1[:,np.arange(int(v1),int(v2))]
            
    return M
        
def im_circle(im_diameter,circ_diameter):
    import numpy as np
    def im_radimap(im_diameter):
        if np.size(np.shape(im_diameter))==1:
            size_1val = float(im_diameter[0]);
            radius = (size_1val-1)/2;
            [X, Y] = np.meshgrid(np.arange(-radius,radius+1),np.arange(-radius,radius+1));
            M = np.sqrt(X**2+Y**2);
        else:
            size_1val = float(np.max(im_diameter))
            radius = (size_1val-1)/2;
            [X, Y] = np.meshgrid(np.arange(-radius,radius+1),np.arange(-radius,radius+1));
            M1 = np.sqrt(X**2+Y**2);
            if np.shape(M1)[0]-float(im_diameter[0])>0:
                v1 = np.floor((np.shape(M1)[0]-float(im_diameter[0]))/2)
                v2 = np.shape(M1)[0] - np.ceil((np.shape(M1)[0]-float(im_diameter[0]))/2)
                M = M1[np.arange(int(v1),int(v2)),:]
            if (np.shape(M1)[1]-im_diameter[1])>0:
                v1 = np.floor((np.shape(M1)[1]-float(im_diameter[1]))/2)
                v2 = np.shape(M1)[1] - np.ceil((np.shape(M1)[1]-float(im_diameter[1]))/2)
                M = M1[:,np.arange(int(v1),int(v2))]
                
        return M
    radii = im_radimap(im_diameter)
    M = np.zeros((im_diameter[0],im_diameter[0]))
    M[radii<=circ_diameter] = 1
    return M

def im_ring(im_diameter,inner_diameter,outer_diameter):
    import numpy as np
    def im_radimap(im_diameter):
        if np.size(np.shape(im_diameter))==1:
            size_1val = float(im_diameter[0]);
            radius = (size_1val-1)/2;
            [X, Y] = np.meshgrid(np.arange(-radius,radius+1),np.arange(-radius,radius+1));
            M = np.sqrt(X**2+Y**2);
        else:
            size_1val = float(np.max(im_diameter))
            radius = (size_1val-1)/2;
            [X, Y] = np.meshgrid(np.arange(-radius,radius+1),np.arange(-radius,radius+1));
            M1 = np.sqrt(X**2+Y**2);
            if np.shape(M1)[0]-float(im_diameter[0])>0:
                v1 = np.floor((np.shape(M1)[0]-float(im_diameter[0]))/2)
                v2 = np.shape(M1)[0] - np.ceil((np.shape(M1)[0]-float(im_diameter[0]))/2)
                M = M1[np.arange(int(v1),int(v2)),:]
            if (np.shape(M1)[1]-im_diameter[1])>0:
                v1 = np.floor((np.shape(M1)[1]-float(im_diameter[1]))/2)
                v2 = np.shape(M1)[1] - np.ceil((np.shape(M1)[1]-float(im_diameter[1]))/2)
                M = M1[:,np.arange(int(v1),int(v2))]
                
        return M
    radii = im_radimap(im_diameter)
    M = np.zeros((im_diameter[0],im_diameter[0]))
    M[radii<=outer_diameter] = 1
    M[radii<=inner_diameter] = 0
    return M


def im_wedge(diameter,ori,width):
    # diameter expects array
    import numpy as np
    M = im_anglemap(diameter)
    M1 = np.zeros(np.shape(M));
    #if ori<0:
    ori = 360+ori;
        
    #statecheck = ori-(width/2) < 0
    if ori-(width/2) < 0:
        M1[M>=360+(ori-(width/2))]   = 1;
        M1[M<=ori+(width/2)]         = 1;
    else:
        M1[(M>=(ori-(width/2))) & (M<=(ori+(width/2)))] = 1;
    return M1

#def im_doublewedge(diameter,ori,width):
#    import numpy as np
#    M1 = im_wedge(diameter,ori,width)
#    #useori = ori+180
#    #M2 = im_wedge(diameter,useori,width)
#    M2 = np.rot90(np.rot90(M1))
#    return M1+M2
    
    
def im_wedge(diameter,ori,width):
    import numpy as np
    M = im_anglemap(diameter)+360
    M1 = np.ones(np.shape(M));
    use_ori = 360+ori;
    if ((use_ori-width/2)<360):
        M1[M>=use_ori+width/2] = 0
        M1[M<=use_ori-width/2] = 0
        M1[M>=360+(use_ori-width/2)] = 1
    else:
        M1[M>=use_ori+width/2] = 0
        M1[M<=use_ori-width/2] = 0

    return M1

def im_doublewedge(diameter,ori,width):
    import numpy as np
    M1 = im_wedge(diameter,ori,width)
    #useori = ori+180
    #M2 = im_wedge(diameter,useori,width)
    M2 = np.rot90(np.rot90(M1))
    return M1+M2







