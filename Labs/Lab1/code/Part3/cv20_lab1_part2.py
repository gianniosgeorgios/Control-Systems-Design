import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, misc
import math
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def disk_strel(n):
    '''
        Return a structural element, which is a disk of radius n.
    '''
    r = int(np.round(n))
    d = 2*r+1
    x = np.arange(d) - r
    y = np.arange(d) - r
    x, y = np.meshgrid(x,y)
    strel = x**2 + y**2 <= r**2
    return strel.astype(np.uint8)

def harrisStephens(I,s,r,k,thetacorn):
    ns = int(2*np.ceil(3*s)+1)
    rs = int(2*np.ceil(3*r)+1)
    
    gauss1Ds = cv2.getGaussianKernel(ns, s)
    gauss2Ds = gauss1Ds @ gauss1Ds.T        

    gauss1Dr = cv2.getGaussianKernel(rs, r) 
    gauss2Dr = gauss1Dr @ gauss1Dr.T        


    Is_1 = cv2.filter2D(I, -1, gauss2Ds)

    [dx,dy] = np.gradient(Is_1)

    
    dxdx=np.multiply(dx,dx)
    dxdy=np.multiply(dx,dy)
    dydy=np.multiply(dy,dy)

    J1 = cv2.filter2D(dxdx, -1, gauss2Dr)
    J2 = cv2.filter2D(dxdy, -1, gauss2Dr)
    J3 = cv2.filter2D(dydy, -1, gauss2Dr)

    J1minusJ3sq = np.multiply((J1-J3),(J1-J3))
    J2sq        = np.multiply(J2,J2)
    lplus       = 0.5*(J1 + J3 + np.sqrt(J1minusJ3sq + 4*J2sq))
    lminus      = 0.5*(J1 + J3 - np.sqrt(J1minusJ3sq + 4*J2sq))

    R = np.multiply(lplus,lminus)-k*np.multiply(lplus+lminus,lplus+lminus)

    #Συνθήκη 1 
    ns = np.ceil(3*s)*2+1
    B_sq = disk_strel(ns)
    Cond1 = (R==cv2.dilate(R,B_sq))
    Cond1=Cond1*1

    #Συνθήκη 2
    maxR=np.amax(R)
    Cond2 = ( R >= thetacorn*maxR);
    Cond2=Cond2*1

    y1,x1 = np.where(Cond1 & Cond2)
    array_sc = np.ones(y1.shape[0])
    param = np.column_stack([x1,y1,array_sc])

    return param

def harrisLaplacian(I,s0,r0,k,thetacorn,scale,N):
    AbsLog = np.zeros((I.shape[0],I.shape[1],N))
    #Compute Abs LoG for every scale 
    for K in range(N):
        s=s0*scale**(K)
        Log_I = ndimage.gaussian_laplace(I,s)
        AbsLog[:,:,K] = s*s * abs(Log_I)

    for i in range(N):
        s=s0*scale**i
        r=r0*scale**i

        ns = int(2*np.ceil(3*s)+1)
        rs = int(2*np.ceil(3*r)+1)

        gauss1Ds = cv2.getGaussianKernel(ns, s) # Column vector
        gauss2Ds = gauss1Ds @ gauss1Ds.T        # Symmetric gaussian kernel

        gauss1Dr = cv2.getGaussianKernel(rs, r) # Column vector
        gauss2Dr = gauss1Dr @ gauss1Dr.T        # Symmetric gaussian kernel

        Is_1 = cv2.filter2D(I, -1, gauss2Ds)
        [dx,dy] = np.gradient(Is_1)

        dxdx=np.multiply(dx,dx)
        dxdy=np.multiply(dx,dy)
        dydy=np.multiply(dy,dy)

        J1 = cv2.filter2D(dxdx, -1, gauss2Dr)
        J2 = cv2.filter2D(dxdy, -1, gauss2Dr)
        J3 = cv2.filter2D(dydy, -1, gauss2Dr)

        J1minusJ3sq = np.multiply((J1-J3),(J1-J3))
        J2sq        = np.multiply(J2,J2)
        lplus       = 0.5*(J1 + J3 + np.sqrt(J1minusJ3sq + 4*J2sq))
        lminus      = 0.5*(J1 + J3 - np.sqrt(J1minusJ3sq + 4*J2sq))

        R = np.multiply(lplus,lminus)-k*np.multiply(lplus+lminus,lplus+lminus)

        ns = np.ceil(3*s)*2+1
        B_sq = disk_strel(ns)
        Cond1 = (R==cv2.dilate(R,B_sq))
        Cond1=Cond1*1

        maxR=np.amax(R)
        Cond2 = ( R >= thetacorn*maxR)
        Cond2=Cond2*1
        
        if (i==0):
            AbsLogCond = AbsLog[:,:,i] >= AbsLog[:,:,i+1]
        elif (i==N-1):
            AbsLogCond= AbsLog[:,:,i] >= AbsLog[:,:,i-1]
        else:
            AbsLogCond = (AbsLog[:,:,i] >= AbsLog[:,:,i+1])& (AbsLog[:,:,i] >= AbsLog[:,:,i-1])

        temp_y,temp_x = np.where(Cond1 & Cond2 & AbsLogCond)

        a=temp_x.shape[0]
        #print(a)

        temp = np.ones((a))*s

        Temp_Parameters= np.concatenate((temp_x, temp_y,temp), axis=0)
        Temp_Parameters= np.reshape(Temp_Parameters,(3,a))
        Temp_Parameters= np.transpose(Temp_Parameters)
        
        if (i==0):
            Parameters = Temp_Parameters
        else:
            Parameters = np.concatenate((Temp_Parameters,Parameters), axis=0)
    return Parameters

def hessian(I,s,threshold):
    ns = int(2*np.ceil(3*s)+1)
  
    gauss1Ds = cv2.getGaussianKernel(ns, s) # Column vector
    gauss2Ds = gauss1Ds @ gauss1Ds.T # Symmetric gaussian kernel

    Is_1 = cv2.filter2D(I, -1, gauss2Ds)
    [dx,dy] = np.gradient(Is_1)

    [hxx,hxy]=np.gradient(dx)
    [hxy,hyy]=np.gradient(dy)

    R = np.multiply(hxx,hyy)-np.multiply(hxy,hxy)

    B_sq = disk_strel(ns)
    maxima = (R==cv2.dilate(R,B_sq))
    Rmax = np.amax(R)

    temp_y,temp_x = np.where(maxima & (R>= (threshold * Rmax)))

    array_sc = np.ones(temp_y.shape[0])*s

    param = np.column_stack([temp_x,temp_y,array_sc])
    return param

def hessianLaplace(I,s0,threshold,scale,N):
    AbsLog = np.zeros((I.shape[0],I.shape[1],N))
  
    #Compute Abs LoG for every scale 
    for K in range(N):
        s=s0*scale**(K)
        Log_I = ndimage.gaussian_laplace(I,s)
        AbsLog[:,:,K] = s*s * abs(Log_I)

    
    for i in range(N):
        s=s0*scale**i

        ns = int(2*np.ceil(3*s)+1)

        gauss1Ds = cv2.getGaussianKernel(ns, s) # Column vector
        gauss2Ds = gauss1Ds @ gauss1Ds.T        # Symmetric gaussian kernel

        Is_1 = cv2.filter2D(I, -1, gauss2Ds)
        
        [dx,dy] = np.gradient(Is_1)
        [hxx,hxy]=np.gradient(dx)
        [hxy,hyy]=np.gradient(dy)


        R = np.multiply(hxx,hyy)-np.multiply(hxy,hxy)

        B_sq = disk_strel(ns)
        maxima = (R==cv2.dilate(R,B_sq))
        Rmax = np.amax(R)

        if (i==0):
            AbsLogCond = AbsLog[:,:,i] >= AbsLog[:,:,i+1]
        elif (i==N-1):
            AbsLogCond= AbsLog[:,:,i] >= AbsLog[:,:,i-1]
        else:
            AbsLogCond = (AbsLog[:,:,i] >= AbsLog[:,:,i+1])& (AbsLog[:,:,i] >= AbsLog[:,:,i-1])

        temp_y,temp_x = np.where(maxima & (R>= (threshold * Rmax)) & AbsLogCond)

        a=temp_x.shape[0]
        #print(a)

        temp = np.ones((a))*s

        Temp_Parameters= np.concatenate((temp_x, temp_y,temp), axis=0)
        Temp_Parameters= np.reshape(Temp_Parameters,(3,a))
        Temp_Parameters= np.transpose(Temp_Parameters)
        
        if (i==0):
            Parameters = Temp_Parameters
        else:
            Parameters = np.concatenate((Temp_Parameters,Parameters), axis=0)

    return Parameters 

def computeSs_Integral(intI, shiftX, shiftY, offsetx, offsety):
    sD = np.roll(intI,-shiftY + offsety,0)
    sD = np.roll(sD,-shiftX + offsetx,1)
    #Left-Top Corner
    sA = np.roll(intI,shiftY + offsety,0)
    sA = np.roll(sA,shiftX + offsetx,1)
    #Right-Top corner
    sB = np.roll(intI,shiftY + offsety,0)
    sB = np.roll(sB,-shiftX + offsetx,1)
    #Left-Bottom Corner
    sC = np.roll(intI,-shiftY + offsety,0)
    sC = np.roll(sC,shiftX + offsetx,1)
    return [sA,sB,sC,sD]

def unpad ( temp_s , pad ):
    sx = temp_s.shape[0]
    sy = temp_s.shape[1]
    s = temp_s[pad:sx - pad+1,pad:sy - pad+1]
    return s

def box_filters_one_scale(I,s,threshold):
    thresh=0.912
    n = np.ceil(3*s)*2+1
    
    xDxx = 2*math.floor(n/6) + 1
    yDxx = 4*math.floor(n/6) + 1
    rDxx = math.floor((n-yDxx)/2)
    
    xDyy = 4*math.floor(n/6) + 1
    yDyy = 2*math.floor(n/6) + 1
    
    xDxy = 2*math.floor(n/6)+1
    yDxy = 2*math.floor(n/6)+1
    rDxy =math.floor((n - xDxy - yDxy)/3)
    
    sDxy = math.floor((n - xDxy - yDxy)/3)
    mDxy = np.ceil((n - xDxy - yDxy)/3)

    if(sDxy + sDxy + mDxy != n-xDxy - yDxy):
        sDxy = np.ceil((n - xDxy - yDxy)/3)
        mDxy = math.floor((n - xDxy - yDxy)/3)

    Ipadded=np.pad(I,(math.floor(n/2),math.floor(n/2)),'edge')
    intI=cv2.integral(Ipadded)

    #Central Box
    magn = -3
    shiftX = (xDxx -1)//2
    shiftY = (yDxx -1)//2
    pad = math.floor(n/2) + 1
    [ tsA,tsB,tsC,tsD ] = computeSs_Integral( intI, shiftX, shiftY , 0, 0)
    sA = unpad(tsA,pad)
    sB = unpad(tsB,pad)
    sC = unpad(tsC,pad)
    sD = unpad(tsD,pad)
    Lxx = (sD + sA - sB - sC) * magn

    #Left and Right Box
    magn = 1
    shiftX = (xDxx -1)//2 + xDxx
    shiftY = (yDxx -1)//2
    [ tsA,tsB,tsC,tsD ] = computeSs_Integral( intI, shiftX, shiftY ,0, 0)
    sA = unpad(tsA,pad)
    sB = unpad(tsB,pad)
    sC = unpad(tsC,pad)
    sD = unpad(tsD,pad)
    Lxx = (sD + sA - sB - sC) * magn + Lxx


    # Lyy
    #Central Box
    magn = -3
    shiftX = (xDyy -1)//2
    shiftY = (yDyy -1)//2
    [ tsA,tsB,tsC,tsD ] = computeSs_Integral( intI, shiftX, shiftY ,0, 0)
    sA = unpad(tsA,pad)
    sB = unpad(tsB,pad)
    sC = unpad(tsC,pad)
    sD = unpad(tsD,pad)
    Lyy = (sD + sA - sB - sC) * magn
    
    #Top and Bottom Box
    magn = 1
    shiftX = (xDyy -1)//2
    shiftY = (yDyy -1)//2 + yDyy
    [ tsA,tsB,tsC,tsD ] = computeSs_Integral( intI, shiftX, shiftY , 0 , 0)
    sA = unpad(tsA,pad)
    sB = unpad(tsB,pad)
    sC = unpad(tsC,pad)
    sD = unpad(tsD,pad)
    Lyy = (sD + sA - sB - sC) * magn + Lyy




    if(( math.floor(np.ceil((n-2*xDxy)/3) % 2)) == 1):
        rDxy = np.ceil((n-2*xDxy)/3)
    else:
        rDxy = math.floor((n-2*xDxy)/3)



    # Lxy
    #Top Right Box
    magn = -1
    offsetx = - math.floor((rDxy-1)/2) -  math.floor((xDxy -1)/2)
    offsety =  math.floor((rDxy-1)/2) +  math.floor((yDxy -1)/2)
    shiftX =  math.floor((xDxy -1)/2)
    shiftY =  math.floor((yDxy -1)/2)
    [ tsA,tsB,tsC,tsD ] = computeSs_Integral( intI, shiftX, shiftY, offsetx, offsety )
    sA = unpad(tsA,pad)
    sB = unpad(tsB,pad)
    sC = unpad(tsC,pad)
    sD = unpad(tsD,pad)
    Lxy = (sD + sA - sB - sC) * magn
    
    #Top Left Box
    magn = 1
    offsetx =  math.floor((rDxy-1)/2) +  math.floor((xDxy -1)/2)
    offsety =  math.floor((rDxy-1)/2) +  math.floor((yDxy -1)/2)
    shiftX =  math.floor((xDxy -1)/2)
    shiftY =  math.floor((yDxy -1)/2)
    [ tsA,tsB,tsC,tsD ] = computeSs_Integral( intI, shiftX, shiftY, offsetx, offsety  )
    sA = unpad(tsA,pad)
    sB = unpad(tsB,pad)
    sC = unpad(tsC,pad)
    sD = unpad(tsD,pad)
    Lxy = (sD + sA - sB - sC) * magn + Lxy
    
    #Bottom Left Box
    magn = -1
    offsetx =  math.floor((rDxy-1)/2) +  math.floor((xDxy -1)/2)
    offsety = - math.floor((rDxy-1)/2) -  math.floor((yDxy -1)/2)
    shiftX =  math.floor((xDxy -1)/2)
    shiftY =  math.floor((yDxy -1)/2)
    [ tsA,tsB,tsC,tsD ] = computeSs_Integral( intI, shiftX, shiftY, offsetx, offsety  )
    sA = unpad(tsA,pad)
    sB = unpad(tsB,pad)
    sC = unpad(tsC,pad)
    sD = unpad(tsD,pad)
    Lxy = (sD + sA - sB - sC) * magn + Lxy
    
    #Bottom Right Box
    magn = 1
    offsetx = - math.floor((rDxy-1)/2) -  math.floor((xDxy -1)/2)
    offsety = - math.floor((rDxy-1)/2) -  math.floor((yDxy -1)/2)
    shiftX  =  math.floor((xDxy -1)/2)
    shiftY  =  math.floor((yDxy -1)/2)
    [ tsA,tsB,tsC,tsD ] = computeSs_Integral( intI, shiftX, shiftY, offsetx, offsety  )
    sA = unpad(tsA,pad)
    sB = unpad(tsB,pad)
    sC = unpad(tsC,pad)
    sD = unpad(tsD,pad)
    Lxy = (sD + sA - sB - sC) * magn + Lxy

    
    
    R=np.multiply(Lxx,Lyy) - np.power((thresh*Lxy),2)

    
    #Finds maxima
    ns=np.ceil(3*s)*2+1
    B_sq = disk_strel(ns)
    maxima = (R==cv2.dilate(R,B_sq))
    Rmax = np.amax(R)

    interest_map = (maxima & (R >= threshold * Rmax)) 

    return interest_map 

def box_filters_multiscale(I,s0,threshold,scale,N):
  
  AbsLog = np.zeros((I.shape[0],I.shape[1],N))
  
  #Compute Abs LoG for every scale 
  for K in range(N):
    s=s0*scale**(K)
    Log_I = ndimage.gaussian_laplace(I,s)
    AbsLog[:,:,K] = s*s * abs(Log_I)
    
  for i in range(N):
    s = s0 * scale**i
    interest_map=box_filters_one_scale(I,s,threshold)

    if (i==0):
      AbsLogCond = AbsLog[:,:,i] >= AbsLog[:,:,i+1]
    elif (i==N-1):
      AbsLogCond= AbsLog[:,:,i] >= AbsLog[:,:,i-1]
    else:
      AbsLogCond = (AbsLog[:,:,i] >= AbsLog[:,:,i+1])& (AbsLog[:,:,i] >= AbsLog[:,:,i-1])
    
    temp_y,temp_x = np.where(interest_map & AbsLogCond)
    a=temp_x.shape[0]

    temp = np.ones((a))*s

    Temp_Parameters= np.concatenate((temp_x, temp_y,temp), axis=0)
    Temp_Parameters= np.reshape(Temp_Parameters,(3,a))
    Temp_Parameters= np.transpose(Temp_Parameters)
    
    if (i==0):
      Parameters = Temp_Parameters
    else:
      Parameters = np.concatenate((Temp_Parameters,Parameters), axis=0)
  
  return Parameters 
