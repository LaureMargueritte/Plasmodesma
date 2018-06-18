# coding: utf8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

NETMODE = 'mieux'
def loadInt1D(epath):
    "charge un fichier csv de 2D"
    # lit le fichier
    ne1 = pd.read_csv(epath, header=1, sep = ', ', usecols=[0, 1])
    x1 = np.array(ne1['center'])
    #xu1 = np.unique(x1)
    y1 = np.array(ne1['bucket'])
    #yu1 = np.unique(y1)
    # calcul la matrice
    #Xr1, Yr1 = np.meshgrid(yu1, xu1)
    #Zr1 = np.reshape(z1,(len(xu1)))
    #netmode = NETMODE
    #if net:
    #    if netmode=='standard':
    #        Zr1 = nettoie(Zr1)
    #    elif netmode=='mieux':
    #        Zr1 = nettoie_mieux(Zr1)
    #    elif netmode=='encore':
    #        Zr1 = nettoie_encore_mieux(Zr1)
    #    else:
    #        raise Exception(netmode + ' : Wrong netmode !')
    #if sym:
    #        Zr1 = symetrise(Zr1)
    return [x1, np.nan_to_num(y1)]

def affiche1D(x1, y1, scale=1.0):
    ax1 = plt.gca()
    levelbase = [0.5,1,2,5,10,20,50,100]
    m1 = y1.max()
    level = [m1*(i/100.0)/scale for i in levelbase ]
    #ax1.contour(X, Y, level, cmap= cmap)
    #ax1.set_xlabel(r"$\delta  (ppm)$")
    #ax1.set_ylabel(r'$\delta  (ppm)$')
    major_ticks = np.arange(0.5, 9.5, 0.5)
    minor_ticks = np.arange(0.5, 9.5, 0.1)
    ax1.set_xticks(major_ticks)
    ax1.set_xticks(minor_ticks, minor = True)
    #ax1.set_yticks(major_ticks)
    #ax1.set_yticks(minor_ticks, minor = True)
    #ax1.grid(b = True, which = 'major', axis = 'both')
    #ax1.grid(b = True, which = 'minor', axis = 'both')
    #ax1.set_xlim(xmax=11,xmin=0)
    #ax1.set_ylim(ymax=11,ymin=0)
    #if reverse:
        #ax1.invert_xaxis()
        #ax1.invert_yaxis()
    return ax1

def loadInt2D(epath, net=False, sym=False):
    "charge un fichier csv de 2D"
    # lit le fichier
    ne1 = pd.read_csv( epath, header=1, sep = ', ', usecols=[0, 1, 2], engine='python')
    x1 = np.array(ne1['centerF1'])
    xu1 = np.unique(x1)
    y1 = np.array(ne1['centerF2'])
    yu1 = np.unique(y1)
    z1 = np.array(ne1['bucket'])
    # calcul la matrice
    Xr1, Yr1 = np.meshgrid(yu1, xu1)
    Zr1 = np.reshape(z1,(len(xu1),len(yu1)))
    netmode = NETMODE
    if net:
        if netmode=='standard':
            Zr1 = nettoie(Zr1)
        elif netmode=='mieux':
            Zr1 = nettoie_mieux(Zr1)
        elif netmode=='encore':
            Zr1 = nettoie_encore_mieux(Zr1)
        else:
            raise Exception(netmode + ' : Wrong netmode !')
    if sym:
            Zr1 = symetrise(Zr1)
    return [Xr1, Yr1, np.nan_to_num(Zr1)]


def loadStd2D(epath, net=False, sym=False):
    "charge un fichier csv de 2D"
    # lit le fichier
    ne1 = pd.read_csv(epath, header=1, sep = ', ', usecols=[0, 1, 5], engine='python')
    x1 = np.array(ne1['centerF1'])
    xu1 = np.unique(x1)
    y1 = np.array(ne1['centerF2'])
    yu1 = np.unique(y1)
    z1 = np.array(ne1['std'])
    # calcul la matrice
    Xr1, Yr1 = np.meshgrid(yu1, xu1)
    Zr1 = np.reshape(z1,(len(xu1),len(yu1)))
    netmode = NETMODE
    if net:
        if netmode=='standard':
            Zr1 = nettoie(Zr1)
        elif netmode=='mieux':
            Zr1 = nettoie_mieux(Zr1)
        elif netmode=='encore':
            Zr1 = nettoie_encore_mieux(Zr1)
        else:
            raise Exception(netmode + ' : Wrong netmode !')
    if sym:
        Zr1 = symetrise(Zr1)
    return [Xr1, Yr1, np.nan_to_num(Zr1)]
    
def loadMax2D(epath, net=False, sym=False):
    "charge un fichier csv de 2D"
    # lit le fichier
    ne1 = pd.read_csv( epath, header=1, sep = ', ', usecols=[0, 1, 3])
    x1 = np.array(ne1['centerF1'])
    xu1 = np.unique(x1)
    y1 = np.array(ne1['centerF2'])
    yu1 = np.unique(y1)
    z1 = np.array(ne1['max'])
    # calcul la matrice
    Xr1, Yr1 = np.meshgrid(yu1, xu1)
    Zr1 = np.reshape(z1,(len(xu1),len(yu1)))
    netmode = NETMODE
    if net:
        if netmode=='standard':
            Zr1 = nettoie(Zr1)
        elif netmode=='mieux':
            Zr1 = nettoie_mieux(Zr1)
        elif netmode=='encore':
            Zr1 = nettoie_encore_mieux(Zr1)
        else:
            raise Exception(netmode + ' : Wrong netmode !')
    if sym:
        Zr1 = symetrise(Zr1)
    return [Xr1, Yr1, np.nan_to_num(Zr1)]

def loadMin2D(epath, net=False, sym=False):
    "charge un fichier csv de 2D"
    # lit le fichier
    ne1 = pd.read_csv( epath, header=1, sep = ', ', usecols=[0, 1, 4])
    x1 = np.array(ne1['centerF1'])
    xu1 = np.unique(x1)
    y1 = np.array(ne1['centerF2'])
    yu1 = np.unique(y1)
    z1 = np.array(ne1['min'])
    # calcul la matrice
    Xr1, Yr1 = np.meshgrid(yu1, xu1)
    Zr1 = np.reshape(z1,(len(xu1),len(yu1)))
    netmode = NETMODE
    if net:
        if netmode=='standard':
            Zr1 = nettoie(Zr1)
        elif netmode=='mieux':
            Zr1 = nettoie_mieux(Zr1)
        elif netmode=='encore':
            Zr1 = nettoie_encore_mieux(Zr1)
        else:
            raise Exception(netmode + ' : Wrong netmode !')
    if sym:
        Zr1 = symetrise(Zr1)
    return [Xr1, Yr1, np.nan_to_num(Zr1)]


def affichegrid(X, Y, Z, scale=1.0, new=True, cmap=None, reverse=True):#Display of homonuclear 2D-NMR + grid
    if new:
        f1, ax1 = plt.subplots(figsize=(10, 8))
    else:
        ax1 = plt.gca()
    levelbase = [0.5,1,2,5,10,20,50,100]
    m1 = Z.max()
    level = [m1*(i/100.0)/scale for i in levelbase ]
    ax1.contour(X, Y, Z, level, cmap= cmap)
    ax1.set_xlabel(r"$\delta  (ppm)$")
    ax1.set_ylabel(r'$\delta  (ppm)$')
    major_ticks = np.arange(0, 10, 0.5)
    minor_ticks = np.arange(0, 10, 0.03)
    ax1.set_xticks(major_ticks)
    ax1.set_xticks(minor_ticks, minor = True)
    ax1.set_yticks(major_ticks)
    ax1.set_yticks(minor_ticks, minor = True)
    #ax1.grid(b = True, which = 'major', axis = 'both')
    #ax1.grid(b = True, which = 'minor', axis = 'both')
    #ax1.set_xlim(xmax=11,xmin=0)
    #ax1.set_ylim(ymax=11,ymin=0)
    ax1.grid(which='minor', alpha=1)
    ax1.grid(which='minor', alpha=1)
    if reverse:
        ax1.invert_xaxis()
        ax1.invert_yaxis()
    return ax1
    #plt.show()

def affiche(X, Y, Z, scale=1.0, new=True, cmap=None, reverse=True): 
    if new:
        f1, ax1 = plt.subplots(figsize=(10, 8))
    else:
        ax1 = plt.gca()
    levelbase = [0.5,1,2,5,10,20,50,100]
    m1 = Z.max()
    level = [m1*(i/100.0)/scale for i in levelbase ]
    ax1.contour(X, Y, Z, level, cmap=cmap)
    ax1.set_xlabel(r"f1 (ppm)")
    ax1.set_ylabel(r'f2 (ppm)')
    ax1.yaxis.set_label_position('right')
    major_ticks = np.arange(0, 9.5, 0.5)
    minor_ticks = np.arange(0, 9.5, 0.1)
    ax1.yaxis.set_ticks_position('right')
    ax1.set_xticks(major_ticks)
    ax1.set_xticks(minor_ticks, minor= True)
    ax1.set_yticks(major_ticks)
    ax1.set_yticks(minor_ticks, minor=True)
    #ax1.grid(b = True, which = 'major', axis = 'both')
    #ax1.grid(b = True, which = 'minor', axis = 'both')
    #ax1.set_xlim(xmax=11,xmin=0)
    #ax1.set_ylim(ymax=11,ymin=0)
    if reverse:
        ax1.invert_xaxis()
        ax1.invert_yaxis()
    return ax1
    #plt.show()
    
def afficheS(X, Y, Z, scale=1.0, new=True, cmap=None, reverse=True): #Display of homonuclear NMR in smaller picture
    if new:
        f1, ax1 = plt.subplots(figsize=(7, 5))
    else:
        ax1 = plt.gca()
    levelbase = [0.5,1,2,5,10,20,50,100]
    m1 = Z.max()
    level = [m1*(i/100.0)/scale for i in levelbase ]
    ax1.contour(X, Y, Z, level, cmap=cmap)
    ax1.set_xlabel(r"ppm")
    ax1.set_ylabel(r'ppm')
    major_ticks = np.arange(0, 10, 0.5)
    minor_ticks = np.arange(0, 10, 0.1)
    ax1.set_xticks(major_ticks)
    ax1.set_xticks(minor_ticks)
    ax1.set_yticks(major_ticks)
    ax1.set_yticks(minor_ticks)
    #ax1.grid(b = True, which = 'major', axis = 'both')
    #ax1.grid(b = True, which = 'minor', axis = 'both')
    #ax1.set_xlim(xmax=11,xmin=0)
    #ax1.set_ylim(ymax=11,ymin=0)
    if reverse:
        ax1.invert_xaxis()
        ax1.invert_yaxis()
    return ax1
    #plt.show()
    
def affiche2(X, Y, Z, scale=1.0, new=True, cmap=None, reverse=True):#Display of heteronuclear NMR
    if new:
        f1, ax1 = plt.subplots(figsize=(10, 8))
    else:
        ax1 = plt.gca()
    levelbase = [0.5,1,2,5,10,20,50,100]
    m1 = Z.max()
    level = [m1*(i/100.0)/scale for i in levelbase ]
    ax1.contour(X, Y, Z, level, cmap= cmap)
    ax1.set_xlabel(r"$\delta  (ppm)$")
    ax1.set_ylabel(r'$\delta  (ppm)$')
    major_ticksx = np.arange(0, 10, 0.5)
    minor_ticksx = np.arange(0, 10, 0.1)
    major_ticksy = np.arange(-15, 140, 10)
    minor_ticksy = np.arange(-15, 140, 2)
    ax1.set_xticks(major_ticksx)
    ax1.set_xticks(minor_ticksx, minor = True)
    ax1.set_yticks(major_ticksy)
    ax1.set_yticks(minor_ticksy, minor = True)
    #ax1.grid(b = True, which = 'major', axis = 'both')
    #ax1.grid(b = True, which = 'minor', axis = 'both')
    #ax1.set_xlim(xmax=11,xmin=0)
    #ax1.set_ylim(ymax=11,ymin=0)
    if reverse:
        ax1.invert_xaxis()
        ax1.invert_yaxis()
    return ax1
    #plt.show()
    
def affichedosy(X, Y, Z, scale=1.0, new=True, cmap=None, reverse=True):#Display of heteronuclear NMR
    if new:
        f1, ax1 = plt.subplots(figsize=(5, 2))
    else:
        ax1 = plt.gca()
    levelbase = [0.5,1,2,5,10,20,50,100]
    m1 = Z.max()
    level = [m1*(i/100.0)/scale for i in levelbase ]
    ax1.contour(X, Y, Z, level, cmap= cmap)
    ax1.set_xlabel(r"$\delta  (ppm)$")
    ax1.set_ylabel(r'$\delta  (D (F2))$')
    major_ticksx = np.arange(0, 9, 0.5)
    minor_ticksx = np.arange(0, 9, 0.1)
    major_ticksy = np.arange(1, 4, 1)
    minor_ticksy = np.arange(1, 4, 0.5)
    ax1.set_xticks(major_ticksx)
    ax1.set_xticks(minor_ticksx, minor = True)
    ax1.set_yticks(major_ticksy)
    ax1.set_yticks(minor_ticksy, minor = True)
    #ax1.grid(b = True, which = 'major', axis = 'both')
    #ax1.grid(b = True, which = 'minor', axis = 'both')
    #ax1.set_xlim(xmax=11,xmin=0)
    #ax1.set_ylim(ymax=11,ymin=0)
    if reverse:
        ax1.invert_xaxis()
        ax1.invert_yaxis()
    return ax1   
def affratio(I1, I2, new=True, scale=1.0, cmap=None):
    "affiche le ratio de deux images"
    # il faudrait tester que les X Y sont les même !!!
    affiche(I1[0], I1[1], I1[2]/(I2[2]+1e-5), scale=scale, new=new, cmap=cmap)
    if new:
        f1, ax1 = plt.subplots(figsize=(10, 8))
    
def affratio2(I1, I2, new=True, scale=1.0, cmap=None):
    "affiche le ratio de deux images"
    # il faudrait tester que les X Y sont les même !!!
    affiche2(I1[0], I1[1], I1[2]/(I2[2]+1e-5), scale=scale, new=new, cmap=cmap)
    
def afflogratio(I1, I2, scale=1.0):
    "affiche le ratio de deux images"
    # il faudrait tester que les X Y sont les même !!!
    affiche(I1[0], I1[1], np.log(I1[2]/(I2[2])), scale=scale)
    
def affsub(I1, I2, new=True, scale=1.0, cmap=None):
    "affiche le ratio de deux images"
    # il faudrait tester que les X Y sont les même !!!
    affiche(I1[0], I1[1], I1[2]-I2[2], scale=scale, new=new, cmap=cmap)
    if new:
        f1, ax1 = plt.subplots(figsize=(10, 8))
    
def affsub2(I1, I2, scale=1.0):
    "affiche le ratio de deux images"
    # il faudrait tester que les X Y sont les même !!!
    affiche2(I1[0], I1[1], I1[2]-I2[2], scale=scale)    
    
def symetrise(ZZ):
    "symetrisation des spectres"
    return np.minimum(ZZ, ZZ.T)
def nettoie(ZZ, factor=2.0):
    " enlève le bruit dans la matrice - hard thresholding"
    ZZr = ZZ.copy()
    thresh = factor*np.median(ZZ)
    print (thresh)
    ZZr[ZZ<thresh] = 1.0
    return ZZr
def nettoie_mieux(ZZ, factor=2.0):
    " enlève le bruit dans la matrice - hard thresholding par colonne"
    ZZr = ZZ.copy()
    for i in range(ZZ.shape[1]):
        iZZ = ZZ[:,i]
        thresh = factor*np.median(iZZ)
        ZZr[iZZ<thresh,i] = 1.0
    return ZZr
def nettoie_encore_mieux(ZZ, factor=2.0):
    " enlève le bruit dans la matrice  - soft thresholding par colonne"
    ZZr = ZZ.copy()
    for i in range(ZZ.shape[1]):
        iZZ = ZZ[:,i]
        thresh = factor*np.median(iZZ)
        ZZr[:,i] = np.where(iZZ<thresh, 1, iZZ-thresh+1) 
    return ZZr
nem = nettoie_encore_mieux
def compare(name, scale=1.0):
    " bon ca va !"
    g = loadStd2D(name, net=False, sym=False)
    d = loadStd2D(name, net=True, sym=True)
    levelbase = [1,2,5,10,20,50,100]
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(12,7))
    m1 = g[2].max()
    level = [m1*(i/100.0)/scale for i in levelbase ]
    ax1.contour(g[0], g[1], g[2], levels=level)

    m1 = d[2].max()
    level = [m1*(i/100.0)/scale for i in levelbase ]
    ax2.contour(d[0], d[1], d[2], level)
    ax1.invert_xaxis()
    ax1.invert_yaxis()
def normalize(Z):
    "normalise les histogrammes"
    ZZ = np.log(Z)
    mu = ZZ.mean()
    sigma = ZZ.std()
    