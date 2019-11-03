#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 11:52:35 2019

@author: nathanzelesko
"""
import numpy as np
import laplacian as laplacian
import pywt
import matplotlib.pyplot as plt
import matplotlib.colors as c
from mpl_toolkits import mplot3d
import math as m
import time
import random as rand
import mrcfile
import scipy as sp
import scipy.ndimage as ndimage
from sklearn.neighbors import radius_neighbors_graph
from sklearn.neighbors import kneighbors_graph
from sklearn.utils.extmath import _deterministic_vector_sign_flip as dvsf
from sklearn.metrics import pairwise_distances
from matplotlib import rcParams
rcParams.update({'figure.autolayout':True})

seed = 2019
std= 0.01644027
#std = 0
level = 5

def voxel(x,y,z,length=3,res=100):
    space = np.zeros(shape=(res,res,res))
    for i in range(length):
        for j in range(length):
            for k in range(length):
                space[x+i,y+j,z+k] = 1.0
    return space


def volume_to_wavelet_domain(p0,l=level):
    p = p0.copy()
    m = np.sum(p)
    p = np.divide(p,m)
    
    wavelet = pywt.Wavelet('coif3')
    
    coeffs = pywt.wavedecn(p,wavelet,mode='zero',level=l)
    
    coeffs = coeffs[1:]
    vect = []
    for j in range(len(coeffs)):
        for entry in coeffs[-1-j]:
            flat = np.asarray(coeffs[-1-j][entry]).flatten(order='C')
            for item in flat:
                vect.append(item*(2**(j*(1+(3/2)))))
    #print(max(np.asarray(vect).flatten()))
    #print(len(np.asarray(vect).flatten()))
    return np.asarray(vect).flatten()


def wave_emd(p1,p2):
    p = np.asarray(p1)-np.asarray(p2)
    p = np.abs(p)
    emd = np.sum(p)
    return emd


def emd(p1,p2):
    start_time = time.time()
    wave1 = volume_to_wavelet_domain(p1)
    wave2 = volume_to_wavelet_domain(p2)
    emd = wave_emd(wave1,wave2)
    end_time = time.time()
    elapsed_time = end_time-start_time
    return emd


def emd_time_trials(trial_num,res):
    times = []
    for i in range(trial_num):
        x1 = rand.randint(0,res-1)
        x2 = rand.randint(0,res-1)
        y1 = rand.randint(0,res-1)
        y2 = rand.randint(0,res-1)
        z1 = rand.randint(0,res-1)
        z2 = rand.randint(0,res-1)
        v1 = voxel(x1,y1,z1,res)
        v2 = voxel(x2,y2,z2,res)
        times.append(emd(v1,v2)[1])
    std = np.std(times)
    avg = np.average(times)
    return std,avg


def emd_test():
    ds = []
    emds = []
    
    v1 = voxel(0,0,0)
    for i in range(50):
        x = i
        y = i
        z = i
        d = m.sqrt(x**2 + y**2 + z**2)
        v2 = voxel(x,y,z)
        ds.append(d)
        emds.append(emd(v1,v2))
    
    fig,ax = plt.subplots()
    plt.plot(ds,emds,c='b',ms=0.5, label = 'WEMD')
    plt.plot(ds,ds,c='k',ms=0.5, label = 'EMD')
    plt.legend()
    plt.show()
    
    
def rotation_test():
    angles = []
    emds = []
    v1 = voxel(8,8,8)
    for i in range(20):
        angles.append(i)
        v2 = rotate_particle(v1,(i*18))
        emds.append(emd(v1,v2))
    
    fig,ax = plt.subplots()
    plt.plot(angles,emds,c='b',ms=0.5, label = 'WEMD')
    plt.show()
    

def real_data_rotation_test():
    angles = []
    emds = []
    template = load_particle()
    v1 = rotate_particle(template,0)
    for i in range(5):
        angles.append(i)
        v2 = rotate_particle(template,((i+1)*72))
        emds.append(emd(v1,v2))
    
    fig,ax = plt.subplots()
    plt.plot(angles,emds,c='b',ms=0.5, label = 'WEMD')
    plt.show()
    
    
def load_particle():
    with mrcfile.open('rotating_shaft_res6.mrc') as mrc:
        particle = mrc.data
    return particle


def save_particle(particle,filename='example.mrc'):
    with mrcfile.new(filename,overwrite=True) as mrc:
        mrc.set_data(particle)


def rotate_particle(particle,angle,noise = True):
    particle = ndimage.rotate(particle,angle,axes=(0,2),reshape=False)
    if noise == True:
        #c = np.linalg.norm(particle)
        #V = np.shape(particle)[0] * np.shape(particle)[1] * np.shape(particle)[2]
        for i in range(np.shape(particle)[0]):
            for j in range(np.shape(particle)[1]):
                for k in range(np.shape(particle)[2]):
                    particle[i][j][k] = particle[i][j][k] + np.random.normal(0,std)
    return particle


def generate_rotated_dataset(points,semirandom=True):
    data = []
    colors = []
    cmap = plt.get_cmap("hsv")
    og = load_particle()
    if semirandom == False:
        count = []
        angles = []
        for i in range(points):
            angle = i*(360/points)
            data.append(rotate_particle(og,angle))
            colors.append(c.to_hex(cmap(angle/360)))
            angles.append(angle)
    else:
        count = [0,0,0,0]
        angles = []
        for i in range(points):
            chance = rand.random()
            if chance < 0.60/3:
                angle = 0 + np.random.normal(0,1)
                if angle < 0:
                    angle = angle+360
                count[0]+=1
            elif chance < 2*0.60/3:
                angle = 120 + np.random.normal(0,1)
                count[1]+=1
            elif chance < 0.60:
                angle = 240 + np.random.normal(0,1)
                count[2]+=1
            else:
                angle = rand.uniform(0,360)
                count[3]+=1
            angles.append(angle)
            
        angles.sort()
        for angle in angles:
            colors.append(c.to_hex(cmap(angle/360)))
            data.append(rotate_particle(og,angle))
    return data,count,angles,colors


def set_diag(M,value):
    M=sp.sparse.csc_matrix(M).tocoo()
    diag_idx = (M.row==M.col)
    M.data[diag_idx] = value
    M=sp.sparse.csc_matrix(M)
    return M


def wave_transform_data(data):
    waves = []
    #counter = 1
    for image in data:
        waves.append(volume_to_wavelet_domain(image))
        #print(str(counter) + ' converted to wave domain.')
        #counter += 1
    return waves

def csr_iterate_keys(m):
    (height, width) = m.shape
    for i in range(height):
        for j in m[i].indices:
            yield (i,j)

def dok_update_min(dok, location, value):
    if location in dok:
        dok[location] = min(dok[location], value)
    else:
        dok[location] = value

def symmetrize_csr_matrix(m):
    (n,n1) = m.shape
    assert n == n1
    dok = sp.sparse.dok_matrix(m.copy())
    for (i,j) in csr_iterate_keys(m):
        dok_update_min(dok, (i,j), m[i,j])
        dok_update_min(dok, (j,i), m[i,j])
    return dok.tocsr()
    

def get_adjmatrix(data,metric,wave_form):
    if metric == 'euclidean':
        new_data = []
        for item in data:
            new_data.append(item.flatten())
        norms = kneighbors_graph(new_data,10,mode='connectivity',metric='minkowski',p=2,include_self=False)
        dist = pairwise_distances(new_data,metric='euclidean')
        
    elif metric == 'emd':
        if wave_form == False:
            waves = wave_transform_data(data)
        else:
            waves = data
        #norms = kneighbors_graph(waves,10,mode='connectivity',metric='minkowski',p=1,include_self=False)
        dist = pairwise_distances(waves,metric='manhattan')
        
    #W = set_diag(norms,0.0)
    #print(norms)
    np.save('dist_matrix_'+metric+'_'+str(len(data))+"_"+str(seed),dist)
    return symmetrize_csr_matrix(sp.sparse.csr_matrix(norms))


def get_degmatrix(W):
    degrees = []
    num = W.shape[0]
    for i in range(num):
        degrees.append(np.sum(W[i,:]))
    return degrees


def get_laplacian(data,metric,wave_form):
    num = len(data)
    
    W = get_adjmatrix(data,metric,wave_form)
    print("W finished.")
    degrees = get_degmatrix(W)
    #print(degrees)
    
    diag = np.sqrt(degrees)
    #print(diag)
    invsqrtdegrees = 1.0/(diag)
    
    D = sp.sparse.spdiags(invsqrtdegrees,0,num,num)
    I = sp.sparse.identity(num)
        
    L = sp.sparse.csr_matrix(I-(np.dot(np.dot(D,W),D)))
    L = (L+np.transpose(L))/2
    print("L finished.")

    return L,diag


def find_eigenvectors(L,diag):
    L *= -1
    vals,vect = sp.sparse.linalg.eigsh(L,3,sigma=1.0,which='LM')
    embedding = dvsf(vect.T[(4)::-1]/diag)
    print("Embedding found.")
    return vals,embedding


def get_manifold(data,metric='emd',wave_form=False):
    start_time = time.time()
    L,diag = get_laplacian(data,metric,wave_form)
    vals,embedding = find_eigenvectors(L,diag)
    end_time = time.time()
    elapsed_time = end_time-start_time
    return embedding,elapsed_time


def plot_embedding(embedding,colors,metric,n):
    fig,ax = plt.subplots()
    x = embedding[:,1]
    y = embedding[:,2]
    plt.axis('off')
    xmax = max(embedding[1])
    xmin = min(embedding[1])
    xmargin = 0.1*(xmax-xmin)
    ymax = max(embedding[2])
    ymin = min(embedding[2])
    ymargin = 0.1*(ymax-ymin)
    plt.xlim(xmin-xmargin,xmax+xmargin)
    plt.ylim(ymin-ymargin,ymax+ymargin)
    plt.gca().set_aspect('equal', adjustable='box')
    ax.scatter(x,y,c=colors,s=10,edgecolors='k',linewidth=0.3)
    plt.savefig(str(metric)+"_graph_"+str(n)+"_std="+str(std)+"_"+str(seed)+".pdf",format='pdf',pad_inches=0)
    
def get_embedding_distance(embedding,i,j):
    return m.sqrt(((embedding[1][i]-embedding[1][j])**2)+((embedding[2][i]-embedding[2][j])**2))

def get_correlation_coefficent(embedding,angles):
    distances = np.zeros(shape=(len(angles),len(angles)))
    angle_diffs = np.zeros(shape=(len(angles),len(angles)))
    for i in range(len(angles)):
        for j in range(len(angles)):
            diff = m.abs(angles[i]-angles[j])
            if diff > 180:
                diff = 360 - diff
            angle_diffs[i,j] = diff
            
            distances[i,j] = get_embedding_distance(embedding,i,j)
    
    corr_coeff = np.corrcoef(angle_diffs,distances)
    return corr_coeff
            
def generate_to_graph(n,metric):
    data,count,angles,colors = generate_rotated_dataset(n)
    if metric == 'euclidean':
        embedding = get_manifold(data,metric='euclidean')
        plot_embedding(embedding,colors)
    if metric == 'emd':
        embedding = get_manifold(data,metric='emd')
        plot_embedding(embedding,colors)
        
def generate_data(ns):
    np.random.seed(seed)
    for n in ns:
        data,count,angles,colors = generate_rotated_dataset(n)
        np.save("raw_data_"+str(n)+"_"+"std="+str(std)+"_"+str(seed),data)
        np.save("counts_"+str(n)+"_"+"std="+str(std)+"_"+str(seed),count)
        np.save("angles_"+str(n)+"_"+"std="+str(std)+"_"+str(seed),angles)
        np.save("colors_"+str(n)+"_"+"std="+str(std)+"_"+str(seed),colors)
        
def generate_diffusion_maps(ns):
    for n in ns:
        data = np.load("raw_data_"+str(n)+"_"+str(seed)+".npy")
        
        embedding,el_time = get_manifold(data,metric='euclidean')
        np.save("euclidean_embedding_"+str(n)+"_"+str(seed),embedding)
        np.save("euclidean_time_"+str(n)+"_"+str(seed),el_time)
        
        embedding,el_time = get_manifold(data,metric='emd')
        np.save("emd_embedding_"+str(n)+"_"+str(seed),embedding)
        np.save("emd_time_"+str(n)+"_"+str(seed),el_time)
        
def full_generation(ns):
    generate_data(ns)
    generate_diffusion_maps(ns)
    
def generate_plots(ns):
    for n in ns:
       emd_embedding = np.load("emd_embedding_"+str(n)+"_"+str(seed)+".npy")
       euclidean_embedding = np.load("euclidean_embedding_"+str(n)+"_"+str(seed)+".npy")
       colors = np.load("colors_"+str(n)+"_"+str(seed)+".npy")
       plot_embedding(emd_embedding,colors,(True,'emd',str(n)))
       plot_embedding(euclidean_embedding,colors,(True,'euclidean',str(n)))
       
def gen_plots(ns):
    for n in ns:
       emd_embedding = np.load("emd_embed"+str(n)+"_std="+str(std)+"_"+str(seed)+".npy")
       euclidean_embedding = np.load("euc_embed"+str(n)+"_std="+str(std)+"_"+str(seed)+".npy")
       colors = np.load("colors_"+str(n)+"_std="+str(std)+"_"+str(seed)+".npy")
       plot_embedding(emd_embedding,colors,(True,'emd',str(n)))
       plot_embedding(euclidean_embedding,colors,(True,'euclidean',str(n)))
      
def get_dist_matrix(data,metric):
    if metric == 'euclidean':
        new_data = []
        for item in data:
            new_data.append(item.flatten())
        start = time.time()
        dist = pairwise_distances(new_data,metric='euclidean')
        end = time.time()
        elapsed = end-start
        np.save('dist_matrix_'+metric+'_'+str(len(data))+"_std="+str(std)+"_"+str(seed),dist)
        return(0,elapsed)
        
    elif metric == 'emd':
        start = time.time()
        waves = wave_transform_data(data)
        end = time.time()
        wave_elapsed = end-start
        start = time.time()
        dist = pairwise_distances(waves,metric='manhattan')
        end = time.time()
        elapsed = end-start
        np.save('dist_matrix_'+metric+'_'+str(len(data))+"_std="+str(std)+"_"+str(seed),dist)
        return(wave_elapsed,elapsed)

     
def calculate_distances(ns):
    euc_dist_times = []
    emd_dist_times = []
    wave_trans_times = []
    for n in ns:
        data = np.load("raw_data_"+str(n)+"_"+"std="+str(std)+"_"+str(seed)+".npy")
        euc_elapsed = get_dist_matrix(data,'euclidean')[1]
        wave_elapsed,emd_elapsed = get_dist_matrix(data,'emd')
        euc_dist_times.append(euc_elapsed)
        emd_dist_times.append(emd_elapsed)
        wave_trans_times.append(wave_elapsed)
    np.save("euc_dist_times_" +"std="+str(std)+"_"+ str(seed) + ".npy",euc_dist_times)
    np.save("emd_dist_times_" +"std="+str(std)+"_"+ str(seed) + ".npy",emd_dist_times)
    np.save("wave_trans_times_" +"std="+str(std)+"_"+ str(seed) + ".npy",wave_trans_times)
   
def gaussian_kernel_distance_matrix(distance_matrix, sigma):
    assert distance_matrix.ndim == 2
    assert distance_matrix.shape[0] == distance_matrix.shape[1]
 
    m = np.exp(-distance_matrix.astype(float)**2/(2*sigma*2))
    np.fill_diagonal(m, 0)
 
    return m    
    
def get_embedding(ns,sigma):
    emd_embed_times = []
    euc_embed_times = []
    for n in ns:
        emd = np.load('dist_matrix_emd_'+str(n)+'_std='+str(std)+'_'+str(seed)+'.npy')
        euc = np.load('dist_matrix_euclidean_'+str(n)+'_std='+str(std)+'_'+str(seed)+'.npy')
        
        start = time.time()
        W = gaussian_kernel_distance_matrix((emd+emd.T)/2.0, sigma)
        L = laplacian.geometric(W)
        (eigvals, eigvecs) = np.linalg.eig(L)
        eigvals,eigvecs = sp.sparse.linalg.eigs(L,3,which='SM')
        end = time.time()
        elapsed = end - start
        emd_embed_times.append(elapsed)
        np.save("emd_embed"+str(n)+"_std="+str(std)+"_"+str(seed)+".npy",eigvecs)
        
        start = time.time()
        W = gaussian_kernel_distance_matrix((euc+euc.T)/2.0, sigma)
        L = laplacian.geometric(W)
        (eigvals, eigvecs) = np.linalg.eig(L)
        eigvals,eigvecs = sp.sparse.linalg.eigs(L,3,which='SM')
        end = time.time()
        elapsed = end - start
        euc_embed_times.append(elapsed)
        np.save("euc_embed"+str(n)+"_std="+str(std)+"_"+str(seed)+".npy",eigvecs)
        
        np.save('euc_embed_times_std=' + str(std)+"_"+str(seed),euc_embed_times)
        np.save('emd_embed_times_std=' + str(std)+"_"+str(seed),emd_embed_times)
        
def data_to_graph(ns,sigma):
    generate_data(ns)
    calculate_distances(ns)
    #get_embedding(ns,sigma)
    #gen_plots(ns)
  
    
def generate_standard_data(n):
    data,count,angles,colors = generate_rotated_dataset(n,semirandom=False)
    np.save("raw_data_"+str(n)+"_standard",data)
    np.save("angles_"+str(n)+"_standard",angles)
    np.save("colors_"+str(n)+"_standard",colors)    
    
def calculate_emds(data,n):
    emds = []
    waves = wave_transform_data(data)
    for i in range(len(data)):
        emds.append(wave_emd(waves[0],waves[i]))
    np.save("standard_"+str(n)+"_emds",emds)

def calculate_euclid_dists(data,n):
    dists = []
    for i in range(len(data)):
        dists.append(np.linalg.norm(data[0]-data[i]))
    np.save("standard_"+str(n)+"_euclidean_dists",dists)
        
def emd_vs_euclid_plot(n):
    angles = np.load("angles_"+str(n)+"_standard.npy")
    for i in range(len(angles)):
        if angles[i] > 180:
            angles[i] = -(360 - angles[i])
    emds = np.load("standard_"+str(n)+"_emds.npy")
    eucs = np.load("standard_"+str(n)+"_euclidean_dists.npy")
    
    sorted_emds = np.array([x for y, x in sorted(zip(angles,emds))])
    sorted_eucs = np.array([x for y, x in sorted(zip(angles,eucs))])
    angles.sort()
    
    maxemd = max(emds)
    maxeuc = max(eucs)
    s = maxemd/maxeuc
    sorted_eucs = s*sorted_eucs
    
    fig,ax = plt.subplots()
    plt.plot(angles,sorted_emds,label="WEMD")
    plt.plot(angles,sorted_eucs,'--',label="Euclidean")
    plt.legend()
    plt.xticks([-180,-150,-120,-90,-60,-30,0,30,60,90,120,150,180],['-180$^{\circ}$','-150$^{\circ}$','-120$^{\circ}$','-90$^{\circ}$','-60$^{\circ}$','-30$^{\circ}$','0$^{\circ}$','30$^{\circ}$','60$^{\circ}$','90$^{\circ}$','120$^{\circ}$','150$^{\circ}$','180$^{\circ}$'])
    plt.yticks([])
    plt.legend(fontsize=12)
    ax.tick_params(labelsize=11)
    plt.xlim(-180,180)
    plt.savefig("L2vsEMD_"+str(level)+".pdf",format='pdf',pad_inches=0)  
    
def emd_vs_euclid(n):
    generate_standard_data(n)
    data = np.load("raw_data_"+str(n)+"_standard.npy")
    calculate_emds(data,n)
    calculate_euclid_dists(data,n)
    emd_vs_euclid_plot(n)
    
    
    
