import os
import random as rand

import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib.colors as c

import mrcfile


def save_particle(particle,filename='example.mrc'):
    with mrcfile.new(filename,overwrite=True) as mrc:
        mrc.set_data(particle)


def load_particle(particle_dir,filename='rotating_shaft_res6.mrc'):
    with mrcfile.open(os.path.join(particle_dir, filename)) as mrc:
        particle = mrc.data
    return particle


def rotate_particle(particle,angle,std):
    particle = ndimage.rotate(particle,angle,axes=(0,2),reshape=False)
    if std != 0:
        for i in range(np.shape(particle)[0]):
            for j in range(np.shape(particle)[1]):
                for k in range(np.shape(particle)[2]):
                    particle[i][j][k] = particle[i][j][k] + np.random.normal(0,std)
    return particle


def generate_rotated_dataset(points,particle_dir,std,semirandom=True):
    data = []
    colors = []
    cmap = plt.get_cmap("hsv")
    og = load_particle(particle_dir)
    if semirandom == False:
        count = []
        angles = []
        for i in range(points):
            angle = i*(360/points)
            data.append(rotate_particle(og,angle,std))
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
            data.append(rotate_particle(og,angle,std))

    return (data, count, angles, colors)
