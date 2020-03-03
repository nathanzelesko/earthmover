import os

import numpy as np
import matplotlib.pyplot as plt

import particle
import wavetransform

n = 360

def generate_standard_data(computation_dir, particle_dir):
    (data, count, angles, colors) = particle.generate_rotated_dataset(n, particle_dir, std=0,semirandom=False)
    np.save(os.path.join(computation_dir, f"Noiseless Data/Raw Data/raw_data_{n}_standard"), data)
    np.save(os.path.join(computation_dir, f"Noiseless Data/Angles/angles_{n}_standard"), angles)
    np.save(os.path.join(computation_dir, f"Noiseless Data/Colors/colors_{n}_standard"), colors)    
    
    
def calculate_emds(data):
    emds = []
    waves = wavetransform.wave_transform_volumes(data)
    for i in range(len(data)):
        emds.append(wavetransform.wave_emd(waves[0],waves[i]))
    return emds


def calculate_euclid_dists(data):
    dists = []
    for i in range(len(data)):
        dists.append(np.linalg.norm(data[0]-data[i]))
    return dists
    
    
def emd_vs_euclid_plot(data,computation_dir,figure_dir):
    angles = np.load(os.path.join(computation_dir, f"Noiseless Data/Angles/angles_{n}_standard.npy"))
    for i in range(len(angles)):
        if angles[i] > 180:
            angles[i] = -(360 - angles[i])
    print('Calculatind EMD distances')
    emds = calculate_emds(data)
    print('Calculatind Euclidean distances')
    eucs = calculate_euclid_dists(data)
    
    sorted_emds = np.array([x for y, x in sorted(zip(angles,emds))])
    sorted_eucs = np.array([x for y, x in sorted(zip(angles,eucs))])
    angles.sort()
    
    maxemd = max(emds)
    maxeuc = max(eucs)
    s = maxemd/maxeuc
    sorted_eucs = s*sorted_eucs
    
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.linewidth"] = 0.5 
    plt.rcParams['xtick.minor.width'] = 0.5
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['text.usetex'] = True
    plt.rcParams['mathtext.fontset'] = 'cm' # Computer-modern font that is used in LaTeX
    plt.rcParams['pdf.fonttype'] = 42 # This gets rid of "Type 3 font" errors in the IEEE compliance system

    fig,ax = plt.subplots(figsize=(4, 2.1631))
    plt.plot(angles,sorted_emds,label="WEMD")
    plt.plot(angles,sorted_eucs,'--',label="Euclidean")
    plt.legend()
    plt.xticks([-180,-150,-120,-90,-60,-30,0,30,60,90,120,150,180],['-180$^{\circ}$','-150$^{\circ}$','-120$^{\circ}$','-90$^{\circ}$','-60$^{\circ}$','-30$^{\circ}$','0$^{\circ}$','30$^{\circ}$','60$^{\circ}$','90$^{\circ}$','120$^{\circ}$','150$^{\circ}$','180$^{\circ}$'])
    plt.yticks([])
    plt.legend(fontsize=9)
    ax.tick_params(labelsize=6)
    plt.xlim(-180,180)
    plt.savefig(os.path.join(figure_dir,f"L2vsEMD.pdf"),pad_inches=0, bbox_inches='tight')  
    
    
def emd_vs_euclid(computation_dir, particle_dir, figure_dir):
    generate_standard_data(computation_dir, particle_dir)
    data = np.load(os.path.join(computation_dir, f"Noiseless Data/Raw Data/raw_data_{n}_standard.npy"))
    emd_vs_euclid_plot(data, computation_dir, figure_dir)

