import numpy as np
import particle
import wavetransform
import matplotlib.pyplot as plt

n = 360

def generate_standard_data(computation_dir):
    data,count,angles,colors = particle.generate_rotated_dataset(n,std=0,semirandom=False)
    np.save(computation_dir+ f"Raw Data/raw_data_"+str(n)+"_standard",data)
    np.save(computation_dir+ f"Angles/angles_"+str(n)+"_standard",angles)
    np.save(computation_dir+ f"Colors/colors_"+str(n)+"_standard",colors)    
    
    
def calculate_emds(data):
    emds = []
    waves = wavetransform.wave_transform_data(data)
    for i in range(len(data)):
        emds.append(wavetransform.wave_emd(waves[0],waves[i]))
    return emds


def calculate_euclid_dists(data):
    dists = []
    for i in range(len(data)):
        dists.append(np.linalg.norm(data[0]-data[i]))
    return dists
    
    
def emd_vs_euclid_plot(data,computation_dir,figure_dir):
    angles = np.load(computation_dir+ f"Angles/angles_"+str(n)+"_standard.npy")
    for i in range(len(angles)):
        if angles[i] > 180:
            angles[i] = -(360 - angles[i])
    emds = calculate_emds(data)
    eucs = calculate_euclid_dists(data)
    
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
    plt.savefig(figure_dir+f"L2vsEMD.pdf",format='pdf',pad_inches=0)  
    
    
def emd_vs_euclid(computation_dir,figure_dir):
    generate_standard_data()
    data = np.load(computation_dir + f"Raw Data/raw_data_"+str(n)+"_standard.npy")
    emd_vs_euclid_plot(data,figure_dir)