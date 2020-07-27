'''
Example script about seppi20 implementation in colossus
'''

#import model and integrals
from HMF_dyn_state_colossus import model_seppi20
from HMF_dyn_state_colossus import integrals

#import colossus
from colossus.cosmology import cosmology
from colossus.lss import mass_function as mf
from colossus.lss import peaks
import numpy as np

#import matplotlib and numpy
import matplotlib.pyplot as plt
import numpy as np

#set cosmology
cosmo = cosmology.setCosmology('multidark-planck')  

#define variance, xoff, spin arrays
mass_ = np.logspace(13.8,16,60)
R = peaks.lagrangianR(mass_)
sigma_ = cosmo.sigma(R,z=0)
#sigma_ = np.linspace(0.3,1.2,60)
xoff_ = np.logspace(-3,-0.3,50)
spin_ = np.logspace(-3.1,-0.4,40)

#create meshgrid
sigma,xoff,spin = np.meshgrid(sigma_,xoff_,spin_,indexing='ij')

#build model and integrate it
h = model_seppi20.seppi20(sigma,xoff,spin,z=0)
g = integrals.int_h_dspin(sigma_,xoff_,spin_,h)
f_sigma = integrals.int_f_sigma(sigma_,xoff_,g)

#use despali16 and comparat17 for comparison
mf_comparat=mf.massFunction(sigma_,q_in='sigma', z=0, mdef = 'vir', model = 'comparat17', q_out = 'f') 
mf_despali=mf.massFunction(sigma_,q_in='sigma', z=0, mdef = 'vir', model = 'despali16', q_out = 'f') 

#function to convert log10(1/sigma) to Mass
def Mass_sigma(x):
    r=cosmo.sigma(1/(10**x),z=0,inverse=True)
    M=peaks.lagrangianM(r)/cosmo.Hz(z=0)*100
    return np.log10(M)

#plots
plt.figure(figsize=(10,10))
x = np.log10(1/sigma_)
plt.plot(x,f_sigma,label='seppi20')
plt.plot(x,mf_comparat,label='comparat17')
plt.plot(x,mf_comparat,label='despali16')
plt.plot(x,h[:,25,20],label='slice seppi20')
plt.yscale('log')
plt.xlabel(r'$\log_{10}\sigma^{-1}$', fontsize=20)
plt.ylabel(r'$\log_{10}f(\sigma)$', fontsize=20)
plt.legend(fontsize=15)
plt.tick_params(labelsize=20)
#ax = plt.gca()
#ax1_sec = plt.twiny()
#xmin,xmax=ax.get_xlim()
#new_x0 = Mass_sigma(xmin)
#new_x1 = Mass_sigma(xmax)
#ax1_sec.set_xlim(left=new_x0,right=new_x1)
#ax1_sec.set_xlabel(r'$\log_{10}M\ [M_{\odot}]$', fontsize=25, labelpad=15)
#ax1_sec.tick_params(labelsize=20)
plt.grid(True)
plt.tight_layout()
plt.show()
