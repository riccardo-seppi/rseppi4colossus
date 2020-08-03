'''
Example script about seppi20 implementation in colossus
'''

#import model and integrals
from HMF_dyn_state_colossus import model_seppi20

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
mass_ = np.logspace(13.5,16,60)
R = peaks.lagrangianR(mass_)
sigma = cosmo.sigma(R,z=0)

xoff = np.logspace(-3.5,-0.3,50)
spin = np.logspace(-3.5,-0.3,50)

xoff2 = np.logspace(-3.5,-1.5,30)
spin2 = np.logspace(-3.5,-1.5,30)

#build model and integrate it
f_sigma = model_seppi20.seppi20(sigma,xoff,spin,z=0)
f_sigma2 = model_seppi20.seppi20(sigma,xoff2,spin2,z=0)

#use bhattacharya11, despali16 and comparat17 for comparison
mf_tinker=mf.massFunction(sigma,q_in='sigma', z=0, mdef = 'vir', model = 'tinker08', q_out = 'f') 
mf_bhattacharya=mf.massFunction(sigma,q_in='sigma', z=0, model = 'bhattacharya11', q_out = 'f') 
mf_comparat=mf.massFunction(sigma,q_in='sigma', z=0, mdef = 'vir', model = 'comparat17', q_out = 'f') 
mf_despali=mf.massFunction(sigma,q_in='sigma', z=0, mdef = 'vir', model = 'despali16', q_out = 'f') 

#function to convert log10(1/sigma) to Mass
def Mass_sigma(x):
    r=cosmo.sigma(1/(10**x),z=0,inverse=True)
    M=peaks.lagrangianM(r)/cosmo.Hz(z=0)*100
    return np.log10(M)

#plots
fig = plt.figure(figsize=(10,10))
gs = fig.add_gridspec(3,1)
ax1 = fig.add_subplot(gs[0:2, :])
ax2 = fig.add_subplot(gs[2, :])
x = np.log10(1/sigma)
r1 = (f_sigma-mf_tinker)/mf_tinker
r2 = (f_sigma-mf_bhattacharya)/mf_bhattacharya
r3 = (f_sigma - mf_despali)/mf_despali
r4 = (f_sigma-mf_comparat)/mf_comparat
ax1.plot(x,mf_tinker,label='tinker08')
ax1.plot(x,mf_bhattacharya,label='bhattacharya11')
ax1.plot(x,mf_despali,label='despali16')
ax1.plot(x,mf_comparat,label='comparat17')
ax1.plot(x,f_sigma,label='seppi20')
ax1.plot(x,f_sigma2,label='seppi20 subsample')
#ax1.plot(x,h[:,25,20],label='slice seppi20')
ax2.plot(x,r1)
ax2.plot(x,r2)
ax2.plot(x,r3)
ax2.plot(x,r4)
ax1.tick_params(labelsize=20)
ax2.tick_params(labelsize=20)
ax2.set_xlabel(r'$\log_{10}\sigma^{-1}$', fontsize=20)
ax1.set_ylabel(r'$\log_{10}f(\sigma)$', fontsize=20)
ax1.legend(fontsize=15)
ax1.set_yscale('log')
ax1.grid(True)
ax = plt.gca()
ax1_sec = ax1.twiny()
xmin,xmax=ax1.get_xlim()
new_x0 = Mass_sigma(xmin)
new_x1 = Mass_sigma(xmax)
ax1_sec.set_xlim(left=new_x0,right=new_x1)
ax1_sec.set_xlabel(r'$\log_{10}M\ [M_{\odot}]$', fontsize=25, labelpad=15)
ax2.set_ylabel(r'$\Delta f/f$', fontsize=20)
ax1_sec.tick_params(labelsize=20)
ax2.grid(True)
plt.tight_layout()
outf = 'figures/MFs.png'
plt.savefig(outf,overwrite=True)
