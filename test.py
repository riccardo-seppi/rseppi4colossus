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
print(sigma)
xoff = np.logspace(-3.5,-0.3,50)
spin = np.logspace(-3.5,-0.3,50)

xoff2 = np.logspace(-3.5,-1.5,30)
spin2 = np.logspace(-3.5,-1.5,30)

xoff3 = np.array([0.1])
spin3 = np.array([0.1])

#build model and integrate it
g_sigma_xoff = model_seppi20.seppi20(sigma,xoff,z=0,int_xoff=False)
g_sigma_spin = model_seppi20.seppi20(sigma,spin,z=0,int_spin=False)
g_xoff_spin = model_seppi20.seppi20(xoff,spin,z=0,int_sigma=True, int_xoff=False, int_spin=False)
f_sigma = model_seppi20.seppi20(sigma,xoff,spin,z=0)
f_xoff = model_seppi20.seppi20(sigma,xoff,spin,z=0,int_xoff=False, int_sigma=True)
f_spin = model_seppi20.seppi20(sigma,xoff,spin,z=0,int_spin=False, int_sigma=True)
f_sigma2 = model_seppi20.seppi20(sigma,xoff2,spin2,z=0)
f_sigma3 = model_seppi20.seppi20(sigma,xoff3,spin3,z=0)

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
ax1.plot(x,f_sigma3,label='seppi20 one value')
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

fig2 = plt.figure(figsize=(10,10))
gs2 = fig2.add_gridspec(3,3)
ax1 = fig2.add_subplot(gs2[0,0])
ax2 = fig2.add_subplot(gs2[0,1])
ax3 = fig2.add_subplot(gs2[0,2])
ax4 = fig2.add_subplot(gs2[1,0])
ax5 = fig2.add_subplot(gs2[1,1])
ax6 = fig2.add_subplot(gs2[1,2])
ax7 = fig2.add_subplot(gs2[2,0])
ax8 = fig2.add_subplot(gs2[2,1])
ax9 = fig2.add_subplot(gs2[2,2])
ax1.plot(xoff,g_sigma_xoff[int(len(sigma)/2),:],label=r'$\log_{10}\sigma ^{-1} = %.3g$'%(np.log10(1/sigma[int(len(sigma)/2)])))
ax1.set_xlabel(r'$X_{off}$')
ax1.set_ylabel(r'$g(\sigma,X_{off})$')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylim(1e-6)
ax1.legend()
ax2.plot(np.log10(1/sigma),g_sigma_xoff[:,int(len(xoff)/2)],label=r'$X_{off} = %.3g$'%(xoff[int(len(xoff)/2)]))
ax2.set_xlabel(r'$\log_{10}\sigma ^{-1}$')
ax2.set_ylabel(r'$g(\sigma,X_{off})$')
ax2.set_yscale('log')
ax2.set_ylim(1e-6)
ax2.legend()
ax3.plot(spin,g_sigma_spin[int(len(sigma)/2),:],label=r'$\log_{10}\sigma ^{-1} = %.3g$'%(np.log10(1/sigma[int(len(sigma)/2)])))
ax3.set_xlabel(r'$\lambda$')
ax3.set_ylabel(r'$g(\sigma,\lambda)$')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_ylim(1e-6)
ax3.legend()
ax4.plot(np.log10(1/sigma),g_sigma_spin[:,int(len(spin)/2)],label=r'$\lambda = %.3g$'%(spin[int(len(spin)/2)]))
ax4.set_xlabel(r'$\log_{10}\sigma^{-1}$')
ax4.set_ylabel(r'$g(\sigma,\lambda)$')
ax4.set_yscale('log')
ax4.set_ylim(1e-6)
ax4.legend()
ax5.plot(xoff,g_xoff_spin[:,int(len(spin)/2)],label=r'$\lambda = %.3g$'%(spin[int(len(spin)/2)]))
ax5.set_xlabel(r'$X_{off}$')
ax5.set_ylabel(r'$g(X_{off},\lambda)$')
ax5.set_xscale('log')
ax5.set_yscale('log')
ax5.set_ylim(1e-6)
ax5.legend()
ax6.plot(spin,g_xoff_spin[int(len(xoff)/2),:],label=r'$X_{off} = %.3g$'%(xoff[int(len(xoff)/2)]))
ax6.set_xlabel(r'$\lambda$')
ax6.set_ylabel(r'$g(X_{off},\lambda)$')
ax6.set_xscale('log')
ax6.set_yscale('log')
ax6.set_ylim(1e-6)
ax6.legend()
ax7.plot(np.log10(1/sigma),f_sigma)
ax7.set_xlabel(r'$\log_{10}\sigma ^{-1}$')
ax7.set_ylabel(r'$f(\sigma)$')
ax7.set_yscale('log')
ax7.set_ylim(1e-6)
ax8.plot(xoff,f_xoff)
ax8.set_xlabel(r'$X_{off}$')
ax8.set_ylabel(r'$fX_{off})$')
ax8.set_xscale('log')
ax8.set_yscale('log')
ax8.set_ylim(1e-6)
ax9.plot(spin,f_spin)
ax9.set_xlabel(r'$\lambda$')
ax9.set_ylabel(r'$f(\lambda)$')
ax9.set_xscale('log')
ax9.set_yscale('log')
ax9.set_ylim(1e-6)
plt.tight_layout()
outfig = 'figures/distributions.png'
plt.savefig(outfig,overwrite=True)
plt.show()

