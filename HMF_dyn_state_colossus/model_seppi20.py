from colossus.lss import peaks
import numpy as np
from scipy import integrate


def seppi20(sigma,z,xoff=None,spin=None,int_sigma=False,int_xoff=True,int_spin=True):
    """
	The mass function model of Seppi et al 2020.
	
	The model is specified in Equation 23.
    Calibrated for M > 4x10^13 Msun at z=0.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	sigma: array_like
		Variance; can be a number or an array.
    xoff: array_like
        Offset parameter; can be a number or array.
    spin: array_like
        Spin parameter; can be a number or an array.
    int_sigma: bool
        Boolean variable to integrate on sigma or not.
    int_xoff: bool
        Boolean variable to integrate on xoff or not.
    int_spin: bool
        Boolean variable to integrate on spin or not.
		
	Returns
	-----------------------------------------------------------------------------------------------
    The output depends on which integrals the user chooses to perform.
    h: 3D meshgrid
        The halo mass-xoff-spin function :math:`h(\\sigma,xoff,\\lambda)`
    g_sigma_xoff: 2D meshgrid
        The halo mass-xoff function :math:`g(\\sigma,xoff)`, integrated on spin
    g_sigma_spin: 2D meshgrid
        The halo mass-spin function :math:`g(\\sigma,\\lambda)`, integrated on xoff
    g_xoff_spin: 2D meshgrid
        The halo xoff-spin function :math:`g(xoff,\\lambda)`, integrated on mass
	f_xoff: array_like
		The halo xoff function :math:`f(xoff)`, integrated on mass and spin.
	f_spin: array_like
		The halo spin function :math:`f(\\sigma)`, integrated on mass and xoff.
	f: array_like
		The halo mass function :math:`f(\\sigma)`, integrated on xoff and spin.
	"""
    zp1 = 1+z
    dc = peaks.collapseOverdensity(z = 0)    

    A = -22.004*(zp1)**-0.0441
    a = 0.886*(zp1)**-0.1611
    q = 2.285*(zp1)**0.0409
    mu = -3.326*(zp1)**-0.1286
    alpha = 5.623*(zp1)**0.1081
    beta = -0.391*(zp1)**-0.3114
    gamma = 3.024*(zp1)**0.0902
    delta = 1.209*(zp1)**-0.0768
    e = -1.105*(zp1)**0.6123
    
    if sigma is None:
        sigma = np.linspace(0.25,1.2,50)
    if xoff is None:
        xoff = np.logspace(-3.5,-0.3,50)
    if spin is None:
        spin = np.logspace(-3.5,-0.3,50)

    sigma_,xoff_,spin_ = np.meshgrid(sigma,xoff,spin,indexing='ij')

    h_log = A+np.log10(np.sqrt(2/np.pi)) + q*np.log10(np.sqrt(a)*dc/sigma_) - a/2/np.log(10)*dc**2/sigma_**2 + (alpha)*np.log10(xoff_/10**(1.83*mu)) - 1/np.log(10)*(xoff_/10**(1.83*mu))**(0.05*alpha) + gamma*np.log10(spin_/(10**(mu))) - 1/np.log(10)*(xoff_/10**(1.83*mu)/sigma_**e)**(beta)*(spin_/(10**(mu)))**(delta)   
    h = 10**h_log

    #compute 2D distributions
    g_xoff_spin = np.zeros((len(xoff),len(spin)))    
    for i in range(len(xoff)):
        for j in range(len(spin)):
            if len(sigma)==1:
                g_xoff_spin[i,j] = h[:,i,j]
            else:    
                g_xoff_spin[i,j] = integrate.simps(h[:,i,j],sigma)

    g_sigma_spin = np.zeros((len(sigma),len(spin)))    
    for i in range(len(sigma)):
        for j in range(len(spin)):
            if len(xoff)==1:
                g_sigma_spin[i,j] = h[i,:,j]
            else:    
                g_sigma_spin[i,j] = integrate.simps(h[i,:,j],np.log10(xoff))

    g_sigma_xoff = np.zeros((len(sigma),len(xoff)))    
    for i in range(len(sigma)):
        for j in range(len(xoff)):
            if len(spin)==1:
                g_sigma_xoff[i,j] = h[i,j,:]
            else:    
                g_sigma_xoff[i,j] = integrate.simps(h[i,j,:],np.log10(spin))

    #compute 1D distributions
    f_xoff = np.zeros(len(xoff))
    for i in range(len(xoff)):
        if len(sigma)==1:
            f_xoff[i] = g_sigma_xoff[:,i]
        else:    
            f_xoff[i] = integrate.simps(g_sigma_xoff[:,i],np.log10(1/sigma))

    f_spin = np.zeros(len(spin))
    for i in range(len(spin)):
        if len(sigma)==1:
            f_spin[i] = g_sigma_spin[:,i]
        else:    
            f_spin[i] = integrate.simps(g_sigma_spin[:,i],np.log10(1/sigma))
    
    f_sigma = np.zeros(len(sigma))
    for i in range(len(sigma)):
        if len(xoff)==1:
            f_sigma[i] = g_sigma_xoff[i,:]
        else:    
            f_sigma[i] = integrate.simps(g_sigma_xoff[i,:],np.log10(xoff))

    if (int_sigma==False)&(int_xoff==False)&(int_spin==False):
        return h

    if (int_sigma==True)&(int_xoff==False)&(int_spin==False):
        return g_xoff_spin
    
    if (int_sigma==False)&(int_xoff==True)&(int_spin==False):
        return g_sigma_spin

    if (int_sigma==False)&(int_xoff==False)&(int_spin==True):
        return g_sigma_xoff

    if (int_sigma==True)&(int_xoff==True)&(int_spin==False):
        return f_spin
    
    if (int_sigma==True)&(int_xoff==False)&(int_spin==True):
        return f_xoff

    if (int_sigma==False)&(int_xoff==True)&(int_spin==True):
        return f_sigma 












