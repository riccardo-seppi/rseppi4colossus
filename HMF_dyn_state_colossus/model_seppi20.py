from colossus.lss import peaks
import numpy as np
from scipy import integrate


def seppi20(sigma,xoff,spin,z):
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
		
	Returns
	-----------------------------------------------------------------------------------------------
	f: array_like
		The halo mass function :math:`f(\\sigma)`, integrated on the given values on xoff and spin. It has the same dimensions as the input meshgrid.
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

    if xoff is None:
        xoff = np.logspace(-3.5,-0.3,50)
    if spin is None:
        spin = np.logspace(-3.5,-0.3,50)

    sigma_,xoff_,spin_ = np.meshgrid(sigma,xoff,spin,indexing='ij')

    h_log = A+np.log10(np.sqrt(2/np.pi)) + q*np.log10(np.sqrt(a)*dc/sigma_) - a/2/np.log(10)*dc**2/sigma_**2 + (alpha)*np.log10(xoff_/10**(1.83*mu)) - 1/np.log(10)*(xoff_/10**(1.83*mu))**(0.05*alpha) + gamma*np.log10(spin_/(10**(mu))) - 1/np.log(10)*(xoff_/10**(1.83*mu)/sigma_**e)**(beta)*(spin_/(10**(mu)))**(delta)   
    h = 10**h_log

    g = np.zeros((len(sigma),len(xoff)))
    for i in range(len(sigma)):
        for j in range(len(xoff)):
            g[i,j] = integrate.simps(h[i,j,:],np.log10(spin))

    f = np.zeros(len(sigma))
    for i in range(len(sigma)):
        f[i] = integrate.simps(g[i,:],np.log10(xoff))
    
    return f













