from colossus.cosmology import cosmology
from colossus.lss import peaks
import numpy as np

cosmo = cosmology.setCosmology('multidark-planck')    


def seppi20(sigma,xoff,spin,z):
    """
	The mass function model of Seppi et al 2020.
	
	The model is specified in Equation 23.
    Needs a meshgrid of sigma,xoff,spin.
    You can create it with numpy.meshgrid.
    Make sure to use indexing 'ij'.
    Calibrated for M > 4x10^13 Msun at z=0.
	
	Parameters
	-----------------------------------------------------------------------------------------------
	sigma: array_like
		Variance; can be a number or a meshgrid with xoff and spin.
    xoff: array_like
        Offset parameter; can be a number or a meshgrid with sigma and spin.
    spin: array_like
        Spin parameter; can be a number or a meshgrid with sigma and xoff.
		
	Returns
	-----------------------------------------------------------------------------------------------
	h: array_like
		The halo mass-xoff-spin function :math:`h(\\sigma,xoff,spin)`, has the same dimensions as the input meshgrid.
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

    model_log = A+np.log10(np.sqrt(2/np.pi)) + q*np.log10(np.sqrt(a)*dc/sigma) - a/2/np.log(10)*dc**2/sigma**2 + (alpha)*np.log10(xoff/10**(1.83*mu)) - 1/np.log(10)*(xoff/10**(1.83*mu))**(0.05*alpha) + gamma*np.log10(spin/(10**(mu))) - 1/np.log(10)*(xoff/10**(1.83*mu)/sigma**e)**(beta)*(spin/(10**(mu)))**(delta)   
    return 10**model_log













