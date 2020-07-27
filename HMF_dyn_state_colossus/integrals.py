import numpy as np
from scipy import integrate
'''
Spin and Xoff cover order of magnitudes.
Therefore, integrals are done in log10 space, to be more precise. 
'''

def int_h_dsigma(sigma,xoff,spin,h):
    '''
    Performs integral of h(sigma,xoff,spin) dsigma

	Parameters
	-----------------------------------------------------------------------------------------------
	sigma: array_like
		Variance; can be a number or a meshgrid with xoff and spin.
    xoff: array_like
        Offset parameter; can be a number or a meshgrid with sigma and spin.
    spin: array_like
        Spin parameter; can be a number or a meshgrid with sigma and xoff.
    h: array_like
		The halo mass-xoff-spin function :math:`h(\\sigma,xoff,spin)`, has the same dimensions as the input meshgrid.
		
	Returns
	-----------------------------------------------------------------------------------------------
	g_xoff_spin: array_like
        The halo xoff-spin function :math:`g(xoff,spin)`, has the same dimension of the meshgrid on the xoff-spin plane
    '''
    g_xoff_spin = np.zeros((len(xoff),len(spin)))
    for i in range(len(xoff)):
        for j in range(len(spin)):
            g_xoff_spin[i,j] = integrate.simps(h[:,i,j],sigma)
    return g_xoff_spin

def int_h_dxoff(sigma,xoff,spin,h):
    '''
    Performs integral of h(sigma,xoff,spin) dxoff

	Parameters
	-----------------------------------------------------------------------------------------------
	sigma: array_like
		Variance; can be a number or a meshgrid with xoff and spin.
    xoff: array_like
        Offset parameter; can be a number or a meshgrid with sigma and spin.
    spin: array_like
        Spin parameter; can be a number or a meshgrid with sigma and xoff.
    h: array_like
		The halo mass-xoff-spin function :math:`h(\\sigma,xoff,spin)`, has the same dimensions as the input meshgrid.
		
	Returns
	-----------------------------------------------------------------------------------------------
	g_sigma_spin: array_like
        The halo mass-spin function :math:`g(\\sigma,spin)`, has the same dimension of the meshgrid on the sigma-spin plane
    '''
    g_sigma_spin = np.zeros((len(sigma),len(spin)))    
    for i in range(len(sigma)):
        for j in range(len(spin)):
            g_sigma_spin[i,j] = integrate.simps(h[i,:,j],np.log10(xoff))
    return g_sigma_spin

def int_h_dspin(sigma,xoff,spin,h):
    '''
    Performs integral of h(sigma,xoff,spin) dspin

	Parameters
	-----------------------------------------------------------------------------------------------
	sigma: array_like
		Variance; can be a number or a meshgrid with xoff and spin.
    xoff: array_like
        Offset parameter; can be a number or a meshgrid with sigma and spin.
    spin: array_like
        Spin parameter; can be a number or a meshgrid with sigma and xoff.
    h: array_like
		The halo mass-xoff-spin function :math:`h(\\sigma,xoff,spin)`, has the same dimensions as the input meshgrid.
		
	Returns
	-----------------------------------------------------------------------------------------------
	g_sigma_xoff: array_like
        The halo mass-xoff function :math:`g(\\sigma,xoff)`, has the same dimension of the meshgrid on the sigma-xoff plane
    '''
    g_sigma_xoff = np.zeros((len(sigma),len(xoff)))
    for i in range(len(sigma)):
        for j in range(len(xoff)):
            g_sigma_xoff[i,j] = integrate.simps(h[i,j,:],np.log10(spin))
    return g_sigma_xoff
        
def int_f_sigma(sigma,xoff,g):
    '''
    Performs integral of g(sigma,xoff) dxoff

	Parameters
	-----------------------------------------------------------------------------------------------
	sigma: array_like
		Variance; can be a number or a meshgrid with xoff and spin.
    xoff: array_like
        Offset parameter; can be a number or a meshgrid with sigma and spin.
    g: array_like
		The halo mass-xoff function :math:`g(\\sigma,xoff)`, has the same dimensions as the input meshgrid.
		
	Returns
	-----------------------------------------------------------------------------------------------
	f_sigma: array_like
        The halo mass function :math:`f(\\sigma)`, has the same dimension of sigma
    '''
    f_sigma = np.zeros(len(sigma))
    for i in range(len(sigma)):
        f_sigma[i] = integrate.simps(g[i,:],np.log10(xoff))
    return f_sigma

def int_f_xoff(sigma,xoff,g):
    '''
    Performs integral of g(sigma,xoff) dsigma

	Parameters
	-----------------------------------------------------------------------------------------------
	sigma: array_like
		Variance; can be a number or a meshgrid with xoff and spin.
    xoff: array_like
        Offset parameter; can be a number or a meshgrid with sigma and spin.
    g: array_like
		The halo mass-xoff function :math:`g(\\sigma,xoff)`, has the same dimensions as the input meshgrid.
		
	Returns
	-----------------------------------------------------------------------------------------------
	f_xoff: array_like
        The halo xoff function :math:`f(xoff)`, has the same dimension of xoff
    '''
    f_xoff = np.zeros(len(xoff))
    for i in range(len(xoff)):
        f_xoff[i] = integrate.simps(g[:,i],sigma)
    return f_xoff

def int_f_spin(sigma,spin,g):
    '''
    Performs integral of g(sigma,spin) dsigma

	Parameters
	-----------------------------------------------------------------------------------------------
	sigma: array_like
		Variance; can be a number or a meshgrid with xoff and spin.
    spin: array_like
        Spin parameter; can be a number or a meshgrid with sigma and xoff.
    g: array_like
		The halo mass-spin function :math:`g(\\sigma,spin)`, has the same dimensions as the input meshgrid.
		
	Returns
	-----------------------------------------------------------------------------------------------
	f_spin: array_like
        The halo spin function :math:`f(spin)`, has the same dimension of spin
    '''
    f_spin = np.zeros(len(spin))
    for i in range(len(sigma)):
        f_spin[i] = integrate.simps(g[:,i],sigma)
    return f_spin
