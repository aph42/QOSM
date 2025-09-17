import numpy as np
import scipy
H = 7000.

def a_T(p):
   z = -H * np.log(p/100) * 1e-3
   return 0.04 + 0.002*z + 0.0003*z**2

def a_O3(p, a1 = 1/80., a2 = 0.9, a3 = 0.5, a4 = 2.7):
   z = -H * np.log(p/100) * 1e-3
   return a1 * z + (a2 + a3*z)*np.exp(-z/a4) 

def d_T(p, e2 = 0.036):
   z = -H * np.log(p/100) * 1e-3
   return -(e2*z)**8

def d_O3(p, d1 = 0.01, d2 = 0.046):
   z = -H * np.log(p/100) * 1e-3
   return d1 + (d2*z)**6

def d_NOx(p, n2 = 0.00007, n3 = 0.0006):
   z = -H * np.log(p/100) * 1e-3
   return n2*z + n3*z**2

def eps_N2O(z):
    # Reaction N2O + O(1D) -> 2 NO
    # Brasseur and Solomon 5.127a (p 329)
    # Rate coefficient from Appendix 3, p 607
    # no temperature dependence
    k_b39 = 6.7e-11 # cm**3 molec**-1 s**-1

    # linearization of rate of production of NO
    # Reaction 5.128
    # P_NO = 2 k_b39 [N2O] [O(1D)]

    number_density_O1D = init_number_density_O1D(z)

    eps_N2O = 2 * k_b39 * number_density_O1D

    return eps_N2O

def init_number_density_O1D(z):
    # INPUT: z [meters]
    # OUTPUT: number density of O(1D) in molec cm-3

    # from Table A.6.2.c of Brasseur and Solomon (2005)
    z_logp_BS05 = np.arange(0,110001,step=5000) # meters
    O1D_BS05 = np.array([0.,0.,0.,0.,0.,2.9e0,1.3e1,4.0e1,1.0e2,1.8e2,1.9e2,1.5e2,9.8e1,5.2e1,2.4e1,1.3e1,1.5e1,3.3e1,7.0e1,8.9e1,1.4e2,3.2e2,7.8e2])
    # extended downwards with zeros when BS05 has no data to lead to, in effect, no N2O oxidation at 20 km

    interp_O1D_BS05 = scipy.interpolate.interp1d(z_logp_BS05,O1D_BS05,bounds_error=False,fill_value=0.,kind='quadratic')
    number_density_O1D = interp_O1D_BS05(z)
    number_density_O1D = np.clip(number_density_O1D,0,None)

    return number_density_O1D


def init_NOx_0(z):
    # INPUT: z [meters]
    # OUTPUT: NOx_0 [vmr]

    # digitization of WACCM NOx profile from Park et al., 2017
    # "Variability of Stratospheric Reactive Nitrogen and Ozone Related to the QBO"

    H = 7000. # meters

    p_Park17= np.array([385.2,208.4,132.9,97.83,72.76,49.03,32.32,22.26,15.67,11.15,7.519,4.849,2.993,1.868,1.128]) # hPa
    ps_ref = 1000. # hPa
    z_logp_Park17 = H * np.log(ps_ref/p_Park17)

    # volume mixing ratio [ppbv]
    NOx_ppb = np.array([0.0450,0.0768,0.1192,0.2195,0.4366,0.731,1.091,1.899,3.56,6.57,11.21,15.51,17.3,17.0,13.83]) # ppbv

    interp_NOx = scipy.interpolate.interp1d(z_logp_Park17,NOx_ppb,bounds_error=False,kind='quadratic')

    NOx_0 = interp_NOx(z)

    ppb2vmr = 1e-9 # conversion factor
    NOx_0_vmr = np.clip(NOx_0,0,None)*ppb2vmr # clip up to zero, convert from ppb to vmr

    return NOx_0_vmr


def init_dNOx_0_dz(z):
    # INPUT: z [meters]
    # OUTPUT: dNOx_0_dz [vmr m**-1]

    z_coarse = np.linspace(17000,50000,21)
    dz_coarse = z_coarse[1]-z_coarse[0]

    NOx_0_vmr = init_NOx_0(z_coarse)

    dNOx_0_vmr_dz = np.gradient(NOx_0_vmr,dz_coarse)

    interp_dNOx_0_vmr_dz= scipy.interpolate.interp1d(z_coarse,dNOx_0_vmr_dz,bounds_error=False,)

    dNOx_0_vmr_dz = interp_dNOx_0_vmr_dz(z)

    return dNOx_0_vmr_dz
