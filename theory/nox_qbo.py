import numpy as np
import pygeode as pyg
from matplotlib import pyplot as plt
import model

pre = pyg.Pres(10**np.linspace(2, 0, 101))
Z = pyg.Height(np.linspace(20, 60, 101))

H = 7
#z = -H*pyg.log(pre/1000.)
pre = 1000 * pyg.exp(-Z / H)

def gamma_n2o():
# {{{
   lt = 6 + 5 * pyg.exp(-(Z - 20)/6)
   gamma = (10**-lt).rename('gamma_N2O')
   return gamma
# }}}

def upwelling():
# {{{
   # Increase  from .3mm/s at 20 km to .5mm/s at 35 km
   w = 0*Z + 0.0005 - (35 - Z) / (35 - 20) * 0.0002 * (Z < 35.) 
   return w.rename('w')
# }}}

def qbo_upwelling():
# {{{
   # Increase  from .0mm/s at 20 km to .2mm/s at 35 km; back to zero at 50 km
   a0 = 0.
   z0 = 20.

   a1 = 0.0002
   z1 = 35.

   a2 = 0.
   z2 = 40.

   zlmsk = (Z > z0) * (Z < z1)
   zumsk = (Z >= z1) * (Z < z2)

   phs = pyg.NamedAxis(np.linspace(0, 2*np.pi, 101), 'phase')

   amp = (Z - z0) / (z1 - z0) * a1 * zlmsk \
       + (z2 - Z) / (z2 - z1) * a1 * zumsk

   p0 = -(Z - z0) / (z1 - z0) * 2 * np.pi
   #p0 = pyg.clip(p0, 0, 2*np.pi)

   wp = pyg.cos(phs - p0) * amp
   return wp.rename("w'")
# }}}

def plot_gamma():
# {{{
   gamma = gamma_n2o()

   plt.ioff()
   ax = pyg.showvar(np.log10(gamma))
   #ax.setp(xscale = 'log')
   ax.setp_xaxis(major_formatter=plt.FormatStrFormatter(r'10$^{%d}$'))
   ax.setp(xlabel = r'Damping rate (s$^{-1}$)', title = r'$\gamma_{N_2O}$')

   plt.ion()
   ax.render(1)
# }}}

def plot_w():
# {{{
   w = upwelling()
   qw = qbo_upwelling()

   plt.ioff()

   ax = pyg.showlines([1e3*w, 1e3*(w + qw)(s_phase = 0), 1e3*(w + qw)(s_phase = np.pi)], labels = ['Background', 'QBO 1', 'QBO 2'])
   ax.setp(xlabel = 'mm/s', title = r'w$^\asterisk$')

   plt.ion()
   ax.render(2)
# }}}

def plot_LN2O():
# {{{
   # Predict vertical structure in N2O
   gamma = gamma_n2o()
   w = upwelling()
   qw= qbo_upwelling()

   def get_N2O(w, g):
      L = w/g
      T = (1/L).integrate('z', dx = Z*1e3)

      N = pyg.exp(-T)
      return L, T, N

   Lb, Tb, Nb = get_N2O(w, gamma)
   L1, T1, N1 = get_N2O(w + qw(s_phase = 0),     gamma)
   L2, T2, N2 = get_N2O(w + qw(s_phase = np.pi), gamma)

   plt.ioff()

   ax0 = pyg.showlines([L*1e-3 for L in [Lb, L1, L2]], labels = ['bkg', 'qbo1', 'qbo2'], size=(4.1, 3.2))
   ax0.setp(xlabel = 'km', title = r'L$_{N_2O}$', xscale='log', ylim = (20, 40))
   ax0.setp_xaxis(major_locator = plt.LogLocator())

   ax1 = pyg.showlines([T for T in [Tb, T1, T2]], labels = ['bkg', 'qbo1', 'qbo2'], size=(4.1, 3.2))
   ax1.setp(xlabel = 'decay factor', title = r'T$_{N_2O}$', ylim = (20, 40), xlim = (-1, 5))
   #ax1.setp_xaxis(major_locator = plt.LogLocator())

   ax2 = pyg.showlines([N for N in [Nb, N1, N2]], labels = ['bkg', 'qbo1', 'qbo2'], size=(4.1, 3.2))
   ax2.setp(xlabel = r'N$_2$O', title = r'N$_2$O', ylim = (20, 40))

   ax = pyg.plot.grid([[ax0, ax1, ax2]])

   plt.ion()
   ax.render(3)
# }}}

def validate_model():
# {{{
   plim = (150., 4.8)

   zb = -H * np.log(plim[0] / 1000.)
   zt = -H * np.log(plim[1] / 1000.)

   St = mdl.BaseState(zb, zt, Nz = 301)

   st_pres = pyg.Pres(St.ps)
   st_zs = pyg.Height(St.zs)

   St.w0 = 0.

   St.T0 = 1.
   St.O30 = 1.
   St.O30 = 1.
# }}}
