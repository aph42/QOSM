import numpy as np
import pygeode as pyg
from matplotlib import pyplot as plt
import model as mdl

from an_fits_ref import *

pre = pyg.Pres(10**np.linspace(2, 0, 101))
Z = pyg.Height(np.linspace(20e3, 60e3, 101))

H = 7000.
#z = -H*pyg.log(pre/1000.)
pre = 1000 * pyg.exp(-Z / H)

datapath = '../data/'

def open_pop_file(run = None):
# {{{ 
   if run == None: run = 'ref'

   fn = 'era5_pops_regressed_onto_u_pop_4_200hPa_5NS.nc'

   dpops = pyg.open(datapath + pth)

   #time = pyg.NamedAxis(dpops.phase[:] * 840. / (2 * np.pi), 'time')
   #time = pyg.modeltime360n(interval[0], n = len(dpops.phase), step = np.diff(dpops.phase[:])[0] * 840. / (2 * np.pi))
   time = pyg.ModelTime360(np.linspace(interval[0], interval[1], len(dpops.phase)), 
                           startdate = dict(year=2000, month=1, day=1), 
                           units='days')
   dsp = dpops.replace_axes(phase = time).squeeze()

   # Convert ozone from mass mixing ratio to volume mixing ratio
   o3 = dsp.o3_pop1 / mo3 * ma
   dsp = dsp.replace_vars(o3_pop1 = o3.rename('o3_pop1'))

   z_axis = -7 * pyg.log(dsp.pres / 1000.)

   if run in ['ref', 'hlf', 'dbl']:
      msk = (1-0.2*(pyg.tanh((z_axis-20))-pyg.tanh((z_axis-28))))
   elif run in ['dwr']:
      msk = (1-0.2*(pyg.tanh((z_axis-20))+1)) * (-0.5*pyg.tanh(5*(z_axis-25.7)) + 0.5)
   else:
      msk = None

   if msk is not None:
      w = (dsp.resw_pop1 * msk).rename('resw_pop1')
      dsp = dsp.replace_vars(resw_pop1 = w.rename('resw_pop1'))

   return dsp

# }}}

def open_rce_file(run = None):
# {{{
   if run == None: run = 'ref'

   pth = datapath + rce_fns[run]
   ds = pyg.openall(pth)

   time_axis = pyg.ModelTime360(values=ds.time[:] - 152., units = 'days', startdate=dict(year=2000, month=1, day=1))
   pres_axis = pyg.Pres(ds.lyP(i_time=0).squeeze()[:]/1e2)

   return ds.replace_axes(time = time_axis, ly = pres_axis)
# }}}

def get_S_dChidz(run = None, recalc = False):
# {{{
   if run == None: run = 'ref'

   Sfn = datapath + 'SdO3dz_%s.nc' % run
   #if recalc or not os.path.exists(Sfn):
      #save_S_dChidz(run)

   ds = pyg.open(Sfn)
   return ds.Sm, ds.Sd, ds.Om, ds.Od, ds.rat, ds.inv, ds.reg, ds.ireg
# }}}

def gamma_n2o(zs):
# {{{
   lt = 6 + 5 * pyg.exp(-(zs - 20e3)/6e3)
   gamma = (10**-lt).rename('gamma_N2O')
   return gamma
# }}}

def upwelling():
# {{{
   # Increase  from .3mm/s at 20 km to .5mm/s at 35 km
   w = 0*Z + 0.0005 - (35 - Z) / (35 - 20) * 0.0002 * (Z < 35.) 
   return w.rename('w')
# }}}

def qbo_upwelling(zs):
# {{{
   # Increase  from .0mm/s at 20 km to .2mm/s at 35 km; back to zero at 50 km
   a0 = 0.
   z0 = 20e3

   a1 = 0.0002
   z1 = 35e3

   a2 = 0.
   z2 = 40e3

   zlmsk = (zs > z0) * (zs < z1)
   zumsk = (zs >= z1) * (zs < z2)

   phs = pyg.NamedAxis(np.linspace(0, 2*np.pi, 101), 'phase')

   amp = (zs - z0) / (z1 - z0) * a1 * zlmsk \
       + (z2 - zs) / (z2 - z1) * a1 * zumsk

   p0 = -(zs - z0) / (z1 - z0) * 2 * np.pi
   #p0 = pyg.clip(p0, 0, 2*np.pi)

   wp = pyg.cos(phs - p0) * amp
   return wp.rename("w'")
# }}}

def plot_gamma():
# {{{
   gamma = gamma_n2o(Z)

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
   gamma = gamma_n2o(Z)
   w = upwelling()
   qw= qbo_upwelling(Z)

   def get_N2O(w, g):
      L = w/g
      T = (1/L).integrate('z', dx = Z)

      N = 270*pyg.exp(-T)
      return L, T, N

   Lb, Tb, Nb = get_N2O(w, gamma)
   L1, T1, N1 = get_N2O(w + qw(s_phase = 0),     gamma)
   L2, T2, N2 = get_N2O(w + qw(s_phase = np.pi), gamma)

   plt.ioff()

   ax0 = pyg.showlines([L*1e-3 for L in [Lb, L1, L2]], labels = ['bkg', 'qbo1', 'qbo2'], size=(4.1, 3.2))
   ax0.setp(xlabel = 'km', title = r'L$_{N_2O}$', xscale='log', ylim = (20e3, 40e3))
   ax0.setp_xaxis(major_locator = plt.LogLocator())

   ax1 = pyg.showlines([T for T in [Tb, T1, T2]], labels = ['bkg', 'qbo1', 'qbo2'], size=(4.1, 3.2))
   ax1.setp(xlabel = 'decay factor', title = r'T$_{N_2O}$', ylim = (20e3, 40e3), xlim = (-1, 5))
   #ax1.setp_xaxis(major_locator = plt.LogLocator())

   ax2 = pyg.showlines([N for N in [Nb, N1, N2]], labels = ['bkg', 'qbo1', 'qbo2'], size=(4.1, 3.2))
   ax2.setp(xlabel = r'N$_2$O', title = r'N$_2$O', ylim = (20e3, 40e3))

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

   St.wp[:] = 0.
   St.w0[:] = 0.00001

   St.T0   = 1.
   St.O30  = 2.
   St.N2O0 = 3.
   St.NOx0 = 4.

   # Turn off any interactions
   St.aT[:]    = 0.01 / 86400.
   St.aO3[:]   = 0.
   St.dT[:]    = 0.
   St.dO3[:]   = 0.02 / 86400.
   St.dNOx[:]  = 0.
   St.gN2O[:]  = 0.03 / 86400. 
   St.gNOx[:]  = 0.
   St.eN2O[:]  = 0.
   St.eNOx[:]  = 0.04 / 86400. 

   T, O3, N2O, NOx = St.solve()

   T   = pyg.Var((st_zs, ), name = 'T',   values = T)
   O3  = pyg.Var((st_zs, ), name = 'O3',  values = O3)
   N2O = pyg.Var((st_zs, ), name = 'N2O', values = N2O)
   NOx = pyg.Var((st_zs, ), name = 'NOx', values = NOx)

   Ta    = St.T0   * pyg.exp(-((1j * St.omega + St.aT[0])   / St.w0[0]) * (st_zs - zb))
   O3a   = St.O30  * pyg.exp(-((1j * St.omega + St.dO3[0])  / St.w0[0]) * (st_zs - zb))
   N2Oa  = St.N2O0 * pyg.exp(-((1j * St.omega + St.gN2O[0]) / St.w0[0]) * (st_zs - zb))
   NOxa  = St.NOx0 * pyg.exp(-((1j * St.omega + St.eNOx[0]) / St.w0[0]) * (st_zs - zb))

   plt.ioff()
   axs = []
   for i, (v, va) in enumerate(zip([T, O3, N2O, NOx], [Ta, O3a, N2Oa, NOxa])):
      ax = pyg.plot.AxesWrapper(size=(2.5, 3))
      pyg.vplot(va.real(), c = 'k',     ls = '-',  lw = 2., axes = ax)
      pyg.vplot(va.imag(), c = 'k',     ls = '--', lw = 2., axes = ax)
      pyg.vplot(v.real(),  c = f'C{i}', ls = '-',  lw = 2., axes = ax)
      pyg.vplot(v.imag(),  c = f'C{i}', ls = '--', lw = 2., axes = ax)
      ax.axvline(x = 0, c = 'k', lw = 1.)
      axs.append(ax)

   ax = pyg.plot.grid([axs])
   plt.ion()

   ax.render(1)
# }}}

def nox_qbo():
# {{{
   #dsp = open_pop_file(run)
   #daW = fit_amp_phase(dsp.resw_pop1*1e3)
   #Wp = to_complex(daW, 'W') / 1e3
   #Sm, Sd, Om, Od, rat, inv, reg, ireg = get_S_dChidz(run)

   plim = (150., 4.8)

   zb = -H * np.log(plim[0] / 1000.)
   zt = -H * np.log(plim[1] / 1000.)

   St = mdl.BaseState(zb, zt, Nz = 301)

   st_pres = pyg.Pres(St.ps)
   st_zs = pyg.Height(St.zs)

   #St.wp[:] = 0.
   St.w0[:] = 0.0003

   #St.S0[:]    = Sm.interpolate('pres', st_pres)[:]
   #St.dO3dz[:] = Om.interpolate('pres', st_pres)[:]

   St.S0[:]    = 12e-3
   St.dO3dz[:] = 5e-4
   St.dNOxdz[:] = init_dNOx_0_dz(St.zs)

   St.T0   = 0.
   St.O30  = 0.
   St.N2O0 = 0.
   St.NOx0 = 0.

   St.aT[:]    = a_T (St.ps)  / 86400.
   St.aO3[:]   = a_O3(St.ps)  / 86400.
   St.dT[:]    = d_T (St.ps)  / 86400.
   St.dO3[:]   = d_O3(St.ps)  / 86400.
   St.dNOx[:]  = d_NOx(St.ps) / 86400.
   St.gN2O[:]  = gamma_n2o(st_zs)[:]
   St.gNOx[:]  = 0.
   St.eN2O[:]  = 0.
   St.eNOx[:]  = eps_N2O(St.zs)

   T, O3, N2O, NOx = St.solve()

   T   = pyg.Var((st_zs, ), name = 'T',   values = T)
   O3  = pyg.Var((st_zs, ), name = 'O3',  values = O3)
   N2O = pyg.Var((st_zs, ), name = 'N2O', values = N2O)
   NOx = pyg.Var((st_zs, ), name = 'NOx', values = NOx)

   plt.ioff()
   axs = []
   for i, v in enumerate([T, O3, N2O, NOx]):
      ax = pyg.plot.AxesWrapper(size=(2.5, 3))
      pyg.vplot(v.real(),  c = f'C{i}', ls = '-',  lw = 2., axes = ax)
      pyg.vplot(v.imag(),  c = f'C{i}', ls = '--', lw = 2., axes = ax)
      ax.axvline(x = 0, c = 'k', lw = 1.)
      axs.append(ax)

   ax = pyg.plot.grid([axs])
   plt.ion()

   ax.render(2)
# }}}
