import numpy as np
import pygeode as pyg
from matplotlib import pyplot as plt
import model as mdl

from an_fits_ref import *

pre = pyg.Pres(10**np.linspace(2, 0, 101))
Z = pyg.Height(np.linspace(12e3, 40e3, 101))

H = 7000.
#z = -H*pyg.log(pre/1000.)
pre = 1000 * pyg.exp(-Z / H)

datapath = '../data/'

def format_rad(den=2):
# {{{
   def fmt(val, pos = None, den=den):
      num = int(np.round(val * den))
      gcd = np.gcd(den, num)

      num = num // gcd
      den = den // gcd
      if num == 0:
         return r'0'
      elif den > 1:
         if num == 1:
            return r'$\frac{\pi}{%d}$' % den
         elif num == -1:
            return r'$\frac{-\pi}{%d}$' % den
         else:
            return r'$\frac{%d \pi}{%d}$' % (num, den)
      else:
         if num == 1:
            return r'$\pi$'
         elif num == -1:
            return r'$-\pi$'
         else:
            return r'$%d \pi$' % num
   return fmt
# }}}

def set_rad_axis(ax, den = 2):
   ax.setp_xaxis(major_locator = plt.MultipleLocator(1 / den), major_formatter = plt.FuncFormatter(format_rad(den)))

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

def upwelling(zs):
# {{{
   # Increase  from .3mm/s at 20 km to .5mm/s at 35 km
   w = 0*zs + 0.0005 - (35 - zs*1e-3) / (35 - 20) * 0.0002 * (zs*1e-3 < 35.) 
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

def fit_amp_phase(v, plot = False, off = 0.):
# {{{
   if v.hasaxis('time'):
      phs = (2 * np.pi * (v.time - interval[0]) / 840.)
      ax = 'time'
   else:
      phs = v.phase
      ax = 'phase'

   ca = (2*pyg.cos(phs) * v).mean(ax)
   sa = (2*pyg.sin(phs) * v).mean(ax)

   amp = pyg.sqrt(ca**2 + sa**2)

   phase = pyg.arctan2(sa, ca)
   df = (np.pi + phase.diff()) % (2 * np.pi) - np.pi
   p0 = phase.slice[:1]
   p1 = df.cumsum(0, v0 = phs[1]).replace_axes(z=v.z.slice[1:])

   phs_c = pyg.concatenate([p0, p1])

   p30 = phs_c(z = 30e3)[0]
   off -= p30 - (p30 % (2 * np.pi))
   phs_c += off

   ds = pyg.asdataset([amp.rename('amp'), 
                       phs_c.rename('phase'),
                       ca.rename('re'),
                       sa.rename('im')])

   if plot:
      w = to_timeseries(ds, phs, 'w')
      pyg.showgrid([v, w], ncol=1, fig=0)

   return ds
# }}}

def to_timeseries(ds, phase, name = 'v'):
   w = ds.amp * np.cos(phase - ds.phase)
   return w.rename(name)

def to_complex(ds, name = 'v'):
# {{{
   v = ds.amp * pyg.exp(1j * ds.phase)
   return v.rename(name)
# }}}

def to_amp_phase(v, off = 0.):
# {{{
   amp = pyg.absolute(v).rename('amp')
   phs = pyg.angle(v).rename('phase')
   df = (np.pi + phs.diff()) % (2 * np.pi) - np.pi
   p0 = phs.slice[:1]
   p1 = df.cumsum(0, v0 = phs[1]).replace_axes(pres=phs.pres.slice[1:])

   phs_c = pyg.concatenate([p0, p1])

   p30 = phs_c(pres = 80)[0]
   off -= p30 - (p30 % (2 * np.pi))
   phs_c += off

   re  = pyg.real(v).rename('re')
   im  = pyg.imag(v).rename('im')

   return pyg.asdataset([amp, phs_c.rename('phase'), re, im])
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
   w = upwelling(Z)
   qw = qbo_upwelling(Z)

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
   w = upwelling(Z)
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
   ax0.setp(xlabel = 'km', title = r'L$_{N_2O}$', xscale='log', ylim = (20e3, 40e3), xlim = (0.5, 10000.))
   ax0.setp_xaxis(major_locator = plt.LogLocator())

   ax1 = pyg.showlines([T for T in [Tb, T1, T2]], labels = ['bkg', 'qbo1', 'qbo2'], size=(4.1, 3.2))
   ax1.setp(xlabel = 'decay scales', title = r'T$_{N_2O}$', ylim = (20e3, 40e3), xlim = (-1, 5))
   #ax1.setp_xaxis(major_locator = plt.LogLocator())

   ax2 = pyg.showlines([N for N in [Nb, N1, N2]], labels = ['bkg', 'qbo1', 'qbo2'], size=(4.1, 3.2))
   ax2.setp(xlabel = r'ppbv', title = r'N$_2$O', ylim = (20e3, 40e3), xscale = 'log')
   ax2.setp_xaxis(major_locator = plt.LogLocator())

   ax = pyg.plot.grid([[ax0, ax1, ax2]])

   plt.ion()
   ax.render(3)
# }}}

def validate_model():
# {{{
   plim = (150., 4.8)

   zb = -H * np.log(plim[0] / 1000.)
   zt = -H * np.log(plim[1] / 1000.)

   St = mdl.BaseState(zb, zt, Nz = 801)

   st_pres = pyg.Pres(St.ps)
   st_zs = pyg.Height(St.zs)

   St.wp[:] = 0.
   St.w0[:] = 0.0003

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

def run_nox_qbo():
# {{{
   #dsp = open_pop_file(run)
   #daW = fit_amp_phase(dsp.resw_pop1*1e3)
   #Wp = to_complex(daW, 'W') / 1e3
   #Sm, Sd, Om, Od, rat, inv, reg, ireg = get_S_dChidz(run)

   plim = (150., 4.8)

   zb = -H * np.log(plim[0] / 1000.)
   zt = -H * np.log(plim[1] / 1000.)

   St = mdl.BaseState(zb, zt, Nz = 81)

   st_pres = pyg.Pres(St.ps)
   st_zs = pyg.Height(St.zs)
   st_zs.plotatts['scalefactor'] = 1e-3
   st_zs.units = 'km'

   #St.wp[:] = 0.0001
   St.wp[:] = to_complex(fit_amp_phase(qbo_upwelling(st_zs)))[:]
   #St.w0[:] = 0.0003
   St.w0[:] = upwelling(st_zs)[:]

   #St.S0[:]    = Sm.interpolate('pres', st_pres)[:]
   #St.dO3dz[:] = Om.interpolate('pres', st_pres)[:]

   St.S0[:]    = 12e-3
   St.dO3dz[:] = 5e-4
   #St.dN2Odz[:] = 1e-11
   St.dN2Odz[:] = init_dN2O_0_dz(St.zs)
   #St.dNOxdz[:] = 1e-12
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
   St.eN2O[:]  = eps_N2O(St.zs)
   St.eNOx[:]  = 0.

   T, O3, N2O, NOx = St.solve()

   T   = pyg.Var((st_pres, ), name = 'T',   values = T)
   O3  = pyg.Var((st_pres, ), name = 'O3',  values = O3)
   N2O = pyg.Var((st_pres, ), name = 'N2O', values = N2O)
   NOx = pyg.Var((st_pres, ), name = 'NOx', values = NOx)

   return St, pyg.asdataset([T, O3, N2O, NOx])
# }}}

def plot_nox_coefs(fig = 4):
# {{{
   St, ds = run_nox_qbo()

   st_pres = pyg.Pres(St.ps)
   dN2Odz = pyg.Var((st_pres, ), name = 'dN2Odz', values = St.dN2Odz)
   dNOxdz = pyg.Var((st_pres, ), name = 'dNOxdz', values = St.dNOxdz)
   gN2O   = pyg.Var((st_pres, ), name = 'gN2O',   values = St.gN2O)
   eN2O   = pyg.Var((st_pres, ), name = 'eN2O',   values = St.eN2O)

   plt.ioff()

   size = (2.8, 3)
   ylims = (110, 4.8)

   axn = pyg.showvar(dN2Odz, size = size)
   axn.setp(ylim = ylims, title = r'$\partial_z$ N$_2$O', xlabel = 'vmr m$^{-1}$')

   axx = pyg.showvar(dNOxdz, size = size)
   axx.setp(ylim = ylims, title = r'$\partial_z$ NO$_x$', xlabel = 'vmr m$^{-1}$')

   axg = pyg.showvar(gN2O, size = size)
   axg.setp(ylim = ylims, title = r'$\gamma_{N_2O}$', xlabel = 's$^{-1}$')

   axe = pyg.showvar(eN2O, size = size)
   axe.setp(ylim = ylims, title = r'$\epsilon_{N_2O}$', xlabel = 'vmr NO$_x$ (vmr N$_2$O s)$^{-1}$')

   axs = pyg.plot.grid([[axn, axx], [axg, axe]])

   plt.ion()
   axs.render(fig)
# }}}

def plot_nox_qbo(fig = 3):
# {{{
   St, ds = run_nox_qbo()

   dsn = to_amp_phase(ds.N2O)
   dsx = to_amp_phase(ds.NOx)

   def phs(d, offset): return (d.phase - offset) / np.pi

   plt.ioff()

   def make_pair(ds, var, unit, c):
      axa = pyg.plot.AxesWrapper(size=(2.5, 3))
      pyg.vplot(ds.amp(),  c = c, ls = '-',  lw = 2., axes = axa)
      axa.setp(title = f'{var}: Amplitude', xlabel = unit)

      axp = pyg.plot.AxesWrapper(size=(2.5, 3))
      pyg.vplot(phs(ds, 0),  c = c, ls = '-', lw = 2., axes = axp)
      axp.setp(title = f'{var}: Phase')
      set_rad_axis(axp)

      return [axa, axp]

   axs = []
   axs.append(make_pair(dsn, r'N$_2$O', 'vmr', 'C0'))
   axs.append(make_pair(dsx, r'NO$_x$', 'vmr', 'C1'))
   
   ax = pyg.plot.grid(axs)
   plt.ion()

   ax.render(fig)
# }}}
