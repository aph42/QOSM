import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

H = 7000.   # Scale height for log-pressure coordinates
cp = 1004.   # J / K kg specific heat at constant pressure
R = 287.    # J / K kg specific gas constant for dry air


class BaseState():
   '''  A simple PDE solver for coupled set of advection/reaction equations
   for the evolution of QBO temperature, ozone, N2O and NOx anomalies given
   a profile of background upwelling and anomalous QBO-related upwelling.
   '''
    def __init__(self, zb, zt, Nz = 800):
        # Grid parameters
        self.__dict__['zb'] = zb
        self.__dict__['zt'] = zt
        self.__dict__['Lz'] = zt - zb
        self.__dict__['Nz'] = Nz

        self.initialize()

        # Upwelling (background, perterbation)
        self.__dict__['w0'] = 0.0003 + 0.0 * self.zs
        self.__dict__['wp'] = 0.0001 + 0j + 0.0 * self.zs

        self.__dict__['S0']     = 0.012  + 0.0 * self.zs
        self.__dict__['dO3dz']  = 0.0005 + 0.0 * self.zs
        self.__dict__['dN2Odz'] = 0.0005 + 0.0 * self.zs
        self.__dict__['dNOxdz'] = 0.0005 + 0.0 * self.zs

        self.__dict__['omega']   = 2 * np.pi / 840. / 86400.

        # Temperature coeffs
        self.__dict__['aT']   = 0.1 / 86400  + 0.0 * self.zs
        self.__dict__['aO3']  = 0.3 / 86400  + 0.0 * self.zs

        # Ozone coeffs
        self.__dict__['dT']   = 0.01 / 86400 + 0.0 * self.zs
        self.__dict__['dO3']  = 0.01 / 86400 + 0.0 * self.zs
        self.__dict__['dNOx'] = 0.01 / 86400 + 0.0 * self.zs

        # N2O coeffs
        self.__dict__['eN2O']  = 0.01 / 86400 + 0.0 * self.zs

        # NOx coeffs
        self.__dict__['nN2O']  = 0.01 / 86400 + 0.0 * self.zs

        self.__dict__['T0'  ] = 0.
        self.__dict__['O30' ] = 0.
        self.__dict__['N2O0'] = 0.
        self.__dict__['NOx0'] = 0.

    def __setattr__(self, name, value):
        if name in self.__dict__.keys(): self.__dict__[name] = value
        else: raise ValueError('%s has no attribute "%s".' % (self, name))
        self.initialize()

    def initialize(self):
        self.__dict__['zs'] = np.linspace(self.zb, self.zt, self.Nz + 1)
        self.__dict__['dz'] = self.Lz / (self.Nz + 1)

        ps = 1000 * np.exp(-self.zs / H)
        self.__dict__['ps'] = ps

    def solve(self):
        def Dz(c): return sparse.diags([-c[1:], c], [-1, 0], shape=(self.Nz, self.Nz), dtype = np.complex64, format = 'csr')
        def C(c):  return sparse.diags([c], [0], shape=(self.Nz, self.Nz), dtype = np.complex64, format = 'csr')

        dz = self.dz

        T0 = self.T0
        X0 = self.X0

        tz = self.w0t
        tc = 1j * self.om + self.al + self.w0t * R / (cp * H)
        gm = -self.gm

        ft = -self.S0 * self.wp

        ep = -self.ep
        xz = self.w0x
        xc = 1j * self.om + self.dl

        fx = -self.dX0dz * self.wp - self.nu * self.NOxp

        Ltt = Dz(tz[1:]) + C(tc[1:] * dz)
        Ltx = C(gm[1:] * dz)
        Lxt = C(ep[1:] * dz)
        Lxx = Dz(xz[1:]) + C(xc[1:] * dz)

        L = sparse.bmat([[Ltt, Ltx], [Lxt, Lxx]], format='csr', dtype = np.complex64)

        Ft = ft[1:] * dz
        Ft[0] += tz[0] * T0

        Fx = fx[1:] * dz
        Fx[0] += xz[0] * X0

        F = np.concatenate([Ft, Fx])

        sln = spsolve(L, F)

        T = np.concatenate([[T0], sln[:self.Nz]])
        X = np.concatenate([[X0], sln[self.Nz:]])

        return T, X
