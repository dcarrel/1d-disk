import numpy as np
from grid import *
from eos import *
from opacity import *
from scipy.special import gamma

class ShastaVariable:
    def __init__(self, grid, data, vf, D):
        self.grid = grid
        self.data = data
        self.vf = vf
        self.D = D

class FullVariable:
    def __init__(self, params):
        self.eos = load_table(params.EOS_TABLE)
        self.params=params
        self.grid = Grid(params=params)

    ## sets up mass distribution
        def am_distribution(x, f):
            mu = np.log(self.params.LC / f - self.params.LMIN)
            sigma2 = 2 * np.log((self.params.LC - self.params.LMIN) / (self.params.LC / f - self.params.LMIN))
            return 1 / (x - self.params.LMIN) / np.sqrt(sigma2 * 2 * np.pi) * np.exp(-(np.log(x - self.params.LMIN) - mu) ** 2 / 2 / sigma2)

        def spec_ang(x):
            return np.sqrt(CONST_G * self.params.MBH * x ** 2 / (x - self.params.RSCH))

        def dl_dr(x):
            return np.sqrt(CONST_G * self.params.MBH / (x - self.params.RSCH)) - 0.5 * x * np.sqrt(CONST_G * self.params.MBH / (x - self.params.RSCH) ** 3)

        def mass_distribution(x, f):
            am_x = spec_ang(x)
            am_dist = am_distribution(am_x, f)
            result = am_dist * dl_dr(x) / 2 / np.pi / x
            result = np.where(np.logical_or(np.isnan(result), np.isinf(result)), 0, result)
            result = np.where(result < 0, 0, result)
            return result

        self.mass_distribution = mass_distribution(self.grid.r_cell, self.params.SIGMAF)
        cutoff = 1#np.maximum(1, np.exp(-(self.grid.r_cell/5/self.params.RC)**2))
        self.mass_distribution *= cutoff

    def update_variables(self, sigma, ts, t=0):
        self.sigma = sigma
        below_floor = self.sigma < self.params.SIGMA_FLOOR

        self.ts = ts
        self.s = np.ones(self.ts.shape)
        self.s[1:-1] = self.ts[1:-1]/self.sigma[1:-1]
        self.s[0] = self.s[1]
        self.s[-1] = self.s[-2]

        self.s[self.s<self.params.ENTROPY_ATOL] = self.params.ENTROPY_ATOL

        #self.s = np.where(self.s<ENTROPY_MIN, ENTROPY_MIN, self.s)
        self.sigma[below_floor] = self.params.SIGMA_FLOOR
        self.sigma[0] = self.sigma[1]
        self.sigma[-1] = self.sigma[-2]
        self.ts = self.s*self.sigma
        self.t = t
        self.chi = self.sigma*self.grid.omgko*np.sqrt((self.grid.r_cell - self.params.RSCH)/self.grid.r_cell)
        
        self.T = self.eos(self.chi, self.s)

        self.rho = entropy_difference(self.T, self.chi, self.s, just_density=True)
        self.U = RADA*self.T**4 + 1.5*self.rho*KB*self.T/mu
        self.P = RADA*self.T**4/3 + self.rho*KB*self.T/mu

        self.be = -1 + 2*(self.U + self.P)/self.rho/self.grid.vk2o

        self.H = self.sigma/2/self.rho
        self.h = self.H/self.grid.r_cell

        if self.params.CONST_NU:
            self.nu = self.params.CONST_NU*self.grid.cell_ones()
        else:
            self.nu = self.params.ALPHA*self.H**2*self.grid.omgko

        self.sigv = sig((self.be-self.params.BE_CRIT)/self.params.DBE)
        ## calculates velocities at inter_faces
        lc_sigma = 2 * np.pi * self.sigma * self.grid.r_cell
        g = np.sqrt(self.grid.r_cell) / (self.nu+1e-20)
        d = 3*self.nu

        lc_sigma_tild, g_tild, d_tild = [], [], []
        if self.params.INTERP.__eq__("LINEAR"):
            lc_sigma_tild = np.interp(self.grid.r_face, self.grid.r_cell, lc_sigma)
            g_tild = np.interp(self.grid.r_face, self.grid.r_cell, g)
            d_tild = np.interp(self.grid.r_face, self.grid.r_cell, d)
        elif self.params.INTERP.__eq__("LOGARITHMIC"):
            lc_sigma_tild = np.interp(np.log10(self.grid.r_face), np.log10(self.grid.r_cell), lc_sigma)
            g_tild = np.interp(np.log10(self.grid.r_face), np.log10(self.grid.r_cell), g)
            d_tild = np.interp(np.log10(self.grid.r_face), np.log10(self.grid.r_cell), d)


        ## defined at the cell centers

        self.vr = -d_tild * g_tild / lc_sigma_tild * (lc_sigma[1:] / g[1:] - lc_sigma[:-1] / g[:-1])/self.grid.ddr
        #self.vr[0] = np.minimum(0, self.vr[0])
        #self.vr[-1] = np.maximum(0, self.vr[-1])
        #self.vr[0] = self.vr[1]


        ## calculate source terms for density
        sigma_wl = self.sigma*self.grid.omgko*self.sigv ## wind loss
        if not self.params.WIND_ON: sigma_wl *= 0

        ## uses angular momentum distribution instead


        sigma_fb= self.mass_distribution * self.params.MDOT(self.t)
        if not self.params.FB_ON: sigma_fb *= 0

        self.sigma_dot = sigma_fb - sigma_wl

        # calculate source terms for entropy
        kappa = kappa_interpolator(self.rho, self.T)
        self.qrad = 4*RADA*self.T**4*c/(1+kappa*self.sigma)  # radiative cooling
        self.qwind = self.params.FWIND*self.grid.omgko*self.sigv*self.sigma*self.grid.vk2o  # wind cooling
        if not self.params.WIND_ON: self.qwind *= 0
        self.qvis = 2.25*self.nu*self.grid.omgko**2*self.sigma  # viscous cooling
        self.qfb = 0.5*sigma_fb*self.grid.vk2o  # fallback heating
        if not self.params.FB_ON: self.qfb *= 0

        self.ts_dot = (self.qvis+self.qfb-self.qrad-self.qwind)/self.T

        self.ts_dt = np.abs(self.ts[1:-1]/self.ts_dot[1:-1])
        self.sigma_dt = np.abs(self.sigma[1:-1]/self.sigma_dot[1:-1])

    def sigma_var(self):
        return ShastaVariable(self.grid, self.sigma, self.vr, self.sigma_dot)
    def ts_var(self):
        return ShastaVariable(self.grid, self.ts, self.vr, self.ts_dot)
