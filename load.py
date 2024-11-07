from grid import *
import os
from eos import *
from consts import *

def vector_interp(r_face, r_cell, v_cell):
    v_face = v_cell[:,:-1]
    v_face += (v_cell[:,1:] - v_cell[:,:-1])/(r_cell[1:]-r_cell[:-1])*(r_face - r_cell[:-1])
    return v_face

def Snu_scattering(T, rho, nu, grid, filter=None):
    if filter is not None:
        T *= filter(grid.r_cell)
    exp_arg = np.einsum("ij,k->ijk", T**-1, HP*nu/KB)
    denom = np.exp(exp_arg) -1

    KR = kappa_interpolator(rho, T)
    eps = KR/(KR + 0.4)
    fact = 2*np.sqrt(eps)/(1+np.sqrt(eps))

    num = 2*HP/c/c*np.einsum("j,k->jk", 2*np.pi*grid.r_cell*grid.dr, nu**3)
    integrand = np.einsum("jk,ijk->ijk", num, denom**-1)
    integrand = np.einsum("ij,ijk->ijk", fact, integrand)
    ## weens's law
    integrand = np.where(exp_arg > 1000, 0, integrand)
    ## Rayleigh-Jeans tail
    integrand = np.where(exp_arg < 1/1000, 2*KB/c/c*np.einsum("k,ij->ijk", nu**2, T*grid.r_cell*2*np.pi*grid.dr), integrand)
    return np.sum(integrand, axis=1)/(4*np.pi*(10*PC)**2)

class LoadSimulation:
    def __init__(self, t0=None, tf=None, dt=0.1*YEAR, mode="CONST", tend=3*MONTH, dts=10*DAY, tstart=10*DAY,
                 params=Params(), file_dir=None, read_dat=False, temp_filter=None, tindex=None):

        if file_dir is None:
            self.sim_dir = os.path.abspath(params.SIM_DIR)
        else:
            self.sim_dir = os.path.abspath(file_dir)

        self.params=params
        self.temp_filter = temp_filter
        print(os.path.join(self.sim_dir,"*.npy"))
        load_from_dat = (not glob.glob(os.path.join(self.sim_dir,"*.npy"))) or read_dat

        if load_from_dat:
            self.glob_sigma = sorted(glob.glob(self.sim_dir+"/sigma.*.dat"))
            self.glob_entropy = sorted(glob.glob(self.sim_dir + "/entropy.*.dat"))
            ## want to make homogeneous
            self.ts_by_file = [np.loadtxt(f, skiprows=1, usecols=0) for f in self.glob_sigma]
            max_num = np.max([np.size(ts) for ts in self.ts_by_file])

            for i, ts in enumerate(self.ts_by_file):
                lts = np.size(ts)

                if not np.size(ts) == max_num:
                    diff = int(max_num-lts)
                    self.ts_by_file[i] = np.append(self.ts_by_file[i], -np.inf*np.ones(diff))

            self.ts_by_file = np.array(self.ts_by_file)
            self.ts = self.ts_by_file.flatten() ## ugly as shit
            self.ts = self.ts[np.logical_not(np.isinf(self.ts))]
        else:
            self.ts = np.load(os.path.abspath(self.sim_dir+"/tsave.npy"), mmap_mode="r")
        if t0 is None:
            t0 = self.ts[0]
        if tf is None:
            tf = self.ts[-1]

        self.grid = Grid(params=params)

        comparr = []

        def am_distribution(x, f):
            mu = np.log(self.params.LC / f + self.params.LMIN)
            sigma2 = 2 * np.log((self.params.LC + self.params.LMIN) / (self.params.LC / f + self.params.LMIN))
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

        if tindex is not None:
            comparr = np.array(self.ts)[tindex]

        elif mode.__eq__("CONST"):
            comparr = np.append(np.arange(t0, tf, dt), tf)
        elif mode.__eq__("LINEAR"):
            # Linear ramp
            slope = (dts-dt)/(tend-tstart)
            first = np.arange(t0, tstart, dt)
            n=0
            trans = [tstart]
            while True:
                dtp = dt + slope * (trans[-1]-tstart)
                trans += [trans[-1] + dtp]
                if trans[-1] > tend:
                    break
            last = np.append(np.arange(tend+dts, tf, dts), tf)
            comparr = np.append(first, trans)
            comparr = np.append(comparr, last)
        elif mode.__eq__("LOGARITHMIC"):
            dtlog = np.log10((self.ts[1]+dt)/self.ts[1])
            logsp = np.append([t0], np.arange(np.log10(self.ts[1]), np.log10(tf)+dtlog, dtlog))
            comparr = 10**logsp

        if load_from_dat:
            ts_max = np.max(self.ts_by_file, axis=1)
            ts_min = np.min(np.abs(self.ts_by_file), axis=1)
            ts_file = -np.inf*np.ones(comparr.shape)
            self.sigma, self.s = np.array([]), np.array([])
            for i, (t0, tf) in enumerate(zip(ts_min, ts_max)):
                betwixt = np.logical_and(comparr >= t0, comparr <= tf)
                ts_file = np.where(betwixt, i, ts_file)

                t_arr = self.ts_by_file[i]
                t_comp = comparr[betwixt]

                ts_proj = np.einsum("i,j->ij", np.ones(t_comp.shape), t_arr)
                ca_proj = np.einsum("i,j->ij", t_comp, np.ones(t_arr.shape))
                indices = np.unique(np.argmin(np.abs(ts_proj - ca_proj), axis=1))

                sigma_read = None; entropy_read = None

                try:
                    sigma_read = np.loadtxt(self.glob_sigma[i], skiprows=1)[:,1:][indices]
                    entropy_read = np.loadtxt(self.glob_entropy[i], skiprows=1)[:,1:][indices]
                except:
                    continue

                if not self.sigma.size and not self.s.size:
                    self.sigma = np.array(sigma_read)
                    self.s = np.array(entropy_read)
                    self.ts = self.ts_by_file[i][indices]
                else: ## just appends to file
                    self.sigma = np.vstack((self.sigma, sigma_read))
                    self.s = np.vstack((self.s, entropy_read))
                    self.ts = np.append(self.ts, self.ts_by_file[i][indices])

        else:
            ts_proj = np.einsum("i,j->ij", np.ones(comparr.shape), self.ts)
            ca_proj = np.einsum("i,j->ij", comparr, np.ones(self.ts.shape))
            indices = np.unique(np.argmin(np.abs(ts_proj - ca_proj), axis=1))

            self.sigma = np.load(os.path.abspath(self.sim_dir+"/sigma.npy"), mmap_mode="r")[indices]
            self.s = np.load(os.path.abspath(self.sim_dir+"/entropy.npy"), mmap_mode="r")[indices]
            self.ts = self.ts[indices]
            self.grid = Grid(grid_array=np.load(os.path.abspath(self.sim_dir+"/r_cell.npy")), params=params)

        self.eos = load_table(params.EOS_TABLE)

        self.chi = self.sigma*self.grid.omgko*np.sqrt((self.grid.r_cell - self.params.RSCH)/self.grid.r_cell)
        self.T = self.eos(self.chi, self.s)

        self.rho = entropy_difference(self.T, self.chi, self.s, just_density=True)

        self.rad_P = RADA*self.T**4/3
        self.gas_P = self.rho*KB*self.T/mu
        self.P = self.rad_P + self.gas_P
        self.U = RADA * self.T ** 4 + 1.5 * self.rho * KB * self.T / mu

        self.be = -1 + 2 * (self.U + self.P) / self.rho/self.grid.vk2o
        self.H = self.sigma / 2 / self.rho
        self.h = self.H/self.grid.r_cell

        if params.CONST_NU:
            self.nu = params.CONST_NU*self.grid.cell_ones()
        else:
            self.nu = params.ALPHA * self.H ** 2 * self.grid.omgko

        self.nuf = np.logspace(13, 18, 100)
        self.S_nu = Snu(self.T, self.nuf, self.grid)
        self.S_nu_scat = Snu_scattering(self.T, self.rho, self.nuf, self.grid, filter=self.temp_filter)

        self.L_nu = 4*np.pi*(10*PC)**2*self.S_nu
        self.L_nu_scat = 4*np.pi*(10*PC)**2*self.S_nu_scat

        self.nuL_nu = np.einsum("j,ij->ij", self.nuf, self.L_nu)
        self.nuL_nu_scat = np.einsum("j,ij->ij", self.nuf, self.L_nu_scat)

        self.sigv = sig((self.be - params.BE_CRIT) / params.DBE)
        ## calculates velocities at inter_faces
        lc_sigma = 2 * np.pi * self.sigma * self.grid.r_cell
        g = np.sqrt(self.grid.r_cell) / (self.nu + 1e-20)
        d = 3 * self.nu

        g_tild, d_tild, lc_sigma_tild = [], [], []
        if params.INTERP.__eq__("LINEAR"):
            r_cell = self.grid.r_cell
            r_face = self.grid.r_face
            g_tild = vector_interp(r_face, r_cell, g)
            d_tild = vector_interp(r_face, r_cell, d)
            lc_sigma_tild = vector_interp(r_face, r_cell, lc_sigma)
        elif params.INTERP.__eq__("LOGARITHMIC"):
            logr_cell = np.log10(self.grid.r_cell)
            logr_face = np.log10(self.grid.r_face)
            g_tild = vector_interp(logr_face, logr_cell, g)
            d_tild = vector_interp(logr_face, logr_cell, d)
            lc_sigma_tild = vector_interp(logr_face, logr_cell, lc_sigma)

        self.vr = -d_tild*g_tild/lc_sigma_tild * (lc_sigma[:,1:]/g[:,1:] - lc_sigma[:,:-1]/g[:,:-1])/self.grid.ddr

        ## calculate source terms for density
        sigma_wl = params.FWIND*self.sigma * self.grid.omgko * self.sigv  ## wind loss
        if not params.WIND_ON: sigma_wl *= 0

        time_part = params.MDOT(self.ts)
        sigma_fb = np.einsum("i,j->ij", time_part, self.mass_distribution)
        if not params.FB_ON: sigma_fb *= 0

        self.sigma_fb = sigma_fb
        self.sigma_wl = sigma_wl
        self.sigma_dot = sigma_fb - sigma_wl

        ## calculate source terms for entropy
        kappa = kappa_interpolator(self.rho, self.T)
        self.kappa = kappa

        qrad = 4 * RADA * self.T ** 4 * c / (1 + kappa * self.sigma)
        qwind = params.FWIND*self.grid.omgko*self.sigv*self.sigma*self.grid.vk2o
        if not params.WIND_ON: qwind *= 0

        qvis = 2.25 * self.nu * self.grid.omgko ** 2*self.sigma
        qfb = params.FSH*0.5*sigma_fb*self.grid.vk2o
        if not params.FB_ON: qfb *= 0

        self.qvis = qvis/self.sigma
        self.qfb = qfb/self.sigma
        self.qrad = qrad/self.sigma
        self.qwind = qwind/self.sigma

        ## calculates advective flux
        sigma_face = vector_interp(self.grid.r_face, self.grid.r_cell, self.sigma)
        sigma_flux = sigma_face*self.vr
        net_sigma_flux = np.zeros(self.sigma.shape)

        net_sigma_flux[:,1:-1] = (sigma_flux[:, :-1]*self.grid.face_area[:-1] - sigma_flux[:, 1:]*self.grid.face_area[1:]) / self.grid.cell_vol[1:-1]
        self.sigma_adv = net_sigma_flux

        ## calculates advective cooling
        ts_face = vector_interp(self.grid.r_face, self.grid.r_cell, self.sigma*self.s)
        ts_flux = ts_face*self.vr
        net_ts_flux = np.zeros(self.sigma.shape)

        net_ts_flux[:, 1:-1] = self.T[:, 1:-1]*(ts_flux[:,:-1]*self.grid.face_area[:-1] - ts_flux[:, 1:]*self.grid.face_area[1:])/self.grid.cell_vol[1:-1]
        self.qadv = net_ts_flux/self.sigma

        self.ts_dot = (qvis + qfb - qrad - qwind) / self.T
