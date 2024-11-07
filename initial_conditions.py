from params import *
import numpy as np
import os, sys, glob
from simulation import *
from grid import *
from consts import *
from scipy.special import iv
import matplotlib.pyplot as plt
from eos import load_table

## analytic solution for a spreading accretion disk with constant viscosity
## TVISC = number of viscous times
def analytic_solution(M0, R0, TVISC):
    def func(r):
        x = r/R0
        tau = 12*TVISC
        retval = M0/np.pi/R0**2/tau*x**-0.25*np.exp(-(1+x**2)/tau)*iv(0.25, 2*x/tau)
        retval = np.where(np.logical_or(np.isnan(retval), np.isinf(retval)), 0, retval)
        return retval
    return func

class InitialCondition:
    def __init__(self, m0=0.01, tv=1, ambf=1e-5, params=Params(), tf=1*MONTH, load_from=None,
                 save_dir=None, verbose=True, evolve=True, progress_message=None):
        self.load_from = load_from
        if save_dir is None and load_from is not None:
            save_dir = load_from
        if save_dir is None and load_from is None:
            save_dir = params.SIM_DIR+"/ics"
        self.save_dir = os.path.join(os.getcwd(), save_dir)
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        if load_from is not None:
            directory = os.path.join(os.getcwd(), load_from)
            self.params = Params(load=directory)
            self.grid = Grid(params=self.params)

            self.sigma0, self.entropy0 = np.load(directory+"/ics.npy")
        else:
            self.grid = Grid(params=params)
            self.sigma0 = analytic_solution(m0*params.MSTAR, 2*params.RP, tv)(self.grid.r_cell)
            self.params=params
            self.eos = load_table(self.params.EOS_TABLE)

            ## sets the power law profile in the ambient region
            sigma0_max = np.max(self.sigma0)
            sigma0_crit = ambf*sigma0_max
            disk_body = np.where(self.sigma0 > ambf*sigma0_max, 1, 0)
            bdy1 = np.argmax(disk_body)
            r1 = self.grid.r_cell[bdy1]
            bdy2 = len(self.sigma0) - np.argmax(disk_body[::-1])-1
            r2 = self.grid.r_cell[bdy2]

            def to_min(r):
                sigma_floor = np.maximum(self.params.SIGMA_FLOOR, 1)
                a = 3
                f = (r[0]/r1)**a
                sigf = sigma_floor-sigma0_crit*f/(1-f)
                return (sigma0_crit - sigf)*(r/r1)**a+sigf


            if r1 < 2*self.params.RP: self.sigma0 = np.where(self.grid.r_cell < r1, to_min(self.grid.r_cell), self.sigma0)
           # if r1 < 2*self.params.RP: self.sigma0 = np.where(self.grid.r_cell < r1, sigma0_crit, self.sigma0)
            if r2 > 2*self.params.RP: self.sigma0 = np.where(self.grid.r_cell > r2, sigma0_crit*(self.grid.r_cell/r2)**-3, self.sigma0)

            ## sets up entropy profile in a thick state
            h0 = np.sqrt((params.BE_CRIT+1)/8)
            h0 *= self.sigma0/sigma0_max+1e-3
            density0 = self.sigma0/2/h0/self.grid.r_cell
            chi0 = self.sigma0*self.grid.omgko*np.sqrt((self.grid.r_cell - self.params.RSCH)/self.grid.r_cell)

            temperature = get_temperature_from_density(1000*self.grid.cell_ones(), chi0, density0)
            self.entropy0 = entropy_difference(temperature, chi0, None, just_estimate=True)


            ic_pdict = self.params._pdict.copy()
            ic_pdict["EVOLVE_SIGMA"] = False
            ic_pdict["FB_ON"] = False
            ic_pdict["SAVE"] = False
            ic_pdict["TF"] = tf
            ic_pdict["SIM_DIR"] = save_dir
            ic_pdict["TOL"] = 1e-2
            self.params = Params(load=ic_pdict)

            self.sim = Simulation(self.sigma0, self.entropy0, params=self.params, verbose=verbose,
                                  progress_message=progress_message)
            self._makefig("before_evolving.png")
            if evolve:
                self.sim.evolve()
                self.sigma0 = self.sim.var0.sigma
                self.entropy0 = self.sim.var0.s
                self._makefig()
            self._save()

    def _save(self):
        self.params.save()
        save_array = np.array([self.sigma0, self.entropy0])
        np.save(self.params.SIM_DIR+"/ics.npy", save_array)

    def _makefig(self, name="ics.pdf"):
        save_dir = os.path.join(os.getcwd(),self.params.SIM_DIR)
        fig, axs = plt.subplots(1,4, figsize=(12, 3))
        axs[0].loglog(self.grid.r_cell, self.sigma0)
        axs[0].set_title(r"$\Sigma$ (g/cm$^2$)", fontsize=15)
        axs[1].loglog(self.grid.r_cell, self.entropy0)
        axs[1].set_title(r"$s$ (erg/g$\cdot$K)", fontsize=15)

        q_wind = self.sim.var0.qwind
        q_visc = self.sim.var0.qvis
        q_rad = self.sim.var0.qrad

        ts_face = np.interp(np.log10(self.grid.r_face), np.log10(self.grid.r_cell), self.sim.var0.ts)
        ts_flux = ts_face*self.sim.var0.vr
        net_flux = self.grid.cell_zeros()
        net_flux[1:-1] = self.sim.var0.T[1:-1]*(-ts_flux[1:]*self.grid.face_area[1:] + ts_flux[:-1]*self.grid.face_area[:-1])/self.grid.cell_vol[1:-1]


        axs[2].loglog(self.grid.r_cell, q_wind/self.sigma0, label="wind")
        axs[2].loglog(self.grid.r_cell, q_visc/self.sigma0, label="visc")
        axs[2].loglog(self.grid.r_cell, q_rad/self.sigma0, label="rad")

        adv_pos = net_flux > 0
        adv_neg = net_flux < 0
        net_flux_pos = np.where(adv_pos, net_flux, -np.inf)
        net_flux_neg = np.where(adv_neg, -net_flux, -np.inf)

        axs[2].loglog(self.grid.r_cell, (net_flux_pos / self.sigma0), label="adv")
        axs[2].loglog(self.grid.r_cell, (net_flux_neg / self.sigma0), color="tab:red", linestyle="--")
        axs[2].legend(frameon=False)
        axs[2].set_title(r"$\dot q$ (erg/s$\cdot$g)", fontsize=15)
        axs[2].set_ylim(bottom=0.1)

        axs[3].loglog(self.grid.r_cell, self.sim.var0.h)
        #axs[3].set_ylim(1e-6, 1)
        axs[3].set_title(r"$h$", fontsize=15)

        for ax in axs:
            ax.set_xlabel("Radius (cm)", fontsize=15)
        for ax in axs:
            ax.axvline(x=2*self.params.RP, linestyle="--", color="black")

        fig.savefig(f"{save_dir}/{name}", dpi=300, bbox_inches="tight")
        return fig, axs









