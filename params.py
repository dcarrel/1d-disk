import numpy as np
from consts import *
import json, os, sys
sys.path.append(os.path.abspath(os.getcwd()))
from STARS_library.SL_run import *

class Params:
    def __init__(self,
                 MBH=1e6*MSUN,          ## BH mass
                 R0=5,                  ## Inner grid radius in Rsch
                 RF=1000,               ## Outer grid radius in Rsch
                 NR=500,                ## Number of grid points
                 T0=0,                  ## Initial time
                 TF=5*YEAR,             ## Final time
                 TS=0.1*DAY,            ## Increment between saves
                 SIM_DIR="MY_SIM",     ## File name
                 FILE_INT=0.1*YEAR,     ## Increment between different files (doesn't really matter)
                 RESTART=False,         ## Whether or not to restart, not really that useful
                 CFLDT = 0.7,           ## CFL number
                 SDT = 0.5,             ## Source number, kind of like CFL number for sources
                 BE_CRIT=-0.1,          ## Wind parameter
                 DBE=1/300,             ## Wind parameter
                 FWIND=0.5,             ## Wind parameter
                 SIGMAF = 2,              ## Fallback parameter
                 FSH=0.5,               ## Fallback parameter
                 ALPHA=0.01,            ## Viscosity parameter
                 TOL=1e-3,              ## Numerical tolerance, approximated by RK1/2 difference
                 MAXIT=1000,            ## Max number of iterations for adapative timestepping
                 load=None,             ## Option to load parameter file from pre-existing one
                 EOS_TABLE="EOS", ## What EOS table to use
                 GEOMETRY = "LOGARITHMIC",     ## Grid geometry
                 MSTAR = MSUN,            ## Stellar mass
                 MAGE = 0.5,               ## like a wizard but more mystical
                 BETA = 1,                 ## tidal disruption parameter RT/RP

                 INTERP="LINEAR",       ## How to interpolate from cells onto faces
                 ## Options: LINEAR: interpolates linearly on r_cell, LOGARITHMIC: interpolates linearly on log r_cell
                 WIND_ON = True,        ## If the wind should be on
                 FB_ON = True,          ## If the fallback should be on
                 CONST_NU=None,         ## Set to test against analytic solution. Does not evolve entropy
                 EVOLVE_SIGMA=True,     ## If false, fixes sigma profile
                 EVOLVE_ENTROPY=True,    ## If false, fixes entropy profile
                 SAVE=True,
                 MAXTIME=None, ## cancels after a certain amount of time
                 PRINT=False,

                 M0=0.01,
                 TV0=0.05
                 ):
        self._pdict = {"MBH": float(MBH),
                           "R0": float(R0),
                           "RF": float(RF),
                           "NR": int(NR),
                           "T0": float(T0),
                           "TF":float(TF),
                           "TS": float(TS),
                           "SIM_DIR": SIM_DIR,
                           "FILE_INT": FILE_INT,
                           "RESTART": RESTART,
                           "CFLDT": float(CFLDT),
                           "SDT": float(SDT),
                           "BE_CRIT": float(BE_CRIT),
                           "DBE": float(DBE),
                           "FWIND": float(FWIND),
                           "SIGMAF": float(SIGMAF),
                           "FSH": float(FSH),
                           "ALPHA":float(ALPHA),
                           "TOL": float(TOL),
                           "MAXIT": int(MAXIT),
                           "EOS_TABLE": EOS_TABLE,
                           "GEOMETRY": GEOMETRY,
                           "MSTAR": float(MSTAR),
                           "MAGE": float(MAGE),
                           "BETA": float(BETA),
                           "INTERP": INTERP,
                           "WIND_ON": WIND_ON,
                           "FB_ON": FB_ON,
                           "CONST_NU": CONST_NU,
                           "EVOLVE_ENTROPY": EVOLVE_ENTROPY,
                           "EVOLVE_SIGMA": EVOLVE_SIGMA,
                           "SAVE": SAVE,
                           "MAXTIME": MAXTIME,
                           "M0": M0,
                           "TV0":TV0
            }

        if load is not None:
            new_dict = []

            ## loads in dictionary
            if isinstance(load, dict):
                new_dict = load.copy()
            if isinstance(load, str):
                with open(os.path.join(os.path.dirname(__file__), load+"/params.json")) as f:
                    new_dict = json.load(f)

            for key, val in new_dict.items():
                if key in self._pdict:
                    self._pdict[key] = val
        ## writes param file

        
        self.MBH = self._pdict["MBH"]

        self.RSCH = 2*CONST_G*self.MBH/c**2
        self.NR = self._pdict["NR"]

        self.T0 = self._pdict["T0"]
        self.TF = self._pdict["TF"]
        self.TS = self._pdict["TS"]

        self.SIM_DIR = os.path.abspath(self._pdict["SIM_DIR"])

        self.FILE_INT = self._pdict["FILE_INT"]
        self.RESTART = self._pdict["RESTART"]

        self.CFLDT = self._pdict["CFLDT"]
        self.SDT = self._pdict["SDT"]

        self.BE_CRIT = self._pdict["BE_CRIT"]
        self.DBE = self._pdict["DBE"]
        self.FWIND = self._pdict["FWIND"]

        self.SIGMAF = self._pdict["SIGMAF"]
        self.FSH = self._pdict["FSH"]

        self.ALPHA = self._pdict["ALPHA"]
        self.TOL = self._pdict["TOL"]
        self.MAXIT = self._pdict["MAXIT"]
        self.EOS_TABLE = self._pdict["EOS_TABLE"]
        self.GEOMETRY = self._pdict["GEOMETRY"]
        self.MSTAR = self._pdict["MSTAR"]
        self.BETA = self._pdict["BETA"]
        self.MAGE = self._pdict["MAGE"]

        self.INTERP = self._pdict["INTERP"]

        self.WIND_ON = self._pdict["WIND_ON"]
        self.FB_ON = self._pdict["FB_ON"]

        self.CONST_NU = self._pdict["CONST_NU"]
        if self.CONST_NU: self._pdict["EVOLVE_ENTROPY"] = False
        self.EVOLVE_ENTROPY = self._pdict["EVOLVE_ENTROPY"]

        self.EVOLVE_SIGMA = self._pdict["EVOLVE_SIGMA"]
        self.SAVE = self._pdict["SAVE"]
        self.MAXTIME = self._pdict["MAXTIME"]

        self.M0 = self._pdict["M0"]
        self.TV0 = self._pdict["TV0"]

        ## c+p from https://dergipark.org.tr/en/download/article-file/1612778
        def stellar_radius(m):
            r = 0
            if m < 1.66:
                r = 1.054*m**0.935
            else:
                r = 1.371*m**0.542
            return r

        self.TFB = 0.11*YEAR*stellar_radius(self.MSTAR/MSUN)**(3/2)*(self.MBH/1e6/MSUN)**(1/6)*(self.MSTAR/MSUN)**-1
        self.RT = 100*RSUN*stellar_radius(self.MSTAR/MSUN)*(MSUN/self.MSTAR)**(1/3)*(self.MBH/1e6/MSUN)**(1/3)
        self.RP = self.RT/BETA
        try:
            self.MDOT = STARS_library().retrieve(self.MSTAR/MSUN, self.MAGE, self.BETA, self.MBH)
            self.FALLBACK_FAILURE=False
        except:
            self.FALLBACK_FAILURE=True

        if not self.FALLBACK_FAILURE:
            self.TOTAL_FALLBACK_MASS = np.trapz(self.MDOT.mdots, self.MDOT.ts)
            self.A_ORB = 7.6e13*(self.MBH/1e6/MSUN)**(1/3)
            self.A_SD = self.TOTAL_FALLBACK_MASS/self.A_ORB**2

        self.CAPTURE=True if self.RT < self.RSCH else False
        self.R0 = 2*self.RSCH
        self.RF = 5000*self.RSCH



        if PRINT and not self.FALLBACK_FAILURE:
            print(f"Simulation parameters: {self.SIM_DIR}: \t R0: {(self.R0/self.RP/2):3.3f} R0 \t RF: {(self.RF/self.RSCH):3.3f}rg\t "
              f"MFB/MSTAR: {(self.TOTAL_FALLBACK_MASS/self.MSTAR):3.3e}\t MFB/A^2:{(self.A_SD):3.3e}")

        self.RC = 2*self.RP
        self.RMB = 2*self.RSCH
        self.LMIN = np.sqrt(CONST_G * self.MBH * self.RMB ** 2 / (self.RMB - self.RSCH))
        self.LC = np.sqrt(CONST_G * self.MBH * self.RC ** 2 / (self.RC - self.RSCH))



    def save(self, file_name=None):
        if not os.path.exists(self.SIM_DIR):
            os.makedirs(self.SIM_DIR)
        if file_name is None: file_name = "params.json"
        with open(os.path.join(self.SIM_DIR, file_name), "w") as f:
            json.dump(self._pdict, f)



