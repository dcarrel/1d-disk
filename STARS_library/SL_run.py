import os, sys
from configparser import RawConfigParser
import time as tm
curr_dir = os.path.abspath(os.path.dirname( __file__ ))
sys.path.append(curr_dir)
sys.path.append(curr_dir+"/..")
from consts import *
from src.interpolation import beta_interpolate, mass_interpolate, age_interpolate
from src.retrieval import retrieval
import numpy as np

config = RawConfigParser()
config.read(os.path.abspath(os.path.join(curr_dir, 'STARS.config')))

class STARS_library:

    def add_options(self, parser=None, usage=None, config=None):
        import optparse

        if parser == None:
            parser = optparse.OptionParser(usage=usage, conflict_handler="resolve")

        parser.add_option("-r", "--retrieve", nargs=3, type="float",
                          help="Tuple to retrieve single model at command line: {mass} {age} {beta}.")

        parser.add_option("-g", "--retrieve_grid", action="store_true", dest="retrieve_grid", default=False, 
                          help="Flag to retrieve grid of models from file, which is RETRIEVE.par by default.\
                           DEFAULT=False")

        return (parser)

    def initialize(self):

        ## BETA INTERPOLATION ##
        all_input_subdirs = [name for name in os.listdir(self.input_dir)
                             if os.path.isdir(os.path.join(self.input_dir, name))]

        for subdir in all_input_subdirs:
            beta_interpolate(self.input_dir, self.output_dir, subdir, self.num_interp_beta)


        ## MASS INTERPOLATION ##
        current_output_subdirs = [name for name in os.listdir(self.output_dir)
                             if os.path.isdir(os.path.join(self.output_dir, name))]

        zams_key = "t0.0"
        tams_key = "t1.0"
        model_directories_by_time = {
            zams_key: None,
            tams_key: None
        }

        zams_masses = []
        tams_masses = []

        zams_mass_vals = []
        tams_mass_vals = []

        for cod in current_output_subdirs:
            dir_name = cod.split("/")[-1]
            age_key = dir_name.split("_")[-1]
            mass_key = dir_name.split("_")[0]

            # need to sort numbers, not strings... X: m10 will sort before m3, which is not what we want
            mass_value = float(mass_key[1:])

            # We're currently not handling the t0.57 case in the mass interpolation, so skip
            if age_key == "t0.57":
                continue

            if zams_key in dir_name:
                zams_masses.append(mass_key)
                zams_mass_vals.append(mass_value)
            else:
                tams_masses.append(mass_key)
                tams_mass_vals.append(mass_value)

        zmv_i = np.argsort(zams_mass_vals)
        tmv_i = np.argsort(tams_mass_vals)

        model_directories_by_time[zams_key] = np.asarray(zams_masses)[zmv_i]
        model_directories_by_time[tams_key] = np.asarray(tams_masses)[tmv_i]

        for age_string, mass_steps in model_directories_by_time.items():

            for m1, m2 in zip(mass_steps[:-1], mass_steps[1:]):
                mass_interpolate(self.output_dir, age_string, m1, m2, self.num_interp_mass)


        ## AGE INTERPOLATION ##
        model_directories_by_mass = {}
        current_output_subdirs = [name for name in os.listdir(self.output_dir)
                                  if os.path.isdir(os.path.join(self.output_dir, name))]
        for sub_dir in current_output_subdirs:
            mass_str = sub_dir.split("_")[0]

            if mass_str not in model_directories_by_mass:
                if mass_str != "m1.0":
                    model_directories_by_mass[mass_str] = ["t0.0", "t1.0"]
                else:
                    model_directories_by_mass[mass_str] = ["t0.0", "t0.57", "t1.0"]

        for mass_string, age_steps in model_directories_by_mass.items():
            for t1, t2 in zip(age_steps[:-1], age_steps[1:]):
                age_interpolate(self.output_dir, mass_string, t1, t2, self.num_interp_age)

    def __init__(self):

        self.input_dir = os.path.join(curr_dir, config.get('general_settings', 'input_dir'))
        self.output_dir = os.path.join(curr_dir, config.get('general_settings', 'output_dir'))
        self.retrieval_grid_file = os.path.join(curr_dir, config.get('general_settings', 'retrieval_grid_file'))
        self.retrieval_input_dir = os.path.join(curr_dir, config.get('general_settings', 'retrieval_input_dir'))
        self.retrieval_scratch_dir = os.path.join(curr_dir, config.get('general_settings', 'retrieval_scratch_dir'))
        self.retrieval_output_dir = os.path.join(curr_dir, config.get('general_settings', 'retrieval_output_dir'))
        self.retrieval_grid_file = os.path.join(curr_dir, config.get('general_settings', 'retrieval_grid_file'))
        self.num_interp_beta = int(config.get('interpolation', 'NUM_BETA_INTERP_POINTS'))
        self.num_interp_mass = int(config.get('interpolation', 'NUM_MASS_INTERP_POINTS'))
        self.num_interp_age = int(config.get('interpolation', 'NUM_AGE_INTERP_POINTS'))

        # Check if initialized, if not, initialize first
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        interpolated_subdirs = [name for name in os.listdir(self.output_dir)
                                if os.path.isdir(os.path.join(self.output_dir, name))]

        if len(interpolated_subdirs) == 0:
            #print("Initializing library...")

            t1 = tm.time()
            self.initialize()
            t2 = tm.time()

            #print("... Initialize complete. [%0.2f sec]" % (t2 - t1))

    def retrieve(self, mass, age, beta, mbh):
        ret_dir =retrieval(mass, age, beta, self.retrieval_input_dir, self.retrieval_scratch_dir, self.retrieval_output_dir)
        #print(f"file located at {ret_dir}")
        return Interpolator(ret_dir, mbh)
        

class Interpolator:
    def __init__(self, ret_dir, mbh):
        self.ts, self.mdots = np.loadtxt(ret_dir).T
        self.ts *= DAY*np.sqrt(mbh/1e6/MSUN)
        self.mdots *= MSUN/YEAR/np.sqrt(mbh/1e6/MSUN)
        self.log_slope = (np.log10(self.mdots[1])-np.log10(self.mdots[0]))/(np.log10(self.ts[1])-np.log10(self.ts[0]))
    def __call__(self, t, arg=1):
        if isinstance(t, np.ndarray):
            ret_vals = np.zeros(t.shape)
            ret_vals = arg*np.where(t < self.ts[0], self.mdots[0]*(t/self.ts[0])**self.log_slope, ret_vals)
            ret_vals = np.where(t > self.ts[-1], self.mdots[-1]*(t/self.ts[-1])**(-5/3), ret_vals)
            betwixt = np.logical_and(t < self.ts[-1], t > self.ts[0])
            ret_vals = np.where(betwixt, np.interp(t, self.ts, self.mdots), ret_vals)
            return ret_vals
        else:
            if t < self.ts[0]:
                 return arg*self.mdots[0]*((t+1e-10)/self.ts[0])**self.log_slope
            elif t > self.ts[-1]:
                return self.mdots[-1]*(t/self.ts[-1])**(-5/3)

            else:
                return np.interp(t, self.ts, self.mdots)

        
        


if __name__ == "__main__" and False:
    useagestring = """python SL_run.py [options]

    ### To run a single model: ###
    `python SL_run.py -r {mass} {age} {beta}`
    
    retrieve params:
        mass [M_sun]: 0.1 - 10.0
        age [fractional; 0.0=ZAMS, 1.0=min(10 Gyr, TAMS)]: 0.0 - 1.0
        beta [impact parameter]: varies (#prints range if outside it)
    
    
    
    ### To run a grid of models: ###
    1) Edit the ./RETRIEVE.par file with parameters you wish:
    
    ex:
    
    mass [M_sun]    age [fractional; 0=ZAMS, 1==TAMS]   beta [r_p/r_t]
    1               0                                   1
    2               0.5                                 3
    ...
    
    2) Run the command:
    `python SL_run.py -g`
    """

    stars_lib = STARS_library()
    parser = stars_lib.add_options(usage=useagestring)
    options, args = parser.parse_args()
    stars_lib.options = options


    if options.retrieve and options.retrieve_grid:
        raise Exception("Use only -r OR -g.")

    if options.retrieve:
        mass = options.retrieve[0]
        age = options.retrieve[1]
        beta = options.retrieve[2]
        stars_lib.retrieve(mass, age, beta)

    if options.retrieve_grid:
        mass, age, beta = np.loadtxt(stars_lib.retrieval_grid_file, skiprows=1, unpack=True)
        for m, a, b in zip(mass, age, beta):
            stars_lib.retrieve(m, a, b)
