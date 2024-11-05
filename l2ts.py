from load import *
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from matplotlib import colormaps
"""

"""
def run_diagnostic(simulation_name, simulation_save_directory, log_filenames, output_directory=None):
    simulation_save_directory = os.path.abspath(simulation_save_directory)
    log_files = sorted(glob(os.path.join(simulation_save_directory,log_filenames+"*.out")))
    sim_files = sorted(glob(os.path.join(simulation_save_directory, f"{simulation_name}_*")))

    if output_directory is None:
        output_directory = os.path.join(simulation_save_directory, f"{simulation_name}_diagnostic")
    else:
        output_directory = os.path.join(simulation_save_directory, output_directory)

    try:
        sim_files.remove(output_directory)
    except:
        print("")

    cmap1 = colormaps["PiYG"]
    cmap2 = colormaps["PRGn"]
    cmap3 = colormaps["BrBG"]
    cmap4 = colormaps["RdBu"]
    cmap5 = colormaps["PuOr"]

    if len(log_files) != len(sim_files):
        print("Log file length and simulation file length incompatible")
        return 0

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)


    all_simulation_files = []
    all_simulation_log = []

    ## reads log files
    for i, file in enumerate(log_files):
        with open(file, "r") as f:
            lines = [l.split("\t") for l in f.readlines()]
            all_simulation_log += [lines]
            all_simulation_files += [[os.path.join(sim_files[i], l[0]) for l in lines]]

    all_simulation_log = [log for asl in all_simulation_log for log in asl]
    all_simulation_files = [f for files in all_simulation_files for f in files]

    for (log, file) in zip(all_simulation_log, all_simulation_files):
        finished=False
        if os.path.exists(os.path.join(file, "sigma.npy")): finished=True

        figure_name = file.split("/")
        if not finished:
            figure_name = figure_name[-2] + "_" + figure_name[-1] + ".pdf"
        else:
            figure_name = figure_name[-2] + "_" + figure_name[-1] + "_DONE.pdf"
        ## check if simulation is finished

        figure_name = os.path.join(output_directory, figure_name)

        if not finished:
            sigma_000_file = os.path.join(file, "sigma.000.dat")
            with open(sigma_000_file) as f:
                lines = f.readlines()
                if len(lines) <= 2: continue

        sim = LoadSimulation(params=Params(load=file), file_dir=file, tindex=[0, -1]) ## load only last two timesteps
        if len(sim.ts) < 2: print("issue with ts"); continue

        fig, axs = plt.subplots(2,7, figsize=(18, 6))

        for i,ax in enumerate(axs):
            ax[0].loglog(sim.grid.r_cell/sim.params.RSCH, sim.sigma[i])
            ax[1].loglog(sim.grid.r_cell/sim.params.RSCH, sim.s[i])
            ax[2].loglog(sim.grid.r_cell/sim.params.RSCH, sim.T[i])
            ax[3].loglog(sim.grid.r_cell/sim.params.RSCH, sim.h[i])

            ax[4].loglog(sim.grid.r_face/sim.params.RSCH, np.where(sim.vr[i] > 0, sim.vr[i], -np.inf), color="tab:blue")
            ax[4].loglog(sim.grid.r_face/sim.params.RSCH, np.where(sim.vr[i] < 0, -sim.vr[i], - np.inf), color="tab:orange")

            ## plot sigma source terms

            ax[5].loglog(sim.grid.r_cell/sim.params.RSCH, sim.sigma_fb[i], color="tab:blue")
            ax[5].loglog(sim.grid.r_cell / sim.params.RSCH,
                         np.where(sim.sigma_adv[i] > 0, sim.sigma_adv[i], -np.inf), color="tab:orange")
            ax[5].loglog(sim.grid.r_cell / sim.params.RSCH,
                         np.where(sim.sigma_adv[i] < 0, sim.sigma_adv[i], -np.inf), color="tab:orange", linestyle="--")
            ax[5].loglog(sim.grid.r_cell / sim.params.RSCH, sim.sigma_wl[i], color="tab:green")

            ax[6].loglog(sim.grid.r_cell / sim.params.RSCH, sim.qfb[i], color="tab:blue")
            ax[6].loglog(sim.grid.r_cell / sim.params.RSCH,
                         np.where(sim.sigma_adv[i] > 0, sim.qadv[i], -np.inf), color="tab:orange")
            ax[6].loglog(sim.grid.r_cell / sim.params.RSCH,
                         np.where(sim.sigma_adv[i] < 0, sim.qadv[i], -np.inf), color="tab:orange", linestyle="--")

            ax[6].loglog(sim.grid.r_cell / sim.params.RSCH, sim.qwind[i], color="tab:green")
            ax[6].loglog(sim.grid.r_cell / sim.params.RSCH, sim.qvis[i], color="tab:purple")
            ax[6].loglog(sim.grid.r_cell / sim.params.RSCH, sim.qrad[i], color="tab:orange")




        fig.savefig(figure_name)

run_diagnostic("ib_runs", "/Users/Shared/", "ib_submit")







