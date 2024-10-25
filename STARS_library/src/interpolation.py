import os
import numpy as np
from scipy.interpolate import interp1d
from src.constants import *

def sort_betas_key(f):
    return float(f.split('.dat')[0])

def beta_interpolate(input_dir, output_dir, current_sub_dir, num_interp_points, sim_beta_files=None, beta_arr=None):

    current_sub_dir = current_sub_dir + "/"
    current_dmdt_dir = input_dir + current_sub_dir

    Beta_slope = []
    Beta_yinter = []
    Mapped_time = []
    Premaptime = []
    Premapdmdt = []

    if sim_beta_files == None:
        sim_beta_files = [f for f in os.listdir(current_dmdt_dir) if not f.startswith('.')]
        sim_beta_files.sort(key=sort_betas_key)

    Sim_beta = [float(b[:-4]) for b in sim_beta_files]

    #print("Processing: betas %s for dir %s" % (Sim_beta, current_dmdt_dir))

    if beta_arr == None:
        beta_arr = np.logspace(np.log10(Sim_beta[0]), np.log10(Sim_beta[-1]), num=num_interp_points)

    time = {}
    dmdt = {}
    ipeak = {}
    mapped_time = {}

    time['lo'], dmdt['lo'] = np.genfromtxt(current_dmdt_dir + sim_beta_files[0], skip_header=1, unpack=True)
    ipeak['lo'] = np.argmax(dmdt['lo'])

    time['lo'] = np.array([
        time['lo'][:ipeak['lo']],
        time['lo'][ipeak['lo']:]], dtype=object)  # peak in array 2

    dmdt['lo'] = np.array([
        dmdt['lo'][:ipeak['lo']],
        dmdt['lo'][ipeak['lo']:]], dtype=object)  # peak in array 2

    Premaptime.append(np.copy(time['lo']))
    Premapdmdt.append(np.copy(dmdt['lo']))

    for i in range(1, len(Sim_beta)):
        # indexing this way bc calculating slope and yintercepts
        # BETWEEN each simulation beta

        time['hi'], dmdt['hi'] = np.genfromtxt(current_dmdt_dir + sim_beta_files[i], skip_header=1, unpack=True)

        ipeak['hi'] = np.argmax(dmdt['hi'])

        # split time_hi and dmdt_hi into pre-peak and post-peak array
        # peak in 2nd array
        time['hi'] = np.array([time['hi'][:ipeak['hi']],
                               time['hi'][ipeak['hi']:]], dtype=object)
        dmdt['hi'] = np.array([dmdt['hi'][:ipeak['hi']],
                               dmdt['hi'][ipeak['hi']:]], dtype=object)
        # will contain time/dmdt arrays
        # (split into pre & post peak times/dmdts)
        # for each beta value
        Premapdmdt.append(np.copy(dmdt['hi']))
        Premaptime.append(np.copy(time['hi']))

        mapped_time['hi'] = []
        mapped_time['lo'] = []

        Beta_slope.append([])

        Beta_yinter.append([])
        Mapped_time.append([])
        for j in [0, 1]:  # once before peak, once after peak
            # choose more densely sampled curve to map times to 0-1
            # less densely sampled curve will be interpolated to match
            if len(time['lo'][j]) < len(time['hi'][j]):
                # hi array more densely sampled
                interp = 'lo'
                nointerp = 'hi'
            else:
                # will also catch case where they have the same lengths
                interp = 'hi'
                nointerp = 'lo'
            # map times from more densely sampled curves
            # (both pre & post peak, might be from diff. dmdts)
            # to 0 - 1
            mapped_time[nointerp].append(
                1. / (time[nointerp][j][-1] - time[nointerp][j][0]) *
                (time[nointerp][j] - time[nointerp][j][0]))
            mapped_time[interp].append(
                1. / (time[interp][j][-1] - time[interp][j][0]) *
                (time[interp][j] - time[interp][j][0]))

            # ensure bounds are same for interp and nointerp
            # before interpolation
            # (should be 0 and 1 from above, but could be slightly off
            # due to rounding errors in python)
            mapped_time[interp][j][0] = 0
            mapped_time[interp][j][-1] = 1
            mapped_time[nointerp][j][0] = 0
            mapped_time[nointerp][j][-1] = 1

            func = interp1d(mapped_time[interp][j], dmdt[interp][j])
            dmdtinterp = func(mapped_time[nointerp][j])

            if interp == 'hi':
                slope = ((dmdtinterp - dmdt['lo'][j]) /
                         (Sim_beta[i] - Sim_beta[
                             i - 1]))
            else:
                slope = ((dmdt['hi'][j] - dmdtinterp) /
                         (Sim_beta[i] - Sim_beta[
                             i - 1]))
            Beta_slope[-1].append(slope)

            yinter1 = (dmdt[nointerp][j] - Beta_slope[-1][j] *
                       Sim_beta[i - 1])
            yinter2 = (dmdtinterp - Beta_slope[-1][j] *
                       Sim_beta[i])
            Beta_yinter[-1].append((yinter1 + yinter2) / 2.0)
            Mapped_time[-1].append(
                np.array(mapped_time[nointerp][j]))

        time['lo'] = np.copy(time['hi'])
        dmdt['lo'] = np.copy(dmdt['hi'])


    interp_index_low = [0 for i in beta_arr]
    interp_index_high = [0 for i in beta_arr]
    for i, b in enumerate(beta_arr):
        for j in range(len(Sim_beta)):
            if b == Sim_beta[j]:
                # no need to interp, already have dmdt & t for this beta
                beta_interp = False
                interp_index_high[i] = j
                interp_index_low[i] = j
                break

            if b < Sim_beta[j]:
                interp_index_high[i] = j
                interp_index_low[i] = j - 1
                beta_interp = True
                break

    if not os.path.exists(output_dir + current_sub_dir):
        os.makedirs(output_dir + current_sub_dir)

    for i in range(len(beta_arr)):
        if interp_index_high[i] == interp_index_low[i]:
            beta_string = '{0:f}'.format(beta_arr[i])[:5]

            time, dmdt = np.genfromtxt(current_dmdt_dir + beta_string + '.dat', skip_header=1, unpack=True)

            file_to_save = output_dir + current_sub_dir + beta_string + '.dat'
            #print("\tSaving %s" % file_to_save)
            np.savetxt(file_to_save, np.transpose([time, dmdt]))
        else:

            dmdtinterpolated = np.array([
                                Beta_yinter[interp_index_low[i]][0] +
                                Beta_slope[interp_index_low[i]][0] * beta_arr[i],
                                Beta_yinter[interp_index_low[i]][1] +
                                Beta_slope[interp_index_low[i]][1] * beta_arr[i]], dtype=object)

                            # map mapped_times back to actual times, requires interpolation
                            # in time
                            # first for pre peak times

            timeinterpolated = []
            for j in [0, 1]:
                # interp_index_low indexes beta
                # mapped time between beta low and beta high
                time_betalo = (
                    Mapped_time[interp_index_low[i]][j] *
                    (Premaptime[interp_index_low[i]][j][-1] -
                     Premaptime[interp_index_low[i]][j][0]) +
                    Premaptime[interp_index_low[i]][j][0])
                time_betahi = (
                    Mapped_time[interp_index_low[i]][j] *
                    (Premaptime[interp_index_high[i]][j][-1] -
                     Premaptime[interp_index_high[i]][j][0]) +
                    Premaptime[interp_index_high[i]][j][0])

                timeinterpolated.append(
                    time_betalo + (time_betahi - time_betalo) *
                    (beta_arr[i] -
                     Sim_beta[interp_index_low[i]]) /
                    (Sim_beta[interp_index_high[i]] -
                     Sim_beta[interp_index_low[i]]))

            timeinterpolated = np.array(timeinterpolated, dtype=object)
            
            beta_string = '{0:f}'.format(beta_arr[i])[:5]

            file_to_save = output_dir + current_sub_dir + beta_string + '.dat'
            #print("\tSaving %s" % file_to_save)
            np.savetxt(file_to_save, np.transpose([np.concatenate([timeinterpolated[0], timeinterpolated[1]]),
                                                    np.concatenate([dmdtinterpolated[0], dmdtinterpolated[1]])]))
            return os.path.abspath(file_to_save)

def mass_interpolate(output_dir, age_string, m1, m2, num_interp_points, input_dir=None, mass_arr=None):

    #print("Processing: [%s, %s] for time %s" % (m1, m2, age_string))

    m1_dir = model_dir_formatter.format(m1, age_string)
    m2_dir = model_dir_formatter.format(m2, age_string)
    dmdt_sub_dirs = [m1_dir, m2_dir]

    # --------- GET SIMULATION BETAS -----------------
    # find lo_interp_beta_files
    if input_dir == None:
        a = [f for f in os.listdir(output_dir + dmdt_sub_dirs[0]) if not f.startswith('.')]
    else:
        a = [f for f in os.listdir(input_dir + dmdt_sub_dirs[0]) if not f.startswith('.')]
    a.sort(key=sort_betas_key)

    lo_interp_beta_files = a
    lo_interp_betas = [float(b[:-4]) for b in lo_interp_beta_files]

    Sim_mass = [float(m1[1:]), float(m2[1:])]
    if mass_arr == None:
        mass_arr = np.linspace(Sim_mass[0], Sim_mass[-1], num=num_interp_points)

    for z, low_interp_beta_file in enumerate(lo_interp_beta_files):
        # ------ DIRECTORY PARAMETERS -------
        Beta_slope = []
        Beta_yinter = []

        Mapped_time = []
        # for converting back from mapped time to actual times and doing
        # interpolation in actual time
        Premaptime = []
        Premapdmdt = []

        # ----- CREATE INTERPOLATION FUNCTIONS; FIND SLOPES & YINTERs -----
        time = {}
        dmdt = {}
        ipeak = {}
        mapped_time = {}
        # get dmdt and t for the lowest beta value
        if input_dir == None:
            time['lo'], dmdt['lo'] = np.genfromtxt(output_dir + dmdt_sub_dirs[0] + "/" + low_interp_beta_file,
                                                   skip_header=1, unpack=True)
        else:
            time['lo'], dmdt['lo'] = np.genfromtxt(input_dir + dmdt_sub_dirs[0] + "/" + low_interp_beta_file,
                                                   skip_header=1, unpack=True)
        ipeak['lo'] = np.argmax(dmdt['lo'])

        # split time['lo'] & dmdt['lo'] into pre-peak and post-peak array
        time['lo'] = np.array([
            time['lo'][:ipeak['lo']],
            time['lo'][ipeak['lo']:]], dtype=object)  # peak in array 2
        dmdt['lo'] = np.array([
            dmdt['lo'][:ipeak['lo']],
            dmdt['lo'][ipeak['lo']:]], dtype=object)  # peak in array 2

        # will contain time/dmdt arrays
        # (split into pre & post peak times/dmdts)
        # for each beta value
        Premaptime.append(np.copy(time['lo']))
        Premapdmdt.append(np.copy(dmdt['lo']))

        # for i in range(1, len(Sim_beta)):
        for i in range(1, len(Sim_mass)):
            if input_dir == None:
                a = [f for f in os.listdir(output_dir + dmdt_sub_dirs[i]) if not f.startswith('.')]
            else:
                a = [f for f in os.listdir(input_dir + dmdt_sub_dirs[i]) if not f.startswith('.')]
            a.sort(key=sort_betas_key)

            hi_interp_beta_files = a
            hi_interp_betas = [float(b[:-4]) for b in hi_interp_beta_files]

            if len(hi_interp_beta_files) != len(lo_interp_beta_files):
                raise Exception('ERROR: not same number of betas in each directory.\
                                Possibly because did not clear output directory before making new interpolated library.')

            if input_dir == None:
                time['hi'], dmdt['hi'] = np.genfromtxt(output_dir + dmdt_sub_dirs[i] + "/" + hi_interp_beta_files[z],
                                                       skip_header=1, unpack=True)
            else:
                time['hi'], dmdt['hi'] = np.genfromtxt(input_dir + dmdt_sub_dirs[i] + "/" + hi_interp_beta_files[z],
                                                       skip_header=1, unpack=True)
            ipeak['hi'] = np.argmax(dmdt['hi'])

            # split time_hi and dmdt_hi into pre-peak and post-peak array
            # peak in 2nd array
            time['hi'] = np.array([time['hi'][:ipeak['hi']],
                                   time['hi'][ipeak['hi']:]], dtype=object)
            dmdt['hi'] = np.array([dmdt['hi'][:ipeak['hi']],
                                   dmdt['hi'][ipeak['hi']:]], dtype=object)
            # will contain time/dmdt arrays
            # (split into pre & post peak times/dmdts)
            # for each beta value
            Premapdmdt.append(np.copy(dmdt['hi']))
            Premaptime.append(np.copy(time['hi']))

            mapped_time['hi'] = []
            mapped_time['lo'] = []

            Beta_slope.append([])

            Beta_yinter.append([])
            Mapped_time.append([])
            for j in [0, 1]:  # once before peak, once after peak
                # choose more densely sampled curve to map times to 0-1
                # less densely sampled curve will be interpolated to match
                if len(time['lo'][j]) < len(time['hi'][j]):
                    # hi array more densely sampled
                    interp = 'lo'
                    nointerp = 'hi'
                else:
                    # will also catch case where they have the same lengths
                    interp = 'hi'
                    nointerp = 'lo'
                # map times from more densely sampled curves
                # (both pre & post peak, might be from diff. dmdts)
                # to 0 - 1
                mapped_time[nointerp].append(
                    1. / (time[nointerp][j][-1] - time[nointerp][j][0]) *
                    (time[nointerp][j] - time[nointerp][j][0]))
                mapped_time[interp].append(
                    1. / (time[interp][j][-1] - time[interp][j][0]) *
                    (time[interp][j] - time[interp][j][0]))

                # ensure bounds are same for interp and nointerp
                # before interpolation
                # (should be 0 and 1 from above, but could be slightly off
                # due to rounding errors in python)
                mapped_time[interp][j][0] = 0
                mapped_time[interp][j][-1] = 1
                mapped_time[nointerp][j][0] = 0
                mapped_time[nointerp][j][-1] = 1

                func = interp1d(mapped_time[interp][j], dmdt[interp][j])
                dmdtinterp = func(mapped_time[nointerp][j])

                if interp == 'hi':
                    slope = ((dmdtinterp - dmdt['lo'][j]) / (Sim_mass[i] - Sim_mass[i - 1]))
                else:
                    slope = ((dmdt['hi'][j] - dmdtinterp) / (Sim_mass[i] - Sim_mass[i - 1]))

                Beta_slope[-1].append(slope)

                yinter1 = (dmdt[nointerp][j] - Beta_slope[-1][j] * Sim_mass[i - 1])
                yinter2 = (dmdtinterp - Beta_slope[-1][j] * Sim_mass[i])
                Beta_yinter[-1].append((yinter1 + yinter2) / 2.0)
                Mapped_time[-1].append(np.array(mapped_time[nointerp][j]))

            time['lo'] = np.copy(time['hi'])
            dmdt['lo'] = np.copy(dmdt['hi'])

        interp_index_low = [0 for i in mass_arr]
        interp_index_high = [0 for i in mass_arr]

        for i, b in enumerate(mass_arr):
            for j in range(len(Sim_mass)):
                if b == Sim_mass[j]:
                    # no need to interp, already have dmdt & t for this beta
                    beta_interp = False
                    interp_index_high[i] = j
                    interp_index_low[i] = j
                    break

                if b < Sim_mass[j]:
                    interp_index_high[i] = j
                    interp_index_low[i] = j - 1
                    beta_interp = True
                    break

        for i in range(len(mass_arr)):
            if interp_index_high[i] == interp_index_low[i]:
                # already have this mass
                pass
            else:
                dmdtinterpolated = np.array([
                    Beta_yinter[interp_index_low[i]][0] +
                    Beta_slope[interp_index_low[i]][0] * mass_arr[i],
                    Beta_yinter[interp_index_low[i]][1] +
                    Beta_slope[interp_index_low[i]][1] * mass_arr[i]], dtype=object)

                # map mapped_times back to actual times, requires interpolation
                # in time
                # first for pre peak times

                timeinterpolated = []
                for j in [0, 1]:
                    # interp_index_low indexes beta
                    # mapped time between beta low and beta high
                    time_betalo = (
                            Mapped_time[interp_index_low[i]][j] *
                            (Premaptime[interp_index_low[i]][j][-1] -
                             Premaptime[interp_index_low[i]][j][0]) +
                            Premaptime[interp_index_low[i]][j][0])
                    time_betahi = (
                            Mapped_time[interp_index_low[i]][j] *
                            (Premaptime[interp_index_high[i]][j][-1] -
                             Premaptime[interp_index_high[i]][j][0]) +
                            Premaptime[interp_index_high[i]][j][0])

                    timeinterpolated.append(
                        time_betalo + (time_betahi - time_betalo) *
                        (mass_arr[i] -
                         Sim_mass[interp_index_low[i]]) /
                        (Sim_mass[interp_index_high[i]] -
                         Sim_mass[interp_index_low[i]]))

                timeinterpolated = np.array(timeinterpolated, dtype=object)

                savesmalldir = 'm' + str(round(mass_arr[i], 3))[:5] + '_' + age_string + '/'
                if not os.path.exists(output_dir + savesmalldir):
                    os.makedirs(output_dir + savesmalldir)

                beta_float = round(lo_interp_betas[z] + ((mass_arr[i] - Sim_mass[0]) / (Sim_mass[-1] - Sim_mass[0])) \
                                   * (hi_interp_betas[z] - lo_interp_betas[z]), 3)

                beta_string = '{0:f}'.format(beta_float)[:5]

                file_to_save = output_dir + savesmalldir + beta_string + '.dat'
                #print("\tSaving %s" % file_to_save)
                np.savetxt(file_to_save,
                           np.transpose([np.concatenate([timeinterpolated[0], timeinterpolated[1]]),
                                         np.concatenate([dmdtinterpolated[0], dmdtinterpolated[1]])]))

def age_interpolate(output_dir, mass_string, t1, t2, num_interp_points, input_dir=None, age_arr=None):
    #print("Processing: [%s, %s] for mass %s" % (t1, t2, mass_string))

    t1_dir = model_dir_formatter.format(mass_string, t1)
    t2_dir = model_dir_formatter.format(mass_string, t2)
    dmdt_sub_dirs = [t1_dir, t2_dir]

    # --------- GET SIMULATION BETAS -----------------
    # find lo_interp_beta_files
    if input_dir == None:
        a = [f for f in os.listdir(output_dir + dmdt_sub_dirs[0]) if not f.startswith('.')]
    else:
        a = [f for f in os.listdir(input_dir + dmdt_sub_dirs[0]) if not f.startswith('.')]
    a.sort(key=sort_betas_key)

    lo_interp_beta_files = a
    lo_interp_betas = [float(b[:-4]) for b in lo_interp_beta_files]

    Sim_age = [float(f.split('t')[1].split('/')[0]) for f in dmdt_sub_dirs]
    if age_arr == None:
        age_arr = np.linspace(Sim_age[0], Sim_age[-1], num=num_interp_points)

    for z, low_interp_beta_file in enumerate(lo_interp_beta_files):
        # ------ DIRECTORY PARAMETERS -------
        Beta_slope = []
        Beta_yinter = []
        Mapped_time = []
        # for converting back from mapped time to actual times and doing
        # interpolation in actual time
        Premaptime = []
        Premapdmdt = []

        # ----- CREATE INTERPOLATION FUNCTIONS; FIND SLOPES & YINTERs -----
        time = {}
        dmdt = {}
        ipeak = {}
        mapped_time = {}
        # get dmdt and t for the lowest beta value
        # energy & dmde (cgs)
        if input_dir == None:
            time['lo'], dmdt['lo'] = np.genfromtxt(output_dir + dmdt_sub_dirs[0] + "/" + low_interp_beta_file,
                                                   skip_header=1, unpack=True)
        else:
            time['lo'], dmdt['lo'] = np.genfromtxt(input_dir + dmdt_sub_dirs[0] + "/" + low_interp_beta_file,
                                                   skip_header=1, unpack=True)
        ipeak['lo'] = np.argmax(dmdt['lo'])

        # split time['lo'] & dmdt['lo'] into pre-peak and post-peak array
        time['lo'] = np.array([
            time['lo'][:ipeak['lo']],
            time['lo'][ipeak['lo']:]], dtype=object)  # peak in array 2
        dmdt['lo'] = np.array([
            dmdt['lo'][:ipeak['lo']],
            dmdt['lo'][ipeak['lo']:]], dtype=object)  # peak in array 2

        # will contain time/dmdt arrays
        # (split into pre & post peak times/dmdts)
        # for each beta value
        Premaptime.append(np.copy(time['lo']))
        Premapdmdt.append(np.copy(dmdt['lo']))

        for i in range(1, len(Sim_age)):
            if input_dir == None:
                a = [f for f in os.listdir(output_dir + dmdt_sub_dirs[i]) if not f.startswith('.')]
            else:
                a = [f for f in os.listdir(input_dir + dmdt_sub_dirs[i]) if not f.startswith('.')]
            a.sort(key=sort_betas_key)

            hi_interp_beta_files = a
            hi_interp_betas = [float(b[:-4]) for b in hi_interp_beta_files]

            if len(hi_interp_beta_files) != len(lo_interp_beta_files):
                raise Exception('ERROR: not same number of betas in each directory.\
                                Possibly because did not clear output directory before making new interpolated library.')

            if input_dir == None:
                time['hi'], dmdt['hi'] = np.genfromtxt(output_dir + dmdt_sub_dirs[i] + "/" + hi_interp_beta_files[z],
                                                       skip_header=1, unpack=True)
            else:
                time['hi'], dmdt['hi'] = np.genfromtxt(input_dir + dmdt_sub_dirs[i] + "/" + hi_interp_beta_files[z],
                                                       skip_header=1, unpack=True)

            ipeak['hi'] = np.argmax(dmdt['hi'])

            # split time_hi and dmdt_hi into pre-peak and post-peak array
            # peak in 2nd array
            time['hi'] = np.array([time['hi'][:ipeak['hi']],
                                   time['hi'][ipeak['hi']:]], dtype=object)
            dmdt['hi'] = np.array([dmdt['hi'][:ipeak['hi']],
                                   dmdt['hi'][ipeak['hi']:]], dtype=object)
            # will contain time/dmdt arrays
            # (split into pre & post peak times/dmdts)
            # for each beta value
            Premapdmdt.append(np.copy(dmdt['hi']))
            Premaptime.append(np.copy(time['hi']))

            mapped_time['hi'] = []
            mapped_time['lo'] = []

            Beta_slope.append([])

            Beta_yinter.append([])
            Mapped_time.append([])
            for j in [0, 1]:  # once before peak, once after peak
                # choose more densely sampled curve to map times to 0-1
                # less densely sampled curve will be interpolated to match
                if len(time['lo'][j]) < len(time['hi'][j]):
                    # hi array more densely sampled
                    interp = 'lo'
                    nointerp = 'hi'
                else:
                    # will also catch case where they have the same lengths
                    interp = 'hi'
                    nointerp = 'lo'
                # map times from more densely sampled curves
                # (both pre & post peak, might be from diff. dmdts)
                # to 0 - 1
                mapped_time[nointerp].append(
                    1. / (time[nointerp][j][-1] - time[nointerp][j][0]) *
                    (time[nointerp][j] - time[nointerp][j][0]))
                mapped_time[interp].append(
                    1. / (time[interp][j][-1] - time[interp][j][0]) *
                    (time[interp][j] - time[interp][j][0]))

                # ensure bounds are same for interp and nointerp
                # before interpolation
                # (should be 0 and 1 from above, but could be slightly off
                # due to rounding errors in python)
                mapped_time[interp][j][0] = 0
                mapped_time[interp][j][-1] = 1
                mapped_time[nointerp][j][0] = 0
                mapped_time[nointerp][j][-1] = 1

                func = interp1d(mapped_time[interp][j], dmdt[interp][j])
                dmdtinterp = func(mapped_time[nointerp][j])

                if interp == 'hi':
                    slope = ((dmdtinterp - dmdt['lo'][j]) / (Sim_age[i] - Sim_age[i - 1]))
                else:
                    slope = ((dmdt['hi'][j] - dmdtinterp) / (Sim_age[i] - Sim_age[i - 1]))

                Beta_slope[-1].append(slope)

                yinter1 = (dmdt[nointerp][j] - Beta_slope[-1][j] * Sim_age[i - 1])
                yinter2 = (dmdtinterp - Beta_slope[-1][j] * Sim_age[i])
                Beta_yinter[-1].append((yinter1 + yinter2) / 2.0)
                Mapped_time[-1].append(np.array(mapped_time[nointerp][j]))

            time['lo'] = np.copy(time['hi'])
            dmdt['lo'] = np.copy(dmdt['hi'])

        interp_index_low = [0 for i in age_arr]
        interp_index_high = [0 for i in age_arr]
        for i, b in enumerate(age_arr):
            for j in range(len(Sim_age)):
                if b == Sim_age[j]:
                    # no need to interp, already have dmdt & t for this beta
                    beta_interp = False
                    interp_index_high[i] = j
                    interp_index_low[i] = j
                    break

                if b < Sim_age[j]:
                    interp_index_high[i] = j
                    interp_index_low[i] = j - 1
                    beta_interp = True
                    break


        for i in range(len(age_arr)):
            if interp_index_high[i] == interp_index_low[i]:
                # already have this mass
                pass
            else:

                dmdtinterpolated = np.array([
                    Beta_yinter[interp_index_low[i]][0] +
                    Beta_slope[interp_index_low[i]][0] * age_arr[i],
                    Beta_yinter[interp_index_low[i]][1] +
                    Beta_slope[interp_index_low[i]][1] * age_arr[i]], dtype=object)

                # map mapped_times back to actual times, requires interpolation
                # in time
                # first for pre peak times

                timeinterpolated = []
                for j in [0, 1]:
                    # interp_index_low indexes beta
                    # mapped time between beta low and beta high
                    time_betalo = (
                            Mapped_time[interp_index_low[i]][j] *
                            (Premaptime[interp_index_low[i]][j][-1] -
                             Premaptime[interp_index_low[i]][j][0]) +
                            Premaptime[interp_index_low[i]][j][0])
                    time_betahi = (
                            Mapped_time[interp_index_low[i]][j] *
                            (Premaptime[interp_index_high[i]][j][-1] -
                             Premaptime[interp_index_high[i]][j][0]) +
                            Premaptime[interp_index_high[i]][j][0])

                    timeinterpolated.append(
                        time_betalo + (time_betahi - time_betalo) *
                        (age_arr[i] -
                         Sim_age[interp_index_low[i]]) /
                        (Sim_age[interp_index_high[i]] -
                         Sim_age[interp_index_low[i]]))

                timeinterpolated = np.array(timeinterpolated, dtype=object)

                savesmalldir = mass_string + '_t' + str(round(age_arr[i], 3))[:5] + '/'
                if not os.path.exists(output_dir + savesmalldir):
                    os.makedirs(output_dir + savesmalldir)

                beta_float = round(lo_interp_betas[z] + ((age_arr[i] - Sim_age[0]) / (Sim_age[-1] - Sim_age[0])) \
                                   * (hi_interp_betas[z] - lo_interp_betas[z]), 3)

                beta_string = '{0:f}'.format(beta_float)[:5]
                file_to_save = output_dir + savesmalldir + beta_string + '.dat'
                #print("\tSaving %s" % file_to_save)

                np.savetxt(file_to_save,
                           np.transpose([np.concatenate([timeinterpolated[0], timeinterpolated[1]]),
                                         np.concatenate([dmdtinterpolated[0], dmdtinterpolated[1]])]))
