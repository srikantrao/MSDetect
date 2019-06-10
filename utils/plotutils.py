import matplotlib.pyplot as plt
from scipy import math
import numpy as np
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import utils.statutils as stu

def plot1trace(trial, plot_interp=True, eqax=False):
    '''
    Plot a trace form our trial set
    Parameters:
        trial - one Eyetrace
        plot_interp (bool) - Flag if we want to plot timepoints marked as bad
        eqax (bool) - Flag if we want to plot all traces on same axis limits
    Returns:
        Nothing just plots stuff
    '''
    if(plot_interp):
        xvals = trial.x
        yvals = trial.y
        tvals = trial.time
    else:
        xvals = trial.x[~trial.interp_idx]
        yvals = trial.y[~trial.interp_idx]
        tvals = trial.time[~trial.interp_idx]

    im = plt.scatter(xvals, yvals,
                    s=3, c=tvals,
                    alpha=0.5, edgecolors='none')
    plt.xlabel('x (degrees)')
    plt.ylabel('y (degrees)')
    plt.title(f'Subject: {trial.subjid}')

    # set axes range
    if(eqax):
        plt.set_xlim(np.min(xvals), np.max(xvals))
        plt.set_ylim(np.min(yvals), np.max(yvals))

    return
    

def plot16traces(trials, plot_interp=True, eqax=False, rand_state=np.random.RandomState(None)):
    '''
    Plot 16 traces from our trial set
    Parmeters:
        trials (list of Eyetraces) Our eyetrace trials for plotting
        plot_interp (bool) - Flag if we want to plot timepoints marked as bad
        eqax (bool) - Flag if we want to plot all traces on same axis limits
    Returns:
        Nothing just plots stuff *TODO* Do this correctly by passing axes
    '''

    ##calc number of rows needed with 2 cols
    ncol = 4
    nrow = int(math.ceil(16 / ncol))
    rand16 = rand_state.choice(trials,size=16,replace=False)

    #Calculate max and min for a trials we are plotting
    minx = np.min([np.min(trial.x) for trial in rand16])
    miny = np.min([np.min(trial.y) for trial in rand16])
    maxx = np.max([np.max(trial.x) for trial in rand16])
    maxy = np.max([np.max(trial.y) for trial in rand16])

    #Show early -> late progression
    fig, ax = plt.subplots(nrow, ncol, figsize=(10,10))
    plt.tight_layout()

    for i, pl in enumerate(ax.flatten()):
        trial = rand16[i]

        if(plot_interp):
            xvals = trial.x
            yvals = trial.y
            tvals = trial.time
        else:
            xvals = trial.x[~trial.interp_idx]
            yvals = trial.y[~trial.interp_idx]
            tvals = trial.time[~trial.interp_idx]

        im = pl.scatter(xvals, yvals,
                        s=3, c=tvals,
                        alpha=0.5, edgecolors='none')
        pl.set_xlabel('x (degrees)')
        pl.set_ylabel('y (degrees)')
        pl.set_title(f'Subject: {trial.subjid}')
        #pl.axis('equal') ##comment this in and out
        ##add colorbar
        #div = make_axes_locatable(pl)
        #cax = div.append_axes("right", size="10%", pad=0.05)
        #cbar = plt.colorbar(im, cax=cax)
        #cbar.set_label('Seconds')

        # set axes range
        if(eqax):
            pl.set_xlim(minx, maxx)
            pl.set_ylim(miny, maxy)

    plt.tight_layout()


def plot_velocity_vec(control_trials, patient_trials, v_dframes=1, drift_only=False, saccade_only=False):
    '''
    Plot the (directional) velocity vector for patient and control trials
    Parameters:
        Patient Trials (list of Eyetraces)
        Control Trials (list of Eyetraces)
        v_dframes (number of frames over which to calculate velocity)
    Returns:
        Plot
    '''

    [con_mean_dvvec_x, con_mean_dvvec_y] = zip(*[stu.get_mean_velocity_vec(trial, v_dframes, drift_only, saccade_only) for trial in control_trials])
    [pat_mean_dvvec_x, pat_mean_dvvec_y] = zip(*[stu.get_mean_velocity_vec(trial, v_dframes, drift_only, saccade_only) for trial in patient_trials])

    dt = v_dframes*control_trials[0].dt #THIS ASSUMES ALL DTS ARE EQUAL.
    
    fig = plt.figure(figsize=(8,8))
    plt.scatter(con_mean_dvvec_x, con_mean_dvvec_y, label='controls')
    plt.scatter(pat_mean_dvvec_x, pat_mean_dvvec_y, label='patients')
    plt.axvline(0,c='k')
    plt.axhline(0,c='k')
    if(drift_only):
        plt.title(f'Mean Drift Velocity Vector, dt={dt:.5f}')
    elif(saccade_only):
        plt.title(f'Mean Saccade Velocity Vector, dt={dt:.5f}')
    else:
        plt.title(f'Mean Total Velocity Vector, dt={dt:.5f}')
    plt.legend()
    return(fig)

