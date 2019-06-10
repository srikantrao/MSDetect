import numpy as np
import os

class EyeTrace:
    """A class to contain eye trace information
    Parameters:
        xlocs (numpy array) set of x positions of eye
        ylocs (numpy array) set of y positions of eye
        time (numpy array) set of timepoints in register with xlocs and ylocs
        velocities (numpy array) set of velocities of eye *IGNORED RIGHT NOW**))
        interptimes (numpy array) set of timepoints that have been interpolated
        bts (numpy array) set of timepoints that have blinks
        nbs (numpy array) number of blinks
        ppd (float) Number of pixels per degrees of eyetraces
        subinfo (Pandas DF) line from pandas df contining info about patients
    Returns:
        Instance of Class eyetrace
    """
    def __init__(self, xlocs, ylocs, time, velocities, interptimes, bts, bns, ppd, fname, subinfo):
        # define filename
        self.fname_full = fname
        self.fname_root = os.path.split(fname)[0]
        self.fname = os.path.split(fname)[-1]

        #not all filenames have 'meanrem' in name, but first 2 are always consistent
        # subjideye_tracenum_meanrem_480hz_####..mat
        split = self.fname.split("_")
        self.subjid = split[0][:-1] #5 numbers
        self.eye = split[0][-1].upper() # 'l' or 'r' for left/right eye
        self.tracenum = split[1] # v001, v002, v003, v004
        self.traceid = self.fname.replace('..mat','') # some filenames have two dots..
        # self.traceid = self.traceid.replace('.mat*.mat', '') # some filenames have this format

        # define scaled time
        self.time_offset = time[0]
        self.time = time - self.time_offset
        # round to 10 decimal places to avoid inconsistencies
        self.time = np.round_(self.time, 10)

        # define dt
        self.dt = time[1]-time[0]
        # define ppd value
        self.ppd = ppd

        # define x and y position attributes
        ## raw values: center at start
        self.xraw = np.subtract(xlocs, xlocs[0])
        self.yraw = np.subtract(ylocs, ylocs[0])
        # calcualte velocities
        self.xvel = (self.xraw[1:] - self.xraw[:-1])
        self.yvel = (self.yraw[1:] - self.yraw[:-1])
        self.vraw = np.sqrt(np.square(self.xvel)+
                            np.square(self.yvel))/self.dt
        self.vraw = np.append(self.vraw, self.vraw[-1]) #same length
        
        # define timepoints as saccades (not saccade=drift)
        self.sac_idx = self.vraw > 10 #10 degrees per second

        # define times that have been interpolated
        self.interp_times = interptimes - self.time_offset
        # indices of self.time where any of self.interp_times is in self.time within tol
        self.interp_idx = self.in1d_close(self.interp_times, self.time, tol=1e-6) # True indicates that there was interpolation

        # define number of blinks and times
        self.blink_times = bts
        self.blink_num = bns
        
        # values that are changeable
        self.x = np.subtract(xlocs, xlocs[0])
        self.y = np.subtract(ylocs, ylocs[0])
        self.v = self.vraw

        #DO THIS INSTEAD OUTSIDE READIN
        #get subject info variables from datasheet
        #pf = rut.read_pstats(patientfile)
        self.subinfo = subinfo
        self.sub_ms = np.array(subinfo['MS']).squeeze()
        self.sub_age = np.array(subinfo['Age']).squeeze()
        self.sub_edss = np.array(subinfo['EDSS']).squeeze()
        self.disease_dur = np.array(subinfo['Diseasedur']).squeeze()
        self.progressive = np.array(subinfo['progressive']).squeeze()
        self.cerebellar = np.array(subinfo['Cerebellar']).squeeze()
        self.brainstem = np.array(subinfo['Brainstem']).squeeze()

    def in1d_close(self, a, b, tol=1e-6):
        '''
        Calculates a in b up to some tolerance.
        i.e. Find all values in a that match any value in b within some tolorance
        Parameters:
          a (numpy array) values to look for
          b (numpy array) array to look in
          tol (float) tolerance
        Returns:
          in1d : ndarray, bool
            The values from a that are in b are located at b[in1d].
        Notes:
            adapted from https://github.com/numpy/numpy/issues/7784
            defines inclusion as being [a[i] - tol, a[i] + tol)
            test:
                a = np.array([1, 1.1, 1.3, 1.6, 2.0, 2.5, 3, 3.4, 3.8])
                b = np.linspace(0, 10, 101)
                c = (((b >= 0.8) & (b < 2.2)) |
                     ((b >= 2.3) & (b < 2.7)) |
                     ((b >= 2.8) & (b < 4.0)))
                assert np.all(in1d_close(a, b, tol=0.2) == c)
        '''
        a = np.unique(a)
        intervals = np.empty(2*a.size, float)
        intervals[::2] = a - tol
        intervals[1::2] = a + tol
        overlaps = intervals[:-1] >= intervals[1:]
        overlaps[1:] = overlaps[1:] | overlaps[:-1]
        keep = np.concatenate((~overlaps, [True]))
        intervals = intervals[keep]
        return np.searchsorted(intervals, b, side='right') & 1 == 1
