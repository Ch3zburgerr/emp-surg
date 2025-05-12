import os
from util import get_time_slice, preprocess
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from joints import JOINT_NAMES
from group_walk import group_walk
folder_link = "D:\\Coding\\data\\joint_out"

def find_extremes(time_slice, use_vertices = False, vertex_step = 1):
    x_min, x_max, y_min, y_max, z_min, z_max = float('inf'), float('-inf'), float('inf'), float('-inf'), float('inf'), float('-inf')
    for i in range(len(time_slice)):
        joints = time_slice[i]['joints3d']
        numPeople = len(time_slice[i]['trackers'])
        for person in range(numPeople):
            for joint in joints[person]:
                x_min = min(x_min, joint[0])
                x_max = max(x_max, joint[0])
                y_min = min(y_min, joint[1])
                y_max = max(y_max, joint[1])
                z_min = min(z_min, joint[2])
                z_max = max(z_max, joint[2])
    if (use_vertices):
        for i in range(len(time_slice)):
            vertices = time_slice[i]['vertices']
            for person in range(0, numPeople, vertex_step):
                for vertex in vertices[person]:
                    x_min = min(x_min, vertex[0])
                    x_max = max(x_max, vertex[0])
                    y_min = min(y_min, vertex[1])
                    y_max = max(y_max, vertex[1])
                    z_min = min(z_min, vertex[2])
                    z_max = max(z_max, vertex[2])
    return x_min, x_max, y_min, y_max, z_min, z_max

#https://github.com/siva82kb/smoothness/blob/master/python/smoothness.py 

def spectral_arclength(movement, fs, padlevel=4, fc=10.0, amp_th=0.05):
    """
    Calcualtes the smoothness of the given speed profile using the modified spectral
    arc length metric.

    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.
    padlevel : integer, optional
               Indicates the amount of zero padding to be done to the movement
               data for estimating the spectral arc length. [default = 4]
    fc       : float, optional
               The max. cut off frequency for calculating the spectral arc
               length metric. [default = 10.]
    amp_th   : float, optional
               The amplitude threshold to used for determing the cut off
               frequency upto which the spectral arc length is to be estimated.
               [default = 0.05]

    Returns
    -------
    sal      : float
               The spectral arc length estimate of the given movement's
               smoothness.
    (f, Mf)  : tuple of two np.arrays
               This is the frequency(f) and the magntiude spectrum(Mf) of the
               given movement data. This spectral is from 0. to fs/2.
    (f_sel, Mf_sel) : tuple of two np.arrays
                      This is the portion of the spectrum that is selected for
                      calculating the spectral arc length.

    Notes
    -----
    This is the modfieid spectral arc length metric, which has been tested only
    for discrete movements.
    It is suitable for movements that are a few seconds long, but for long
    movements it might be slow and results might not make sense (like any other
    smoothness metric).

    Examples
    --------
    >>> t = np.arange(-1, 1, 0.01)
    >>> move = np.exp(-5*pow(t, 2))
    >>> sal, _, _ = spectral_arclength(move, fs=100.)
    >>> '%.5f' % sal
    '-1.41403'

    """
    # Number of zeros to be padded.
    nfft = int(pow(2, np.ceil(np.log2(len(movement))) + padlevel))

    # Frequency
    f = np.arange(0, fs, fs/nfft)
    # Normalized magnitude spectrum
    Mf = abs(np.fft.fft(movement, nfft))
    Mf = Mf/max(Mf)

    # Indices to choose only the spectrum within the given cut off frequency Fc.
    # NOTE: This is a low pass filtering operation to get rid of high frequency
    # noise from affecting the next step (amplitude threshold based cut off for
    # arc length calculation).
    fc_inx = ((f <= fc)*1).nonzero()
    f_sel = f[fc_inx]
    Mf_sel = Mf[fc_inx]

    # Choose the amplitude threshold based cut off frequency.
    # Index of the last point on the magnitude spectrum that is greater than
    # or equal to the amplitude threshold.
    inx = ((Mf_sel >= amp_th)*1).nonzero()[0]
    fc_inx = range(inx[0], inx[-1]+1)
    f_sel = f_sel[fc_inx]
    Mf_sel = Mf_sel[fc_inx]

    # Calculate arc length
    new_sal = -sum(np.sqrt(pow(np.diff(f_sel)/(f_sel[-1] - f_sel[0]), 2) +
                           pow(np.diff(Mf_sel), 2)))
    return new_sal, (f, Mf), (f_sel, Mf_sel)


def dimensionless_jerk(movement, fs):
    """
    Calculates the smoothness metric for the given speed profile using the dimensionless jerk 
    metric.
    
    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.

    Returns
    -------
    dl       : float
               The dimensionless jerk estimate of the given movement's smoothness.

    Notes
    -----
    

    Examples
    --------
    >>> t = np.arange(-1, 1, 0.01)
    >>> move = np.exp(-5*pow(t, 2))
    >>> dl = dimensionless_jerk(move, fs=100.)
    >>> '%.5f' % dl
    '-335.74684'

    """
    # first enforce data into an numpy array.
    movement = np.array(movement)

    # calculate the scale factor and jerk.
    movement_peak = max(abs(movement))
    dt = 1./fs
    movement_dur = len(movement)*dt
    jerk = np.diff(movement, 2)/pow(dt, 2)
    scale = pow(movement_dur, 3)/pow(movement_peak, 2)

    # estimate dj
    return - scale * sum(pow(jerk, 2)) * dt


def log_dimensionless_jerk(movement, fs):
    """
    Calculates the smoothness metric for the given speed profile using the log dimensionless jerk 
    metric.
    
    Parameters
    ----------
    movement : np.array
               The array containing the movement speed profile.
    fs       : float
               The sampling frequency of the data.

    Returns
    -------
    ldl      : float
               The log dimensionless jerk estimate of the given movement's smoothness.

    Notes
    -----
    

    Examples
    --------
    >>> t = np.arange(-1, 1, 0.01)
    >>> move = np.exp(-5*pow(t, 2))
    >>> ldl = log_dimensionless_jerk(move, fs=100.)
    >>> '%.5f' % ldl
    '-5.81636'

    """
    dimensionless_jerk_value = dimensionless_jerk(movement, fs)
    return -np.log(abs(dimensionless_jerk_value) + 1e-10) # Avoid log(0) error

def find_speed_profiles(group_walks, trackers):
    """
    Finds the speed profiles for each group walk.

    Parameters
    ----------
    group_walks : list
                  A list of group walks.

    Returns
    -------
    speed_profiles : list
                     A list of speed profiles for each group walk.
    """

    speed_profiles = {} # person_id : speed_profile
    for i in range(len(group_walks)):
        # Get the joints or vertices based on the use_vertices flag.
        numPeople = len(group_walks[i])
        for person in range(numPeople):
            # Get the movement data for the person.
            movement = group_walks[i][person]
            if (movement == 0 or movement == -1):
                continue
            tracker = trackers[i][person]
            if tracker not in speed_profiles:
                speed_profiles[tracker] = [[(movement, i)]]
            else:
                if (speed_profiles[tracker][-1][-1][1] + 1 == i): #consecutive frame
                    speed_profiles[tracker][-1].append((movement, i))
                else:
                    speed_profiles[tracker].append([(movement, i)])
    return speed_profiles

def group_smooth_walk(time_slice, fs=30.0):

    group_walks = group_walk(time_slice, 0, len(time_slice), threshold = True, full = True)
    trackers = []
    for i in range(len(time_slice)):
        trackers.append(time_slice[i]['trackers'])

    speed_profiles = find_speed_profiles(group_walks, trackers)

    # Initialize variables to store the smoothness values.
    dl_values = []
    
    # Loop through each frame in the time slice.
    jerkiness_profiles = {}

    for key in speed_profiles.keys():
        speed_profile = speed_profiles[key]
        jerkiness_profiles[key] = []
        num_consecutives = len(speed_profile)
        for i in range(num_consecutives):
            # Get the movement data without the frame numbers

            num_consecutive_frames = len(speed_profile[i])
            
            if (num_consecutive_frames == 1):
                # If there is only one frame, skip it
                continue
            movement = []
            for j in range(num_consecutive_frames):
                movement.append(speed_profile[i][j][0])
            movement = np.array(movement)

            # Get the start and end frame numbers
            start_frame = speed_profile[i][0][1]
            end_frame = speed_profile[i][-1][1]

            # Use the log dimensionless jerk metric to calculate the smoothness

            #print (movement, start_frame, end_frame)
            ldl = log_dimensionless_jerk(movement, fs)
            if (key not in jerkiness_profiles):
                jerkiness_profiles[key] = [(ldl, start_frame, end_frame)]
            else:
                jerkiness_profiles[key].append((ldl, start_frame, end_frame))
    return jerkiness_profiles
            


def main():
    testing = False
    isFull = False
    # Load the data
    if (testing):
        print ("Testing")
        return
    elif (os.path.exists(folder_link)):
        time_slice = get_time_slice("frame_000000.pkl", "frame_001000.pkl", folder_link, step=1, full=isFull, debug=False)        
        time_slice = preprocess(time_slice)
    else:
        print ("Folder does not exist")
        return
    # Get the extremes of the data
    """
    extremes = find_extremes(time_slice)
    print (extremes)
    """

    group_smooth_walks = group_smooth_walk(time_slice, fs=30.0)

if __name__ == "__main__":
    main()
