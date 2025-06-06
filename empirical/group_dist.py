# -*- coding: utf-8 -*-
"""group_distance_traversal.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/17wp5_E3nd_C9QC0nImUAVfHUVVOd0UVF
"""

import os
import torch
import numpy as np
import pickle
import io
import matplotlib.pyplot as plt
import time
from util import get_time_slice, apply_transform, preprocess
from scipy.spatial.distance import euclidean

"""
Applies padding and transformation to every frame in timeslice
"""

""" 
- Could look at average group distances moved over windows/phases
"""

def get_distance_diff(data, trackers):
    res = [] # total difference across frames
    id_dict = {} # key: id; value: previous location
    for i in range(1, len(data)): # Loop framges 
        res.append(0)
        for j in range(0, len(data[i])): # Iterate people 
            # getting the corresponding id from trackers
            id = trackers[i][j]
            if id not in id_dict:
                id_dict[id] = data[i][j][0][[0, 2]]
            else:
                diff = euclidean(data[i][j][0][[0, 2]], id_dict[id])
                res[i-1] += diff
                id_dict[id] = data[i][j][0][[0, 2]]
    res.append(0) # Last frame is 0 
    return res

def plot_distance_change(data):
    """
    Plots a bar chart of distance change over time steps.

    Parameters:
    data (list or array): A sequence of distance change values.
    """
    time_steps = range(1, len(data) + 1)  # Generate time steps

    plt.figure(figsize=(8, 5))
    plt.bar(time_steps, data)

    # Labels and title
    plt.xlabel("Time Step")
    plt.ylabel("Distance Changed")
    plt.title("Group Distance Traversal")

    # Show plot
    plt.show()


def tester():
    frames = ['frame_000000.pkl', 'frame_009790.pkl']
    time_slice = get_time_slice(frames[0], frames[1], "joint_out/", 10)
    time_slice = preprocess(time_slice)
    data = []
    trackers = []
    for frame in time_slice:
        data.append(frame['joints3d'])
        trackers.append(frame['trackers'])
    time_start = time.time()
    dist_diffs = get_distance_diff(data, trackers)
    time_end = time.time()
    print(f"Execution time: {time_end - time_start} seconds")
    plot_distance_change(dist_diffs)

def pipeline_tester(): 
    dataset = get_time_slice(0, 0, "joint_out/", debug=True)
    dataset = preprocess(dataset) 
    data = []
    trackers = []
    print(len(dataset))
    for frame in dataset:
        data.append(frame['joints3d'])
        trackers.append(frame['trackers'])
    group_distances = get_distance_diff(data, trackers)

if __name__ == "__main__":
    pipeline_tester()