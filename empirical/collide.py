import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
import random
import io
import os
import time
from util import CPU_Unpickler, read_pickle, apply_transform, get_time_slice, preprocess, apply_padding

def get_pelvis(joints):
    """
    Extract the pelvis joint from the joints array.
    """
    return joints[:, 0, (0, 2)]  # Extract the pelvis joint for all people in the frame

def calculate_velocity(prev_position, curr_position):
    """
    Calculate the velocity between two positions.
    """
    return np.linalg.norm(curr_position - prev_position)

def is_walking(velocity, velocity_threshold=0.1):
    """
    Determine if a person is walking based on velocity.
    """
    return velocity > velocity_threshold

def detect_collisions(frames, radius, velocity_threshold):
    """
    Detect collision events based on bubble overlap and walking status.
    
    Args:
        frames (list): List of frames, where each frame is a dictionary containing 'joints3d' and 'trackers'.
        radius (float): Radius of each person's bubble.
        velocity_threshold (float): Threshold to determine if a person is walking.
    
    Returns:
        collision_events (list): A list of collision events, where each event is a dictionary containing:
            - frame_idx: Index of the frame where the collision occurred.
            - person_1: ID of the first person involved in the collision.
            - person_2: ID of the second person involved in the collision.
            - distance: Distance between the two people's pelvis joints.
            - walking_status: Tuple indicating if person_1 and person_2 were walking.
    """
    frame_collisions = [] 
    collision_events = []
    total_collisions = 0
    prev_pelvis = None  # Store pelvis positions from the previous frame
    prev_trackers = None  # Store trackers from the previous frame

    for frame_idx, frame in enumerate(frames):
        frame_collisions_i = 0 
        joints3d = frame['joints3d']
        trackers = frame['trackers']
        pelvis = get_pelvis(joints3d)  # Get pelvis joints for all people in the current frame

        # Initialize walking status for the current frame
        walking_status = {tracker: False for tracker in trackers}

        # Calculate walking status for each person
        if frame_idx > 0:
            for i, tracker in enumerate(trackers):
                # If the tracker existed in the previous frame, calculate velocity and check walking status
                if prev_trackers and tracker in prev_trackers:
                    velocity = calculate_velocity(prev_pelvis[prev_trackers.index(tracker)], pelvis[i])
                    walking_status[tracker] = is_walking(velocity, velocity_threshold)

        # Check for bubble overlaps (collisions)
        for i in range(len(trackers)):
            for j in range(i + 1, len(trackers)):
                distance = np.linalg.norm(pelvis[i] - pelvis[j])
                if distance < 2 * radius:  # Sum of radii
                    # Check if either person was walking
                    frame_collisions_i += 1 
                    if walking_status[trackers[i]] or walking_status[trackers[j]]:
                        total_collisions += 1
                        collision_events.append({
                            'frame_idx': frame_idx,
                            'person_1': trackers[i],
                            'person_2': trackers[j],
                            'distance': distance,
                            'walking_status': (walking_status[trackers[i]], walking_status[trackers[j]])
                        })
        # Update previous pelvis positions and trackers for the next frame
        prev_pelvis = pelvis
        prev_trackers = trackers
        frame_collisions.append(frame_collisions_i)

    return collision_events, total_collisions, frame_collisions

def execute_collision_detection(dataset, start_frame, end_frame, step, radius, v_thres):
    """
    Execute collision detection on a time slice of the dataset.
    
    Args:
        dataset (str): Path to the dataset directory.
        start_frame (int): Start frame index.
        end_frame (int): Stop frame index.
        time_slice (list): List of frames to process.
        radius (float): Radius of each person's bubble.
        v_thres (float): Threshold to determine if a person is walking.
    
    Returns:
        collision_events (list): A list of collision events, as described in detect_collisions().
        total_collisions (int): Total number of collisions detected.
    """
    time_slice = get_time_slice(start_frame, end_frame, dataset, step)
    time_slice = preprocess(time_slice)  
    collision_events, total_collisions = detect_collisions(time_slice, radius, v_thres)
    return collision_events, total_collisions
def calculate_chest_facing_direction(pelvis, left_shoulder, right_shoulder, spine2):
    """
    Calculate the facing direction of the chest based on the positions of the pelvis, middle spine, and shoulders.
    Args:
        pelvis (array): 3D coordinates of the pelvis.
        left_shoulder (array): 3D coordinates of the left shoulder.
        right_shoulder (array): 3D coordinates of the right shoulder.
        spine2 (array): 3D coordinates of the middle spine joint.
    Returns:
        array: A unit vector representing the facing direction of the chest.
    """
    # Calculate the vector between the shoulders
    shoulder_vector = right_shoulder - left_shoulder
    # Calculate the vector from the pelvis to the middle spine joint
    spine_vector = spine2 - pelvis
    facing_direction = np.cross(shoulder_vector, spine_vector)
    # Normalize the vector to get the direction
    facing_direction = facing_direction / np.linalg.norm(facing_direction)
    
    return facing_direction
def dist3D(joint1, joint2):
    """
    Calculate the displacement (Euclidean distance) between two joints in 3D space.
    Args:
        joint1 (size 3 array): 3D coordinates of the first joint.
        joint2 (size 3 array): 3D coordinates of the second joint.
    Returns:
        float: The displacement between the two joints.
    """
    displacement = np.sqrt((joint1[0] - joint2[0])**2 + (joint1[1] - joint2[1])**2 + (joint1[2] - joint2[2])**2)
    return displacement
def distXYZ(joint1, joint2, x = True, y = True, z = True):
    """
    Calculate the displacement (Euclidean distance) between two joints in 3D space.
    Args:
        joint1 (size 3 array): 3D coordinates of the first joint.
        joint2 (size 3 array): 3D coordinates of the second joint.
        x (bool): Whether to include x-axis displacement.
        y (bool): Whether to include y-axis displacement.
        z (bool): Whether to include z-axis displacement.
    Returns:
        float: The displacement between the two joints in the specified axes.
    """
    if (not x and not y and not z):
        return -1
    displacement = 0
    if (x):
        displacement += (joint1[0] - joint2[0])**2
    if (y):
        displacement += (joint1[1] - joint2[1])**2
    if (z):
        displacement += (joint1[2] - joint2[2])**2
    return np.sqrt(displacement)
def calculate_joint_displacement(time_slice, person_tracker, joint_name = 'spine2', x = True, y = True, z = True):
    """
    Calculate the displacement of the chest based on the positions of the pelvis and middle spine joint.
    Args:
        time_slice (list): A list of dictionaries containing joint data for each frame.
        person_tracker (int): The index of the person to track.
        joint_name (str): The name of the joint to calculate displacement for.
        x (bool): Whether to include x-axis displacement.
        y (bool): Whether to include y-axis displacement.
        z (bool): Whether to include z-axis displacement.
    Returns:
        array: A unit vector representing the displacement of the chest.
    

    Notes:
    I will use both spine2 and pelvis to check for walking, with a high velocity boundary for spine2 and
    a moderate boundary for the pelvis in order to check for sitting.
    
    To check for quickly standing, turning, or crouching, I will also check in the x and z directions 
    compared to the y direction specifically.

    To cases I hope to discard to isolate walking include:
        Bending over, Stretching, Reaching over, etc. in a chair
        Standing up quickly from a chair, crouching, creating quick vertical movement without walking
    """
    #velocity threshould should be 0.03 axis units per frame
    if (not x and not y and not z):
        return None
    displacements = []
    if (person_tracker not in time_slice[0]['trackers']):  
        print ("Person not found in time slice 1")
        return None
    index0 = time_slice[0]['trackers'].index(person_tracker)
    frame0 = time_slice[0]['joints3d'].cpu().numpy()
    previousJoint = frame0[index0][JOINT_NAMES.index(joint_name)]
    for i in range(1, len(time_slice)):
        frameI = time_slice[i]['joints3d'].cpu().numpy()
        if (person_tracker not in time_slice[i]['trackers']):
            print ("Person tracker not in time slice " + str(i + 1))
            return displacements
        indexI = time_slice[i]['trackers'].index(person_tracker)
        currentJoint = frameI[indexI][JOINT_NAMES.index(joint_name)]
        displacement = distXYZ(previousJoint, currentJoint, x, y, z)

        #print (displacement, previousJoint, currentJoint)

        displacements.append(displacement)
        previousJoint = currentJoint
    return displacements
def calculate_walking_frames(time_slice, person_tracker, velocity_threshold_spine = 0.05, velocity_threshold_pelvis):
    """
    Calculate the frames where the person is walking based on the displacement of the chest.
    Args:
        time_slice (list): A list of dictionaries containing joint data for each frame.
        person_tracker (int): The index of the person to track.
        velocity_threshold (float): The threshold for considering a frame as walking.
        joint_name (str): The name of the joint to calculate displacement for.
        x (bool): Whether to include x-axis displacement.
        y (bool): Whether to include y-axis displacement.
        z (bool): Whether to include z-axis displacement.

    Returns:
        list: A list of indices representing the frames where the person is walking.
    """
    displacementsSpine = calculate_joint_displacement(time_slice, person_tracker, 'spine2', x = True, y = True, z = False)
    #print (*displacements, sep = "\n")
    displacementsPelvis = calculate_joint_displacement(time_slice, person_tracker, 'pelvis', x = True, y = True, z = False)
    if (displacementsSpine == None):
        return None
    if (displacementsPelvis == None):
        return None
    walking_frames = [i for i in range(len(displacementsSpine)) if (displacementsSpine[i] > velocity_threshold_spine and displacementsPelvis[i] > velocity_threshold)]
    return walking_frames
# Example usage:
def tester():
    time_start = time.time()

    collision_events, total_collisions = execute_collision_detection(dataset = "/Users/jamesignacio/Downloads/HMR Test Folder 2/joint_out/",
                                                                    start_frame='frame_000000.pkl',
                                                                    end_frame='frame_049640.pkl',
                                                                    step=1,
                                                                    radius=0.25, #Shoulder-to-shoulder 
                                                                    v_thres=2)

    time_end = time.time()
    print(f"Execution time: {time_end - time_start} seconds")
    print(collision_events)
    print(f"Total Collisions: {total_collisions}")

def pipeline_tester():
    dataset = get_time_slice(0, 0, "/pasteur/data/ghent_surg/full_hmr_outputs/220610_22011_test_5000/joint_out/", debug=True)
    dataset = preprocess(dataset)
    collision_events, total_collisions, frame_collisions = detect_collisions(dataset, 0.25, 2) 
    print(len(collision_events))
    print(total_collisions)
    print(frame_collisions)
    breakpoint()

if __name__ == "__main__":
    pipeline_tester()
