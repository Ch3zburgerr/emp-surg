import os
from util import get_time_slice, preprocess
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from joints import JOINT_NAMES

PELVIS_INDEX = JOINT_NAMES.index('pelvis')
LEFT_SHOULDER_INDEX = JOINT_NAMES.index('left_shoulder')
RIGHT_SHOULDER_INDEX = JOINT_NAMES.index('right_shoulder')
SPINE2_INDEX = JOINT_NAMES.index('spine2')

def calculate_chest_facing_direction(pelvis, left_shoulder, right_shoulder, spine2):
    """
    calculates the direction the chest is facing
    """
    # Calculate the vector between the shoulders
    shoulder_vector = right_shoulder - left_shoulder
    # Calculate the vector from the pelvis to the middle spine joint
    spine_vector = spine2 - pelvis
    #cross the two vectors (find the vector perpendicular to both vectors)
    facing_direction = np.cross(shoulder_vector, spine_vector)
    # Normalize the vector to get the direction
    facing_direction = facing_direction / np.linalg.norm(facing_direction)
    
    return facing_direction
def calculate_walking_directions(time_slice, start, end, full = False):
    """
    Calculates the walking directions for a group of people in a given time slice
    The walking directions are defined as the deviation between chest facing directions between adjacent frames.
    """
    start = max(0, start)
    end = min(len(time_slice), end)
    if (full):
        start = 0
        end = len(time_slice)
    walking_directions = [[] for _ in range(len(time_slice))]
    prev_facing_direction = {}
    for i in range(start, end):
        num_people = len(time_slice[i]['trackers'])
        for j in range(num_people):
            joints = time_slice[i]['joints3d'][j]
            pelvis = joints[PELVIS_INDEX]
            left_shoulder = joints[LEFT_SHOULDER_INDEX]
            right_shoulder = joints[RIGHT_SHOULDER_INDEX]
            spine2 = joints[SPINE2_INDEX]
            tracker = time_slice[i]['trackers'][j]
            # Calculate the chest facing direction
            current_facing_direction = calculate_chest_facing_direction(pelvis, left_shoulder, right_shoulder, spine2)
            
            if (tracker not in prev_facing_direction):
                prev_facing_direction[tracker] = current_facing_direction
                walking_directions[i].append([-1])
                #even if there is no previous frame/tracker/anything, we still append -1 so that the trackers work in indexing
                continue
            # Store the facing direction in the time slice
            walking_directions[i].append((current_facing_direction + prev_facing_direction[tracker]) / 2)
            prev_facing_direction[tracker] = current_facing_direction
        keys = list(prev_facing_direction.keys())
        for key in keys:
            if (key not in time_slice[i]['trackers']):
                prev_facing_direction.pop(key)
    return walking_directions
def dist3D(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)
def distXY(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
def calculate_joint_displacementsXY(time_slice, start, end, joint_name = 'spine2', full = False):
    #calculates the displacement of a joint in the XY plane (not up or down, just right/left/forward/backward)
    joint_ind = JOINT_NAMES.index(joint_name)
    start = max(0, start)
    end = min(len(time_slice), end)
    if (full):
        start = 0
        end = len(time_slice)
    joint_displacements = [[] for _ in range(len(time_slice))]
    prev_joints = {}
    for i in range(start, end):
        num_people = len(time_slice[i]['trackers'])
        for j in range(num_people):
            joints = time_slice[i]['joints3d'][j]
            joint = joints[joint_ind]
            tracker = time_slice[i]['trackers'][j]
            if (tracker not in prev_joints):
                prev_joints[tracker] = joint
                joint_displacements[i].append([-1])
                continue
            joint_displacements[i].append((joint + prev_joints[tracker]) / 2)
            prev_joints[tracker] = joint
        keys = list(prev_joints.keys())  
        for key in keys:
            if (key not in time_slice[i]['trackers']):
                prev_joints.pop(key)
    return joint_displacements
def calculate_velocity(displacement_vector):
    return np.sqrt(displacement_vector[0]**2 + displacement_vector[1]**2)
def group_walk(time_slice, start, end, full = False, spine_threshold = 3.3, pelvis_threshold = 0.5):
    start = max(0, start)
    end = min(len(time_slice), end)
    if (full):
        start = 0
        end = len(time_slice)
    group_walks = [[] for _ in range(len(time_slice))]
    pelvis_displacements = calculate_joint_displacementsXY(time_slice, start, end, joint_name = 'pelvis', full = full)
    spine_displacements = calculate_joint_displacementsXY(time_slice, start, end, joint_name = 'spine2', full = full)
    #walking_directions = calculate_walking_directions(time_slice, start, end, full = full)
    for i in range(start, end):
        num_people = len(time_slice[i]['trackers'])
        for j in range(num_people):
            spine_d = spine_displacements[i][j]
            pelvis_d = pelvis_displacements[i][j]
            if (spine_d[0] == -1 or pelvis_d[0] == -1):
                group_walks[i].append(-1)
                continue
            spine_velocity = calculate_velocity(spine_d)
            pelvis_velocity = calculate_velocity(pelvis_d)
            group_walks[i].append(spine_velocity if (spine_velocity > spine_threshold and pelvis_velocity > pelvis_threshold) else 0)

    return group_walks
            
def main():
    folder_link = "D:\\Coding\\data\\joint_out"
    if (os.path.exists(folder_link)):
        time_slice = get_time_slice("frame_000000.pkl", "frame_000100.pkl", folder_link, step=1, full=True, debug=False)        
        time_slice = preprocess(time_slice)
        f = open("group_walk_results.txt", "w")
        f.truncate(0)
        f.write("Group Walk Results: \n")
        f.write("The Results are in this format: Each frame is either a -1 if that person did not appear in the frevious frame, \n")
        f.write("a 0 if they are not walking, and a decimal representing velocity followed by a 3d vector showing direction\n")
        f.write("The order of numbers of each frame matches with the order of the trackers in each frame.\n")
        group_walks = group_walk(time_slice, 0, len(time_slice), full = True)
        trajectories = calculate_walking_directions(time_slice, 0, len(time_slice), full = True)
        for i in range(len(group_walks)):
            f.write("Frame " + str(i + 1) + ": ")
            ind = 0
            upTo = len(group_walks[i]) - 1
            for veloc in group_walks[i]:
                f.write(str(veloc))
                if (veloc != 0 and trajectories[i][ind][0] != -1):
                    f.write(' (' + str(trajectories[i][ind][0]) + ' ' + str(trajectories[i][ind][1]) + ' ' + str(trajectories[i][ind][2]) + ')')
                if (ind != upTo):
                    f.write(", ")
                ind += 1
            f.write("\n")
            
        f.close()
    else:
        print ("Folder does not exist")
if __name__ ==  "__main__":
    main()
