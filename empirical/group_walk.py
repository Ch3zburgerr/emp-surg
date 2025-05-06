import os
from util import get_time_slice, preprocess
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from joints import JOINT_NAMES
from test_pipeline import find_extremes

PELVIS_INDEX = JOINT_NAMES.index('pelvis')
LEFT_SHOULDER_INDEX = JOINT_NAMES.index('left_shoulder')
RIGHT_SHOULDER_INDEX = JOINT_NAMES.index('right_shoulder')
SPINE2_INDEX = JOINT_NAMES.index('spine2')
def rotation_matrix(axis, theta):
    """
    Compute the rotation matrix for a given axis and angle.
    Args:
        axis (array): A 3-element array representing the axis of rotation (must be a unit vector).
        theta (float): Rotation angle in radians.

    Returns:
        ndarray: A 3x3 rotation matrix.
    """
    axis = axis / np.linalg.norm(axis)  # Ensure the axis is a unit vector
    ux, uy, uz = axis
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    R = np.array([
        [cos_theta + ux**2 * (1 - cos_theta),
         ux*uy*(1 - cos_theta) - uz*sin_theta,
         ux*uz*(1 - cos_theta) + uy*sin_theta],

        [uy*ux*(1 - cos_theta) + uz*sin_theta,
         cos_theta + uy**2 * (1 - cos_theta),
         uy*uz*(1 - cos_theta) - ux*sin_theta],

        [uz*ux*(1 - cos_theta) - uy*sin_theta,
         uz*uy*(1 - cos_theta) + ux*sin_theta,
         cos_theta + uz**2 * (1 - cos_theta)]
    ])
    return R
def find_extremes(time_slice, use_vertices = False):
    x_min, x_max, y_min, y_max, z_min, z_max = float('inf'), float('-inf'), float('inf'), float('-inf'), float('inf'), float('-inf')
    for i in range(len(time_slice)):
        joints = time_slice[i]['joints3d']
        numPeople = len(time_slice[i]['trackers'])
        for person in range(numPeople):
            for joint in joints[person]:
                x_min = min(x_min, joint[0])
                x_max = max(x_max, joint[0])
                y_min = min(y_min, joint[2])
                y_max = max(y_max, joint[2])
                z_min = min(z_min, joint[1])
                z_max = max(z_max, joint[1])
        if (use_vertices):
            vertices = time_slice[i]['vertices']
            for person in range(numPeople):
                for vertex in vertices[person]:
                    x_min = min(x_min, vertex[0])
                    x_max = max(x_max, vertex[0])
                    y_min = min(y_min, vertex[2])
                    y_max = max(y_max, vertex[2])
                    z_min = min(z_min, vertex[1])
                    z_max = max(z_max, vertex[1])
    return x_min - 1, x_max + 1, y_min, y_max, z_min, z_max
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
            joint_displacements[i].append(prev_joints[tracker] - joint)
            prev_joints[tracker] = joint
        keys = list(prev_joints.keys())  
        for key in keys:
            if (key not in time_slice[i]['trackers']):
                prev_joints.pop(key)
    return joint_displacements
def calculate_velocity(displacement_vector):
    return np.sqrt(displacement_vector[0]**2 + displacement_vector[1]**2)
def group_walk(time_slice, start, end, full = False, threshold = True, spine_threshold = 0.26, pelvis_threshold = 0.2):
    f = open("group_walk_other_stuff.txt", "w")
    f.truncate(0)
    f.write("Group Walk Results: \n")
    f.write("The Results are in this format: Each frame is either a -1 if that person did not appear in the frevious frame, \n")
    f.write("a 0 if they are not walking, and a decimal representing velocity followed by a 3d vector showing direction\n")
    f.write("The order of numbers of each frame matches with the order of the trackers in each frame.\n")
       
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
        f.write("Frame " + str(i + 1) + ": \n")
        for j in range(num_people):
            spine_d = spine_displacements[i][j]
            pelvis_d = pelvis_displacements[i][j]
            if (spine_d[0] == -1 or pelvis_d[0] == -1):
                f.write("-1 ")
                group_walks[i].append(-1)
                continue
            spine_velocity = calculate_velocity(spine_d)
            pelvis_velocity = calculate_velocity(pelvis_d)
            f.write(str(spine_velocity) + " " + str(pelvis_velocity) + "\n")
            if (threshold):
                if (spine_velocity > spine_threshold and pelvis_velocity > pelvis_threshold):
                    group_walks[i].append(spine_velocity)
                else:
                    group_walks[i].append(0)
            else:
                group_walks[i].append(spine_velocity)
        f.write("\n")
    return group_walks
def group_walk_plot(time_slice, start, end, frame_start_num, group_walks, full = False): 
    start = max(0, start)
    end = min(len(time_slice), end)
    if (full):
        start = 0
        end = len(time_slice)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_min, x_max, y_min, y_max, z_min, z_max = find_extremes(time_slice)
    for i in range(start, end):
        numPeople = len(time_slice[i]['trackers'])
        joints = time_slice[i]['joints3d']
        vertices = time_slice[i]['vertices']
        tracker = time_slice[i]['trackers']
        rotation_matrix180 = np.array([[-1,0,0],[0,-1,0], [0,0,-1]])
        xz_matrix = rotation_matrix(np.array([0,1,0]), np.radians(10))
        yz_matrix = rotation_matrix(np.array([1,0,0]), np.radians(-75))
        matrix_final = np.dot(rotation_matrix180, xz_matrix)
        matrix_final = np.dot(matrix_final, yz_matrix)
        personNjoints3d = [np.dot(joints[index], matrix_final) for index in range(numPeople)]
        personNvertices = [np.dot(vertices[index], matrix_final) for index in range(len(vertices))]
        xJoints = np.array(personNjoints3d)[:, :, 0]
        yJoints = np.array(personNjoints3d)[:, :, 1]
        zJoints = np.array(personNjoints3d)[:, :, 2]
        xVertices = [personNvertices[index][:,0] for index in range(0, len(personNvertices))]
        yVertices = [personNvertices[index][:,1] for index in range(0, len(personNvertices))]
        zVertices = [personNvertices[index][:,2] for index in range(0, len(personNvertices))]
        xVertices = np.array([xVertices[i][::17] for i in range(len(xVertices))])
        yVertices = np.array([yVertices[i][::17] for i in range(len(yVertices))])
        zVertices = np.array([zVertices[i][::17] for i in range(len(zVertices))])
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])
        for person in range(numPeople):
            ax.scatter(xJoints[person], yJoints[person], zJoints[person])
            ax.scatter(xVertices[person], yVertices[person], zVertices[person])
            spine2 = personNjoints3d[person][SPINE2_INDEX]
            pelvis = personNjoints3d[person][PELVIS_INDEX]
            left_shoulder = personNjoints3d[person][LEFT_SHOULDER_INDEX]
            right_shoulder = personNjoints3d[person][RIGHT_SHOULDER_INDEX]
            # Calculate the chest facing direction
            current_facing_direction = calculate_chest_facing_direction(pelvis, left_shoulder, right_shoulder, spine2)
            ax.quiver(spine2[0], spine2[1], spine2[2], current_facing_direction[0], current_facing_direction[1], current_facing_direction[2], color='b', length=0.5, normalize=True, label = 'Facing Direction ' + str(tracker[person]))
            if (group_walks[i][person] == -1 or group_walks[i][person] == 0):
                fig.text(0, 0.20 - person * 0.05, "Tracker " + str(tracker[person]) + " is not walking")
            else:
                fig.text(0, 0.20 - person * 0.05, "Tracker " + str(tracker[person]) + ": " + str(group_walks[i][person]))
        #ax.quiver(spine2[0], spine2[1], spine2[2], current_facing_direction[0], current_facing_direction[1], current_facing_direction[2], color='r', length=0.5, normalize=True, label = 'Facing Direction')
        ax.view_init(elev = 20, azim = 45)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        frame_start_num += 1
        plt.title('Frame ' + str(frame_start_num))
        plt.savefig('frame_' + str(frame_start_num) + '.png')
        plt.cla()
        fig.texts.clear()
def main():
    folder_link = "D:\\Coding\\data\\joint_out"
    if (os.path.exists(folder_link)):
        isFull = False
        time_slice = get_time_slice("frame_000000.pkl", "frame_002000.pkl", folder_link, step=1, full=isFull, debug=False)        
        time_slice = preprocess(time_slice)
        #f = open("group_walk_results.txt", "w")
        #f.truncate(0)
        #f.write("Group Walk Results: \n")
        #f.write("The Results are in this format: Each frame is either a -1 if that person did not appear in the frevious frame, \n")
        #f.write("a 0 if they are not walking, and a decimal representing velocity followed by a 3d vector showing direction\n")
        #f.write("The order of numbers of each frame matches with the order of the trackers in each frame.\n")
        group_walks = group_walk(time_slice, 0, len(time_slice), threshold = True, full = isFull)
        #print (*group_walks, sep = "\n")
        #trajectories = calculate_walking_directions(time_slice, 0, len(time_slice), full = isFull)
        group_walk_plot(time_slice, 0, len(time_slice), 1, group_walks, full = isFull)
        #for i in range(len(group_walks)):
        #    f.write("Frame " + str(i + 1) + ": ")
        #    ind = 0
        #    upTo = len(group_walks[i]) - 1
        #    for veloc in group_walks[i]:
        #        f.write(str(veloc))
        #        if (veloc != 0 and trajectories[i][ind][0] != -1):
        #            f.write(' (' + str(trajectories[i][ind][0]) + ' ' + str(trajectories[i][ind][1]) + ' ' + str(trajectories[i][ind][2]) + ')')
        #        if (ind != upTo):
        #            f.write(", ")
        #        ind += 1
        #    f.write("\n")
            
        #f.close()
    else:
        print ("Folder does not exist")
if __name__ ==  "__main__":
    main()
