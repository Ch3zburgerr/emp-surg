import os
import io
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional, Tuple

# === CONFIG ===
JOINT_DIR = "joint_out/"
START_FRAME = "frame_000000.pkl"
END_FRAME = "frame_049640.pkl"
BATCH_SIZE = 10
OUTPUT_FILE = "facing_pairs.txt"
VISUALIZATION_DIR = "facing_pairs_visualizations/"
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Color setup
SKELETON_COLOR = 'limegreen'
ARROW_COLORS = [
    '#FF0000',  # Red
    '#0000FF',  # Blue
    '#FF00FF',  # Magenta
    '#00FFFF',  # Cyan
    '#FFA500',  # Orange
    '#800080',  # Purple
    '#8B0000',  # Dark Red
    '#000080'   # Navy
]

# SMPL joint indices and connections
SMPL_JOINT_INDICES = {
    'pelvis': 0, 'left_hip': 1, 'right_hip': 2,
    'spine1': 3, 'left_knee': 4, 'right_knee': 5,
    'spine2': 6, 'left_ankle': 7, 'right_ankle': 8,
    'spine3': 9, 'left_foot': 10, 'right_foot': 11,
    'neck': 12, 'left_collar': 13, 'right_collar': 14,
    'head': 15, 'left_shoulder': 16, 'right_shoulder': 17,
    'left_elbow': 18, 'right_elbow': 19,
    'left_wrist': 20, 'right_wrist': 21,
    'jaw': 22, 'nose': 23, 'left_eye': 24,
    'right_eye': 25, 'left_ear': 26, 'right_ear': 27
}

BODY_CONNECTIONS = [
    ('pelvis', 'left_hip'), ('pelvis', 'right_hip'),
    ('left_hip', 'left_knee'), ('right_hip', 'right_knee'),
    ('left_knee', 'left_ankle'), ('right_knee', 'right_ankle'),
    ('pelvis', 'spine1'), ('spine1', 'spine2'), ('spine2', 'spine3'),
    ('spine3', 'neck'), ('neck', 'head'), ('head', 'nose'),
    ('neck', 'left_shoulder'), ('neck', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'), ('right_shoulder', 'right_elbow'),
    ('left_elbow', 'left_wrist'), ('right_elbow', 'right_wrist'),
    ('left_eye', 'right_eye'),
    ('left_eye', 'nose'),
    ('right_eye', 'nose')
]

ESSENTIAL_JOINTS = ['pelvis', 'left_shoulder', 'right_shoulder', 
                   'left_hip', 'right_hip', 'neck', 'left_eye', 'right_eye']

# === CORE FUNCTIONS ===
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=torch.device('cpu'))
        return super().find_class(module, name)

def load_pickle(file_path: str) -> Optional[Dict]:
    try:
        with open(file_path, 'rb') as f:
            data = CPU_Unpickler(f).load()
            if isinstance(data, dict):
                # Convert all tensors to numpy arrays immediately
                for key in data:
                    if isinstance(data[key], torch.Tensor):
                        data[key] = data[key].detach().cpu().numpy()
            return data
    except Exception as e:
        print(f"Failed to load {os.path.basename(file_path)}: {str(e)}")
        return None

def apply_transform(data, flip_yz=False):
    # Ensure all inputs are torch tensors
    vertices = torch.from_numpy(data['vertices']) if isinstance(data['vertices'], np.ndarray) else data['vertices']
    joints3d = torch.from_numpy(data['joints3d']) if isinstance(data['joints3d'], np.ndarray) else data['joints3d']
    pred_cam_t = torch.from_numpy(data['pred_cam_t']) if isinstance(data['pred_cam_t'], np.ndarray) else data['pred_cam_t']
    
    vertices = (vertices + pred_cam_t.unsqueeze(1)).detach().cpu().numpy()
    joints = (joints3d + pred_cam_t.unsqueeze(1)).detach().cpu().numpy()
    if flip_yz:
        vertices = vertices[..., [0, 2, 1]]
        joints = joints[..., [0, 2, 1]]
    return joints, vertices

def extract_people_joints(frame_data: Dict) -> List[Dict[str, np.ndarray]]:
    if not isinstance(frame_data, dict) or 'joints3d' not in frame_data:
        return []
    
    # Apply transform to the data before processing
    joints3d, _ = apply_transform(frame_data, flip_yz=True)
    
    people = []
    for person_joints in joints3d:
        person_data = {}
        for joint in ESSENTIAL_JOINTS:
            idx = SMPL_JOINT_INDICES[joint]
            if idx < len(person_joints):
                person_data[joint] = person_joints[idx]
            else:
                print(f"Joint {joint} (index {idx}) out of bounds")
                break
        else:  # Only add if all joints were found
            people.append(person_data)
    return people

def compute_gaze_vector(joints: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    L = joints['left_eye']
    R = joints['right_eye']
    N = joints['neck']
    mu = (L + R) / 2  # Midpoint between eyes
    initial_gaze = mu - N
    eye_vector = L - R
    plane_normal = np.cross(eye_vector, initial_gaze)
    plane_normal = plane_normal / (np.linalg.norm(plane_normal) + 1e-8)
    corrected_gaze = initial_gaze - np.dot(initial_gaze, plane_normal) * plane_normal
    corrected_gaze = corrected_gaze / (np.linalg.norm(corrected_gaze) + 1e-8)
    return mu, corrected_gaze

def calculate_mutual_angles(p1_forward: np.ndarray, p1_pos: np.ndarray, 
                           p2_forward: np.ndarray, p2_pos: np.ndarray) -> Tuple[float, float, float]:
    p1_to_p2 = (p2_pos - p1_pos) / (np.linalg.norm(p2_pos - p1_pos) + 1e-8)
    angle1 = np.degrees(np.arccos(np.clip(np.dot(p1_forward, p1_to_p2), -1.0, 1.0)))
    angle2 = np.degrees(np.arccos(np.clip(np.dot(p2_forward, -p1_to_p2), -1.0, 1.0)))
    angle_between = np.degrees(np.arccos(np.clip(np.dot(p1_forward, p2_forward), -1.0, 1.0)))
    return angle1, angle2, angle_between

def visualize_interaction(frame_name: str, people_joints: List[Dict[str, np.ndarray]], facing_pairs: List[Tuple[int, int]]):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot skeletons and joint labels
    for person in people_joints:
        for joint1, joint2 in BODY_CONNECTIONS:
            if joint1 in person and joint2 in person:
                ax.plot(*zip(person[joint1], person[joint2]), color=SKELETON_COLOR, linewidth=2, alpha=0.7)
        
        for joint in ['pelvis', 'neck', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 'left_eye', 'right_eye']:
            if joint in person:
                ax.text(*person[joint], joint, color='darkgreen', fontsize=8)

    angle_info = []
    for i, person in enumerate(people_joints):
        _, gaze_vector = compute_gaze_vector(person)
        pos = person['neck']
        color = ARROW_COLORS[i % len(ARROW_COLORS)]
        
        lw = 3 if any(i in pair for pair in facing_pairs) else 2
        ax.quiver(*pos, *gaze_vector, length=0.3, color=color, 
                 arrow_length_ratio=0.15, linewidth=lw, label=f'Person {i}')
        angle_info.append(f"P{i}: Gaze direction")

    for (i, j) in facing_pairs:
        _, p1_gaze = compute_gaze_vector(people_joints[i])
        _, p2_gaze = compute_gaze_vector(people_joints[j])
        p1_pos = people_joints[i]['neck']
        p2_pos = people_joints[j]['neck']
        
        angle1, angle2, angle_between = calculate_mutual_angles(p1_gaze, p1_pos, p2_gaze, p2_pos)
        
        ax.plot(*zip(p1_pos, p2_pos), '--', color='purple', linewidth=3, alpha=0.5)
        mid_point = (p1_pos + p2_pos) / 2
        ax.text(*mid_point, f"{angle_between:.1f}°", 
               color='black', fontsize=10,
               bbox=dict(facecolor='white', alpha=0.7))

        angle_info.append(f"\nFacing Pair ({i},{j}):")
        angle_info.append(f"  P{i}→P{j}: {angle1:.1f}°")
        angle_info.append(f"  P{j}→P{i}: {angle2:.1f}°")
        angle_info.append(f"  Gaze Angle: {angle_between:.1f}°")

    ax.text2D(0.05, 0.95, "\n".join(angle_info), transform=ax.transAxes,
              bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    ax.set_title(f"Frame {frame_name}\nFacing Pairs: {facing_pairs}", pad=20)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATION_DIR}/{frame_name.replace('.pkl', '.png')}", dpi=120, bbox_inches='tight')
    plt.close()

def process_frame(frame_path: str, output_file: str) -> bool:
    frame_name = os.path.basename(frame_path)
    frame_data = load_pickle(frame_path)
    if not frame_data:
        return False
    
    people_joints = extract_people_joints(frame_data)
    if len(people_joints) < 2:
        return False
    
    facing_pairs = []
    for i in range(len(people_joints)):
        for j in range(i+1, len(people_joints)):
            _, p1_gaze = compute_gaze_vector(people_joints[i])
            _, p2_gaze = compute_gaze_vector(people_joints[j])
            p1_pos = people_joints[i]['neck']
            p2_pos = people_joints[j]['neck']
            
            angle1, angle2, _ = calculate_mutual_angles(p1_gaze, p1_pos, p2_gaze, p2_pos)
            
            if angle1 < 45 and angle2 < 45:
                facing_pairs.append((i, j))
    
    if facing_pairs:
        with open(output_file, 'a') as f:
            f.write(f"Frame: {frame_name}\nFacing Pairs: {facing_pairs}\n\n")
        visualize_interaction(frame_name, people_joints, facing_pairs)
        return True
    return False

def main():
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    
    start_id = int(START_FRAME.split('_')[1].split('.')[0])
    end_id = int(END_FRAME.split('_')[1].split('.')[0])
    
    total_facing = 0
    for i in range(start_id, end_id + 1, BATCH_SIZE):
        frame_path = os.path.join(JOINT_DIR, f"frame_{i:06d}.pkl")
        if os.path.exists(frame_path) and process_frame(frame_path, OUTPUT_FILE):
            total_facing += 1
    
    with open(OUTPUT_FILE, 'a') as f:
        f.write(f"\n=== SUMMARY ===\nFrames with facing pairs: {total_facing}\n")
    
    print(f"Processing complete.\n- Text output: {OUTPUT_FILE}\n- Visualizations: {VISUALIZATION_DIR}")

if __name__ == "__main__":
    main()