import os
import io
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional
from collections import defaultdict

# === CONFIG ===
JOINT_DIR = "joint_out/"
START_FRAME = 0
END_FRAME = 49640
FRAME_INTERVAL = 10  # Process every 10th frame
OUTPUT_FILE = "elbow_bends.txt"
VISUALIZATION_DIR = "elbow_bends_visualizations/"
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Visualization settings - different color for each person (up to 8 people)
PERSON_COLORS = [
    '#FF0000', '#00AA00', '#0000FF', '#FF00FF',  # Red, Green, Blue, Magenta
    '#00AAAA', '#FFA500', '#800080', '#008080'   # Cyan, Orange, Purple, Teal
]
LEFT_STYLE = '--'       # Dashed for all left arms
RIGHT_STYLE = '-'       # Solid for all right arms
LINE_WIDTH = 2
JOINT_LABEL_SIZE = 8
ANGLE_LABEL_SIZE = 10

# SMPL joint indices
JOINT_INDICES = {
    'left_shoulder': 16,
    'left_elbow': 18,
    'left_wrist': 20,
    'right_shoulder': 17,
    'right_elbow': 19,
    'right_wrist': 21
}

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def load_pickle(file_path: str) -> Optional[Dict]:
    try:
        with open(file_path, 'rb') as f:
            data = CPU_Unpickler(f).load()
            if isinstance(data, dict):
                for key in data:
                    if isinstance(data[key], torch.Tensor):
                        data[key] = data[key].cpu().numpy()
            return data
    except Exception as e:
        print(f"Failed to load {os.path.basename(file_path)}: {str(e)}")
        return None

def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculate joint angle in degrees between vectors ba and bc"""
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def visualize_frame_elbows(frame_num: int, people_data: List[Dict]):
    """Create single 3D visualization showing all people's elbow bends for a frame"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set viewing angle for better visibility
    ax.view_init(elev=20, azim=45)
    
    for person_data in people_data:
        person_id = person_data['person_id']
        person_color = PERSON_COLORS[person_id % len(PERSON_COLORS)]
        person_joints = person_data['joints']
        
        # Extract joint positions
        ls = person_joints[JOINT_INDICES['left_shoulder']]
        le = person_joints[JOINT_INDICES['left_elbow']]
        lw = person_joints[JOINT_INDICES['left_wrist']]
        rs = person_joints[JOINT_INDICES['right_shoulder']]
        re = person_joints[JOINT_INDICES['right_elbow']]
        rw = person_joints[JOINT_INDICES['right_wrist']]
        
        # Plot left arm (dashed)
        ax.plot(*zip(ls, le, lw), color=person_color, linestyle=LEFT_STYLE,
                linewidth=LINE_WIDTH, alpha=0.8, 
                label=f'P{person_id} Left')
        
        # Plot right arm (solid)
        ax.plot(*zip(rs, re, rw), color=person_color, linestyle=RIGHT_STYLE,
                linewidth=LINE_WIDTH, alpha=0.8,
                label=f'P{person_id} Right')
        
        # Label joints
        for pos, label in [
            (ls, f'LS{person_id}'), (le, f'LE{person_id}'), (lw, f'LW{person_id}'),
            (rs, f'RS{person_id}'), (re, f'RE{person_id}'), (rw, f'RW{person_id}')
        ]:
            ax.text(*pos, label, color=person_color, 
                   fontsize=JOINT_LABEL_SIZE, ha='center', va='center')
        
        # Display angle text
        ax.text(*le, f"{person_data['left_elbow_angle']:.0f}°", 
               color=person_color, fontsize=ANGLE_LABEL_SIZE,
               ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        ax.text(*re, f"{person_data['right_elbow_angle']:.0f}°", 
               color=person_color, fontsize=ANGLE_LABEL_SIZE,
               ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    ax.set_title(f"Frame {frame_num}\nElbow Bend Angles (L=dashed, R=solid)", pad=20)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # Create custom legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    ax.legend(unique_handles, unique_labels, loc='upper right')
    
    plt.tight_layout()
    
    vis_path = os.path.join(VISUALIZATION_DIR, f"frame_{frame_num:06d}_elbows.png")
    plt.savefig(vis_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization: {vis_path}")

def process_frame(frame_data: Dict, frame_num: int) -> List[Dict]:
    """Process all people in a single frame"""
    if not frame_data or 'joints3d' not in frame_data:
        return []

    people_data = []
    joints = frame_data['joints3d']
    
    for person_id, person_joints in enumerate(joints):
        try:
            if isinstance(person_joints, torch.Tensor):
                person_joints = person_joints.cpu().numpy()
                
            ls = person_joints[JOINT_INDICES['left_shoulder']]
            le = person_joints[JOINT_INDICES['left_elbow']]
            lw = person_joints[JOINT_INDICES['left_wrist']]
            rs = person_joints[JOINT_INDICES['right_shoulder']]
            re = person_joints[JOINT_INDICES['right_elbow']]
            rw = person_joints[JOINT_INDICES['right_wrist']]
            
            people_data.append({
                'person_id': person_id,
                'joints': person_joints,
                'left_elbow_angle': calculate_angle(ls, le, lw),
                'right_elbow_angle': calculate_angle(rs, re, rw)
            })
            
        except Exception as e:
            print(f"Frame {frame_num} Person {person_id} error: {str(e)}")
    
    # Create single visualization for all people in this frame
    if people_data:
        visualize_frame_elbows(frame_num, people_data)
    
    return people_data

def main():
    # Write header with clear column labels
    with open(OUTPUT_FILE, 'w') as f:
        f.write(
            "FrameNumber,PersonID,LeftElbowAngle(degrees),RightElbowAngle(degrees)\n"
            "# FrameNumber: The frame number (e.g., 000010 = frame 10)\n"
            "# PersonID: Unique identifier for each person in the frame\n"
            "# Angles: Measured in degrees (0°=fully bent, 180°=fully extended)\n"
        )
    
    # Process every 10th frame
    for frame_num in range(START_FRAME, END_FRAME + 1, FRAME_INTERVAL):
        frame_name = f"frame_{frame_num:06d}.pkl"
        frame_path = os.path.join(JOINT_DIR, frame_name)
        
        if not os.path.exists(frame_path):
            continue
            
        frame_data = load_pickle(frame_path)
        if not frame_data:
            continue
            
        people_data = process_frame(frame_data, frame_num)
        
        # Append data with clear columns
        with open(OUTPUT_FILE, 'a') as f:
            for person_data in people_data:
                f.write(
                    f"{frame_num:06d},"
                    f"{person_data['person_id']},"
                    f"{person_data['left_elbow_angle']:.1f},"
                    f"{person_data['right_elbow_angle']:.1f}\n"
                )
        
        if frame_num % 100 == 0:  # Progress update every 100 frames
            print(f"Processed frame {frame_num} ({len(people_data)} people detected)")

if __name__ == "__main__":
    main()