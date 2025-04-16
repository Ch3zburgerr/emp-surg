import os
import io
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional, DefaultDict, Tuple
from collections import defaultdict

# === CONFIG ===
JOINT_DIR = "joint_out/"
START_FRAME = 0        # frame_000000.pkl
END_FRAME = 49640      # frame_049640.pkl
FRAME_INTERVAL = 10    # Process every 10th frame
OUTPUT_FILE = "attention_switches.txt"
VISUALIZATION_DIR = "attention_switches_visualizations/"
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Visualization parameters
SKELETON_COLOR = 'lightgray'
ARROW_ALPHA = 0.8
HEAD_ARROW_COLORS = [
    '#FF0000',  # Red
    '#0000FF',  # Blue
    '#FF00FF',  # Magenta
    '#00FFFF',  # Cyan
    '#FFA500',  # Orange
    '#800080',  # Purple
    '#8B0000',  # Dark Red
    '#000080'   # Navy
]

# Head orientation parameters
HEAD_ANGLE_CHANGE_THRESHOLD = 30  # degrees

# SMPL joint indices used for attention detection
HEAD_JOINT_INDICES = {
    'neck': 12,     # Base of head
    'nose': 23,     # Forward direction
    'left_eye': 24, # Used for head plane calculation
    'right_eye': 25,
}

# Head connections for visualization
HEAD_CONNECTIONS = [
    ('neck', 'nose'),
    ('nose', 'left_eye'),
    ('nose', 'right_eye'),
    ('left_eye', 'right_eye')
]

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def load_pickle(file_path: str) -> Optional[Dict]:
    try:
        with open(file_path, 'rb') as f:
            return CPU_Unpickler(f).load()
    except Exception as e:
        print(f"Failed to load {os.path.basename(file_path)}: {str(e)}")
        return None

def extract_head_orientation(frame_data: Dict) -> List[Dict[str, np.ndarray]]:
    if not isinstance(frame_data, dict) or 'joints3d' not in frame_data:
        return []
    
    joints3d = frame_data['joints3d']
    if isinstance(joints3d, torch.Tensor):
        joints3d = joints3d.cpu().numpy()
    
    return [{joint: person_joints[idx] for joint, idx in HEAD_JOINT_INDICES.items()}
            for person_joints in joints3d]

def compute_head_direction(head_joints: Dict[str, np.ndarray]) -> np.ndarray:
    """Calculate head forward direction using nose-neck vector and eye plane"""
    head_vec = head_joints['nose'] - head_joints['neck']
    eye_vec = head_joints['right_eye'] - head_joints['left_eye']
    head_normal = np.cross(head_vec, eye_vec)
    return head_normal / (np.linalg.norm(head_normal) + 1e-8)

def visualize_head_switch(frame_num: int, 
                        people: List[Dict[str, np.ndarray]], 
                        tracker: 'HeadOrientationTracker'):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot head skeletons
    for person_id, head_joints in enumerate(people):
        color = HEAD_ARROW_COLORS[person_id % len(HEAD_ARROW_COLORS)]
        
        for joint1, joint2 in HEAD_CONNECTIONS:
            if joint1 in head_joints and joint2 in head_joints:
                ax.plot(*zip(head_joints[joint1], head_joints[joint2]), 
                       color=SKELETON_COLOR, linewidth=1.5, alpha=0.7)
        
        ax.text(*head_joints['neck'], 'neck', color='black', fontsize=8)
        ax.text(*head_joints['nose'], 'nose', color='black', fontsize=8)

    # Draw head direction arrows
    info_text = []
    for person_id, head_joints in enumerate(people):
        color = HEAD_ARROW_COLORS[person_id % len(HEAD_ARROW_COLORS)]
        current_dir = compute_head_direction(head_joints)
        
        if tracker.previous_directions[person_id] is not None:
            prev_dir = tracker.previous_directions[person_id]
            ax.quiver(*head_joints['neck'], *prev_dir, length=0.2,
                     color=color, arrow_length_ratio=0.1, linestyle=':',
                     linewidth=3, alpha=ARROW_ALPHA, label=f'P{person_id} Previous')
            
            dot_product = np.dot(current_dir, prev_dir)
            angle_change = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
            info_text.append(f"P{person_id}: Angle change = {angle_change:.1f}°")
        
        ax.quiver(*head_joints['neck'], *current_dir, length=0.25,
                 color=color, arrow_length_ratio=0.15, linewidth=4,
                 alpha=ARROW_ALPHA, label=f'P{person_id} Current')
        
        info_text.append(f"P{person_id}: Total switches = {tracker.switch_counts[person_id]}")

    ax.text2D(0.05, 0.95, "\n".join(info_text), transform=ax.transAxes,
              bbox=dict(facecolor='white', alpha=0.8))
    
    ax.set_title(f"Frame {frame_num}\nHead Orientation Switch Detection", pad=20)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend(loc='upper right')
    plt.tight_layout()
    
    vis_path = os.path.join(VISUALIZATION_DIR, f"frame_{frame_num:06d}_switch.png")
    plt.savefig(vis_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved switch visualization: {vis_path}")

class HeadOrientationTracker:
    def __init__(self):
        self.switch_counts = defaultdict(int)
        self.previous_directions = defaultdict(lambda: None)
        self.last_directions = defaultdict(lambda: None)
    
    def update(self, person_id: int, head_dir: np.ndarray, frame_num: int) -> bool:
        switch_detected = False
        
        if self.last_directions[person_id] is not None:
            # Calculate angle change from last frame
            dot_product = np.dot(head_dir, self.last_directions[person_id])
            angle_change = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
            
            if angle_change > HEAD_ANGLE_CHANGE_THRESHOLD:
                self.previous_directions[person_id] = self.last_directions[person_id]
                self.switch_counts[person_id] += 1
                switch_detected = True
        
        self.last_directions[person_id] = head_dir
        return switch_detected

def main():
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    
    tracker = HeadOrientationTracker()
    processed_count = 0
    missing_count = 0
    switch_visualization_count = 0

    with open(OUTPUT_FILE, 'w') as f:
        f.write("Frame\tTotalSwitches\tPersonDetails\n")
        
        for frame_num in range(START_FRAME, END_FRAME + 1, FRAME_INTERVAL):
            frame_name = f"frame_{frame_num:06d}.pkl"
            frame_path = os.path.join(JOINT_DIR, frame_name)
            
            if not os.path.exists(frame_path):
                missing_count += 1
                continue
            
            frame_data = load_pickle(frame_path)
            if frame_data is None:
                missing_count += 1
                continue
            
            people = extract_head_orientation(frame_data)
            if not people:
                continue
                
            switch_in_frame = False
            for person_id, head_joints in enumerate(people):
                head_dir = compute_head_direction(head_joints)
                if tracker.update(person_id, head_dir, frame_num):
                    switch_in_frame = True
            
            if switch_in_frame:
                visualize_head_switch(frame_num, people, tracker)
                switch_visualization_count += 1
            
            frame_switches = sum(1 for p in range(len(people)) 
                              if tracker.switch_counts[p] > 0 and 
                              tracker.last_directions[p] is not None)
            switchers = [f"P{pid}({tracker.switch_counts[pid]})" 
                        for pid in range(len(people)) 
                        if tracker.switch_counts[pid] > 0]
            
            f.write(f"{frame_num}\t{frame_switches}\t{'|'.join(switchers) if switchers else '-'}\n")
            processed_count += 1
            
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} frames...")

        summary = [
            "\n=== SUMMARY ===",
            f"Total frames processed: {processed_count}",
            f"Total attention switches: {sum(tracker.switch_counts.values())}",
            f"Switch visualizations created: {switch_visualization_count}",
            "Per-person switch counts:"
        ] + [f"Person {pid}: {count} switches" for pid, count in sorted(tracker.switch_counts.items())]
        
        f.write("\n".join(summary))

    print(f"\nProcessing complete.")
    print(f"Text results saved to: {OUTPUT_FILE}")
    print(f"Visualizations saved to: {VISUALIZATION_DIR}")
    print(f"Frames processed: {processed_count} (missing: {missing_count})")
    print(f"Switch visualizations created: {switch_visualization_count}")

if __name__ == "__main__":
    main()