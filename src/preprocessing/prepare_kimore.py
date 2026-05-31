"""
Preprocessing script for KIMORE dataset.
Loads the pkl file, extracts 3D joint positions,
splits into body parts, normalizes, and saves for training.
"""

import pickle
import numpy as np
import os

# 1. Load data
print("Loading KIMORE dataset...")
with open('data/kimore/kimore_exercise_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

# Joint columns (25 Kinect joints) - we'll use positions only
JOINT_NAMES = [
    'spinebase', 'spinemid', 'neck', 'head',
    'shoulderleft', 'elbowleft', 'wristleft', 'handleft',
    'shoulderright', 'elbowright', 'wristright', 'handright',
    'hipleft', 'kneeleft', 'ankleleft', 'footleft',
    'hipright', 'kneeright', 'ankleright', 'footright',
    'spineshoulder', 'handtipleft', 'thumbleft', 'handtipright', 'thumbright'
]

# Body part groupings (indices into JOINT_NAMES)
BODY_PARTS = {
    'trunk': ['spinebase', 'spinemid', 'neck', 'head', 'spineshoulder'],
    'left_arm': ['shoulderleft', 'elbowleft', 'wristleft', 'handleft', 'handtipleft', 'thumbleft'],
    'right_arm': ['shoulderright', 'elbowright', 'wristright', 'handright', 'handtipright', 'thumbright'],
    'left_leg': ['hipleft', 'kneeleft', 'ankleleft', 'footleft'],
    'right_leg': ['hipright', 'kneeright', 'ankleright', 'footright'],
}

# 2. Extract 3D positions from raw data
def extract_positions(joint_data):
    """
    Each joint has shape (n_frames, 7).
    Columns are likely: [qw, qx, qy, qz, x, y, z] or similar.
    We take the last 3 columns as (x, y, z) positions.
    """
    arr = np.array(joint_data)
    # Take last 3 columns as x, y, z
    return arr[:, -3:]


def normalize_skeleton(positions_dict, n_frames):
    """
    Normalize all joints relative to spine base (center of body).
    positions_dict: {joint_name: (n_frames, 3)}
    """
    spine_base = positions_dict['spinebase']  # (n_frames, 3)
    normalized = {}
    for joint, pos in positions_dict.items():
        normalized[joint] = pos - spine_base  # center around spine base
    return normalized


def pad_or_truncate(sequence, target_length):
    """Pad with zeros or truncate to target_length frames."""
    current_length = sequence.shape[0]
    if current_length >= target_length:
        return sequence[:target_length]
    else:
        padding = np.zeros((target_length - current_length, sequence.shape[1]))
        return np.vstack([sequence, padding])



# 3. Process all exercises
TARGET_LENGTH = 300  # frames - pad/truncate all sequences to this

processed_data = {}

for ex_name in ['ex1', 'ex2', 'ex3', 'ex4', 'ex5']:
    print(f"\nProcessing {ex_name}...")
    df = data[ex_name]
    
    exercise_samples = []
    exercise_labels = []
    exercise_body_parts = {part: [] for part in BODY_PARTS}
    
    valid_count = 0
    skip_count = 0
    
    for idx in range(len(df)):
        try:
            # Extract positions for all joints
            positions = {}
            n_frames = None
            valid = True
            
            for joint in JOINT_NAMES:
                if joint not in df.columns:
                    valid = False
                    break
                joint_data = df.iloc[idx][joint]
                pos = extract_positions(joint_data)
                
                if n_frames is None:
                    n_frames = pos.shape[0]
                positions[joint] = pos
            
            if not valid or n_frames is None or n_frames < 10:
                skip_count += 1
                continue
            
            # Normalize relative to spine base
            positions = normalize_skeleton(positions, n_frames)
            
            # Build full skeleton array (n_frames, n_joints * 3)
            full_skeleton = []
            for joint in JOINT_NAMES:
                full_skeleton.append(positions[joint])
            full_skeleton = np.hstack(full_skeleton)  # (n_frames, 75)
            
            # Pad/truncate
            full_skeleton = pad_or_truncate(full_skeleton, TARGET_LENGTH)
            
            # Build body part arrays
            for part_name, part_joints in BODY_PARTS.items():
                part_data = []
                for joint in part_joints:
                    part_data.append(positions[joint])
                part_data = np.hstack(part_data)  # (n_frames, n_joints_in_part * 3)
                part_data = pad_or_truncate(part_data, TARGET_LENGTH)
                exercise_body_parts[part_name].append(part_data)
            
            # Get label (clinical total score)
            label = df.iloc[idx]['cTS']
            
            exercise_samples.append(full_skeleton)
            exercise_labels.append(label)
            valid_count += 1
            
        except Exception as e:
            skip_count += 1
            continue
    
    print(f"  Valid samples: {valid_count}, Skipped: {skip_count}")
    
    # Convert to numpy arrays
    exercise_samples = np.array(exercise_samples)  # (n_subjects, TARGET_LENGTH, 75)
    exercise_labels = np.array(exercise_labels)     # (n_subjects,)
    
    body_parts_arrays = {}
    for part_name in BODY_PARTS:
        body_parts_arrays[part_name] = np.array(exercise_body_parts[part_name])
    
    processed_data[ex_name] = {
        'full_skeleton': exercise_samples,
        'labels': exercise_labels,
        'body_parts': body_parts_arrays,
    }
    
    print(f"  Full skeleton shape: {exercise_samples.shape}")
    print(f"  Labels shape: {exercise_labels.shape}")
    print(f"  Label range: {exercise_labels.min():.3f} to {exercise_labels.max():.3f}")
    for part_name, arr in body_parts_arrays.items():
        print(f"  {part_name} shape: {arr.shape}")

# 4. Save processed data
output_path = 'data/processed/kimore_processed.pkl'
os.makedirs('data/processed', exist_ok=True)

with open(output_path, 'wb') as f:
    pickle.dump(processed_data, f)

print(f"\n{'='*50}")
print(f"Processed data saved to {output_path}")
print(f"{'='*50}")