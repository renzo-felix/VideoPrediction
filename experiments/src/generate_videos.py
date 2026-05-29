import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pybullet as p
import pybullet_data
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import argparse

from config import *
from data.pybullet_objects import create_object

def generate_video(shape, size_scale, color_rgba, material_specular, texture_type, target_speed, output_path):
    p.connect(p.DIRECT)

    # --- ACTIVATE GPU ACCELERATION IF WORKING GPU IS AVAILABLE ---
    import os
    import pkgutil
    # Only load EGL if an NVIDIA GPU driver is active and communicating
    if os.system("nvidia-smi > /dev/null 2>&1") == 0:
        egl = pkgutil.get_loader('eglRenderer')
        if egl:
            p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
    # -------------------------------------------------------------

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1./240.)

    p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_PLANE),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_PLANE, rgbaColor=[0.9, 0.9, 0.9, 1])
    )

    obj_id = create_object(shape, size_scale, color_rgba, material_specular, texture_type)
    p.resetBaseVelocity(obj_id, linearVelocity=[target_speed / 10.0, 0, 0])

    frames = []
    speeds = []

    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=CAMERA_TARGET, distance=CAMERA_DISTANCE,
        yaw=CAMERA_YAW, pitch=CAMERA_PITCH, roll=0, upAxisIndex=2
    )
    proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1.0, nearVal=0.1, farVal=100.0)

    for _ in range(NUM_FRAMES):
        p.stepSimulation()
        vel, _ = p.getBaseVelocity(obj_id)
        speeds.append(np.linalg.norm(vel))

        _, _, rgb, _, _ = p.getCameraImage(
            width=IMG_SIZE, height=IMG_SIZE,
            viewMatrix=view_matrix, projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        frames.append(rgb[:, :, :3])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, FPS, (IMG_SIZE, IMG_SIZE))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    p.disconnect()

    return {"actual_speed": np.mean(speeds), "speed_std": np.std(speeds), "num_frames": NUM_FRAMES}

def generate_training(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    speeds = [5, 15, 25, 35, 45]
    sizes = ['small', 'medium', 'large']
    colors = ['red', 'blue', 'green']
    repetitions = 3

    metadata = []
    video_counter = 0

    for size_name in sizes:
        for color_name in colors:
            for base_speed in speeds:
                for rep in range(repetitions):
                    speed_noisy = base_speed + np.random.uniform(-1.5, 1.5)
                    speed_noisy = np.clip(speed_noisy, 3, 50)

                    size_scale = SIZE_MAPPING[size_name]
                    color_rgba = COLOR_MAPPING[color_name]

                    video_id = f"sphere_{size_name}_{color_name}_{int(base_speed*10):02d}_{rep:02d}"
                    video_path = output_dir / f"{video_id}.mp4"

                    meta = generate_video("sphere", size_scale, color_rgba,
                                         MATERIAL_MAPPING['matte'], None, speed_noisy, video_path)

                    metadata.append({
                        "video_id": video_id,
                        "target_speed": base_speed,
                        "actual_speed": meta["actual_speed"],
                        "speed_std": meta["speed_std"],
                        "size": size_name,
                        "color": color_name,
                        "shape": "sphere",
                        "material": "matte",
                        "texture": "plain",
                        "num_frames": meta["num_frames"],
                        "video_path": str(video_path)
                    })
                    video_counter += 1

    df = pd.DataFrame(metadata)
    df.to_csv(DATA_DIR / "training_metadata.csv", index=False)
    print(f"✓ Generated {len(metadata)} training videos")
    print(f"  Speeds: {speeds}")
    print(f"  Sizes: {sizes}")
    print(f"  Colors: {colors}")
    print(f"  Repetitions: {repetitions}")
    print(f"  Total: {len(speeds)} × {len(sizes)} × {len(colors)} × {repetitions} = {len(metadata)}")

def generate_test(category, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if category == "size":
        baseline = {
            'shape': 'cube',
            'size_scale': 1.0,
            'color_rgba': COLOR_MAPPING['yellow'],
            'material_specular': MATERIAL_MAPPING['glossy'],
            'texture_type': None
        }
        variations = [{**baseline, 'size_scale': SIZE_MAPPING[s], 'condition': s}
                     for s in OBJECT_PROPERTIES['size']]
    elif category == "color":
        baseline = {
            'shape': 'cylinder',
            'size_scale': SIZE_MAPPING['xlarge'],
            'color_rgba': COLOR_MAPPING['red'],
            'material_specular': MATERIAL_MAPPING['metallic'],
            'texture_type': None
        }
        variations = [{**baseline, 'color_rgba': COLOR_MAPPING[c], 'condition': c}
                     for c in OBJECT_PROPERTIES['color']]
    elif category == "shape":
        baseline = {
            'shape': 'sphere',
            'size_scale': SIZE_MAPPING['xlarge'],
            'color_rgba': COLOR_MAPPING['yellow'],
            'material_specular': MATERIAL_MAPPING['glossy'],
            'texture_type': None
        }
        variations = [{**baseline, 'shape': s, 'condition': s}
                     for s in OBJECT_PROPERTIES['shape']]
    elif category == "material":
        baseline = {
            'shape': 'capsule',
            'size_scale': SIZE_MAPPING['xlarge'],
            'color_rgba': COLOR_MAPPING['yellow'],
            'material_specular': MATERIAL_MAPPING['matte'],
            'texture_type': None
        }
        variations = [{**baseline, 'material_specular': MATERIAL_MAPPING[m], 'condition': m}
                     for m in OBJECT_PROPERTIES['material']]
    elif category == "texture":
        baseline = {
            'shape': 'cube',
            'size_scale': SIZE_MAPPING['xlarge'],
            'color_rgba': COLOR_MAPPING['yellow'],
            'material_specular': MATERIAL_MAPPING['matte'],
            'texture_type': None
        }
        variations = [{**baseline, 'texture_type': t if t != 'plain' else None, 'condition': t}
                     for t in OBJECT_PROPERTIES['texture']]
    else:
        raise ValueError(f"Unknown category: {category}")

    metadata = []
    speeds = np.linspace(SPEED_RANGE[0], SPEED_RANGE[1], N_VIDEOS_PER_CONDITION)

    for var in variations:
        cond = var.pop('condition')
        cond_dir = output_dir / cond
        cond_dir.mkdir(exist_ok=True)

        for i, speed in enumerate(tqdm(speeds, desc=f"{category}_{cond}")):
            video_id = f"{category}_{cond}_{i:03d}"
            video_path = cond_dir / f"{video_id}.mp4"

            meta = generate_video(var['shape'], var['size_scale'], var['color_rgba'],
                                 var['material_specular'], var['texture_type'], speed, video_path)

            metadata.append({
                "video_id": video_id, "category": category, "condition": cond,
                "target_speed": speed, "actual_speed": meta["actual_speed"],
                "speed_std": meta["speed_std"], "num_frames": meta["num_frames"],
                "video_path": str(video_path)
            })

    pd.DataFrame(metadata).to_csv(DATA_DIR / f"test_metadata_{category}.csv", index=False)
    print(f"✓ Generated {len(metadata)} test videos for {category}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["training", "test"])
    parser.add_argument("--category", choices=TEST_CATEGORIES)
    parser.add_argument("--output_dir")
    args = parser.parse_args()

    if args.mode == "training":
        output_dir = args.output_dir or (VIDEOS_DIR / "training")
        generate_training(output_dir)
    elif args.mode == "test":
        if not args.category:
            parser.error("--category required for test mode")
        output_dir = args.output_dir or (VIDEOS_DIR / f"test_{args.category}")
        generate_test(args.category, output_dir)

if __name__ == "__main__":
    main()
