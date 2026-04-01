import os
import argparse
import glob
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.collections import PolyCollection

import common

parser = argparse.ArgumentParser()
parser.add_argument("--case_dir", type=str, required=True)
parser.add_argument("--sim_dir", type=str, required=True)
parser.add_argument("--vid_path", type=str, required=True)
args = parser.parse_args()

def main():
    parsed_case_dir = common.parse_case_dir(args.case_dir)

    face_o, face_n = parsed_case_dir["face_o"], parsed_case_dir["face_n"]
    points, face_points = parsed_case_dir["points"], parsed_case_dir["face_points"]
    regions = parsed_case_dir["regions"]
    face_centers, cell_centers, l_faces = parsed_case_dir["face_centers"], parsed_case_dir["cell_centers"], parsed_case_dir["l_faces"]

    us = np.stack([np.load(x) for x in sorted(glob.glob(os.path.join(args.sim_dir, "u_*.npy")))], axis=0)
    metadatas = sorted(glob.glob(os.path.join(args.sim_dir, "metadata_*.json")))
    assert len(us) == len(metadatas)

    speeds = np.linalg.norm(us, axis=-1)
    us_direction = us / np.where(speeds > 0, speeds, 1)[:, :, None]

    u_boundary = np.zeros((len(face_o), 3))

    for r in regions:
        if r.boundary: 
            if r.velocity_boundary_type == 'fixedValue':
                u_boundary[r.start_face:r.start_face+r.n_faces] = r.velocity_boundary_value
            elif r.velocity_boundary_type == 'zeroGradient':
                u_boundary[r.start_face:r.start_face+r.n_faces] = us[face_o[r.start_face:r.start_face+r.n_faces]]
            elif r.velocity_boundary_type == 'empty':
                pass # keep as zeros
            else:
                assert False

    u_boundary = u_boundary[len(face_n):]

    u_boundary_speed = np.linalg.norm(u_boundary, axis=-1)
    u_boundary_direction = u_boundary / np.where(u_boundary_speed > 0, u_boundary_speed, 1)[:, None]

    fig, ax = plt.subplots(figsize=(10, 8))

    min_x, max_x = points[:, 0].min(), points[:, 0].max()
    pad_x = 0.05 * (max_x - min_x)
    min_y, max_y = points[:, 1].min(), points[:, 1].max()
    pad_y = 0.05 * (max_y - min_y)
    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    mesh = PolyCollection(points[face_points][:, :, :2], facecolor=(0.6, 0.7, 0.9, 0.05), edgecolor=(0, 0, 0, 0.2), linewidth=0.3)
    ax.add_collection(mesh)

    arrow_length = 0.8 * l_faces[0]

    speed_min = min(speeds.min().item(), u_boundary_speed.min().item())
    speed_max = max(speeds.max().item(), u_boundary_speed.max().item(), speed_min+1e-12)
    norm = Normalize(vmin=speed_min, vmax=speed_max)
    cmap = plt.get_cmap("coolwarm")

    quiver = ax.quiver(
        cell_centers[:, 0], cell_centers[:, 1], us_direction[0, :, 0], us_direction[0, :, 1], speeds[0],
        cmap=cmap, norm=norm,
        angles="xy", scale_units="xy", scale=1.0 / arrow_length,
        width=0.004, headwidth=4, headlength=5, headaxislength=4.5,
        pivot="tail",
    )

    ax.quiver(
        face_centers[len(face_n):, 0], face_centers[len(face_n):, 1], 
        u_boundary_direction[:, 0], u_boundary_direction[:, 1], u_boundary_speed,
        cmap=cmap, norm=norm,
        angles="xy", scale_units="xy", scale=1.0 / arrow_length,
        width=0.004, headwidth=4, headlength=5, headaxislength=4.5,
        pivot="tail",
    )

    cb = fig.colorbar(quiver, ax=ax, shrink=0.9, pad=0.02)
    cb.set_label("speed")

    title = ax.set_title("")

    pbar = tqdm(total=len(us), desc="Making video")

    def update(i):
        quiver.set_UVC(us_direction[i, :, 0], us_direction[i, :, 1], speeds[i])

        with open(metadatas[i], "r") as f:
            m = json.load(f)
        t = m["t"]
        title.set_text(f"t = {t:.4g}")

        pbar.update(1)

        return (quiver, title)

    anim = animation.FuncAnimation(fig, update, frames=len(us), blit=False)
    anim.save(args.vid_path, fps=50, dpi=72)
    plt.close(fig)
    pbar.close()

if __name__ == "__main__":
    main()
