import os
import re
from typing import Optional, Literal
from dataclasses import dataclass
import numpy as np

@dataclass
class Region:
    name: str
    start_face: int
    n_faces: int

    boundary: bool

    velocity_boundary_type: Optional[Literal["fixedValue", "zeroGradient", "empty"]]
    pressure_boundary_type: Optional[Literal["fixedValue", "zeroGradient", "empty"]]

    # leading 1 dim lets array broadcast to all faces in boundary region code
    # and helps simplify/remove some boundary region specific code
    velocity_boundary_value: Optional[np.ndarray] # 1x3
    pressure_boundary_value: Optional[np.ndarray] # 1x1

def parse_case_dir(case_dir):
    with open(os.path.join(case_dir, "constant/polyMesh/boundary"), "r") as f:
        poly_boundary = f.read()

    with open(os.path.join(case_dir, "constant/polyMesh/faces"), "r") as f:
        poly_faces = f.read()

    with open(os.path.join(case_dir, "constant/polyMesh/owner"), "r") as f:
        poly_owner = f.read()

    with open(os.path.join(case_dir, "constant/polyMesh/neighbour"), "r") as f:
        poly_neighbour = f.read()

    with open(os.path.join(case_dir, "constant/polyMesh/points"), "r") as f:
        poly_points = f.read()

    with open(os.path.join(case_dir, "system/controlDict"), "r") as f:
        control_dict = f.read()

    with open(os.path.join(case_dir, "constant/physicalProperties"), "r") as f:
        physical_properties = f.read()

    with open(os.path.join(case_dir, "0/U"), "r") as f:
        U0 = f.read()

    with open(os.path.join(case_dir, "0/p"), "r") as f:
        p0 = f.read()

    points = np.array([[float(x) for x in match] for match in re.findall(r"\((\d+(?:\.\d+)?) (\d+(?:\.\d+)?) (\d+(?:\.\d+)?)\)\n", poly_points)], dtype=np.float64)
    face_points = np.array([[int(x) for x in match] for match in re.findall(r"4\((\d+) (\d+) (\d+) (\d+)\)\n", poly_faces)], dtype=np.uint64)
    face_o = np.array([int(x) for x in re.findall(r"\d+\n", poly_owner[poly_owner.find("("):])], dtype=np.uint64)
    face_n = np.array([int(x) for x in re.findall(r"\d+\n", poly_neighbour[poly_neighbour.find("("):])], dtype=np.uint64)

    regions = [Region(
        name="internal",
        start_face=0,
        n_faces=len(face_n),
        boundary=False,
        velocity_boundary_type=None,
        velocity_boundary_value=None,
        pressure_boundary_type=None,
        pressure_boundary_value=None,
    )]

    for name, n_faces, start_face in re.findall(r"(\w+)\s+{[^}]*nFaces\s+(\d+)[^}]*startFace\s+(\d+)[^}]*}", poly_boundary, re.MULTILINE | re.DOTALL):
        if (match := re.search(name + r"[^}]*{[^}]*type\s+fixedValue[^}]*value\s+uniform\s+\((.+)\s+(.+)\s+(.+)\)[^}]*}", U0, re.MULTILINE | re.DOTALL)) is not None:
            velocity_boundary_type = "fixedValue"
            velocity_boundary_value = tuple(float(x) for x in match.groups())
        elif (match := re.search(name + r"[^}]*{[^}]*type\s+noSlip[^}]*}", U0, re.MULTILINE | re.DOTALL)) is not None:
            velocity_boundary_type = "fixedValue"
            velocity_boundary_value = (0, 0, 0)
        elif (match := re.search(name + r"[^}]*{[^}]*type\s+empty[^}]*}", U0, re.MULTILINE | re.DOTALL)) is not None:
            velocity_boundary_type = "empty"
            velocity_boundary_value = None
        elif (match := re.search(name + r"[^}]*{[^}]*type\s+zeroGradient[^}]*}", U0, re.MULTILINE | re.DOTALL)) is not None:
            velocity_boundary_type = "zeroGradient"
            velocity_boundary_value = None
        else:
            assert False

        # now do the same for pressure
        if (match := re.search(name + r"[^}]*{[^}]*type\s+fixedValue[^}]*value\s+uniform\s+([\S]+);[^}]*}", p0, re.MULTILINE | re.DOTALL)) is not None:
            pressure_boundary_type = "fixedValue"
            pressure_boundary_value = float(match.groups()[0])
        elif (match := re.search(name + r"[^}]*{[^}]*type\s+zeroGradient[^}]*}", p0, re.MULTILINE | re.DOTALL)) is not None:
            pressure_boundary_type = "zeroGradient"
            pressure_boundary_value = None
        elif (match := re.search(name + r"[^}]*{[^}]*type\s+empty[^}]*}", p0, re.MULTILINE | re.DOTALL)) is not None:
            pressure_boundary_type = "empty"
            pressure_boundary_value = None
        else:
            assert False

        if velocity_boundary_value is not None:
            velocity_boundary_value = np.array(velocity_boundary_value, dtype=np.float64).reshape(1, 3)

        if pressure_boundary_value is not None:
            pressure_boundary_value = np.array(pressure_boundary_value, dtype=np.float64).reshape(1, 1)

        regions.append(Region(
            name=name,
            start_face=int(start_face),
            n_faces=int(n_faces),
            boundary=True,
            velocity_boundary_type=velocity_boundary_type,
            velocity_boundary_value=velocity_boundary_value,
            pressure_boundary_type=pressure_boundary_type,
            pressure_boundary_value=pressure_boundary_value,
        ))


    nu = float(re.search(r"nu\s+(\S+)\s", physical_properties, re.MULTILINE | re.DOTALL).group(1))
    dt = float(re.search(r"deltaT\s+(\d+(?:\.\d+)?);",control_dict, re.MULTILINE | re.DOTALL).group(1))
    start_time = float(re.search(r"startTime\s+(\d+(?:\.\d+)?);", control_dict, re.MULTILINE | re.DOTALL).group(1))
    end_time = float(re.search(r"endTime\s+(\d+(?:\.\d+)?);", control_dict, re.MULTILINE | re.DOTALL).group(1))

    face_centers = points[face_points].mean(axis=1)

    # face points are ordered by right hand rule so this is outward facing w.r.t owner cell
    dA = np.cross(points[face_points[:, 1]] - points[face_points[:, 0]], points[face_points[:, 3]] - points[face_points[:, 0]])
    SA_faces = np.linalg.norm(dA, axis=1)
    
    n_cells = np.max(face_o).item() + 1

    cells = [[] for _ in range(n_cells)]
    cell_vols = np.zeros(n_cells, dtype=np.float64)

    for face_idx, o in enumerate(face_o):
        cells[o].append(points[face_points[face_idx]])
        cell_vols[o] += np.dot(dA[face_idx], face_centers[face_idx])

    for face_idx, n in enumerate(face_n):
        cells[n].append(points[face_points[face_idx]])
        cell_vols[n] -= np.dot(dA[face_idx], face_centers[face_idx])

    cells = np.array(cells, dtype=np.float64)
    cell_centers = cells.mean(axis=(1,2))
    cell_vols *= 1/3

    # distances for finite differences
    l_faces = np.concatenate([
        # faces between cells
        np.linalg.norm(cell_centers[face_n] - cell_centers[face_o[:len(face_n)]], axis=1),

        # faces between cell and boundary
        np.linalg.norm(face_centers[len(face_n):] - cell_centers[face_o[len(face_n):]], axis=1)
    ], axis=-1)

    return dict(
        face_o=face_o,
        face_n=face_n,

        points=points,
        face_points=face_points,

        regions=regions,

        nu=nu,
        dt=dt,
        start_time=start_time,
        end_time=end_time,

        face_centers=face_centers,
        dA=dA,
        SA_faces=SA_faces,
        cell_centers=cell_centers,
        cell_vols=cell_vols,
        l_faces=l_faces,
    )
