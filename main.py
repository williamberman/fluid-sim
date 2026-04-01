import os
import argparse
from tqdm import tqdm
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import json

import common

parser = argparse.ArgumentParser()
parser.add_argument("--case_dir", type=str, required=True)
parser.add_argument("--pressure_correction_steps", default=2, type=int)
parser.add_argument("--sim_dir", type=str, default=None, required=False)
args = parser.parse_args()

def main():
    parsed_case_dir = common.parse_case_dir(args.case_dir)

    face_o, face_n = parsed_case_dir["face_o"], parsed_case_dir["face_n"]
    regions = parsed_case_dir["regions"]
    nu, dt, start_time, end_time = parsed_case_dir["nu"], parsed_case_dir["dt"], parsed_case_dir["start_time"], parsed_case_dir["end_time"]
    dA, SA_faces, cell_centers, cell_vols, l_faces = parsed_case_dir["dA"], parsed_case_dir["SA_faces"], parsed_case_dir["cell_centers"], parsed_case_dir["cell_vols"], parsed_case_dir["l_faces"]

    n_cells = len(cell_centers)
    cell_vols_inv = 1/cell_vols
    l_faces_inv = 1/l_faces

    pbar = tqdm(total=(end_time - start_time) // dt, desc="running sim", postfix=f"time: {start_time:.2f}s")

    t = start_time

    u_prev, p_prev = np.zeros((n_cells, 3)), np.zeros((n_cells, 1)) # not general case but just assume starting fields are zero
            
    if args.sim_dir is not None:
        os.makedirs(args.sim_dir, exist_ok=True)

    sim_step = 0

    while t < end_time:
        if args.sim_dir is not None:
            np.save(os.path.join(args.sim_dir, f"u_{sim_step:06d}.npy"), u_prev)
            np.save(os.path.join(args.sim_dir, f"p_{sim_step:06d}.npy"), p_prev)
            with open(os.path.join(args.sim_dir, f"metadata_{sim_step:06d}.json"), "w") as f:
                json.dump({"t": t}, f, indent=4)

        # sparse system components
        diagonal_terms    = np.zeros((n_cells, 3))
        constant_terms    = np.zeros((n_cells, 3))
        face_terms_o_to_n = np.zeros((len(face_n), 3)) # outward facing normal from face.o to face.n
        face_terms_n_to_o = np.zeros((len(face_n), 3)) # inward facing normal from face.n to face.o

        # momentum term: volume_integral(D_t (u_i)) = V/dt*u_i - V/dt*u_prev_i
        diagonal_terms +=  cell_vols[:, None]/dt
        constant_terms += -cell_vols[:, None]/dt*u_prev

        for r in regions:
            face_o_ = face_o[r.start_face:r.start_face+r.n_faces]
            face_n_ = face_n[r.start_face:r.start_face+r.n_faces]
            dA_ = dA[r.start_face:r.start_face+r.n_faces]
            l_inv_ = l_faces_inv[r.start_face:r.start_face+r.n_faces]
            SA_ = SA_faces[r.start_face:r.start_face+r.n_faces]
        
            # convection term: surface_integral(u_face_i * sum_j (u_prev_face_j * dA_j)) = sum_faces (u_face_i * phi_prev)

            if r.velocity_boundary_type is None:
                # u_face_i * phi_prev = u_o_i * phi_prev / 2 + u_n_i * phi_prev / 2
                u_prev_face = 0.5*(u_prev[face_o_] + u_prev[face_n_])
                phi_prev = np.einsum('fj,fj->f', u_prev_face, dA_)

                # if phi_prev at a face is positive, then the owner cell is upwind -> choose the value at the
                # boundary as the owner cell value

                owner_is_upwind = phi_prev > 0
                owner_is_downwind = ~owner_is_upwind

                upwind_coeff = 0.5 # TODO - locally determine upwind coefficient

                diagonal_terms += upwind_coeff*np.bincount(face_o_[owner_is_upwind], phi_prev[owner_is_upwind], n_cells)[:, None]
                face_terms_o_to_n[r.start_face:r.start_face+r.n_faces][owner_is_downwind] += upwind_coeff*phi_prev[owner_is_downwind][:, None]

                diagonal_terms -= upwind_coeff*np.bincount(face_n_[owner_is_downwind], phi_prev[owner_is_downwind], n_cells)[:, None]
                face_terms_n_to_o[r.start_face:r.start_face+r.n_faces][owner_is_upwind] -= upwind_coeff*phi_prev[owner_is_upwind][:, None]


                half_phi_prev = 0.5*phi_prev
                diagonal_terms += (1 - upwind_coeff)*np.bincount(face_o_, half_phi_prev, n_cells)[:, None]
                face_terms_o_to_n[r.start_face:r.start_face+r.n_faces] += (1 - upwind_coeff)*half_phi_prev[:, None]
                diagonal_terms -= (1 - upwind_coeff)*np.bincount(face_n_, half_phi_prev, n_cells)[:, None]
                face_terms_n_to_o[r.start_face:r.start_face+r.n_faces] -= (1 - upwind_coeff)*half_phi_prev[:, None]

            elif r.velocity_boundary_type == 'fixedValue':
                # u_face_i * phi_prev = c_i * phi_prev
                u_prev_face = r.velocity_boundary_value
                phi_prev = np.einsum('fj,fj->f', u_prev_face, dA_)
                u_face = r.velocity_boundary_value
                u_face_phi_prev = u_face * phi_prev[:, None]
                for i in range(3):
                    constant_terms[:, i] += np.bincount(face_o_, u_face_phi_prev[:, i], n_cells)
            elif r.velocity_boundary_type == 'zeroGradient':
                # u_face_i * phi_prev = u_o_i * phi_prev
                u_prev_face = u_prev[face_o_]
                phi_prev = np.einsum('fj,fj->f', u_prev_face, dA_)
                diagonal_terms += np.bincount(face_o_, phi_prev, n_cells)[:, None]
            elif r.velocity_boundary_type == 'empty':
                pass
            else:
                assert False
        
            # deviatoric stress term1: -surface_integral(nu * sum_j (D_xj u_i * dA_j))
            # = -sum_faces (nu * sum_j (D_xj u_face_i * dA_j))

            if r.velocity_boundary_type is None:
                # D_xj u_face_i = 1/l * (u_n_i - u_o_i)
                # face term: -nu * 1/l * SA * (u_n_i - u_o_i), because orthogonal mesh
                term = -nu * l_inv_ * SA_
                diagonal_terms -= np.bincount(face_o_, term, n_cells)[:, None]
                face_terms_o_to_n[r.start_face:r.start_face+r.n_faces] += term[:, None]

                diagonal_terms -= np.bincount(face_n_, term, n_cells)[:, None]
                face_terms_n_to_o[r.start_face:r.start_face+r.n_faces] += term[:, None]
            elif r.velocity_boundary_type == 'fixedValue':
                # D_xj u_face_i = (c_i - u_o_i) / l
                # face term: -nu / l * SA (c_i - u_o_i), because orthogonal mesh
                term = -nu * l_inv_ * SA_
                for i in range(3):
                    constant_terms[:, i] += np.bincount(face_o_, term*r.velocity_boundary_value[:, i], n_cells)
                diagonal_terms -= np.bincount(face_o_, term, n_cells)[:, None]
            elif r.velocity_boundary_type == 'zeroGradient':
                # D_xj u_face_i = 0, so no contrib
                pass
            elif r.velocity_boundary_type == 'empty':
                pass
            else:
                assert False

            # deviatoric stress term2: TODO

        constant_terms_no_pressure = constant_terms.copy()

        for r in regions:
            face_o_ = face_o[r.start_face:r.start_face+r.n_faces]
            face_n_ = face_n[r.start_face:r.start_face+r.n_faces]
            dA_ = dA[r.start_face:r.start_face+r.n_faces]

            # pressure term: surface_integral(p_face * dA_i) = sum_faces (p_prev_face * dA_i)

            if r.pressure_boundary_type is None:
                # p_prev_face * dA_i = 0.5 * (p_prev_o + p_prev_n) * dA_i
                p_prev_face = 0.5*(p_prev[face_o_] + p_prev[face_n_])
                p_prev_face_dA = p_prev_face * dA_
                for i in range(3):
                    constant_terms[:, i] += np.bincount(face_o_, p_prev_face_dA[:, i], n_cells)
                    constant_terms[:, i] -= np.bincount(face_n_, p_prev_face_dA[:, i], n_cells)
            elif r.pressure_boundary_type == 'fixedValue':
                # p_prev_face * dA_i = c_i * dA_i
                p_prev_face = r.pressure_boundary_value
                p_prev_face_dA = p_prev_face * dA_
                for i in range(3):
                    constant_terms[:, i] += np.bincount(face_o_, p_prev_face_dA[:, i], n_cells)
            elif r.pressure_boundary_type == 'zeroGradient':
                # p_prev_face * dA_i = p_o_i * dA_i
                p_prev_face = p_prev[face_o_]
                p_prev_face_dA = p_prev_face * dA_
                for i in range(3):
                    constant_terms[:, i] += np.bincount(face_o_, p_prev_face_dA[:, i], n_cells)
            elif r.pressure_boundary_type == 'empty':
                pass
            else:
                assert False

        # solve momentum equation

        A_flat = np.concatenate([diagonal_terms, face_terms_o_to_n, face_terms_n_to_o])
        row_ind = np.concatenate([np.arange(n_cells, dtype=np.uint64), face_o[:len(face_n)], face_n])
        col_ind = np.concatenate([np.arange(n_cells, dtype=np.uint64), face_n,               face_o[:len(face_n)]])
        
        u = np.zeros((n_cells, 3))

        for i in range(3):
            A = csr_matrix((A_flat[:, i], (row_ind, col_ind)), shape=(n_cells, n_cells))
            u[:, i] = spsolve(A, -constant_terms[:, i])

        # pressure correction

        A_off_diag_flat = np.concatenate([face_terms_o_to_n, face_terms_n_to_o])
        row_ind_off_diag = np.concatenate([face_o[:len(face_n)], face_n])
        col_ind_off_diag = np.concatenate([face_n,               face_o[:len(face_n)]])

        for _ in range(args.pressure_correction_steps):
            # incompressible flow, zero divergence constraint: 
            # sum_i D_x_i u_i = 0
            # surface_integral(sum_i u_i dA_i) = 0
            # sum_faces (sum_i u_face_i dA_i) = 0

            # momentum equation w/ undiscretized p and u's represented all off diagonal u's separated: 
            # a_i * u_i + H_i(u_off_diagonal_i) + D_x_i p = 0 
            # rearrange and say H_i = H_i(u_off_diagonal_i)
            # u_i = -1/a_i * D_x_i p - 1/a_i * H_i
            # a_0 == a_1 == a_2, will remove index later in code

            a = diagonal_terms
            a_inv = 1/a

            H = np.zeros((n_cells, 3))

            for i in range(3):
                A_off_diag = csr_matrix((A_off_diag_flat[:, i], (row_ind_off_diag, col_ind_off_diag)), shape=(n_cells, n_cells))
                H[:, i] = A_off_diag @ u[:, i] + constant_terms_no_pressure[:, i]

            H_by_a = H * a_inv

            p_diagonal_terms = np.zeros(n_cells)
            p_constant_terms = np.zeros(n_cells)
            p_face_terms_o_to_n = np.zeros(len(face_n))
            p_face_terms_n_to_o = np.zeros(len(face_n))

            for r in regions:
                face_o_ = face_o[r.start_face:r.start_face+r.n_faces]
                face_n_ = face_n[r.start_face:r.start_face+r.n_faces]
                dA_ = dA[r.start_face:r.start_face+r.n_faces]
                l_inv_ = l_faces_inv[r.start_face:r.start_face+r.n_faces]
                SA_ = SA_faces[r.start_face:r.start_face+r.n_faces]

                if r.boundary:
                    # for velocity specified boundary types, u_face_i is given and so the contributions from 
                    # sum_i u_face_i dA_i = 0 are only constant terms

                    if r.velocity_boundary_type == 'fixedValue':
                        # u_face_i = c_i
                        u_face = r.velocity_boundary_value
                        phi = np.einsum('fj,fj->f', u_face, dA_)
                        p_constant_terms += np.bincount(face_o_, phi, n_cells)
                    elif r.velocity_boundary_type == 'zeroGradient':
                        # u_face_i = u_o_i
                        u_face = u[face_o_]
                        phi = np.einsum('fj,fj->f', u_face, dA_)
                        p_constant_terms += np.bincount(face_o_, phi, n_cells)
                    elif r.velocity_boundary_type == 'empty':
                        pass
                    else:
                        assert False
                else:
                    # a_inv_face = 1/a_i interpolated to the face
                    # H_by_a_face = H_i/a_i interpolated to the face
                    # u_face_i = -a_inv_face / l * (p_n - p_o) - H_by_a_face

                    a_inv_face = 0.5 * (a_inv[face_o_] + a_inv[face_n_])
                    H_by_a_face = 0.5 * (H_by_a[face_o_] + H_by_a[face_n_])

                    # sum_i u_face_i dA_i = 0
                    # face term: -a_inv_face / l * SA * (p_n - p_o) - sum_i H_by_a_face dA_i = 0

                    coeff_term = -a_inv_face[:, 0] * l_inv_ * SA_

                    p_diagonal_terms -= np.bincount(face_o_, coeff_term, n_cells)
                    p_face_terms_o_to_n[r.start_face:r.start_face+r.n_faces] += coeff_term

                    p_diagonal_terms -= np.bincount(face_n_, coeff_term, n_cells)
                    p_face_terms_n_to_o[r.start_face:r.start_face+r.n_faces] += coeff_term

                    constant_term = np.einsum('fj,fj->f', H_by_a_face, dA_)

                    p_constant_terms -= np.bincount(face_o_, constant_term, n_cells)
                    p_constant_terms += np.bincount(face_n_, constant_term, n_cells)

            A_flat = np.concatenate([p_diagonal_terms, p_face_terms_o_to_n, p_face_terms_n_to_o])
            row_ind = np.concatenate([np.arange(n_cells, dtype=np.uint64), face_o[:len(face_n)], face_n])
            col_ind = np.concatenate([np.arange(n_cells, dtype=np.uint64), face_n,               face_o[:len(face_n)]])

            A = csr_matrix((A_flat, (row_ind, col_ind)), shape=(n_cells, n_cells))

            # set reference pressure
            A = A.tolil()
            A[0, :] = 0
            A[0, 0] = 1
            p_constant_terms[0] = 0
            A = A.tocsr()

            p = spsolve(A, -p_constant_terms).reshape(-1, 1)

            # u_i = -1/a_i * D_x_i p - 1/a_i * H_i
            # volume_integral(D_x_i p dV) = surface_integral(p dA_i)
            # V * D_x_i p = sum_faces p_face dA_i
            # D_x_i p = (1/V) * sum_faces p_face dA_i

            Dp = np.zeros((n_cells, 3))
            
            for r in regions:
                face_o_ = face_o[r.start_face:r.start_face+r.n_faces]
                face_n_ = face_n[r.start_face:r.start_face+r.n_faces]
                dA_ = dA[r.start_face:r.start_face+r.n_faces]

                if r.pressure_boundary_type is None:
                    p_face = 0.5*(p[face_o_] + p[face_n_])
                    term = p_face * dA_
                    for i in range(3):
                        Dp[:, i] += np.bincount(face_o_, term[:, i], n_cells)
                        Dp[:, i] -= np.bincount(face_n_, term[:, i], n_cells)
                elif r.pressure_boundary_type == 'fixedValue':
                    p_face = r.pressure_boundary_value
                    term = p_face * dA_
                    for i in range(3):
                        Dp[:, i] += np.bincount(face_o_, term[:, i], n_cells)
                elif r.pressure_boundary_type == 'zeroGradient':
                    p_face = p[face_o_]
                    term = p_face * dA_
                    for i in range(3):
                        Dp[:, i] += np.bincount(face_o_, term[:, i], n_cells)
                elif r.pressure_boundary_type == 'empty':
                    pass
                else:
                    assert False

            Dp *= cell_vols_inv[:, None]

            u = -a_inv * Dp - H_by_a

        u_prev = u
        p_prev = p
        t += dt
        sim_step += 1
        pbar.update(1)

if __name__ == "__main__":
    main()