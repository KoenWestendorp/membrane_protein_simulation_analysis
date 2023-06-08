"""
Align residues to a rigid set of points within their structere.

Koen Westendorp, 2023.
"""


import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisFromFunction as AFF


def translate_to_origin(P):
    M = P.mean(axis=1)
    O = P - M[:, None, ...]
    return O


def hat(a):
    """
    Return the normalized vector of `a`.

    `a` is normalized such that its direction remains the same, but its length
    is one.
    """
    return a / np.linalg.norm(a)


def determine_rotation(O, a_index, b_index, c_index, d_index):
    """
    Determine the rotation vectors u, v, and w for four points.

    Note
    ----
    Care must be taken to prevent vectors w (ab) and v' (cd) from being
    (close to) colinear.

    Parameters
    ----------
    O : ndarray
        The positions for which to determine the u, v, w axes.
    a_index, b_index, c_index, d_index : int
        Indices into `O` to the points a, b, c, and d.
        Points a and b form the w (z) axis, c and d form the v' (y')
        pseudo-axis. The plane formed by w and v' must be the same plane that
        the vw plane should form.

    Returns
    -------
    uvw : ndarray
        Rotation matrix of u, v, w as normalized vectors.
    """
    # The vector between a and b becomes the z axis.
    w = O[:, a_index] - O[:, b_index]

    # v' is the vector between c and d.
    # We consider v' a pseudo-axis because w and v' are not perpendicular, but
    # they do together form the plane that w and v lie on.
    vp = O[:, c_index] - O[:, d_index]

    # u is perpendicular to the plane formed by v' and w
    # (i.e., the plane on which both vp and w lie).
    u = np.cross(vp, w, axis=1)
    # Calculate the true v to become the y axis from w and u.
    v = np.cross(w, u, axis=1)

    uhat = hat(u)
    vhat = hat(v)
    what = hat(w)

    uvw = np.array([uhat, vhat, what]).swapaxes(0, 1)
    return uvw


def align_residue(positions, a_index, b_index, c_index, d_index):
    """
    Align a single residue.

    Parameters
    ----------
    positions : ndarray
        The positions of the atoms in the residue as a n_atoms by 3 array.
    a_index, b_index, c_index, d_index : int
        Indices into `positions` to the points a, b, c, and d.
        Points a and b form the w (z) axis, c and d form the v' (y')
        pseudo-axis. The plane formed by w and v' must be the same plane that
        the vw plane should form.

    Returns
    -------
    ndarray
        The translated and rotated positions aligned to the axes that are
        derived from ab and cd as a n_atoms by 3 matrix.
    """
    O = translate_to_origin(positions)
    R = determine_rotation(O, a_index, b_index, c_index, d_index)
    return O @ R.T


def align_residues(residues, a_index, b_index, c_index, d_index):
    """
    Align many residues.

    Note
    ----
    Assumes `residues` contains the positions of the same kind of residue.
    They must be the same size (number of atoms) and the vectors ab and cd must
    be meaningful to the residue.

    Parameters
    ----------
    residues : ndarray
        The positions of the atoms in the residues as a n_residues by n_atoms by 3 array.
    a_index, b_index, c_index, d_index : int
        Indices into `positions` to the points a, b, c, and d.
        Points a and b form the w (z) axis, c and d form the v' (y')
        pseudo-axis. The plane formed by w and v' must be the same plane that
        the vw plane should form.

    Returns
    -------
    ndarray
        The translated and rotated residues aligned to the axes that are
        derived from ab and cd as a n_residues by n_atoms by 3 matrix.
    """
    return np.array(
        [align_residue(R, a_index, b_index, c_index, d_index) for R in residues]
    )


def align_residues_from_universe(
    u, positions_selector, a_index, b_index, c_index, d_index
):
    """
    Align many residues from a Universe.

    Note
    ----
    Assumes the residues selected by the positions_selector contain the same
    kind of residue. They must be the same size (number of atoms) and the
    vectors ab and cd must be meaningful to the residue.

    Assumes `residues` contains the positions of the same kind of residue.
    They must be the same size (number of atoms) and the vectors ab and cd must
    be meaningful to the residue.

    Parameters
    ----------
    u : Universe
        An MDAnalysis Universe to analyze.
    position_selector : str
        Selector for the residue positions.
    a_index, b_index, c_index, d_index : int
        Indices into `positions` to the points a, b, c, and d.
        Points a and b form the w (z) axis, c and d form the v' (y')
        pseudo-axis. The plane formed by w and v' must be the same plane that
        the vw plane should form.

    Returns
    -------
    ndarray
        The translated and rotated residues aligned to the axes that are
        derived from ab and cd as a n_residues by n_atoms by 3 matrix.
    """

    positions = u.select_atoms(positions_selector)

    residue_size = len(positions.residues[0].atoms)

    R_f = AFF(lambda x: x.positions.copy(), positions).run().results.timeseries
    n_frames = len(R_f)
    R_frax = R_f.reshape(n_frames, -1, residue_size, 3)

    RO_f = []
    for frame in R_frax:
        O = translate_to_origin(frame)
        R = determine_rotation(O, a_index, b_index, c_index, d_index)
        RO = O @ R
        # We need to check for the case that there is simply no residue near the protein.
        if len(RO) > 0:
            RO_f.append(RO)
    RO_f = np.concatenate(RO_f)

    print(f"{RO_f.reshape(n_frames, -1, residue_size, 3).shape = }")
    return RO_f.reshape(n_frames, -1, residue_size, 3)


def align_residues_from_trajectory(
    structure_path,
    trajectory_path,
    position_selector,
    a_index,
    b_index,
    c_index,
    d_index,
    in_memory=True,
):
    """
    Align many residues from a Universe loaded from the trajectory data
    specified by `structure_path` and `trajectory_path`.

    Note
    ----
    Assumes the residues selected by the positions_selector contain the same
    kind of residue. They must be the same size (number of atoms) and the
    vectors ab and cd must be meaningful to the residue.
    Parameters

    ----------
    structure_path : str
        Path to the structure file for the simulation, such as a gro file.
    trajectory_path : str
        Path to the trajectory file for the simulation, such as an xtc file.
    position_selector : str
        Selector for the residue positions.
    a_index, b_index, c_index, d_index : int
        Indices into `positions` to the points a, b, c, and d.
        Points a and b form the w (z) axis, c and d form the v' (y')
        pseudo-axis. The plane formed by w and v' must be the same plane that
        the vw plane should form.
    in_memory : bool, optional
        Whether the Universe should be loaded into memory from the paths. In
        most cases the analysis is significantly faster and more consistent
        across runs with `in_memory` set to True.
        (The default is True.)

    Returns
    -------
    ndarray
        The translated and rotated residues aligned to the axes that are
        derived from ab and cd as a n_residues by n_atoms by 3 matrix.
    """
    u = mda.Universe(gro_path, xtc_path, in_memory=in_memory)
    return align_residues_from_universe(
        u, position_selector, a_index, b_index, c_index, d_index
    )
