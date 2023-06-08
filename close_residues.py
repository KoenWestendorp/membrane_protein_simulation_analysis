"""
An efficient method for obtaining an array of residues that are closer than
some threshold to some target.

Koen Westendorp, 2023.
"""


import MDAnalysis as mda
from MDAnalysis.analysis.base import AnalysisFromFunction as AFF
import numpy as np
from numpy import linalg


# Cutoff distance for residue to be in close proximity to the target.
THRESHOLD = 6.0  # Angstrom


def mda_pbc_to_gro(m):
    """
    Generates Gro style pbc dimensions from MDA style dimensions.

    Converts `[a, b, c, alpha, beta, gamma]`
    where alpha = angle(a, b), beta = angle(a, c), gamma = angle(a, b), to

    ```
    [[xx, 0., 0.],
     [yx, yy, 0.],
     [zx, zy, zz]]
    ```

    Parameters
    ----------
    m : collection of 6 floats
        MDA style dimensions in the order [a, b, c, alpha, beta, gamma] where alpha, beta, gamma are in degrees.

    Returns
    -------
    ndarray
        Gro style dimensions as a 3 by 3 matrix.
    """
    a, b, c, alpha, beta, gamma = m
    # Angles are provided in degrees. Convert them to radians, first.
    alpha, beta, gamma = np.array([alpha, beta, gamma]) * np.pi / 180

    xx = a
    xy = 0.0
    xz = 0.0
    yx = b * np.cos(gamma)
    yy = b * np.sin(gamma)
    yz = 0.0
    zx = c * np.cos(beta)
    zy = (c / np.sin(gamma)) * (np.cos(alpha) - np.cos(beta) * np.cos(gamma))
    zz = np.sqrt(c**2 - zx**2 - zy**2)

    return np.array([[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]])


def close_atoms(target, positions, pbc=None, threshold=THRESHOLD):
    """
    Returns a mask into `positions` that captures the positions that are closer
    than `threshold` to at least one atom of `target`.

    Parameters
    ----------
    target : ndarray
        Array of the xyz positions for a group of atoms that constitute the
        target structure.

        The shape of `target` is (number of atoms, 3).
    positions : ndarray
        Array of the xyz positions for a set of atoms.

        The shape of `positions` is (number of atoms, 3).
    pbc : None or ndarray, optional
        A 3 by 3 ndarray describing the periodic boundary conditions the
        coordinates in `target` and `positions` are subject to. If None, the
        PBC adjustment is ignored.
        (The default is None.)
    threshold : float, optional
        Distance threshold in Å.
        (The default is `THRESHOLD`, which is equal to 6.0 Å.)

    Returns
    -------
    close_mask_per_atom : ndarray
        A mask into `positions` of positions that are sufficiently close to at
        least one atom of `target`.
    """
    if not (pbc is None):
        L = pbc
        Linv = linalg.inv(pbc)

        D_prime = (target @ Linv)[:, None] - (positions @ Linv)[None, :]
        D_prime_tpx = np.divmod(D_prime + 0.5, 1)[1] - 0.5
        D_tpx = D_prime_tpx @ L
    else:
        D_tpx = target[:, None] - positions[None, :]

    # Euclidian distance of xyz axis.
    D2_tp = (D_tpx**2).sum(axis=2)
    # Take the minimum distance over the target axis.
    Dmin2_d = D2_tp.min(axis=0)

    # Create a mask of the entries where the minimum distance between a target
    # atom and a residue atom is smaller than `threshold`.
    close_mask_per_atom = Dmin2_d < threshold**2

    return close_mask_per_atom


def close_residues(target, residues, residue_size, pbc=None, threshold=THRESHOLD):
    """
    Returns a mask into `residues` that captures the residues that have at
    least one atom that are closer than `threshold` to at least one atom of
    `target`.

    Note
    ----
    It is assumed that the `residues` array contains the positions for the
    atoms in a group of residues of the same kind. More precisely, it is
    assumed that all residues have the same size---i.e., all residues have the
    same number of atoms or beads.

    Parameters
    ----------
    target : ndarray
        Array of the xyz positions for a group of atoms that constitute the
        target structure.

        The shape of `target` is (number of atoms, 3).
    residues : ndarray
        Array of the xyz positions for a set of atoms of a group of residues.

        The shape of `residues` is (number of residues * `residue_size`, 3).
    residue_size : int
        The number of atoms or beads in the residue.
    pbc : None or ndarray, optional
        A 3 by 3 ndarray describing the periodic boundary conditions the
        coordinates in `target` and `positions` are subject to. If None, the
        PBC adjustment is ignored.
        (The default is None.)
    threshold : float, optional
        Distance threshold in Å.
        (The default is `THRESHOLD`, which is equal to 6.0 Å.)

    Returns
    -------
    close_mask_per_res_atom : ndarray
        A mask into `residues` of residues that are sufficiently close to at
        least one atom of `target`.
    """
    close_mask_per_atom = close_atoms(target, residues, pbc=pbc, threshold=threshold)
    close_mask_per_res = close_mask_per_atom.reshape((-1, residue_size)).sum(axis=1) > 0
    close_mask_per_res_atom = close_mask_per_res.repeat(residue_size)

    return close_mask_per_res_atom


def close_residues_universe(
    u, target_selector, positions_selector, upto=None, threshold=THRESHOLD
):
    """
    Return a timeseries list of arrays containing the set of residues with
    positions within `threshold` of the target.

    Note
    ----
    It is assumed that the positions selected by `positions_selector` describe
    a group of residues of the same kind. More precisely, it is assumed that
    all residues have the same size---i.e., all residues have the same number
    of atoms or beads.

    Parameters
    ----------
    u : Universe
    target_selector : str
        Selector into the Universe to retrieve the target positions.
    positions_selector : str
        Selector into the Universe to retrieve the residue positions.
    upto : None or int, optional
        If None all frames from the trajectory will be analyzed. If an integer
        value is provided, the frames up to that number will be considered.
        (The default is None.)
    threshold : float, optional
        Distance threshold in Å.
        (The default is `THRESHOLD`, which is equal to 6.0 Å.)

    Returns
    -------
    list of ndarrays
        The set of close residues as a n_close_residues by n_residue_atoms by 3
        (xyz) array for each frame in the trajectory.
    """
    target = u.select_atoms(target_selector)
    positions = u.select_atoms(positions_selector)

    residue_size = len(positions.residues[0].atoms)

    T_f = AFF(lambda x: x.positions.copy(), target).run().results.timeseries
    R_f = AFF(lambda x: x.positions.copy(), positions).run().results.timeseries
    frames = R_f.shape[0]

    # In case upto is specified, do this analysis for frames up to that number.
    if upto != None:
        T_f = T_f[:upto, ...]
        R_f = R_f[:upto, ...]

    pbc = mda_pbc_to_gro(u.dimensions)

    close_chols = []
    for current_frame, (P, C) in enumerate(zip(T_f, R_f)):
        close_a_mask = close_residues(P, C, residue_size, pbc=pbc, threshold=threshold)
        close_a = C[close_a_mask, :]
        close_a_sliced = close_a.reshape((-1, residue_size, 3))
        close_chols.append(close_a_sliced)

    return close_chols
