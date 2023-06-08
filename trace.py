"""
Display a trace and hydrophobic moments.

Koen Westendorp, 2023.
"""

from pymol.cgo import SAUSAGE, COLOR, SPHERE, CYLINDER
from pymol import cmd
import numpy as np
from matplotlib import colormaps


HYDROPHOBICITY_CMAP = colormaps["viridis"]


# Eisenberg, D., Schwarz, E., Komaromy, M. & Wall, R. (1984). Analysis of
# membrane and surface protein sequences with the hydrophobic moment plot.
# _Journal of Molecular Biology_, 179(1), 125--142.
# https://doi.org/10.1016/0022-2836(84)90309-7
# fmt: off
EISENBERG_NORMALIZED_CONSENSUS = {
    "ALA":  0.62,  # A
    "ARG": -2.53,  # R
    "ASN": -0.78,  # N
    "ASP": -0.90,  # D
    "CYS":  0.29,  # C
    "GLN": -0.85,  # Q
    "GLU": -0.74,  # E
    "GLY":  0.48,  # G
    "HIS": -0.40,  # H
    "ILE":  1.38,  # I
    "LEU":  1.06,  # L
    "LYS": -1.50,  # K
    "MET":  0.64,  # M
    "PHE":  1.19,  # F
    "PRO":  0.12,  # P
    "SER": -0.18,  # S
    "THR": -0.05,  # T
    "TYR":  0.26,  # Y
    "TRP":  0.81,  # W
    "VAL":  1.08,  # V
}
# fmt: on


def hydrophobic_moments(
    selection="backbone and name CA",
    name="hydrophobic_moments",
    scale=EISENBERG_NORMALIZED_CONSENSUS,
):
    """
    Create cgo objects of an averaged trace of the backbone and the hydrophobic
    moments of the residues projected onto it as lines extending from the
    trace. Spheres indicating the tips of the hydrophobic moments are also
    drawn, as well as spheres colored according to the hydrophobicity at the
    position of the residue's C-alpha.
    """
    # TODO parametarize. For now it is hard-coded because we also do the
    # calculation of hydrophobic moment using 3 due to hard implementation.
    # However, we can actually decouple the tracing from the calculation of the
    # hydrophobic moment.
    n_res = 3
    atoms = cmd.get_model(selection).atom

    residues = [atom.resn for atom in atoms]
    positions = np.array([atom.coord for atom in atoms])

    # Calculate the average positions of the alpha carbons by looking at a
    # point, the preceding one, and the following one.
    trace = [
        (prev + curr + next) / 3.0
        for prev, curr, next in zip(positions[:-2], positions[1:-1], positions[2:])
    ]
    # Hydrophobicity (such as Eisenberg) in same order as positions.
    hydros = np.array([scale[resn] for resn in residues])

    # Hydrophobic moment is calculated here to be the direction of the alpha C
    # from the trace point with the magintude of the hydrophobicity of the
    # residue of which the alpha C is a part.
    contributions = []
    for trace_point, residue_point, hydrophobicity in zip(trace, positions[1:], hydros):
        # Direction away from trace to residue position.
        v = residue_point - trace_point
        direction = v / np.linalg.norm(v)
        moment_contribution = direction * hydrophobicity
        contributions.append(moment_contribution)
    moments = contributions
    starts = trace  # All of the trace points with the first and last cut off.
    ends = [start + moment for start, moment in zip(starts, moments)]

    # Draw the alpha carbon locations.
    cgo_alpha_c = [
        u
        for hy, ca in zip(hydros, positions)
        for u in [
            COLOR,
            *HYDROPHOBICITY_CMAP(hy / 2),
            SPHERE,
            *ca,
            0.3,
        ]
    ]
    # Draw the trace.
    cgo_trace = [
        u
        for start, end in zip(starts[:-1], starts[1:])
        for u in [SAUSAGE, *start, *end, 0.1, 0.4, 1, 0.6, 0.4, 1, 0.6]
        # Don't show any traces that are unrealistically long.
        if np.linalg.norm(end - start) < 4
    ]
    # Finally, draw the hydrophobic moments.
    cgo_moments = [
        u
        for start, end in zip(starts, ends)
        for u in [SAUSAGE, *start, *end, 0.1, 1, 1, 0, 1, 0, 0]
    ]
    cgo_moment_end = [u for end in ends for u in [COLOR, 0, 0, 1, SPHERE, *end, 0.2]]
    # The names have been prepended by a dot in order to be able to quickly
    # delete the objects in PyMOL using `delete .*`.
    cmd.load_cgo(cgo_trace, f".{name}_trace")
    cmd.load_cgo(cgo_moments, f".{name}_moments")
    cmd.load_cgo(cgo_moment_end, f".{name}_moment_end")
    cmd.load_cgo(cgo_alpha_c, f".{name}_alpha_c")
    return cgo


cmd.extend("hydrophobic_moments", hydrophobic_moments)
