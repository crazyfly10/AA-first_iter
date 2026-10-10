#!/usr/bin/env python3
"""
Extract CG junction bonded terms from the atomistic fibril+linker trajectory.

The terms involve CG BEADS, not atoms:
    F bead  = centre of mass of 4 HET-s monomers   (b = 3.824 nm)
    M bead  = succinimide ring + amide
    PEG bead= one ethylene-oxide repeat
    B bead  = biotin (head/tail split not applied here)

So gmx gangle cannot be used - the F bead centroid sits ~2 nm from the
junction and corresponds to no single atom. This script maps first.

Terms extracted (per linker):
    bond      M - F1
    angle     PEG1 - M - F1
    angle     M - F1 - F2
    dihedral  PEG2 - PEG1 - M - F1
    dihedral  PEG1 - M - F1 - F2

Requires: pip install MDAnalysis

Usage:
    python3 extract_junction.py prod.tpr prod.xtc
"""

import sys
import numpy as np

try:
    import MDAnalysis as mda
except ImportError:
    sys.exit("pip install MDAnalysis")

TPR = sys.argv[1] if len(sys.argv) > 1 else "prod.tpr"
XTC = sys.argv[2] if len(sys.argv) > 2 else "prod.xtc"
STRIDE = int(sys.argv[3]) if len(sys.argv) > 3 else 1

RES_PER_CHAIN = 73
N_CHAINS = 12
MONOMERS_PER_BEAD = 4          # b = 4 x 9.56 A = 3.824 nm

u = mda.Universe(TPR, XTC)
print(f"{len(u.trajectory)} frames, {len(u.atoms)} atoms")

# ---------------------------------------------------------------- CYL sites
cyl = u.select_atoms("resname CYL")
cyl_res = cyl.residues
print(f"CYL residues found: {len(cyl_res)}  "
      f"(resids {[r.resid for r in cyl_res]})")
if len(cyl_res) != 2:
    sys.exit("expected 2 CYL residues")

# ------------------------------------------------------- bead atom groups
# M bead: succinimide ring + the amide that joins it to PEG.
# Atom names come from your prep file; adjust if yours differ.
M_NAMES  = "C4 C5 C6 O1 N1 C30 O9 C7 C8 C9 O2 N2"
# PEG beads: each repeat is  C-C-O.  PEG1 is nearest the maleimide.
PEG_SETS = ["C10 C11 O3", "C12 C13 O4", "C14 C15 O5", "C16 C17 O6"]

def bead(res, names):
    """centre of mass of the named atoms within one CYL residue"""
    sel = res.atoms.select_atoms("name " + names)
    if sel.n_atoms == 0:
        sys.exit(f"no atoms matched '{names}' in CYL {res.resid}")
    return sel

def fibril_bead(first_chain):
    """COM group for MONOMERS_PER_BEAD consecutive monomers"""
    lo = first_chain * RES_PER_CHAIN + 1
    hi = (first_chain + MONOMERS_PER_BEAD) * RES_PER_CHAIN
    return u.select_atoms(f"protein and resid {lo}:{hi} and backbone")

# which fibril beads flank each linker: chain-A linker -> chains A-D then E-H
#                                        chain-L linker -> chains I-L then E-H
SITES = []
for res in cyl_res:
    chain_idx = (res.resid - 1) // RES_PER_CHAIN
    if chain_idx < N_CHAINS / 2:                 # near the A end
        f1, f2 = fibril_bead(0), fibril_bead(4)
        label = "A-end"
    else:                                        # near the L end
        f1, f2 = fibril_bead(8), fibril_bead(4)
        label = "L-end"
    SITES.append(dict(label=label, res=res, F1=f1, F2=f2,
                      M=bead(res, M_NAMES),
                      PEG=[bead(res, s) for s in PEG_SETS]))
    print(f"  {label}: CYL resid {res.resid}, chain index {chain_idx}, "
          f"M bead {SITES[-1]['M'].n_atoms} atoms, "
          f"F1 {f1.n_atoms} atoms")

# ------------------------------------------------------------- geometry
def ang(a, b, c):
    v1, v2 = a - b, c - b
    cosv = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cosv, -1, 1)))

def dih(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x, y = np.dot(v, w), np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))

TERMS = ["bond_M_F1", "angle_PEG_M_F1", "angle_M_F1_F2",
         "dih_PEG2_PEG1_M_F1", "dih_PEG1_M_F1_F2"]
data = {s["label"]: {t: [] for t in TERMS} for s in SITES}

for ts in u.trajectory[::STRIDE]:
    for s in SITES:
        M  = s["M"].center_of_mass()
        F1 = s["F1"].center_of_mass()
        F2 = s["F2"].center_of_mass()
        P1 = s["PEG"][0].center_of_mass()
        P2 = s["PEG"][1].center_of_mass()
        d = data[s["label"]]
        d["bond_M_F1"].append(np.linalg.norm(M - F1) / 10.0)   # nm
        d["angle_PEG_M_F1"].append(ang(P1, M, F1))
        d["angle_M_F1_F2"].append(ang(M, F1, F2))
        d["dih_PEG2_PEG1_M_F1"].append(dih(P2, P1, M, F1))
        d["dih_PEG1_M_F1_F2"].append(dih(P1, M, F1, F2))

# --------------------------------------------------------------- report
def transitions(series, is_dih):
    """count basin changes; only meaningful for dihedrals"""
    if not is_dih:
        return None
    s = np.array(series)
    basin = np.digitize(s, [-120, 0, 120])
    return int(np.sum(basin[1:] != basin[:-1]))

print(f"\n{'site':<8}{'term':<22}{'mean':>10}{'std':>9}"
      f"{'min':>9}{'max':>9}{'trans':>7}")
print("-" * 74)

for s in SITES:
    lab = s["label"]
    for t in TERMS:
        v = np.array(data[lab][t])
        tr = transitions(v, t.startswith("dih"))
        print(f"{lab:<8}{t:<22}{v.mean():>10.3f}{v.std():>9.3f}"
              f"{v.min():>9.3f}{v.max():>9.3f}"
              f"{'' if tr is None else tr:>7}")
        np.savetxt(f"{lab}_{t}.dat", v)

print("""
Reading this
  bond and angle terms have no basins - a single peak is correct and the
  std is the thermal width you will invert.

  dihedrals need basin transitions. Under ~10 the relative populations are
  not trustworthy, and replicas started from rotated torsions are needed.

  Compare the two sites. They sit at opposite fibril ends in different local
  environments; if a term differs markedly between them, that is worth
  understanding before you invert it.

  Raw series written to <site>_<term>.dat for histogramming and Boltzmann
  inversion. Remember the Jacobians: r^2 for bonds, sin(theta) for angles,
  none for dihedrals.
""")
