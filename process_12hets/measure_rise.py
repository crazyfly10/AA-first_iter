#!/usr/bin/env python3
"""
Measure the inter-monomer rise of a stacked fibril across an MD trajectory.

The rise is the distance between the centres of mass of the ORDERED CORE of
consecutive monomers. Using the core (the four beta-strands) rather than the
whole monomer matters: the disordered N-terminal region and the unstructured
segment between beta2 and beta3 wander, and including them adds noise that has
nothing to do with the stacking spacing.

Requires: pip install MDAnalysis

Usage:
    python3 measure_rise.py prod.tpr nojump.xtc
"""

import sys
import numpy as np

try:
    import MDAnalysis as mda
except ImportError:
    sys.exit("MDAnalysis not found.  pip install MDAnalysis")

# ----------------------------------------------------------------- config
TPR = sys.argv[1] if len(sys.argv) > 1 else "prod.tpr"
XTC = sys.argv[2] if len(sys.argv) > 2 else "nojump.xtc"

RES_PER_CHAIN = 73          # HET-s 217-289 after His-tag removal
N_CHAINS      = 12

# Ordered core, as OFFSETS from the first residue of each chain.
# Author numbering 226-234, 237-245, 262-270, 273-282 with chain start 217.
CORE_OFFSETS = (list(range(9, 18)) + list(range(20, 29)) +
                list(range(45, 54)) + list(range(56, 66)))

INTERIOR = range(3, 9)      # chains D-I (0-indexed 3..8): ignore fraying ends
STRIDE   = 10               # analyse every Nth frame

# ------------------------------------------------------------------ setup
u = mda.Universe(TPR, XTC)
print(f"{len(u.trajectory)} frames, {len(u.atoms)} atoms")

prot = u.select_atoms("protein")
n_res = prot.n_residues
print(f"protein residues: {n_res}  (expected {RES_PER_CHAIN * N_CHAINS})")

if n_res != RES_PER_CHAIN * N_CHAINS:
    print("\n!! residue count mismatch - check RES_PER_CHAIN / N_CHAINS")
    print("   first/last residue of the protein selection:")
    print("  ", prot.residues[0].resname, prot.residues[0].resid)
    print("  ", prot.residues[-1].resname, prot.residues[-1].resid)

# build one CA selection per chain, restricted to the ordered core
cores = []
for c in range(N_CHAINS):
    idx = [c * RES_PER_CHAIN + o for o in CORE_OFFSETS]
    sel = prot.residues[idx].atoms.select_atoms("name CA")
    cores.append(sel)
    if c == 0:
        print(f"core CA atoms per chain: {sel.n_atoms}  (expected 37)")

# ---------------------------------------------------------------- measure
pairs = [(c, c + 1) for c in INTERIOR if c + 1 < N_CHAINS]
rises = {p: [] for p in pairs}
times = []

for ts in u.trajectory[::STRIDE]:
    times.append(ts.time)
    for a, b in pairs:
        d = cores[b].center_of_mass() - cores[a].center_of_mass()
        rises[(a, b)].append(np.linalg.norm(d))

times = np.array(times)
print(f"\nanalysed {len(times)} frames "
      f"({times[0]/1000:.1f} - {times[-1]/1000:.1f} ns)\n")

# ----------------------------------------------------------------- report
letters = "ABCDEFGHIJKL"
print(f"{'pair':<8}{'mean (A)':>11}{'std':>9}{'min':>9}{'max':>9}")
print("-" * 46)

allv = []
for (a, b) in pairs:
    v = np.array(rises[(a, b)])
    allv.append(v)
    print(f"{letters[a]}-{letters[b]:<6}{v.mean():>11.3f}{v.std():>9.3f}"
          f"{v.min():>9.3f}{v.max():>9.3f}")

allv = np.concatenate(allv)
print("-" * 46)
print(f"{'ALL':<8}{allv.mean():>11.3f}{allv.std():>9.3f}"
      f"{allv.min():>9.3f}{allv.max():>9.3f}")

# drift over the trajectory, per interface then pooled
pooled = np.mean([rises[p] for p in pairs], axis=0)
slope = np.polyfit(times, pooled, 1)[0] * 1000.0        # A per ns
print(f"\ndrift: {slope:+.4f} A/ns "
      f"({slope * (times[-1]-times[0]) / 1000:+.3f} A over the window)")

# ---------------------------------------------------------------- verdict
BUILD = 9.29
off = allv.mean() - BUILD
print(f"\nbuild value        {BUILD:.2f} A")
print(f"trajectory mean    {allv.mean():.3f} +/- {allv.std():.3f} A")
print(f"difference         {off:+.3f} A  ({100*off/BUILD:+.2f} %)")

print()
if abs(off) < 0.15:
    print("VERDICT: rise preserved. Bead spacing b = 4 x rise stands.")
else:
    print("VERDICT: rise has shifted. Recompute b and k_theta = l_p*kT/b,")
    print("         since both scale with the bead spacing.")

print("""
Note on interpretation
  A standard deviation of ~0.1-0.2 A is normal thermal breathing of the
  stacking interface and is not drift. What matters is the mean, and whether
  the slope is consistent across the window rather than trending one way.

  Only interior chains are used. Terminal monomers fray - that is real
  physics, but their spacing is not representative of a long fibril.
""")

# optional plot
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 4.5))
    for (a, b) in pairs:
        plt.plot(times / 1000, rises[(a, b)], lw=0.7, alpha=0.5,
                 label=f"{letters[a]}-{letters[b]}")
    plt.plot(times / 1000, pooled, lw=2, color="k", label="mean")
    plt.axhline(BUILD, ls="--", color="r", label=f"build = {BUILD} A")
    plt.xlabel("time (ns)")
    plt.ylabel("inter-monomer rise (A)")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("rise.png", dpi=150)
    print("wrote rise.png")
except ImportError:
    pass
