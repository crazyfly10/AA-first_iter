#!/usr/bin/env python3
"""
rename_backbone.py

Rename the backbone atoms of the capped modified cysteine to standard PDB
names, so prepgen can find them.

WHAT IT CHANGES
    Only column 2 of the @<TRIPOS>ATOM block, and only for 11 atoms.
    Coordinates, GAFF2 atom types, and AM1-BCC charges are untouched.

WHY
    prepgen reads the mainchain file (cyl.mc), which says HEAD_NAME N and
    TAIL_NAME C. It then looks in the MOL2 for atoms literally named "N" and
    "C". Avogadro named them "N" and "C31", so prepgen would not find them.

    Later, tleap matches your fibril PDB (which already uses N, CA, CB, SG)
    against the CYL template. The names have to agree.

USAGE
    python3 rename_backbone.py cyl_bcc.mol2 cyl_named.mol2
"""

import sys

# ---------------------------------------------------------------------------
# Atom number -> new name.
# Derived from the bond list: atom 5 bonds to N, CB and C (an alpha carbon),
# atom 7 is the thioether sulfur (GAFF2 type "ss"), and so on.
#
# CA / CB / SG are POSITIONS along the side chain (alpha, beta, gamma),
# not element symbols. This is standard PDB naming.
# ---------------------------------------------------------------------------
NEW_NAMES = {
    # --- the six that matter: prepgen and tleap both need these ---
    4:  "N",     # backbone amide nitrogen      -> HEAD atom
    5:  "CA",    # alpha carbon                 -> MAIN_CHAIN
    6:  "CB",    # beta carbon
    7:  "SG",    # side-chain sulfur (thioether)
    50: "C",     # backbone carbonyl carbon     -> TAIL atom
    51: "O",     # backbone carbonyl oxygen

    # --- the caps: named only so cyl.mc can list them for deletion ---
    1:  "CH3",   # acetyl methyl
    2:  "CAY",   # acetyl carbonyl carbon
    3:  "OY",    # acetyl oxygen
    52: "NT",    # N-methylamide nitrogen
    53: "CAT",   # N-methylamide methyl
}

# expected element for each, as a safety check
EXPECTED = {1: "C", 2: "C", 3: "O", 4: "N", 5: "C", 6: "C",
            7: "S", 50: "C", 51: "O", 52: "N", 53: "C"}


def main(infile, outfile):
    lines = open(infile).readlines()
    output = []
    in_atom_block = False
    renamed = 0

    for line in lines:

        if line.startswith("@<TRIPOS>ATOM"):
            in_atom_block = True
            output.append(line)
            continue

        if line.startswith("@<TRIPOS>"):          # any other section header
            in_atom_block = False
            output.append(line)
            continue

        if in_atom_block and line.strip():
            fields = line.split()
            atom_number = int(fields[0])

            if atom_number in NEW_NAMES:
                old_name  = fields[1]
                new_name  = NEW_NAMES[atom_number]
                gaff_type = fields[5]
                charge    = fields[8]

                # safety check: does the GAFF type start with the element
                # we expect?  (c3/c/c5 -> C,  ns/n -> N,  o -> O,  ss -> S)
                element = gaff_type[0].upper()
                if element != EXPECTED[atom_number]:
                    print(f"  !! atom {atom_number}: expected "
                          f"{EXPECTED[atom_number]}, found type '{gaff_type}'")
                    print("     Numbering does not match. Stopping.")
                    sys.exit(1)

                print(f"  {atom_number:3d}  {old_name:>5s} -> {new_name:<5s}"
                      f"  (type {gaff_type}, charge {charge})")

                # rebuild the line with the new name, keeping everything else
                fields[1] = new_name
                line = ("{:>7s} {:<8s} {:>9s} {:>9s} {:>9s} {:<8s} "
                        "{:>3s} {:<8s} {:>10s}\n").format(*fields[:9])
                renamed += 1

        output.append(line)

    open(outfile, "w").writelines(output)
    print(f"\nrenamed {renamed} atoms (expected {len(NEW_NAMES)})")
    print(f"wrote {outfile}")

    if renamed != len(NEW_NAMES):
        print("\n!! not all atoms were found - check the numbering")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python3 rename_backbone.py in.mol2 out.mol2")
    main(sys.argv[1], sys.argv[2])
