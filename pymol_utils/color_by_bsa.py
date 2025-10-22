from pymol import cmd

def color_by_bsa(pdb_object, antigen_chains, antibody_chains):
    """
    Calculates per-residue BSA and colors the antigen surface accordingly.
    
    USAGE: run color_by_bsa.py
           color_by_bsa 1a14_cleaned, N, H+L
    """
    # Sanitize inputs
    antigen_selection = f"({pdb_object}) and chain {antigen_chains}"
    antibody_selection = f"({pdb_object}) and chain {antibody_chains}"

    print("Step 1: Calculating per-residue SASA for the full complex...")
    # Store per-residue SASA in the b-factor property for the complex
    cmd.alter(pdb_object, "b=0.0") # Reset b-factors
    sasa_complex = cmd.get_area(pdb_object, load_b=1)

    # Store these complex SASA values in a dictionary
    sasa_complex_dict = {}
    cmd.iterate(pdb_object, "sasa_complex_dict[(chain, resi)] = b", space=locals())

    print("Step 2: Calculating per-residue SASA for the isolated antigen...")
    # Create a temporary object for the isolated antigen
    cmd.create("temp_antigen", antigen_selection)
    cmd.alter("temp_antigen", "b=0.0")
    sasa_isolated = cmd.get_area("temp_antigen", load_b=1)

    # Store isolated SASA values in a dictionary
    sasa_isolated_dict = {}
    cmd.iterate("temp_antigen", "sasa_isolated_dict[(chain, resi)] = b", space=locals())
    cmd.delete("temp_antigen") # Clean up

    print("Step 3: Calculating per-residue BSA and applying to structure...")
    # Calculate BSA and load it into the b-factor of the original antigen selection
    for (chain, resi), sasa_iso in sasa_isolated_dict.items():
        sasa_com = sasa_complex_dict.get((chain, resi), 0.0)
        bsa_residue = sasa_iso - sasa_com
        print(resi, bsa_residue)
        # Update the b-factor of the original object with the BSA value
        cmd.alter(f"{antigen_selection} and resi {resi}", f"b={bsa_residue}")

    print("Step 4: Coloring the antigen by the calculated BSA (stored in b-factor)...")
    cmd.show("surface", antigen_selection)
    # This will color the antigen from blue (low BSA) to red (high BSA)
    cmd.spectrum("b", "blue_white_red", selection=antigen_selection, minimum=0, maximum=150)
    cmd.ramp_new("bsa_ramp", pdb_object, [0, 50, 150], ["white", "yellow", "red"])
    cmd.color("bsa_ramp", antigen_selection)
    
    print("\nColoring complete. Residues with high BSA are now red.")

# Make the function available as a command in PyMOL
cmd.extend("color_by_bsa", color_by_bsa)