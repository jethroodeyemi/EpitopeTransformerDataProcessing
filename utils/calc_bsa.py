# --- CONFIGURATION (Change these for your complex) ---
PDB_FILE = "C:/Users/rye164/Downloads/1a14_cleaned.pdb"
PDB_OBJECT_NAME = "1a14_cleaned"
ANTIGEN_CHAINS = "chain N"
ANTIBODY_CHAINS = "(chain H or chain L)"
# --- END CONFIGURATION ---

# 0. Set Correct Parameters for SASA Calculation
set dot_solvent, on      # Crucial: enables calculation of solvent accessible surface
set solvent_radius, 1.4  # Standard water probe radius

# 1. Load the PDB and remove water molecules
load PDB_FILE
remove solvent

# 2. Calculate Complex SASA
sasa_complex = cmd.get_area(PDB_OBJECT_NAME)
print(f"SASA of full complex: {sasa_complex:.2f} Å²")

# 3. Calculate Antigen SASA
create antigen, f"{PDB_OBJECT_NAME} and {ANTIGEN_CHAINS}"
sasa_antigen = cmd.get_area('antigen')
print(f"SASA of isolated antigen: {sasa_antigen:.2f} Å²")

# 4. Calculate Antibody SASA
create antibody, f"{PDB_OBJECT_NAME} and {ANTIBODY_CHAINS}"
sasa_antibody = cmd.get_area('antibody')
print(f"SASA of isolated antibody: {sasa_antibody:.2f} Å²")

# 5. Calculate and Print Final BSA
bsa = (sasa_antigen + sasa_antibody) - sasa_complex
print(f"\n-------------------------------------------------")
print(f"ACCURATE Buried Surface Area (BSA): {bsa:.2f} Å²")
print(f"-------------------------------------------------")

# 6. Clean up
delete antigen
delete antibody