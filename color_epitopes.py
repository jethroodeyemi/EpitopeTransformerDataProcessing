# color_epitopes.py

import csv
from pymol import cmd
import os

def color_by_prediction(pdb_file, chain, csv_file, threshold=0.6):
    """
    Loads a PDB structure and colors it based on prediction results from a CSV file.

    Usage in PyMOL:
    run color_epitopes.py
    color_by_prediction pdb_id, chain, csv_file, [threshold]

    Example:
    color_by_prediction 4gxu, A, 4gxu_A_predictions.csv, threshold=0.6
    """
    threshold = float(threshold)
    pdb_id = os.path.basename(pdb_file).split('.')[0]
    # --- 1. Load and prepare the structure ---
    print(f"Loading structure for {pdb_id}...")
    cmd.load(pdb_file, pdb_id)
    cmd.remove("solvent")
    cmd.show_as("surface")
    cmd.color("gray80", pdb_id) # Set a default color

    # --- 2. Read the prediction data ---
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            # Must convert to a list to iterate multiple times if needed
            predictions = list(reader)
    except FileNotFoundError:
        print(f"Error: Cannot find the CSV file '{csv_file}'. Make sure it's in the correct directory.")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    print("Coloring residues based on prediction outcomes...")
    # --- 3. Define color schemes and apply colors ---
    colors = {
        'TP': 'green',  # True Positive
        'FP': 'red',    # False Positive
        'FN': 'yellow', # False Negative
    }

    # Create empty selections for each category first
    for category in colors:
        cmd.select(category, 'none')

    for row in predictions:
        try:
            res_id = row['res_id']
            # Important: PyMOL selections don't need quotes for insertion codes like '133A'
            selection_string = f"chain {chain} and resi {res_id}"
            
            score = float(row['prediction_score'])

            category = None
            if score >= threshold:
                category = 'TP'
            elif score >= threshold:
                category = 'FP'
            elif score < threshold:
                category = 'FN'
            
            if category:
                # Color the residue immediately
                cmd.color(colors[category], selection_string)
                # Add this residue to the named selection for the category
                cmd.select(category, f"{category} or ({selection_string})")

        except (KeyError, ValueError) as e:
            print(f"Warning: Skipping row due to missing data or parsing error: {row}. Error: {e}")
            continue

    # --- 4. Final Touches ---
    cmd.deselect() # Clear the current selection
    cmd.zoom("all")
    print("\nVisualization Complete!")
    print(f"  {colors['TP'].capitalize()}: True Positives (Correct predictions)")
    print(f"  {colors['FP'].capitalize()}: False Positives (Over-predictions)")
    print(f"  {colors['FN'].capitalize()}: False Negatives (Missed epitopes)")
    print("  Gray: True Negatives (Correctly ignored)")

# Make this script a callable command in PyMOL
cmd.extend("color_by_prediction", color_by_prediction)