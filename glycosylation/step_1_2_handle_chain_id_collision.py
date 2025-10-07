import os
import requests
import time
import csv
from pathlib import Path
from tqdm import tqdm
import sys

# --- Configuration ---
EMBEDDING_DIR = Path("./embedding_cache")
OUTPUT_CSV_FILE = "glycosylation_report.csv"

# --- API Endpoints ---
PDB_GRAPHQL_API = "https://data.rcsb.org/graphql"
UNIPROT_API = "https://rest.uniprot.org/uniprotkb/{}.json"


def get_unique_protein_chains(directory: Path) -> list:
    """Parses filenames in a directory to get a sorted list of (PDB_ID, Chain_ID) tuples."""
    if not directory.is_dir():
        print(f"Error: Directory '{directory}' not found.")
        sys.exit(1)
        
    unique_pairs = set()
    for filename in directory.iterdir():
        if filename.suffix == '.npy':
            parts = filename.stem.split('_')
            if len(parts) >= 2:
                pdb_id = parts[0].upper()
                chain_id = parts[1]
                unique_pairs.add((pdb_id, chain_id))
    return sorted(list(unique_pairs))

def build_graphql_query(pdb_ids: list[str]) -> str:
    formatted_pdb_ids = ", ".join(f'"{id}"' for id in pdb_ids)
    query = f"""
    {{
      entries(entry_ids: [{formatted_pdb_ids}]) {{
        rcsb_id
        polymer_entities {{
          rcsb_polymer_entity_container_identifiers {{
            reference_sequence_identifiers {{
              database_accession
              database_name
            }}
          }}
          polymer_entity_instances {{
            rcsb_polymer_entity_instance_container_identifiers {{
              asym_id
              auth_asym_id
            }}
          }}
        }}
      }}
    }}
    """
    return query

def fetch_pdb_data_in_batches(pdb_ids: list[str]) -> dict:
    pdb_data_map = {}
    batch_size = 100

    for i in tqdm(range(0, len(pdb_ids), batch_size), desc="Fetching PDB Data"):
        batch_ids = pdb_ids[i:i+batch_size]
        query = build_graphql_query(batch_ids)
        try:
            response = requests.post(PDB_GRAPHQL_API, json={'query': query})
            response.raise_for_status()
            data = response.json()
            for entry in data.get("data", {}).get("entries", []):
                pdb_data_map[entry['rcsb_id']] = entry['polymer_entities']
        except requests.exceptions.RequestException as e:
            print(f"Error fetching PDB data for batch {batch_ids}: {e}")
            
    return pdb_data_map

def get_uniprot_id_robust(entity_data: list, chain_id_from_file: str) -> str | None:
    """
    Robustly parses GraphQL data to find the UniProt ID, prioritizing author chain ID.
    """
    auth_match = None
    asym_match = None

    for entity in entity_data:
        # First, find the UniProt ID for this entity, if it exists
        uniprot_id = None
        identifiers_container = entity.get('rcsb_polymer_entity_container_identifiers')
        if identifiers_container:
            ref_seqs = identifiers_container.get('reference_sequence_identifiers')
            if ref_seqs:
                for ref in ref_seqs:
                    if ref.get('database_name') == 'UniProt':
                        uniprot_id = ref.get('database_accession')
                        break
        
        # Now check if any of its instances match our target chain ID
        for instance in entity.get('polymer_entity_instances', []):
            ids = instance.get('rcsb_polymer_entity_instance_container_identifiers', {})
            if ids.get('auth_asym_id') == chain_id_from_file:
                auth_match = uniprot_id
            if ids.get('asym_id') == chain_id_from_file:
                asym_match = uniprot_id

    # Prioritize the match from the author-provided chain ID, as it's less ambiguous
    return auth_match if auth_match else asym_match

def get_glycosylation_sites(uniprot_id: str) -> list[dict]:
    if not uniprot_id: return []
    url = UNIPROT_API.format(uniprot_id)
    glycosylation_info = []
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            for feature in data.get("features", []):
                if feature.get("type") == "Glycosylation":
                    location = feature.get('location', {})
                    position = location.get('start', {}).get('value', 'N/A')
                    description = feature.get("description", "No description")
                    full_description_str = f"Residue {position}: {description}"
                    glycosylation_info.append({"position": str(position), "description": full_description_str})
            return glycosylation_info
        else: return []
    except requests.exceptions.RequestException: return []

def process_chains(protein_chains: list) -> dict:
    """Takes a list of (pdb, chain) tuples and returns a results dictionary."""
    results = {}
    if not protein_chains:
        return results

    unique_pdb_ids = sorted(list(set(pdb_id for pdb_id, chain_id in protein_chains)))
    pdb_data_map = fetch_pdb_data_in_batches(unique_pdb_ids)
    
    print("\nProcessing individual chains and querying UniProt...")
    for pdb_id, chain_id in tqdm(protein_chains, desc="Analyzing Chains"):
        key = f"{pdb_id}_{chain_id}"
        entity_data = pdb_data_map.get(pdb_id)
        if not entity_data:
            results[key] = {"uniprot_id": None, "glycosylation": ["PDB data not found"]}
            continue
        
        uniprot_id = get_uniprot_id_robust(entity_data, chain_id)
        if not uniprot_id:
            results[key] = {"uniprot_id": None, "glycosylation": ["UniProt ID not found for this chain"]}
            continue
        
        time.sleep(0.05)
        glycosylation_sites = get_glycosylation_sites(uniprot_id)
        results[key] = {"uniprot_id": uniprot_id, "glycosylation": glycosylation_sites}
    return results

def main():
    """Main function to run or update the glycosylation report."""
    output_path = Path(OUTPUT_CSV_FILE)
    chains_to_process = []
    
    # --- Mode Detection: Initial Run vs. Update ---
    if output_path.exists():
        print(f"Found existing report: '{OUTPUT_CSV_FILE}'. Checking for entries to fix.")
        chains_to_fix = []
        with open(output_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get('UniProt_ID') == 'N/A':
                    chains_to_fix.append((row['PDB_ID'], row['Chain_ID']))
        
        if not chains_to_fix:
            print("No entries with 'N/A' UniProt ID found. Nothing to do. Exiting.")
            return
            
        print(f"Found {len(chains_to_fix)} entries to fix.")
        chains_to_process = chains_to_fix
        
    else:
        print("No existing report found. Starting initial run.")
        chains_to_process = get_unique_protein_chains(EMBEDDING_DIR)
        if not chains_to_process:
            print("No valid protein files found in cache. Exiting.")
            return

    # --- Run Processing ---
    new_results = process_chains(chains_to_process)
    
    # --- Write/Update CSV File ---
    if not new_results and output_path.exists():
        print("\nNo new data was successfully fetched. The report remains unchanged.")
        return

    print(f"\nUpdating results in {OUTPUT_CSV_FILE}...")
    
    # Read existing data if in update mode, otherwise start fresh
    final_data = []
    if output_path.exists():
        with open(output_path, 'r', newline='', encoding='utf-8') as csvfile:
            final_data = list(csv.DictReader(csvfile))
        # Update the rows with new results
        for i, row in enumerate(final_data):
            key = f"{row['PDB_ID']}_{row['Chain_ID']}"
            if key in new_results:
                print(f"Updating row for {key}...")
                # Re-create the row with updated info
                data = new_results[key]
                uniprot_id_str = data.get('uniprot_id') or 'N/A'
                sites_data = data.get('glycosylation', [])
                
                is_glycosylated = 'No'
                residue_numbers_str = ''
                full_sites_str = ''
                
                if isinstance(sites_data, list) and len(sites_data) > 0 and isinstance(sites_data[0], dict):
                    is_glycosylated = 'Yes'
                    residue_numbers = [site['position'] for site in sites_data if site['position'] != 'N/A']
                    residue_numbers_str = ", ".join(residue_numbers)
                    full_descriptions = [site['description'] for site in sites_data]
                    full_sites_str = "; ".join(full_descriptions)
                elif not data.get('uniprot_id'):
                    full_sites_str = sites_data[0] if isinstance(sites_data, list) and sites_data else 'Update Failed'
                
                final_data[i] = {
                    'PDB_ID': row['PDB_ID'], 'Chain_ID': row['Chain_ID'], 'UniProt_ID': uniprot_id_str,
                    'Is_Glycosylated': is_glycosylated, 'Glycosylation_Residue_Numbers': residue_numbers_str,
                    'Glycosylation_Sites': full_sites_str
                }
    else: # Initial run mode
        for key, data in sorted(new_results.items()):
            pdb_id, chain_id = key.split('_', 1)
            uniprot_id_str = data.get('uniprot_id') or 'N/A'
            sites_data = data.get('glycosylation', [])
            is_glycosylated = 'No'
            residue_numbers_str = ''
            full_sites_str = ''
            if isinstance(sites_data, list) and len(sites_data) > 0 and isinstance(sites_data[0], dict):
                is_glycosylated = 'Yes'
                residue_numbers = [site['position'] for site in sites_data if site['position'] != 'N/A']
                residue_numbers_str = ", ".join(residue_numbers)
                full_descriptions = [site['description'] for site in sites_data]
                full_sites_str = "; ".join(full_descriptions)
            elif not data.get('uniprot_id'):
                full_sites_str = sites_data[0] if isinstance(sites_data, list) and sites_data else 'Error'
            final_data.append({
                'PDB_ID': pdb_id, 'Chain_ID': chain_id, 'UniProt_ID': uniprot_id_str,
                'Is_Glycosylated': is_glycosylated, 'Glycosylation_Residue_Numbers': residue_numbers_str,
                'Glycosylation_Sites': full_sites_str
            })

    # Write the final combined data to the CSV
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            header = ['PDB_ID', 'Chain_ID', 'UniProt_ID', 'Is_Glycosylated', 'Glycosylation_Residue_Numbers', 'Glycosylation_Sites']
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
            writer.writerows(final_data)
        print("Successfully saved the report.")
    except IOError as e:
        print(f"Error: Could not write to file {output_path}. Reason: {e}")

if __name__ == "__main__":
    main()