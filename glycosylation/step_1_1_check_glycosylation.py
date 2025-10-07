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


def get_unique_protein_chains(directory: Path) -> set:
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
    return unique_pairs

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

def get_uniprot_id(entity_data: list, chain_id_from_file: str) -> str | None:
    """
    Parses the polymer entity data from GraphQL to find the correct UniProt ID
    for a given chain ID. This version is robust against missing data keys.
    """
    for entity in entity_data:
        for instance in entity.get('polymer_entity_instances', []):
            ids = instance.get('rcsb_polymer_entity_instance_container_identifiers', {})
            
            if ids.get('asym_id') == chain_id_from_file or ids.get('auth_asym_id') == chain_id_from_file:
                # This is the correct entity. Now safely find its UniProt ID.
                
                # Step 1: Safely get the container dictionary
                identifiers_container = entity.get('rcsb_polymer_entity_container_identifiers')
                
                # Step 2: Check if the container itself exists before proceeding
                if identifiers_container:
                    # Step 3: Get the list of reference sequences. This could be None.
                    ref_seqs = identifiers_container.get('reference_sequence_identifiers')
                    
                    # Step 4: CRITICAL FIX - Check if ref_seqs is a list before looping
                    if ref_seqs:
                        for ref in ref_seqs:
                            if ref.get('database_name') == 'UniProt':
                                return ref.get('database_accession')
                
                # If we get here, no UniProt ID was found for this matching chain
                return None
    
    return None # Return None if the chain itself was never found

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
                    glycosylation_info.append({
                        "position": str(position),
                        "description": full_description_str
                    })
            return glycosylation_info
        else: return []
    except requests.exceptions.RequestException: return []

def main():
    protein_chains = get_unique_protein_chains(EMBEDDING_DIR)
    if not protein_chains:
        print("No valid protein files found. Exiting.")
        return
        
    print(f"Found {len(protein_chains)} unique PDB/chain pairs.")
    unique_pdb_ids = sorted(list(set(pdb_id for pdb_id, chain_id in protein_chains)))
    
    pdb_data_map = fetch_pdb_data_in_batches(unique_pdb_ids)
    
    results = {}
    print("\nProcessing individual chains and querying UniProt...")
    for pdb_id, chain_id in tqdm(protein_chains, desc="Analyzing Chains"):
        entity_data = pdb_data_map.get(pdb_id)
        if not entity_data:
            results[f"{pdb_id}_{chain_id}"] = {"uniprot_id": None, "glycosylation": ["PDB data not found"]}
            continue
        
        uniprot_id = get_uniprot_id(entity_data, chain_id)
        if not uniprot_id:
            results[f"{pdb_id}_{chain_id}"] = {"uniprot_id": None, "glycosylation": ["UniProt ID not found for this chain"]}
            continue
        
        time.sleep(0.05) # Small delay to be polite to UniProt API
        glycosylation_sites = get_glycosylation_sites(uniprot_id)
        results[f"{pdb_id}_{chain_id}"] = {"uniprot_id": uniprot_id, "glycosylation": glycosylation_sites}

    print(f"\nWriting results to {OUTPUT_CSV_FILE}...")
    try:
        with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['PDB_ID', 'Chain_ID', 'UniProt_ID', 'Is_Glycosylated', 'Glycosylation_Residue_Numbers', 'Glycosylation_Sites'])
            
            for key, data in sorted(results.items()):
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
                    full_sites_str = sites_data[0] if isinstance(sites_data, list) and sites_data else 'Unknown Error'

                writer.writerow([pdb_id, chain_id, uniprot_id_str, is_glycosylated, residue_numbers_str, full_sites_str])
        print("Successfully saved the report.")
    except IOError as e:
        print(f"Error: Could not write to file {OUTPUT_CSV_FILE}. Reason: {e}")

    print("\n\n--- Glycosylation Report (Console Summary) ---")
    glycosylated_count = 0
    for key, data in sorted(results.items()):
        print(f"\n--- Protein: {key} ---")
        if data['uniprot_id']:
            print(f"  UniProt ID: {data['uniprot_id']}")
            if data['glycosylation']:
                glycosylated_count += 1
                print("  Glycosylation Sites Found.")
            else:
                print("  No glycosylation sites listed in UniProt.")
        else:
            error_msg = data['glycosylation'][0] if isinstance(data['glycosylation'], list) and data['glycosylation'] else "Unknown Error"
            print(f"  Could not map to a UniProt ID. Reason: {error_msg}")
    
    print("\n--- Summary ---")
    print(f"Total proteins checked: {len(results)}")
    print(f"Proteins with glycosylation data: {glycosylated_count}")

if __name__ == "__main__":
    main()