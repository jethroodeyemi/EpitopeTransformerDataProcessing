import json
import requests
import time
import logging
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_JSON = "output/data_splits_on_spike_proteins_cluster.json"

# JSON Outputs (for training)
OUTPUT_JSON_STRICT = "output/data_splits_STRICT_no_viruses.json"
OUTPUT_JSON_MODERATE = "output/data_splits_MODERATE_no_sars_mers.json"

# CSV Outputs (for inspection)
CSV_FILES = {
    "SARS2":       "output/list_sars_cov_2.csv",
    "SARS1":       "output/list_sars_cov_1_removed.csv",
    "MERS":        "output/list_mers_cov_removed.csv",
    "OTHER_VIRUS": "output/list_other_viruses.csv",
    "CLEAN":       "output/list_clean_non_viral.csv",
    "MISSING":     "output/list_errors_missing.csv"
}

LOG_FILE = "processing.log"

BATCH_SIZE = 50
MAX_WORKERS = 20
MAX_RETRIES = 5

# ==========================================
# 0. LOGGING SETUP
# ==========================================
with open(LOG_FILE, 'w'): pass
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logger = logging.getLogger()

# ==========================================
# 1. GRAPHQL WORKER
# ==========================================
def fetch_metadata_batch(batch_ids):
    url = "https://data.rcsb.org/graphql"
    query = """
    query($ids: [String!]!) {
      entries(entry_ids: $ids) {
        rcsb_id
        struct {
          title
          pdbx_descriptor
        }
        polymer_entities {
          rcsb_entity_source_organism {
            scientific_name
            ncbi_taxonomy_id
          }
        }
      }
    }
    """
    clean_ids = [pid.upper() for pid in batch_ids]
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(url, json={"query": query, "variables": {"ids": clean_ids}}, timeout=10)
            if response.status_code == 200:
                return response.json()
            elif response.status_code in [429, 503, 504]:
                time.sleep(2 ** attempt)
            else:
                return None
        except Exception:
            time.sleep(1)
    return None

# ==========================================
# 2. CLASSIFICATION LOGIC
# ==========================================
def parse_and_classify(entry_data):
    # API always returns ID in Uppercase
    pid = entry_data["rcsb_id"]
    
    struct = entry_data.get("struct") or {}
    title = (struct.get("title") or "").strip()
    desc = (struct.get("pdbx_descriptor") or "").strip()
    
    # Store clean title/desc for CSV output
    # Normalize to avoid CSV breaking if needed (csv module handles quotes usually)
    csv_title = title.replace("\n", " ")
    csv_desc = desc.replace("\n", " ")

    organisms = []
    tax_ids = []
    
    entities = entry_data.get("polymer_entities") or []
    for entity in entities:
        orgs = entity.get("rcsb_entity_source_organism") or []
        for org in orgs:
            s_name = (org.get("scientific_name") or "").upper()
            if s_name: organisms.append(s_name)
            t_id = org.get("ncbi_taxonomy_id")
            if t_id: tax_ids.append(t_id)

    full_search_string = (csv_title + " " + csv_desc + " " + " ".join(organisms)).upper()

    # Classification Logic
    classification = "CLEAN" # Default

    # 1. SARS-CoV-2 (Keep)
    if 2697049 in tax_ids or \
       "SARS-COV-2" in full_search_string or \
       "COVID-19" in full_search_string or \
       "2019-NCOV" in full_search_string:
        classification = "SARS2"

    # 2. SARS-CoV-1 (Remove) - Overwrites Clean
    elif 694009 in tax_ids or \
         ("SARS" in full_search_string and "SARS-COV-2" not in full_search_string) or \
         "SEVERE ACUTE RESPIRATORY SYNDROME" in full_search_string:
        classification = "SARS1"

    # 3. MERS (Remove) - Overwrites Clean
    elif 1335626 in tax_ids or \
         "MERS" in full_search_string or \
         "MIDDLE EAST RESPIRATORY SYNDROME" in full_search_string:
        classification = "MERS"

    # 4. Other Virus - Overwrites Clean
    else:
        viral_keywords = ["VIRUS", "VIRAL", "CORONAVIRUS", "HCOV", "FLU", "INFLUENZA", "HIV", "EBOLA", "FUSION PROTEIN", "SPIKE", "RBD"]
        if any("VIRUS" in org for org in organisms) or any(k in full_search_string for k in viral_keywords):
            classification = "OTHER_VIRUS"

    # Return dictionary with all info needed for CSV and Splitting
    return {
        "id": pid,
        "class": classification,
        "title": csv_title,
        "desc": csv_desc
    }

# ==========================================
# 3. MAIN
# ==========================================
def main():
    start_time = time.time()
    logger.info("Starting Processing with CSV Generation...")
    
    with open(INPUT_JSON, "r") as f:
        original_data = json.load(f)

    # Flatten and get unique IDs
    all_ids = list(set(pid for split in original_data.values() for pid in split))
    logger.info(f"Loaded {len(all_ids)} unique PDB IDs.")

    # Create Batches
    batches = [all_ids[i:i + BATCH_SIZE] for i in range(0, len(all_ids), BATCH_SIZE)]
    
    # Store full metadata: { "1ABC": { "id": "1ABC", "class": "SARS2", "title": "...", "desc": "..." } }
    full_data_map = {}
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_batch = {executor.submit(fetch_metadata_batch, batch): batch for batch in batches}
        
        for i, future in enumerate(as_completed(future_to_batch)):
            result = future.result()
            if result:
                entries = result.get("data", {}).get("entries", [])
                for entry in entries:
                    data_obj = parse_and_classify(entry)
                    full_data_map[data_obj["id"]] = data_obj
            
            if (i+1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{len(batches)} batches processed.")

    logger.info(f"Metadata fetch complete. {len(full_data_map)} IDs classified.")

    # ==========================================
    # STEP A: GENERATE CSV FILES
    # ==========================================
    logger.info("Generating CSV files...")
    
    # Initialize lists for each CSV category
    csv_rows = {k: [] for k in CSV_FILES.keys()}
    
    # Populate lists
    for pid in all_ids:
        clean_id = pid.upper()
        if clean_id in full_data_map:
            obj = full_data_map[clean_id]
            cls = obj["class"]
            # Add tuple (ID, Title, Desc) to the specific list
            csv_rows[cls].append([obj["id"], obj["title"], obj["desc"]])
        else:
            # Handle IDs that failed to download
            csv_rows["MISSING"].append([pid, "N/A", "Metadata fetch failed"])

    # Write files
    for cls, filename in CSV_FILES.items():
        rows = csv_rows[cls]
        if rows:
            try:
                with open(filename, mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(["PDB_ID", "Title", "Description"]) # Header
                    writer.writerows(rows)
                logger.info(f"  -> Wrote {len(rows)} rows to {filename}")
            except Exception as e:
                logger.error(f"Failed to write {filename}: {e}")

    # ==========================================
    # STEP B: GENERATE JSON SPLITS
    # ==========================================
    logger.info("Generating JSON splits...")
    
    model_strict = {"train": [], "val": [], "test": []}
    model_moderate = {"train": [], "val": [], "test": []}
    stats = {k: len(csv_rows[k]) for k in csv_rows} # Pre-fill stats from CSV counts

    for split in ["train", "val", "test"]:
        for pid in original_data.get(split, []):
            lookup_key = pid.upper()
            
            # Default to CLEAN if missing (rare case of API failure)
            # Use the already stored map
            if lookup_key in full_data_map:
                cls = full_data_map[lookup_key]["class"]
            else:
                cls = "CLEAN" 

            # If it is SARS-CoV-2, NEVER allow it in the 'train' split.
            # (We only want SARS-CoV-2 in 'val' or 'test')
            if split == "train" and cls == "SARS2":
                continue  # Skip this PDB, effectively deleting it from training
            
            # --- FILTERING ---
            # Strict: No SARS1, No MERS, No Other Virus
            if cls in ["SARS2", "CLEAN"]:
                model_strict[split].append(pid)
            
            # Moderate: No SARS1, No MERS (Other Virus is OK)
            if cls in ["SARS2", "CLEAN", "OTHER_VIRUS"]:
                model_moderate[split].append(pid)

    with open(OUTPUT_JSON_STRICT, "w") as f:
        json.dump(model_strict, f, indent=4)
    with open(OUTPUT_JSON_MODERATE, "w") as f:
        json.dump(model_moderate, f, indent=4)

    duration = time.time() - start_time
    logger.info("="*40)
    logger.info(f"Processing Complete.")
    logger.info(f"SARS-CoV-2:      {stats['SARS2']}")
    logger.info(f"Other Virus:     {stats['OTHER_VIRUS']}")
    logger.info(f"Clean:           {stats['CLEAN']}")
    logger.info(f"SARS-CoV-1:      {stats['SARS1']}")
    logger.info(f"MERS-CoV:        {stats['MERS']}")
    logger.info("="*40)
    logger.info(f"Total Time: {duration:.2f}s")

if __name__ == "__main__":
    main()