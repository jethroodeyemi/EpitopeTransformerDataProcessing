import requests
import json
from Bio import Align
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
import time

# --- Reference Sequence (SARS-CoV-2 Spike - UniProt P0DTC2) ---
REF_SEQ = (
    "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHVSGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPFLGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPINLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYNENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFLPFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLTPTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLGAENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGIAVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDCLGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIGVTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDILSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLMSFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNTFVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVAKNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDDSEPVLKGVKLHYT"
)

def fetch_fasta(pdb_id):
    """Fetch FASTA data for a single PDB ID."""
    clean_id = pdb_id.upper()
    url = f"https://www.rcsb.org/fasta/entry/{clean_id}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return (clean_id, r.text, None)
        return (clean_id, None, "Download Failed")
    except Exception as e:
        return (clean_id, None, str(e))

def analyze_sequence(pdb_id, fasta_data):
    """Analyze a single PDB's FASTA data against reference sequence."""
    aligner = Align.PairwiseAligner()
    aligner.mode = 'local'
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    
    chains = fasta_data.split('>')
    best_identity = 0.0
    
    for chain in chains:
        if not chain.strip():
            continue
        lines = chain.split('\n')
        seq = "".join(lines[1:])
        
        if len(seq) < 30:
            continue
        
        score = aligner.score(REF_SEQ, seq)
        ratio = score / len(seq) if len(seq) > 0 else 0
        if ratio > best_identity:
            best_identity = ratio
    
    # Classify
    if best_identity >= 0.85:
        classification = "SARS-CoV-2"
    elif best_identity >= 0.40:
        classification = "Other CoV"
    else:
        classification = "Not Found"
    
    return (pdb_id, best_identity, classification)

def process_single_pdb(pdb_id):
    """Process a single PDB: fetch and analyze."""
    clean_id, fasta_data, error = fetch_fasta(pdb_id)
    if error:
        return (clean_id, None, "Error", error)
    return analyze_sequence(clean_id, fasta_data) + (None,)

def check_cov_sequence_counts_parallel(pdb_list, max_workers=64):
    """Parallelized version using ThreadPool for I/O and ProcessPool for CPU."""
    stats = {"SARS-CoV-2": 0, "Other CoV": 0, "Not Found": 0, "Errors": 0}
    results = []
    
    # Use ThreadPoolExecutor for I/O-bound fetch operations
    # Then ProcessPoolExecutor for CPU-bound alignment (but threads work well here too)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_pdb, pdb_id): pdb_id for pdb_id in pdb_list}
        
        for future in as_completed(futures):
            try:
                pdb_id, identity, classification, error = future.result()
                if error:
                    stats["Errors"] += 1
                    results.append((pdb_id, "N/A", "Error"))
                else:
                    if classification == "SARS-CoV-2":
                        stats["SARS-CoV-2"] += 1
                    elif classification == "Other CoV":
                        stats["Other CoV"] += 1
                    else:
                        stats["Not Found"] += 1
                    results.append((pdb_id, identity, classification))
            except Exception as e:
                stats["Errors"] += 1
    
    return stats, len(pdb_list), results

# --- Run the function ---
if __name__ == "__main__":
    start_time = time.time()
    
    # Load PDB list from JSON file
    with open("output/data_splits_MODERATE_no_sars_mers.json", "r") as f:
        data_splits = json.load(f)

    # Process each split separately with aggressive parallelization
    all_stats = {}
    total_counts = {}
    all_results = {}
    
    # Use 64 workers for maximum parallelization
    MAX_WORKERS = 64

    for split_name in ["train", "val", "test"]:
        pdb_list = data_splits.get(split_name, [])
        if pdb_list:
            print(f"\n{'='*65}")
            print(f"  Processing {split_name.upper()} split ({len(pdb_list)} PDBs) with {MAX_WORKERS} workers...")
            print(f"{'='*65}")
            split_start = time.time()
            stats, count, results = check_cov_sequence_counts_parallel(pdb_list, max_workers=MAX_WORKERS)
            split_time = time.time() - split_start
            print(f"  Completed in {split_time:.2f}s ({len(pdb_list)/split_time:.1f} PDBs/sec)")
            all_stats[split_name] = stats
            total_counts[split_name] = count
            all_results[split_name] = results

    # Print overall summary across all splits
    print("\n" + "="*70)
    print("                    OVERALL SUMMARY ACROSS ALL SPLITS")
    print("="*70)
    print(f"{'Split':<10} | {'Total':<8} | {'SARS-CoV-2':<12} | {'Other CoV':<12} | {'No Match':<10} | {'Errors':<8}")
    print("-"*70)

    grand_total = 0
    grand_sars2 = 0
    grand_other = 0
    grand_none = 0
    grand_errors = 0

    for split_name in ["train", "val", "test"]:
        if split_name in all_stats:
            stats = all_stats[split_name]
            total = total_counts[split_name]
            print(f"{split_name.upper():<10} | {total:<8} | {stats['SARS-CoV-2']:<12} | {stats['Other CoV']:<12} | {stats['Not Found']:<10} | {stats['Errors']:<8}")
            grand_total += total
            grand_sars2 += stats['SARS-CoV-2']
            grand_other += stats['Other CoV']
            grand_none += stats['Not Found']
            grand_errors += stats['Errors']

    print("-"*70)
    print(f"{'TOTAL':<10} | {grand_total:<8} | {grand_sars2:<12} | {grand_other:<12} | {grand_none:<10} | {grand_errors:<8}")
    print("="*70)
    
    # Print detailed breakdown for TRAIN set (the 7 non-standard entries)
    if "train" in all_results:
        train_results = all_results["train"]
        
        sars2_pdbs = [(pdb, identity) for pdb, identity, classification in train_results if classification == "SARS-CoV-2"]
        other_cov_pdbs = [(pdb, identity) for pdb, identity, classification in train_results if classification == "Other CoV"]
        error_pdbs = [pdb for pdb, identity, classification in train_results if classification == "Error"]
        
        print("\n" + "="*70)
        print("         TRAIN SET - DETAILED BREAKDOWN OF SPECIAL CASES")
        print("="*70)
        
        print(f"\n  SARS-CoV-2 Matches ({len(sars2_pdbs)}):")
        for pdb, identity in sars2_pdbs:
            print(f"    - {pdb} (identity: {identity:.2%})")
        
        print(f"\n  Other CoV Matches ({len(other_cov_pdbs)}):")
        for pdb, identity in other_cov_pdbs:
            print(f"    - {pdb} (identity: {identity:.2%})")
        
        print(f"\n  Errors ({len(error_pdbs)}):")
        for pdb in error_pdbs:
            print(f"    - {pdb}")
        
        print("="*70)
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f}s")