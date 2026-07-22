# --- Standard library imports ---
import subprocess
import sys

PIPELINE = [
    "00_setup.py",                              # REAL -- creates directory structure
    "01_download_genomes.py",                   # REAL -- downloads real genomic data (NCBI)
    "02_download_structures.py",                # REAL -- downloads real PDB/AlphaFold structures
    "03_download_images.py",                    # REAL -- downloads real microscopy images (DIBaS/OpenFungi)
    "04_call_resistance.py",                    # REAL -- real AMRFinderPlus resistance calling
    "05_build_ledger.py",                       # REAL -- builds ground-truth ledger from real files
    "06_tokenize_genomes.py",                   # REAL -- real gene-targeted k-mer tokenization
    "07_build_structural_graphs.py",            # REAL -- real protein graph construction
    "08_sanity_check.py",                       # REAL -- validates real tensors/graphs
    "09_build_architecture.py",                 # REAL -- real transformer + GNN architecture
    "10_test_wiring.py",                        # REAL -- gradient + real-data smoke test
    "11_build_baselines.py",                    # REAL -- real sequence-only/structure-only baselines
    "12_build_classical_image_baselines.py",    # REAL -- real k-mer frequency + image baselines
    "13_execute_training.py",                   # REAL -- main training run (resistance number here is INVALID -- see 13b/13c)
    "13b_execute_resistance_blind.py",          # REAL -- honest blind resistance re-evaluation
    "13c_execute_resistance_kmer_nn.py",        # REAL -- honest whole-genome k-mer NN re-evaluation (trustworthy number)
    "14_extract_interpretability.py",           # REAL -- real gradient-based saliency, genomic + structural
    "15_robustness_stress_test.py",             # REAL -- real missing-structure/noise/OOD stress tests
    "16_generate_final_report.py",              # REAL -- assembles FINAL_REPORT.md from real result files
]


def run_pipeline():
    for script in PIPELINE:
        print(f"\n>>> Executing: {script}")
        try:
            subprocess.run([sys.executable, script], check=True)
        except subprocess.CalledProcessError:
            print(f"\n!!! Pipeline halted: {script} failed.")
            sys.exit(1)

    print("\n>>> Pipeline complete: all phases executed and real results logged.")
    print(">>> IMPORTANT: the resistance accuracy printed by 13_execute_training.py")
    print(">>> is INVALID (circular -- see its docstring). The honest resistance")
    print(">>> results are in 13b's and 13c's output and in FINAL_REPORT.md.")


if __name__ == "__main__":
    run_pipeline()