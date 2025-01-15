import os
from tqdm import tqdm
import subprocess
import time

base_dir = "D:\Code\Precursor Detection for Thermoacoustic Instability"
scripts = ["rom_segmentation_classification.py", "cao_theorem.py", "average_mutual_information.py", "recurrence_matrix_generation.py", "convolutional_neural_network.py"]  # Add your script names here

for script in tqdm(scripts, desc="Running scripts", unit="script"):
    script_path = os.path.join(base_dir, script)
    print(f"\nRunning {script_path}...")
    start_time = time.time()
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    elapsed_time = time.time() - start_time
    print(f"Finished {script} in {elapsed_time:.2f} seconds")
    if result.returncode != 0:
        print(f"Error occurred while running {script}")
        print(result.stderr)
        break
