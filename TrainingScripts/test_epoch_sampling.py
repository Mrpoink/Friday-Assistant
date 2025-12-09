import subprocess
import sys
import os

if __name__ == "__main__":
    repo_dir = os.path.dirname(__file__)
    sampler = os.path.join(repo_dir, "epoch_sampler.py")
    # Generate 3 epochs with default 100% pool sizes
    cmd = [sys.executable, sampler, "--epochs", "3"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Done. Check TrainingData/epochs/")
