# Run all demos sequentially
import subprocess, sys

def run(cmd):
    print("\n" + "="*80)
    print("Running:", " ".join(cmd))
    print("="*80)
    subprocess.run([sys.executable] + cmd, check=False)

if __name__ == "__main__":
    run(["react_demo.py"])
    run(["tot_demo.py"])
    run(["reflexion_demo.py"])