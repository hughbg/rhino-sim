import subprocess
import numpy as np
import time, sys, glob
from datetime import datetime


def run_command(command):
    proc = subprocess.run(command, capture_output=True, text=True)
    
    if proc.returncode != 0:
        print(proc.stderr)
        exit(proc.returncode)
        
    else:
        return proc.stdout.split("\n")
    
def in_queue(jobid):
    output = run_command(["squeue", "--me", "-h"])
    for line in output:
        words = line.split()
        if len(words) > 0 and int(words[0]) == jobid:
            return True

    return False
    
        
        

output = run_command(["sbatch"]+sys.argv[1:])         # if fails then everything stops
jobid = None

for line in output:
    if "Submitted batch job" in line:
        print(line, " ".join(sys.argv[1:])); sys.stdout.flush()
        jobid = int(line.split()[-1])

assert jobid is not None, "Failed to get job id from sbatch output"

print("Submitted job", jobid, "at", datetime.now()); sys.stdout.flush()

while in_queue(jobid):
    time.sleep(120)

status = 0
logfile = glob.glob("*"+str(jobid)+"*")
for lf in logfile:
  for line in open(lf):
    if "error" in line or "Error" in line or "failed" in line or "Failed" in line or "Traceback" in line:
      status = 1

print(jobid, "exited queue status", "success" if status==0 else "failed", "at", datetime.now()); sys.stdout.flush()
exit(status)
