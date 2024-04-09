import subprocess
import numpy as np
import time, sys, glob
from datetime import datetime
import subprocess

def sbatch_available():
    result = subprocess.run(['which', 'sbatch'], stdout=subprocess.PIPE)
    return len(result.stdout) > 0
    
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

def in_process_list(jobid):
    output = run_command(["ps", "uxh"])
    for line in output:
        words = line.split()
        if len(words) > 1 and int(words[1]) == jobid:
            return True

    return False


if sbatch_available():  
    submitter = "sbatch"
else:
    submitter = "./sbatch_inline"                # Run the prog as a normal process

output = run_command([submitter]+sys.argv[1:])         # if fails then everything stops
jobid = None

for line in output:
    if "Submitted batch job" in line:
        #print(line, " ".join(sys.argv[1:])); sys.stdout.flush()
        jobid = int(line.split()[-1])

assert jobid is not None, "Failed to get job id from sbatch output"

print("Submitted job", jobid, "at", datetime.now()); sys.stdout.flush()

if sbatch_available(): 
    while in_queue(jobid):
        time.sleep(120)
else:
    while in_process_list(jobid):
        time.sleep(120)

status = 0
logfile = glob.glob("*"+str(jobid)+"*")
for lf in logfile:
  for line in open(lf):
    if "error" in line or "Error" in line or "failed" in line or "Failed" in line or "Traceback" in line:
      status = 1

print(jobid, "exited queue status", "success" if status==0 else "failed", "at", datetime.now()); sys.stdout.flush()
exit(status)
