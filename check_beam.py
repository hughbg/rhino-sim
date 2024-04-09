import numpy as np
import yaml, sys


with open('beams.yaml', 'r') as file:
    beams = yaml.safe_load(file)


assert sys.argv[1] in beams, "Beam "+sys.argv[1]+" not specified"
    
