#!/bin/bash
#SBATCH --job-name=slurm_a_script
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=a.p.s.susaiyah@tilburguniversity.edu
#SBATCH --output=temp/slurm_a_script_output.log
#SBATCH --error=temp/slurm_a_script_error.log

# Load Python if needed
module load python/3.10.18 

# Run your script
python s03_evaluatedetections.py
