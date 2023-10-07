export LOGS=$(echo "logs/black_box_`date`.out" | tr -d '[:blank:]')
sbatch -o $LOGS --export= scripts/preprocessing/job_def.slurm
