# Add any other relevant paths here
export PYTHONPATH=`pwd`:`pwd`/src:$PYTHONPATH
# python scripts/run_exp.py --experiment exp1 --model simclr --partition_name batch  --max_runs_per_scripts 2 --use_slurm False --yaml_sweep_file simclr_sweep.yaml
python -u $@