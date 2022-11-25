# Add any other relevant paths here
export PYTHONPATH=`pwd`:`pwd`/src:$PYTHONPATH
# TEMPLATE scripts/run_exp.py --partition_name batch  --max_runs_per_scripts 1 --run_immediately False --yaml_sweep_file exps/sweep.yaml
python -u $@