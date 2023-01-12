#!/bin/bash
#SBATCH -p stampede
#SBATCH -N 1
#SBATCH -t 72:00:00
#SBATCH -x mscluster65 
#SBATCH -J entropy
#SBATCH -o /home-mscluster/mbeukman/school/NER-Annotations-Density-vs-Pre-trained-Models/logs/outputs_slurm/model.%N.%j.out
#SBATCH -e /home-mscluster/mbeukman/school/NER-Annotations-Density-vs-Pre-trained-Models/logs/errors_slurm/model.%N.%j.err

cd /home-mscluster/mbeukman/school/NER-Annotations-Density-vs-Pre-trained-Models
conda activate manualner
export PYTHONPATH=$PYTHONPATH:`pwd`
export PATH=$PATH:/usr/local/cuda-12.0/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.0/lib64
