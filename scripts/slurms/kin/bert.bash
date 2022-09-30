#!/bin/bash
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 72:00:00
 

#SBATCH -J kin_bert
#SBATCH -o /home/arnol/Masters/ner/logs/kin/outputs_slurm/bert.%N.%j.out
#SBATCH -e /home/arnol/Masters/ner/logs/kin/errors_slurm/bert.%N.%j.err

cd /home/arnol/Masters/ner

python /home/arnol/Masters/ner/scripts/train_ner.py --config-path=../exps/kin --config-name=bert  device.seed=42

