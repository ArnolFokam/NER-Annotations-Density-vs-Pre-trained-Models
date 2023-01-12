#!/bin/bash
#SBATCH -p bigbatch
#SBATCH -N 1
#SBATCH -t 72:00:00
#SBATCH -x mscluster65 
#SBATCH -J local_swap_labels_like_cap_9_yor_xlmr_3_model
#SBATCH -o /home-mscluster/mfokam/ner/logs/local_swap_labels_like_cap_9_yor_xlmr_3/outputs_slurm/model.%N.%j.out
#SBATCH -e /home-mscluster/mfokam/ner/logs/local_swap_labels_like_cap_9_yor_xlmr_3/errors_slurm/model.%N.%j.err

cd /home-mscluster/mfokam/ner
export PYTHONPATH=$PYTHONPATH:`pwd`
export PATH=$PATH:/usr/local/cuda-12.0/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.0/lib64

python /home-mscluster/mfokam/ner/scripts/train_eval_ner.py --config-path=../exps
--config-name=base data.param=2 data.method=local_swap_labels_like_cap data.language=conll_2003_en model.conf=[bert,bert-base-multilingual-cased,mbert] device.seed=3
