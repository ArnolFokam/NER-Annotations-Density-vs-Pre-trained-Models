count=0
for model in afriberta afro_xlmr mbert xlmr; do
	#for corruption in global_cap_labels global_swap_labels global_cap_sentences global_cap_sentences_seed1 global_cap_sentences_seed1; do
	for corruption in global_cap_labels global_swap_labels global_cap_sentences global_cap_sentences_seed1; do
		for param in 0.1 0.01 0.2 0.3 0.4 0.5 0.05 0.6 0.7 0.8 0.9 1.0; do	
			for lang in amh conll_2003_en hau ibo kin lug luo pcm swa wol yor; do
				NAME=$model"_"$corruption"_"$param"_"$lang
				cat base_entropy.sh > allslurms/$NAME.batch
				for seed in 1 2; do # 2
                    i=$((seed-1))
					echo "CUDA_VISIBLE_DEVICES=$i ./run.sh scripts/get_entropies_good.py --model_type $model --corruption_name $corruption --param $param --language $lang --number_of_predictions 50 --seed $seed &" >> allslurms/$NAME.batch
					# count=$((count+1))
					# if [ $((count % 6)) = 0  ]; then 
					# 	wait
					# fi
				done
				echo "wait" >>  allslurms/$NAME.batch
			done
		done
	done
done
