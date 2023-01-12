count=0

for model in xlmr; do
	for corruption in local_swap_labels_like_cap; do
		for param in  1 2 3 4 5 6 7 8 9 10; do	
			for lang in amh conll_2003_en hau ibo kin lug luo pcm swa wol yor; do
				for seed in 1 2 3; do
					./run.sh scripts/get_entropies.py --model_type $model --corruption_name $corruption --param $param --language $lang --number_of_predictions 50 --seed $seed &
					count=$((count+1))
					if [ $((count % 6)) = 0  ]; then 
						wait
					fi
				done
			done
		done
	done
done
