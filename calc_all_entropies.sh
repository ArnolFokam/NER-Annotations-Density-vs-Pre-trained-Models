count=0
#for model in xlmr; do
#afro_xlmr mbert 
for model in afriberta xlmr; do
#    for corruption in global_cap_labels global_cap_sentences global_cap_sentences_seed1 global_cap_sentences_seed2 global_swap_labels local_cap_labels local_swap_labels original; do
    for corruption in global_cap_labels global_cap_sentences global_cap_sentences_seed1 global_cap_sentences_seed2 global_swap_labels local_cap_labels local_swap_labels original; do
        for param in 0.1 0.01 0.2 0.3 0.4 0.5 0.05 0.6 0.7 0.8 0.9 1.0; do
            for lang in amh conll_2003_en hau ibo kin lug luo pcm swa wol yor; do
            #for lang in swa; do
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
