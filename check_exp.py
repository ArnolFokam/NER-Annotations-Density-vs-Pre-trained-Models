index = 2
from itertools import product
import os 

cap = {
        1: [str(i) for i in range(1, 11)],
        2: ['0.01', '0.05'] +  [str(i/10) for i in range(1, 11)]
        }

lang = ["amh",
"conll_2003_en",
"hau",
"ibo",
"kin",
"lug",
"luo",
"pcm",
"swa",
"wol",
"yor"
]

models = ["afriberta","afro_xlmr","mbert","xlmr"]
  
  
method = {
        2: ["global_cap_sentences_seed2", "global_cap_sentences_seed1"], 
        1: ["local_swap_labels"]
        }

seed = ['1', '2', '3']

for t in list(product(models, method[index], cap[index], lang, seed)):
    path_to_results = os.path.join('/home-mscluster/mfokam/ner/results', *t, 'test_results.txt')
    if os.path.exists(path_to_results):
        continue 
    else:
        print(f"model_{t[1]}_{t[2]}_{t[3]}_{t[0]}_{t[4]}_.bash")

