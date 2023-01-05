import glob

import pandas as pd

from ner.dataset import read_examples_from_file

# pcm: 2124
# conll_2003_en: 14042
# lug: 1428
# luo: 644
# wol: 1871
# amh: 1750
# ibo: 2235
# hau: 1912
# swa: 2109
# kin: 2116
# yor: 2171

def main():
    
    data_path = "data/original/1/*"
    D = {
        'Language': [],
        'Number of Sentences': []
    }
    for folder in glob.glob(data_path):
        name = folder.split("/")[-1]
        examples = read_examples_from_file(folder, "train")
        print(f"{name}:", len(examples))
        D['Language'].append(name)
        D['Number of Sentences'].append(len(examples))
    pd.DataFrame(D).to_latex('analysis/num_sentences.tex', index=False)
        
if __name__ == "__main__":
    main()