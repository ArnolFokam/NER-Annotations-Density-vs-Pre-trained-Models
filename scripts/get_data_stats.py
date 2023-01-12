import glob
import numpy as np

from ner.dataset import read_examples_from_file

# name,number_of_sentences,ner_density
# pcm,2124,0.051440955060451675
# en,14042,0.06918245171175862
# lug,1428,0.05699481865284974
# luo,644,0.04866232437960919
# wol,1871,0.027577774758864283
# amh,1750,0.06883735336249952
# ibo,2235,0.06219714880966315
# hau,1912,0.0514815488093074
# swa,2109,0.05378186893761374
# kin,2116,0.051531975288028053
# yor,2171,0.03763727476276789

def main():
    
    data_path = "data/original/1/*"
    print("name,number_of_sentences,ner_density")
    for folder in glob.glob(data_path):
        name = folder.split("/")[-1]
        if name == "conll_2003_en":
            name = "en"
        examples = read_examples_from_file(folder, "train")
        num_words = sum([len(ex.words) for ex in examples])
        labels = list(map(lambda x: [label.split('-')[-1] for label in x.labels if label != 'O'], examples))
        num_labels = sum([len(np.unique(l)) for l in labels])
        print(f"{name},{len(examples)},{num_labels/num_words}")
        
        
if __name__ == "__main__":
    main()