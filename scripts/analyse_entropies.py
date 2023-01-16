from collections import defaultdict
import json
import os
import pathlib
import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scripts.plot_graphs import CLEAN_MODES, savefig, clean_model

def get_vals(folder):
    A = folder.split("/")
    # print(folder)
    original_data = os.path.join(*(A[:2] + ['global_cap_sentences', '1.0'] + A[4:]))
    ent = np.load(os.path.join(folder, 'entropies.npz'))['arr_0']
    
    try:
        labs = np.load(os.path.join(folder, 'labels.npz'))['arr_0']
    except:
        assert 'global_cap_sentences_seed' in folder
        labs = np.load(os.path.join(folder.replace("global_cap_sentences_seed1", 'global_cap_sentences').replace("global_cap_sentences_seed2", 'global_cap_sentences'), 'labels.npz'))['arr_0']
    with open(os.path.join(folder, 'labels.p'), 'rb') as f:
        lab2 = pickle.load(f)

    good_ents = []
    good_labs = []
    good_labs2 = []
    for index, (example, example_labels) in enumerate(zip(ent, labs)):
        for j in range(len(example_labels) - 1, -1, -1):
            if example_labels[j] != -100:
                break
        
        example_labels = example_labels[:j + 1]
        new = []
        new_ent = []
        # Ignoring the -100 tokens
        for idx, (i, myent) in enumerate(zip(example_labels, example)):
            if i == -100:
                continue
            else:
                new.append(i)
                new_ent.append(myent)
        example = new_ent
        
        old = new
        new = [n if n == 0 else 1 for n in new]
        
        x_0 = []
        x_1 = []

        good_ents.extend(example)
        good_labs.extend(new)
        good_labs2.extend(old)
    x_0 = []
    x_1 = []
    for n, ex in zip(good_labs, good_ents):
        if n == 0:   x_0.append(ex)
        elif n == 1: x_1.append(ex)
        else: assert False
    if 0:
        return x_1, [1] * len(x_1), lab2
    good_ents = [np.mean(x_0) if len(x_0) else 0, np.mean(x_1) if len(x_1) else 0][1:]
    good_labs = [0, 1][1:]
    return good_ents, good_labs, lab2


MODELS = ['xlmr', 'afriberta', 'afro_xlmr', 'mbert']
LANGS = ['amh','hau','ibo','kin','lug','luo','pcm','swa','wol','yor',]
# LANGS = ['swa',]
ALL_ENTS = defaultdict(lambda: defaultdict(lambda: {}))
def main(MODEL='afro_xlmr', CORRUPTION='global_swap_labels'):

    axs = [plt.gca()]
    X = None
    for MODEL in MODELS:
        T = f'{MODEL}/{CORRUPTION}'
        print(T)
        XXXX = [None] * 9
        for I, ax in enumerate(axs):
            x = []
            vals = []
            mi, ma = [], []
            for i in range(1, 11):
                II = i / 10
                if 'local' in CORRUPTION: II = i
                x.append(II)
                goods = [[], [], []]
                for t in range(1, 4):
                    for l in LANGS:
                        if CORRUPTION == 'global_cap_sentences' and 1:
                            for C in ['global_cap_sentences', 'global_cap_sentences_seed1', 'global_cap_sentences_seed2']:
                                test = get_vals(f"entropies/{MODEL}/{C}/{II}/{l}/{t}")
                                goods[0].extend(test[0])
                                goods[1].extend(test[1])
                                goods[2] = test[2]
                        else:
                            test = get_vals(f"entropies/{T}/{II}/{l}/{t}")
                            goods[0].extend(test[0])
                            goods[1].extend(test[1])
                            goods[2] = test[2]
                goods = (np.array(goods[0]), np.array(goods[1]), goods[2])
                ogv = v = goods
                if X is None: X = np.unique(v[1])[::-1]
                idx = (v[1] == X[I])
                v = v[0][idx]
                
                if 1:
                    print("LEN V", len(v))
                    vals.append(m:= np.mean(v))
                    s = np.std(v)
                    mi.append(m - s)
                    ma.append(m + s)
                else:
                    if XXXX[I] is None:
                        XXXX[I] = v
                    vals.append(m:= np.mean(v))
                    s = 0
                    mi.append(np.min(v))
                    ma.append(np.max(v))
                    
            ALL_ENTS[CORRUPTION][MODEL] = [float(v) for v in vals]
            ax.plot(x, vals, label=clean_model(MODEL))
            ax.fill_between(x, mi, ma, alpha=0.1)
            ax.set_ylabel("Entropy")
            ax.set_xlabel(mytest(CORRUPTION))
            ax.set_ylim(0, 1.3)
            ax.set_title(CLEAN_MODES[CORRUPTION])
    plt.legend()
    savefig(f'analysis/plots/entropies/ALL_{CORRUPTION}.png')
    
    plt.close()

def mytest(mode):
    if mode == 'global_cap_sentences':
        return ("Fraction of Sentences Kept")
    if mode == 'global_cap_labels':
        return ("Fraction of Labels Kept")
    if mode == 'global_swap_labels':
        return ("Fraction of Labels Kept")
    if mode == 'local_cap_labels':
        return ("Maximum number of labels kept per sentence")
    if mode == 'local_swap_labels':
        return ("Maximum number of labels swapped per sentence")
    assert False, mode


def entropy_vs_f1():
    old = pd.read_csv("analysis/main_results_v2.csv", index_col=0)
    for N in ['global_cap_sentences', 'global_cap_labels', 'global_swap_labels']:
        df = old.copy(deep=True)
        df = df[df['model'] == 'Afro-XLM-R']
        df = df[df['mode'] == N]
        df = df.groupby('num').mean()
        with open('test.json', 'r') as f:
            ALL_ENTS = json.load(f)
        t = ALL_ENTS[N]['afro_xlmr']
        print(t)
        
        print(df)
        df['good']
        y = df['good'].tolist()
        if N == 'global_cap_sentences':
            y = df['good'].tolist()[2:]
        # plt.plot(t, y, label=N)
        s = np.diff(t)
        plt.plot(s, label=N)
    plt.xlabel("entropy")
    plt.ylabel("performance")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    entropy_vs_f1()
    exit()
    for C in ['global_swap_labels', 'global_cap_labels', 'global_cap_sentences', 'local_cap_labels'][:]:
        main(None, C)
        
    with open('test.json', 'w+') as f:
        json.dump(ALL_ENTS, f)
        