import os
import pathlib
import pickle
from matplotlib import pyplot as plt
import numpy as np
from scripts.plot_graphs import savefig, clean_model

def get_vals(folder):
    # "entropies/xlmr/global_swap_labels/0.5/swa/3/"
    ent = np.load(os.path.join(folder, 'entropies.npz'))['arr_0']
    labs = np.load(os.path.join(folder, 'labels.npz'))['arr_0']
    with open(os.path.join(folder, 'labels.p'), 'rb') as f:
        lab2 = pickle.load(f)
        
    # print(labs[0])
    # print(ent.shape, folder)
    # print(labs[0])
    # exit()
    good_ents = []
    good_labs = []
    for example, example_labels in zip(ent, labs):
        for j in range(len(example_labels) - 1, -1, -1):
            if example_labels[j] != -100:
                break
        
        example_labels = example_labels[:j + 1]
        new = []
        curr = -100
        for i in example_labels:
            if i == -100:
                new.append(curr)
            else:
                curr = i
                new.append(i)
        new = new[1:]
        
        new = [n if n == 0 else 1 for n in new]
        
        x_0 = []
        x_1 = []
        # for n, ex in zip(new, example):
        #     if n == 0:   x_0.append(ex)
        #     elif n == 1: x_1.append(ex)
        #     else: assert False

        # good_ents.extend([np.mean(x_0) if len(x_0) else 0, np.mean(x_1) if len(x_1) else 0])
        # good_labs.extend([0, 1])

        good_ents.extend(example[1:j + 1])
        good_labs.extend(new)
        # good_labs.extend(example_labels[:j + 1])
        # print("NEW", new)
        # print(new)
        # exit()
    x_0 = []
    x_1 = []
    for n, ex in zip(good_labs, good_ents):
        if n == 0:   x_0.append(ex)
        elif n == 1: x_1.append(ex)
        else: assert False

    good_ents = [np.mean(x_0) if len(x_0) else 0, np.mean(x_1) if len(x_1) else 0][1:]
    good_labs = [0, 1][1:]
    return good_ents, good_labs, lab2
    return ent, labs, lab2
MODELS = ['xlmr', 'afriberta', 'afro_xlmr', 'mbert']
LANGS = ['amh','hau','ibo','kin','lug','luo','pcm','swa','wol','yor',]
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
                        test = get_vals(f"entropies/{T}/{II}/{l}/{t}")
                        goods[0].extend(test[0])
                        goods[1].extend(test[1])
                        goods[2] = test[2]
                goods = (np.array(goods[0]), np.array(goods[1]), goods[2])
                ogv = v = goods
                if X is None: X = np.unique(v[1])
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
                    print("LEN V", len(v))
                    vals.append(m:= np.mean(v))
                    s = 0
                    mi.append(np.min(v))
                    ma.append(np.max(v))
            ax.plot(x, vals, label=clean_model(MODEL))
            PPP = ogv[2]
            PPP = ["O", "Entities"]
            LLL = X[I]
            print(PPP)
            print(LLL)
            # ax.set_title(str((PPP[LLL]) if LLL >= 0 else LLL) + " -- " + str(idx.sum()))
            ax.fill_between(x, mi, ma, alpha=0.1)
            ax.set_ylabel("Entropy")
            # ax.set_xlabel("Param")
            ax.set_xlabel(mytest(CORRUPTION))
            ax.set_ylim(0, 1.3)
            # ax.show()
    plt.legend()
    # plt.suptitle(CORRUPTION)
    savefig(f'analysis/plots/entropies/ALL_{CORRUPTION}.png')
    
    plt.close()

def mytest(mode):
    if mode == 'global_cap_sentences':
        return ("Fraction of Sentences Kept")
    if mode == 'global_cap_labels':
        return ("Fraction of Labels Kept")
    if mode == 'global_swap_labels':
        return ("Fraction of Labels Swapped")
    if mode == 'local_cap_labels':
        return ("Maximum number of labels kept per sentence")
    if mode == 'local_swap_labels':
        return ("Maximum number of labels swapped per sentence")
    assert False, mode


if __name__ == '__main__':
    for M in ['xlmr', 'afriberta', 'afro_xlmr', 'mbert']: 
        for C in ['global_swap_labels', 'global_cap_labels', 'global_cap_sentences', 'local_cap_labels']:
            main(M, C)