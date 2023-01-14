import os
import pathlib
import pickle
from matplotlib import pyplot as plt
import numpy as np
from scripts.plot_graphs import CLEAN_MODES, savefig, clean_model

def get_vals(folder):
    # "entropies/xlmr/global_swap_labels/0.5/swa/3/"
    A = folder.split("/")
    original_data = os.path.join(*(A[:2] + ['global_cap_sentences', '1.0'] + A[4:]))
    og_ent = np.load(os.path.join(original_data, 'entropies.npz'))['arr_0']
    og_labs = np.load(os.path.join(original_data, 'labels.npz'))['arr_0']
    ent = np.load(os.path.join(folder, 'entropies.npz'))['arr_0']
    labs = np.load(os.path.join(folder, 'labels.npz'))['arr_0']
    assert np.all(labs == og_labs)
    with open(os.path.join(folder, 'labels.p'), 'rb') as f:
        lab2 = pickle.load(f)

    good_ents = []
    good_labs = []
    good_labs2 = []
    for index, (example, example_labels) in enumerate(zip(ent, labs)):
        for j in range(len(example_labels) - 1, -1, -1):
            if example_labels[j] != -100:
                break
        
        if 0:
            # Mike this is old, but correct
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
            example = example[1:]
        else:
            example_labels = example_labels[:j + 1]
            new = []
            new_ent = []
            og_new_ent = []
            curr = -100
            for idx, (i, myent) in enumerate(zip(example_labels, example)):
                if i == -100:
                    continue
                else:
                    curr = i
                    new.append(i)
                    new_ent.append(myent)
                    og_new_ent.append(og_ent[index][idx])
            # new = new[1:]
            example = new_ent
        
        old = new
        new = [n if n == 0 else 1 for n in new]
        
        x_0 = []
        x_1 = []
        # for n, ex in zip(new, example):
        #     if n == 0:   x_0.append(ex)
        #     elif n == 1: x_1.append(ex)
        #     else: assert False

        # good_ents.extend([np.mean(x_0) if len(x_0) else 0, np.mean(x_1) if len(x_1) else 0])
        # good_labs.extend([0, 1])

        
        # a, b = example[:j + 1], og_ent[index][:j + 1]
        # a, b = np.array(example), np.array(og_new_ent) #og_ent[index]
        a, b = (example), (og_new_ent) #og_ent[index]
        # print(a[:2], b[:2], (a / b)[:2])
        good_ents.extend(a)# / b)
        good_labs.extend(new)
        good_labs2.extend(old)
        # good_labs.extend(example_labels[:j + 1])
        # print("NEW", new)
        # print(new)
        # exit()
    x_0 = []
    x_1 = []
    if 1:
        for n, ex in zip(good_labs, good_ents):
            if n == 0:   x_0.append(ex)
            elif n == 1: x_1.append(ex)
            else: assert False
        if 0: #mike hacks
            good_ents = np.array(good_ents)
            for label in np.unique(good_labs2):
                break
                a = good_ents[np.array(good_labs2) == label]
                print(f'{label}, {np.min(a):<25}, {np.max(a):<25}, {np.mean(a):<25}, {np.std(a):<25}')
            # exit()
            print(min(x_0), max(x_0), np.mean(x_0), np.std(x_0))
            print(min(x_1), max(x_1), np.mean(x_1), np.std(x_1))
            exit()
        if   0:
            good_ents = x_1
            good_labs = [1] * len(x_1)
        elif 1:
            good_ents = [np.mean(x_0) if len(x_0) else 0, np.mean(x_1) if len(x_1) else 0][1:]
            good_labs = [0, 1][1:]
        else:
            good_ents = [np.mean(x_0) if len(x_0) else 0, np.mean(x_1) if len(x_1) else 0][:1]
            good_labs = [0, 1][:1]
    return good_ents, good_labs, lab2
    return ent, labs, lab2


MODELS = ['xlmr', 'afriberta', 'afro_xlmr', 'mbert']
LANGS = ['amh','hau','ibo','kin','lug','luo','pcm','swa','wol','yor',]
# LANGS = ['swa',]
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
            ax.set_title(CLEAN_MODES[CORRUPTION])
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
        return ("Fraction of Labels Kept")
    if mode == 'local_cap_labels':
        return ("Maximum number of labels kept per sentence")
    if mode == 'local_swap_labels':
        return ("Maximum number of labels swapped per sentence")
    assert False, mode


if __name__ == '__main__':
    for M in ['xlmr', 'afriberta', 'afro_xlmr', 'mbert'][:1]: 
        for C in ['global_swap_labels', 'global_cap_labels', 'global_cap_sentences', 'local_cap_labels']:
            main(M, C)