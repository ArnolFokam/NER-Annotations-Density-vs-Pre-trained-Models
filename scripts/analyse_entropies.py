import os
import pathlib
import pickle
from matplotlib import pyplot as plt
import numpy as np
from scripts.plot_graphs import savefig

def get_vals(folder):
    # "entropies/xlmr/global_swap_labels/0.5/swa/3/"
    ent = np.load(os.path.join(folder, 'entropies.npz'))['arr_0']
    labs = np.load(os.path.join(folder, 'labels.npz'))['arr_0']
    with open(os.path.join(folder, 'labels.p'), 'rb') as f:
        lab2 = pickle.load(f)
    return ent, labs, lab2
def mainold():
    # vals_og   = get_vals("entropies/xlmr/global_cap_labels/1.0/swa/3")
    # vals_less = get_vals("entropies/xlmr/global_cap_sentences/0.1/swa/1")
    # vals_less2 = get_vals("entropies/xlmr/global_cap_labels/0.1/swa/2")
    
    vals_og   = get_vals("entropies/xlmr/global_cap_labels/1.0/swa/3")
    vals_less = get_vals("entropies/xlmr/global_cap_labels/0.8/swa/2")
    vals_less2 = get_vals("entropies/xlmr/global_cap_labels/0.1/swa/2")
    
    print(vals_og[0].shape)
    print(vals_less[0].shape)
    print(vals_og[2])
    
    assert np.all(vals_og[1] == vals_less[1])
    assert np.all(vals_og[1] == vals_less2[1])
    
    # idx = (vals_og[1] != 0)
    idx = (vals_og[1] == 1)
    
    print(np.mean(vals_og[0][idx]), np.std(vals_og[0][idx]))
    print(np.mean(vals_less[0][idx]), np.std(vals_less[0][idx]))
    print(np.mean(vals_less2[0][idx]), np.std(vals_less2[0][idx]))
    pass
# 'conll_2003_en',
LANGS = ['amh','hau','ibo','kin','lug','luo','pcm','swa','wol','yor',]
# LANGS = ['swa']
def main(MODEL='afro_xlmr', CORRUPTION='global_swap_labels'):
    # vals_og   = get_vals("entropies/xlmr/global_cap_labels/1.0/swa/3")
    # vals_less = get_vals("entropies/xlmr/global_cap_sentences/0.1/swa/1")
    # vals_less2 = get_vals("entropies/xlmr/global_cap_labels/0.1/swa/2")
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    axs = axs.ravel()
    X = None
    # MODEL = 'afriberta'
    # MODEL = 'afro_xlmr'
    # CORRUPTION = 'global_swap_labels'
    # CORRUPTION = 'global_cap_labels'
    # CORRUPTION = 'global_cap_sentences'
    # T = 'afriberta/global_cap_labels'
    T = f'{MODEL}/{CORRUPTION}'
    # T = 'afriberta/global_cap_sentences'
    XXXX = [None] * 9
    for I, ax in enumerate(axs):
        x = []
        vals = []
        mi, ma = [], []
        for i in range(1, 11):
            x.append(i/10)
            # vals.append(get_vals(f"entropies/xlmr/global_cap_labels/{i/10}/swa/3"))
            goods = [[], [], []]
            for t in range(1, 4):
                for l in LANGS:
                    test = get_vals(f"entropies/{T}/{i/10}/{l}/{t}")
                    goods[0].extend(test[0])
                    goods[1].extend(test[1])
                    goods[2] = test[2]
            goods = (np.array(goods[0]), np.array(goods[1]), goods[2])
            ogv = v = goods
            if X is None: X = np.unique(v[1])
            # print(np.unique(v[1])); exit()
            idx = (v[1] == X[I])
            v = v[0][idx]
            
            if 1:
                vals.append(m:= np.mean(v))
                s = np.std(v)
                mi.append(m - s)
                ma.append(m + s)
            else:
                if XXXX[I] is None:
                    XXXX[I] = v
                
                # v = v - XXXX[I]
                # vals.append(m:= v[0])
                vals.append(m:= np.mean(v))
                s = 0
                mi.append(np.min(v))
                ma.append(np.max(v))
        ax.plot(x, vals)
        PPP = ogv[2]
        LLL = X[I]
        print(PPP)
        print(LLL)
        ax.set_title((PPP[LLL]) if LLL >= 0 else LLL)
        ax.fill_between(x, mi, ma, alpha=0.1)
        ax.set_ylabel("Entropy")
        ax.set_xlabel("Param")
        # ax.show()
    plt.suptitle(T)
    savefig(f'analysis/plots/entropies/{MODEL}_{CORRUPTION}.png')
    
    plt.close()
    return
    plt.show()
    
    # vals_less = get_vals("entropies/xlmr/global_cap_labels/0.8/swa/2")
    # vals_less2 = get_vals("entropies/xlmr/global_cap_labels/0.1/swa/2")
    
    print(vals_og[0].shape)
    print(vals_less[0].shape)
    print(vals_og[2])
    
    assert np.all(vals_og[1] == vals_less[1])
    assert np.all(vals_og[1] == vals_less2[1])
    
    # idx = (vals_og[1] != 0)
    idx = (vals_og[1] == 1)
    
    print(np.mean(vals_og[0][idx]), np.std(vals_og[0][idx]))
    print(np.mean(vals_less[0][idx]), np.std(vals_less[0][idx]))
    print(np.mean(vals_less2[0][idx]), np.std(vals_less2[0][idx]))
    pass

if __name__ == '__main__':
    for M in ['xlmr']: # 'afriberta', 'afro_xlmr', 'mbert', 
        for C in ['global_swap_labels', 'global_cap_labels', 'global_cap_sentences']:
            main(M, C)