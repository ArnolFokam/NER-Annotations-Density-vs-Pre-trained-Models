import copy
import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
LANGS = ['amh', 'conll_2003_en', 'hau', 'ibo', 'kin', 'lug', 'luo', 'pcm', 'swa', 'wol', 'yor']
MODELS = ['afriberta', 'mbert', 'xlmr', 'afro_xlmr']
MODES = ['global_cap_labels', 'global_cap_sentences', 'global_swap_labels', 'original']



def savefig(name, pad=0):
    if '/' in name:
        path = '/'.join(name.split('/')[:-1])
        os.makedirs(path, exist_ok=True)
    # consistent saving
    plt.savefig(name, bbox_inches='tight', pad_inches=pad, dpi=200)
    # Save pdf file too
    name = name.split(".png")[0] + ".pdf"
    plt.savefig(name, bbox_inches='tight', pad_inches=pad, dpi=200)

def main():
    model = 'afriberta'
    mode = 'global_cap_labels'
    # for nums in range(1, 11):
    all_dics = {
        'mode':     [],
        'num':      [],
        'lang':     [],
        'model':    [],
        'seed':     [],
        'f1':       [],
    }
    for mode in MODES:
        break
        for model in MODELS:
            ROOT = f'results/{model}/{mode}'
            ranges = range(1, 11)
            if mode == 'original': ranges = [1]
            for nums in ranges:
                perc = nums / 10
                if mode == 'original': perc = 1
                for lang in LANGS:
                    for seed in range(1, 4):
                        dir = f"{ROOT}/{perc}/{lang}/{seed}/test_results.txt"
                        try:
                            lines = pathlib.Path(dir).read_text().split("\n")
                        except Exception as e:
                            print("BAD", dir, e)
                            continue
                        line = [l for l in lines if 'f1 = ' in l]
                        assert len(line) == 1
                        f1 = float(line[0].split("f1 = ")[-1])
                        print(f1)
                        all_dics['mode'].append(mode)
                        all_dics['num'].append(perc)
                        all_dics['lang'].append(lang)
                        all_dics['model'].append(model)
                        all_dics['seed'].append(seed)
                        all_dics['f1'].append(f1)
    if 0:
        df = pd.DataFrame(all_dics)
        df = df.groupby(['mode', 'num', 'lang', 'model'], as_index=False).mean()
        df.to_csv("analysis/main_results.csv")
    df = pd.read_csv("analysis/main_results.csv")
    
    for lang in LANGS:
        for model in MODELS:
            
            temp = df.loc[df['lang'] == lang]
            temp = temp.loc[temp['model'] == model]
            temp = temp.loc[temp['mode'] == 'original']
            assert len(temp) == 1
            opt = temp['f1'].item()
            df.loc[np.logical_and(df['lang'] == lang, df['model'] == model), 'good'] = df.loc[np.logical_and(df['lang'] == lang, df['model'] == model), 'f1'] / opt
            # T['good'] = T['f1'] / opt
            # good['good'] = good['f1'] / opt
    # df = good
        # df['good']
    
    # df = df[df['model'] == 'afriberta']
    # df = df[df['mode'] == 'global_cap_labels']
    # df = df[df['mode'] == 'global_cap_sentences']
    
    old = df
    for mode in MODES[:-1]:
        df = copy.deepcopy(old)
        df = df[df['mode'] == mode]
        df = df.groupby(['mode', 'num', 'model', 'lang'], as_index=False).mean()
        print(len(df))
        print(df)             
        # plt.plot(df['num'], df['good'])
        sns.lineplot(df, x='num', y='good', hue='model')
        plt.plot(df['num'], df['num'])
        plt.xlabel("Level of Quality")
        plt.ylabel("Fraction of Optimal")
        
        savefig(f'analysis/plots/corruption/{mode}.png')
        plt.close()
        # plt.show()
        
    pass
if __name__ == '__main__':
    main()