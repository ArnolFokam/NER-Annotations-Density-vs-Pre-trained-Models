from collections import defaultdict
import copy
import os
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ner.dataset import read_examples_from_file
from scripts.create_corrupted_datasets import ALL_FUNCS_PARAMS
import seaborn as sns
sns.set_theme()
LANGS = ['amh', 'conll_2003_en', 'hau', 'ibo', 'kin', 'lug', 'luo', 'pcm', 'swa', 'wol', 'yor']
MODELS = ['afriberta', 'mbert', 'xlmr', 'afro_xlmr']
MODES = ['global_cap_labels', 'global_cap_sentences', 'global_swap_labels', 
         'local_cap_labels',
         'local_swap_labels',
         'original']



def savefig(name, pad=0):
    plt.tight_layout()
    if '/' in name:
        path = '/'.join(name.split('/')[:-1])
        os.makedirs(path, exist_ok=True)
    # consistent saving
    plt.savefig(name, bbox_inches='tight', pad_inches=pad, dpi=200)
    # Save pdf file too
    name = name.split(".png")[0] + ".pdf"
    plt.savefig(name, bbox_inches='tight', pad_inches=pad, dpi=200)

def main(lang_to_use=None):
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
    SHOULD_READ = True
    langs_to_use = LANGS  #[l for l in LANGS if l == lang_to_use or lang_to_use is None]
    for mode in MODES:
        if not SHOULD_READ: break
        for model in MODELS:
            ROOT = f'results/{model}/{mode}'
            ranges = range(1, 11)
            if mode == 'original': ranges = [1]
            if mode.startswith('local_swap_labels'): ranges = range(0, 11)
            if mode.startswith('global_cap_sentences'): ranges = [0.1, 0.5] + list(range(1, 11))
            for nums in ranges:
                perc = nums / 10
                if mode.startswith('local'): perc = nums
                if mode == 'original': perc = 1
                for lang in langs_to_use:
                    for seed in range(1, 4):
                        dir = f"{ROOT}/{perc}/{lang}/{seed}/test_results.txt"
                        if mode.startswith('local_swap_labels') and perc == 0:
                            dir = f"results/{model}/original/1/{lang}/{seed}/test_results.txt"
                            # assert False
                        try:
                            lines = pathlib.Path(dir).read_text().split("\n")
                        except Exception as e:
                            _, model_val, perturb, param_val, lang_val, seed_val, _ =  dir.split("/")
                            print("BAD", dir, e, f"model_{perturb}_{param_val}_{lang_val}_{model_val}_{seed_val}_.bash")
                            
                            
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
    if SHOULD_READ:
        df = pd.DataFrame(all_dics)
        df = df.groupby(['mode', 'num', 'lang', 'model'], as_index=False).mean()
        df.to_csv("analysis/main_results.csv")
    # exit()
    df = pd.read_csv("analysis/main_results.csv")
    old = copy.deepcopy(df)
    for lang in langs_to_use:
        for model in MODELS:
            
            temp = df.loc[df['lang'] == lang]
            # temp = temp.loc[temp['model'] == model]
            temp = temp.loc[temp['model'] == model]
            temp = temp.loc[temp['mode'] == 'original']
            assert len(temp) == 1
            opt = temp['f1'].item()
            df.loc[np.logical_and(df['lang'] == lang, df['model'] == model), 'good'] = df.loc[np.logical_and(df['lang'] == lang, df['model'] == model), 'f1'] / opt
    old = df
    def single_plot(mode, AX, lang=None):
        df = copy.deepcopy(old)
        df = df[df['mode'] == mode]
        
        if lang is not None: 
            df = df[df['lang'] == lang]
            AX.set_title(lang)
        
        df = df.groupby(['mode', 'num', 'model', 'lang'], as_index=False).mean()
        sns.lineplot(df, x='num', y='good', hue='model', errorbar='sd', ax=AX)
        x = np.unique(df['num'])
        # AX = plt.gca()
        
        if mode == 'local_swap_labels':
            AX.plot(x, 1 - np.arange(0, 11)/10)
        elif mode == 'global_cap_sentences':
            AX.plot(x, np.array([0.1, 0.5] + list(np.arange(1, 11)))/10)
        else:
            AX.plot(x, np.arange(1, 11)/10, label='Linear Relationship')
        AX.set_xlabel("Level of Quality")
        if mode == 'global_cap_sentences':
            AX.set_xlabel("Fraction of Sentences Kept")
        if mode == 'global_cap_labels':
            AX.set_xlabel("Fraction of Labels Kept")
        if mode == 'global_swapped_labels':
            AX.set_xlabel("Fraction of Labels Swapped")
        if mode == 'local_cap_labels':
            AX.set_xlabel("Maximum number of labels kept per sentence")
        if mode == 'local_swap_labels':
            AX.set_xlabel("Maximum number of labels swapped per sentence")
        
        AX.set_ylabel("Fraction of F1 when using original dataset")
        
    for mode in MODES[:-1]:
        single_plot(mode, plt.gca())
        savefig(f'analysis/plots/corruption/{mode}.png')
        plt.close()
    if lang_to_use is not None:
        for mode in MODES[:-1]:
            fig, axs = plt.subplots(4, 3, figsize=(18, 24))
            for ax, lang in zip(axs.ravel(), LANGS):
                single_plot(mode, ax, lang)
            savefig(f'analysis/plots/corruption/langs/all_{mode}.png')
            plt.close()


def plot_dataset_stats():
    root_dir = 'data'
    ALL_FUNCS_PARAMS['original'] = (None, [{'number': i} for i in range(1, 2)])
    df = {}
    for key, value in ALL_FUNCS_PARAMS.items():
        if key != 'original': continue
        for lang in LANGS:
            for param in value[1]:
                stats = defaultdict(lambda : 0)
                li = list(param.values())
                assert len(li) == 1
                examples = read_examples_from_file(f'{root_dir}/{key}/{li[0]}/{lang}/', 'train')
                for ex in examples:
                    for l in ex.labels:
                        if l == 'O': continue
                        stats[l.split("-")[-1]] += 1
                        # stats['Total'] += 1
                stats = dict(stats)
                if lang == 'conll_2003_en': lang = 'en'
                df[lang] = stats
                print(f"{lang:<10} {li[0]} {stats}")
    df = pd.DataFrame(df)
    df = df.T
    df['Total'] = df['LOC'] + df['ORG'] + df['DATE'].fillna(0) + df['PER']
    df = df.fillna(0).astype(np.int32)
    print(df)
    df.to_latex("analysis/number_entities.tex")

def plot_entity_frequency():
    root_dir = 'data'
    ALL_FUNCS_PARAMS['original'] = (None, [{'number': i} for i in range(1, 2)])

    df = {key: np.zeros(70) for key in LANGS}
    for key, value in ALL_FUNCS_PARAMS.items():
        if key != 'original': continue
        for lang in LANGS:
            for param in value[1]:
                stats = defaultdict(lambda : 0)
                li = list(param.values())
                assert len(li) == 1
                examples = read_examples_from_file(f'{root_dir}/{key}/{li[0]}/{lang}/', 'train')
                for ex in examples:
                    num_entities = 0
                    for l in ex.labels:
                        if l == 'O': continue
                        num_entities += 1
                    df[lang][num_entities] += 1
    df = pd.DataFrame(df)      
    print(df)
    df['Nums'] = df.index
    # L = df.loc[:, 'swa']
    L = df
    print(L)
    for l in LANGS:
        if 'conll' in l: continue
        sns.lineplot(L, x='Nums', y=l)
    # sns.lineplot(L, x='Nums', y='kin')
    # plt.plot(L['Nums'], L['swa'])
    plt.show()
if __name__ == '__main__':
    main(True)
    # plot_dataset_stats()
    # plot_entity_frequency()