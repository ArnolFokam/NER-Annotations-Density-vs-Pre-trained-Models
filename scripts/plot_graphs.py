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
         'global_cap_sentences_seed1',
         'global_cap_sentences_seed2',
         'original',
         ]
def clean_model(model):
    return {'afriberta': "AfriBERTa", 'mbert': "mBERT", 'xlmr': "XLM-R", 'afro_xlmr': "Afro-XLM-R"}[model]


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
    global MODELS
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
                        all_dics['model'].append(clean_model(model))
                        all_dics['seed'].append(seed)
                        all_dics['f1'].append(f1)
    MODELS = [clean_model(m) for m in MODELS]
    if SHOULD_READ:
        df = pd.DataFrame(all_dics)
        # df_std = df.groupby(['mode', 'num', 'lang', 'model'], as_index=False).std()
        # df = df.groupby(['mode', 'num', 'lang', 'model'], as_index=False).mean()
        df = df.groupby(['mode', 'num', 'lang', 'model'], as_index=False).agg(['mean','std'])
        df.columns = ['_'.join(col) for col in df.columns.values]
        df = df.reset_index(['num', 'lang', 'model', 'mode'])
        df['f1'] = df['f1_mean']
        print(df)
        print(df.columns)
        df.to_csv("analysis/main_results.csv")
        # exit()
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
        if mode == 'global_cap_sentences':
            df.loc[df['mode'] == 'global_cap_sentences_seed1', 'mode'] = 'global_cap_sentences'
            df.loc[df['mode'] == 'global_cap_sentences_seed2', 'mode'] = 'global_cap_sentences'
        df = df[df['mode'] == mode]
        print("MMODE", mode, len(df))
        
        if lang is not None: 
            df = df[df['lang'] == lang]
            AX.set_title(lang)
        
        df = df.groupby(['mode', 'num', 'model', 'lang'], as_index=False).mean()
        sns.lineplot(df, x='num', y='good', hue='model', errorbar='sd', ax=AX)
        x = np.unique(df['num'])
        # AX = plt.gca()
        
        if mode == 'local_swap_labels':
            AX.plot(x, 1 - np.arange(0, 11)/10)
        elif 'global_cap_sentences' in mode:
            AX.plot(x, np.array([0.1, 0.5] + list(np.arange(1, 11)))/10)
        else:
            print("MODE", mode, x)
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
        
    if 1:
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
    # Make table for the langs
    temp     = df[df['mode'] == 'original']
    # temp_std = df_std[df_std['mode'] == 'original']
    # print((len(temp)))
    # print(temp)
    
    # temp['two'] = (((temp['f1'].round(2).astype(str) + " (").str.cat(temp['f1_std'].round(2).astype(str))) + ")")
    # print(temp['two'])
    # exit()
    # temp.loc[:, 'actual'] == alls 
    X = pd.pivot_table(temp, columns='model', index='lang', values='f1')
    X2 = ' (' + pd.pivot_table(temp, columns='model', index='lang', values='f1_std', aggfunc=np.sum).round(2).astype(str) + ')'
    # X = X.round(2)
    # print(X.mean(axis=0))
    X['avg'] = X.mean(axis=1)
    X.loc['avg'] = X.mean()
    
    X2['avg'] = ''
    X2.loc['avg'] = ''
    
    X = (X.round(2).astype(str) + X2)
    s = X.to_latex(column_format='lllll|l')
    print(s)
    splits = s.split("\n")
    
    splits.insert(-4, r'\midrule')
    s = '\n'.join(splits)
    print(s)
    with open("analysis/performance_all.tex", 'w+') as f:
        f.write(s)


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
                num_words = sum([len(ex.words) for ex in examples])
                labels = list(map(lambda x: [label.split('-')[-1] for label in x.labels if label != 'O'], examples))
                num_labels = sum([len(np.unique(l)) for l in labels])
                stats = {
                    "Number of Sentences": len(examples),
                    "num_labels": num_labels,
                    "num_words": num_words,
                    **stats,
                }
                if lang == 'conll_2003_en': lang = 'en'
                df[lang] = stats
                print(f"{lang:<10} {li[0]} {stats}")
    df = pd.DataFrame(df)
    df = df.T
    df['Total Entities'] = df['LOC'] + df['ORG'] + df['DATE'].fillna(0) + df['PER']
    df = df.fillna(0).astype(np.int32)
    # df["Entity Density"] = df["Entity Density"].astype(np.float32)
    df["Entity Density $(10^2)$"] = round((df["num_labels"] * 100)/df["num_words"], 2)
    df = df.drop("num_labels", axis=1)
    df = df.drop("num_words", axis=1)
    df = df.sort_values('Total Entities', ascending=False)
    print(df)
    df.to_latex("analysis/number_entities.tex")
    # sns.barplot(df, x='lang', 
    df = df.drop('Total Entities', axis=1)
    df.plot(kind='bar', stacked=True)
    plt.xlabel('Language')
    plt.ylabel("Number of Entities")
    plt.yscale('log')
    plt.tight_layout()
    savefig("analysis/number_entities.png")

def plot_entity_frequency():
    root_dir = 'data'
    ALL_FUNCS_PARAMS['original'] = (None, [{'number': i} for i in range(1, 2)])

    df = {key: np.zeros(70) for key in LANGS}
    df_temp = {key: 0 for key in LANGS}
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
                    df_temp[lang] += 1
    for l in df:
        for i in range(len(df[l])):
            df[l][i] = df[l][i] / df_temp[l]
    df = pd.DataFrame(df)      
    print(df)
    df['Number of Entities'] = df.index
    temp = pd.melt(df, id_vars='Number of Entities', var_name='lang', value_name='Fraction of Sentences')
    sns.lineplot(temp, x='Number of Entities', y='Fraction of Sentences', hue='lang', palette=sns.color_palette("Paired"))
    plt.xlim(0, 15)
    # plt.show()
    savefig("analysis/entity_distribution.png")
    # L = df.loc[:, 'swa']
    print(temp)
    exit()
    L = df
    print(L)
    for l in LANGS:
        if 'conll' in l: continue
        sns.lineplot(L, x='Nums', y=l)
    # sns.lineplot(L, x='Nums', y='kin')
    # plt.plot(L['Nums'], L['swa'])
    
    
    plt.show()
if __name__ == '__main__':
<<<<<<< HEAD
    main(True)
    # plot_dataset_stats()
=======
    # main(True)
    plot_dataset_stats()
>>>>>>> 420d2900 (good)
    # plot_entity_frequency()