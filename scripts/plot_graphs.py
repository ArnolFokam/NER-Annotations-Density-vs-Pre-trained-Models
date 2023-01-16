from collections import defaultdict
import copy
import os
import pathlib
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ner.dataset import get_labels_position, read_examples_from_file
from scripts.create_corrupted_datasets import ALL_FUNCS_PARAMS
import seaborn as sns

from scripts.message import DATA_DICT, DATA_DICT2
DO_JOINT_SUBPLOTS = True
sns.set_theme()
CLEAN_MODES = {
        'global_cap_sentences': 'Globally Capping Sentences',
        'global_cap_labels'   : 'Globally Capping Labels',
        'global_swap_labels'  : 'Globally Swapping Labels',
        'local_cap_labels'    : 'Locally Capping Labels',
        'local_swap_labels'   : 'Locally Swapping Labels',
        'local_swap_labels_like_cap'   : 'Locally Swapping Labels',
}
SHORT=False
YLABEL = "Fraction of F1 when using original dataset"
PER_LANG = False
DO_SUBPLOTS_THING = True
LANGS = ['amh', 'conll_2003_en', 'hau', 'ibo', 'kin', 'lug', 'luo', 'pcm', 'swa', 'wol', 'yor']
MODELS = ['afriberta', 'mbert', 'xlmr', 'afro_xlmr']
MODES = [
    
         'global_cap_sentences', 'global_cap_labels', 'global_swap_labels', 
         'local_cap_labels', 'local_swap_labels_like_cap',
         'local_swap_labels',
         
         'global_cap_sentences_seed1',
         'global_cap_sentences_seed2',
         'original',
         ]
# MODES = [
#          'local_swap_labels_like_cap',
#          'original'
#          ]
def clean_model(model):
    return {'afriberta': "AfriBERTa", 'mbert': "mBERT", 'xlmr': "XLM-R", 'afro_xlmr': "Afro-XLM-R"}[model]

def clean_lang(lang):
    if lang == 'conll_2003_en': return 'en'
    return lang

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
    SHOULD_READ =  True
    langs_to_use = LANGS  #[l for l in LANGS if l == lang_to_use or lang_to_use is None]
    for mode in MODES:
        if not SHOULD_READ: break
        for model in MODELS:
            ROOT = f'results/{model}/{mode}'
            ranges = range(1, 11)
            if mode == 'original': ranges = [1]
            if mode.startswith('local_swap_labels'): ranges = range(0, 11)
            if mode.startswith('local_swap_labels_like_cap'): ranges = range(0, 11)
            if mode.startswith('global_cap_sentences'): ranges = [0.1, 0.5] + list(range(1, 11))
            for nums in ranges:
                perc = nums / 10
                if mode.startswith('local'): perc = nums
                if mode == 'original': perc = 1
                for lang in langs_to_use:
                    for seed in range(1, 4):
                        dir = f"{ROOT}/{perc}/{lang}/{seed}/test_results.txt"
                        if mode.startswith('local_swap_labels') and 'like_cap' not in mode and perc == 0:
                            dir = f"results/{model}/original/1/{lang}/{seed}/test_results.txt"
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
        X = ['mode', 'num', 'lang', 'model']  + (['seed'] if PER_LANG else [])
        # df_std = df.groupby(['mode', 'num', 'lang', 'model'], as_index=False).std()
        # df = df.groupby(['mode', 'num', 'lang', 'model'], as_index=False).mean()
        df = df.groupby(X, as_index=False).agg(['mean','std'])
        df.columns = ['_'.join(col) for col in df.columns.values]
        df = df.reset_index(X)
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
            if PER_LANG:
                for seed in [1, 2, 3]:
                    temp = df.loc[df['lang'] == lang]
                    temp = temp.loc[temp['model'] == model]
                    temp = temp.loc[temp['mode'] == 'original']
                    temp = temp.loc[temp['seed'] == seed]
                    assert len(temp) == 1
                    opt = temp['f1'].item()
                    II = np.logical_and(df['lang'] == lang, df['model'] == model)
                    II = np.logical_and(II, df['seed'] == seed)
                    df.loc[II, 'good'] = df.loc[II, 'f1'] / opt
            else:
                temp = df.loc[df['lang'] == lang]
                temp = temp.loc[temp['model'] == model]
                temp = temp.loc[temp['mode'] == 'original']
                assert len(temp) == 1
                opt = temp['f1'].item()
                II = np.logical_and(df['lang'] == lang, df['model'] == model)
                df.loc[II, 'good'] = df.loc[np.logical_and(df['lang'] == lang, df['model'] == model), 'f1'] / opt
    df.to_csv("analysis/main_results_v2.csv", index=False)
    old = df
    def single_plot(mode, AX, lang=None, average_per_model=True):
        df = copy.deepcopy(old)
        if mode == 'global_cap_sentences':
            df.loc[df['mode'] == 'global_cap_sentences_seed1', 'mode'] = 'global_cap_sentences'
            df.loc[df['mode'] == 'global_cap_sentences_seed2', 'mode'] = 'global_cap_sentences'
        df.loc[df['lang'] == 'conll_2003_en', 'lang'] = 'en'
            
        df = df[df['mode'] == mode]
        print("MMODE", mode, len(df))
        if lang is not None: 
            df = df[df['lang'] == lang]
            AX.set_title(lang)
        print(df)
        df = df.groupby(['mode', 'num', 'model', 'lang'] + (['seed'] if lang is not None else []), as_index=False).mean()
        df = df.rename({"model": "Model"}, axis=1)
        df = df.rename({"lang": "Language"}, axis=1)
        # sns.lineplot(df, x='num', y='good', hue='Model' if average_per_model else 'Language', errorbar='sd', ax=AX, palette=sns.color_palette("Paired") if not average_per_model else None)
        sns.lineplot(df, x='num', y='good', hue='Model' if average_per_model else 'Language', errorbar='sd', ax=AX, palette=(["#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#ffee65", "#beb9db", "#fdcce5", "#8bd3c7"] + ["#e60049", "#0bb4ff",])if not average_per_model else None)
        x = np.unique(df['num'])
        plt.ylim(bottom=0, top=1.1)
        # AX = plt.gca()
        
        # if mode == 'local_swap_labels':
        #     AX.plot(x, 1 - np.arange(0, 11)/10)
        # elif mode == 'local_swap_labels_like_cap':
        #     AX.plot(x, 1 - np.arange(1, 11)/10)
        # elif 'global_cap_sentences' in mode:
        #     AX.plot(x, np.array([0.1, 0.5] + list(np.arange(1, 11)))/10)
        # else:
        #     print("MODE", mode, x)
        #     AX.plot(x, np.arange(1, 11)/10, label='Linear Relationship')
        AX.set_xlabel("Level of Quality")
        if mode == 'global_cap_sentences':
            AX.set_xlabel("Fraction of Sentences Kept")
        if mode == 'global_cap_labels':
            AX.set_xlabel("Fraction of Labels Kept")
        if mode == 'global_swap_labels':
            AX.set_xlabel("Fraction of Labels Kept")
        if mode == 'local_cap_labels':
            AX.set_xlabel("Maximum number of labels kept per sentence" if not SHORT else "Max. # of labels kept per sentence")
        if mode == 'local_swap_labels':
            AX.set_xlabel("Maximum number of labels swapped per sentence")
        if mode == 'local_swap_labels_like_cap':
            AX.set_xlabel("Maximum number of labels not swapped per sentence"  if not SHORT else "Max. # of labels kept per sentence")
        AX.set_ylabel(YLABEL)
        
    if 1:
        NN = 1.5
        if not PER_LANG and 0:
            for mode in MODES[:-1]:
                # fig = plt.figure()#figsize=(5,5))
                fig = plt.figure(figsize=(3.2 * NN, 2.8 * NN))#figsize=(5,5))
                # print(fig.get_size_inches())
                # exit()
                single_plot(mode, plt.gca())
                savefig(f'analysis/plots/corruption/{mode}.png')
                plt.close()
        N_CORRUPTS = 3
        N_SUBPLOTS = 1 + DO_JOINT_SUBPLOTS
        if DO_SUBPLOTS_THING:
            SHORT=True
            NN = 1.25
            fig, axs = plt.subplots(N_SUBPLOTS, N_CORRUPTS, figsize=(3.2 * N_CORRUPTS * NN, 2.4 * NN * N_SUBPLOTS), sharey='row', sharex='col')
            axs = axs.reshape(N_SUBPLOTS, -1)
            for mode, ax in zip(MODES[:-1], axs[0]):
                single_plot(mode, ax)
                if DO_JOINT_SUBPLOTS: ax.set_xlabel('')
                if mode != MODES[0]:
                    ax.set_ylabel("")
                else:
                    ax.set_ylabel("Fraction of F1 compared\nto using original dataset", fontsize=13)
                if mode != 'global_cap_sentences':
                    ax.get_legend().remove()
                else:
                    leg = ax.legend(title="Model")
                    for line in leg.get_lines():
                        line.set_linewidth(4.0)
                ax.set_title(CLEAN_MODES[mode])
            if not DO_JOINT_SUBPLOTS:
                savefig(f'analysis/plots/corruption/subplots_all_models.png')
                plt.close()
            if DO_JOINT_SUBPLOTS:
                axs = axs[1]
            else:
                fig, axs = plt.subplots(1, N_CORRUPTS, figsize=(3.2 * N_CORRUPTS * NN, 2.4 * NN), sharey=True)
            for mode, ax in zip(MODES[:-1], axs):
                single_plot(mode, ax, average_per_model=False)
                if mode != MODES[0]:
                    ax.set_ylabel("")
                else:
                    ax.set_ylabel("Fraction of F1 compared\nto using original dataset", fontsize=13)
                if mode != 'global_cap_sentences':
                    ax.get_legend().remove()
                else:
                    # ax.legend(fontsize=8, title='Language')
                    leg = ax.legend(title='Language', ncol=2)#, fontsize=10)#, fontsize=16)
                    # leg = ax.legend()

                    # change the line width for the legend
                    for line in leg.get_lines():
                        line.set_linewidth(4.0)

                if not DO_JOINT_SUBPLOTS: ax.set_title(CLEAN_MODES[mode])
            if not DO_JOINT_SUBPLOTS:
                savefig(f'analysis/plots/corruption/subplots_all_langs.png')
            else:
                savefig(f'analysis/plots/corruption/subplots_all_all.png')
            plt.close()
        if PER_LANG:
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
    temp = temp.rename({'model': "Model", 'lang': "Language"}, axis=1)
    # temp['Model'] = temp['Model'].map(clean_model)
    temp['Language'] = temp['Language'].apply(clean_lang)
    print(temp)
    # exit()
    temp['f1'] *= 100.0
    temp['f1'] = temp['f1'].round(1)
    temp['f1_std'] *= 100.0
    X = pd.pivot_table(temp, columns='Model', index='Language', values='f1')
    X2 = ' (' + pd.pivot_table(temp, columns='Model', index='Language', values='f1_std', aggfunc=np.sum).round(1).astype(str) + ')'
    # X = X.round(2)
    # print(X.mean(axis=0))
    X['Average'] = X.mean(axis=1)
    X.loc['Average'] = X.mean()
    
    X2['Average'] = ''
    X2.loc['Average'] = ''
    
    goods = []
    # now do textbf per row
    for i, lang in enumerate(X.index):
        test = (-1, -1)
        for j, model in enumerate(X.columns):
            if model == 'Average' or lang == 'Average': continue
            perf = X.iloc[i, j]
            test = max(test, (perf, j))
            # if lang == 'en':print("EN", model, perf)
            # print("MODEL", model, lang)
        # TTT = X.loc[i, X.columns[j]]
        if lang == 'Average': continue
        goods.append((i, test[1]))
        
    # X = X.round(2)
    for v in X.columns:
        X[v] = X[v].apply(lambda x: f"{x:.1f}")
    X = X.astype(str)
    # exit()
    
    X = (X + X2)
    for i, j in goods:
        TTT = X.iloc[i, j]
        X.iloc[i, j] = "\textbf{" + str(TTT) + "}"
    s = X.to_latex(column_format='lllllr', escape=False)
    splits = s.split("\n")
    
    splits.insert(-4, r'\midrule')
    s = '\n'.join(splits)
    with open("analysis/performance_all.tex", 'w+') as f:
        f.write(s)


def plot_dataset_stats():
    root_dir = 'data'
    nice_names = {
        'amh': 'Amharic',
        'hau': 'Hausa',
        'ibo': 'Igbo',
        'kin': 'Kinyarwanda',
        'lug': 'Luganda',
        'luo': 'Luo',
        'pcm': 'Nigerian Pidgin',
        'swa': 'Swahili',
        'wol': 'Wolof',
        'yor': 'Yorùbá',
        'en': 'English',
    }
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
                # + read_examples_from_file(f'{root_dir}/{key}/{li[0]}/{lang}/', 'test') + read_examples_from_file(f'{root_dir}/{key}/{li[0]}/{lang}/', 'dev')
                num_labels = 0
                num_words = 0
                for ex in examples:
                    for l in ex.labels:
                        num_words += 1
                        if l == 'O': continue
                        num_labels += 1
                        stats[l.split("-")[-1]] += 1
                        # stats['Total'] += 1
                # num_words = sum([len(ex.words) for ex in examples])
                labels = list(map(lambda x: [label.split('-')[-1] for label in x.labels if label != 'O'], examples))
                # num_labels = sum([len(np.unique(l)) for l in labels])
                if lang == 'conll_2003_en': lang = 'en'
                stats = {
                    "Number of Sentences": len(examples),
                    "num_labels": num_labels,
                    "num_words": num_words,
                    'Code': lang, # nice_names[lang]
                    **stats,
                }
                df[nice_names[lang]] = stats
                print(f"{lang:<10} {li[0]} {stats}")
    df = pd.DataFrame(df)
    df = df.T
    df['Total Entities'] = df['LOC'] + df['ORG'] + df['DATE'].fillna(0) + df['PER']
    df = df.drop(["LOC", "ORG", "DATE", "PER"], axis=1)
    T = df['Code']
    df = df.drop('Code', axis=1)
    df = df.fillna(0).astype(np.int32)
    df['Language Code'] = T
    # df["Entity Density (%)"] = df["Entity Density (%)"].astype(np.float32)
    df["Entity Density (%)"] = ((df["num_labels"])/df["num_words"]).round(3) * 100
    df = df.drop("num_labels", axis=1)
    df = df.drop("num_words", axis=1)
    df = df.sort_values('Entity Density (%)', ascending=False)
    print(df.columns)
    # exit()
    df = df[['Language Code', 'Number of Sentences', 'Total Entities', 'Entity Density (%)']]
    print(df)
    df.to_latex("analysis/dataset_info.tex")
    # sns.barplot(df, x='lang', 
    df = df.drop('Total Entities', axis=1)
    df.plot(kind='bar', stacked=True)
    plt.xlabel('Language')
    plt.ylabel("Number of Entities")
    plt.yscale('log')
    plt.tight_layout()
    savefig("analysis/number_entities.png")
    
def plot_enity_dist_frequency():
    root_dir = 'data'
    ALL_FUNCS_PARAMS['original'] = (None, [{'number': i} for i in range(1, 2)])

    df = {key: np.zeros(70) for key in LANGS}
    df_temp = {key: 0 for key in LANGS}
    for key, value in ALL_FUNCS_PARAMS.items():
        if key != 'original': continue
        for lang in LANGS:
            for param in value[1]:
                # stats = defaultdict(lambda : 0)
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
    

def plot_entity_frequency():
    root_dir = 'data'
    ALL_FUNCS_PARAMS['original'] = (None, [{'number': i} for i in range(1, 2)])

    df = {key: np.zeros(70) for key in LANGS}
    df_temp = {key: 0 for key in LANGS}
    for key, value in ALL_FUNCS_PARAMS.items():
        if key != 'original': continue
        for lang in LANGS:
            for param in value[1]:
                # stats = defaultdict(lambda : 0)
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
    
def comparison_plots(a):
    df = pd.read_csv("analysis/main_results_v2.csv")
    df.loc[df['mode'] == 'global_cap_sentences_seed1', 'mode'] = 'global_cap_sentences'
    df.loc[df['mode'] == 'global_cap_sentences_seed2', 'mode'] = 'global_cap_sentences'
    XX = []
    def rename_thing(old_mode, new_mode):
        XX.append(new_mode)
        df.loc[df['mode'] == old_mode, 'mode'] = new_mode
    if a == 0:
        df = df[np.logical_or(np.logical_or(df['mode'] == 'global_cap_sentences', df['mode'] == 'global_cap_labels'), df['mode'] == 'global_swap_labels')]
        rename_thing('global_cap_sentences', 'Capping Sentences')
        rename_thing('global_cap_labels', 'Capping Labels')
        rename_thing('global_swap_labels', 'Swapping Labels')
        
        # rename_thing('local_cap_labels', 'Capping Labels')
        # rename_thing('local_swap_labels', 'Swapping Labels')
        
        df = df.rename({"num": "Level of Quality", "good": "Fraction of F1 when using original dataset"}, axis=1)

        # df = df.groupby(['mode', 'num']).mean()
        print(df)
        # df = df.groupby(['mode', 'num', 'model', 'lang'] + (['seed'] if lang is not None else []), as_index=False).mean()
        sns.lineplot(df, x="Level of Quality", y="Fraction of F1 when using original dataset", hue='mode', errorbar='sd', hue_order=XX)
        # plt.show()
        savefig(f'analysis/plots/compare/compare_global.png')
        pass
    elif a == 1:
        df = df[np.logical_or(df['mode'] == 'local_cap_labels', df['mode'] == 'local_swap_labels_like_cap')]
        rename_thing('local_cap_labels', 'Capping Labels')
        rename_thing('local_swap_labels_like_cap', 'Swapping Labels')
        
        df = df.rename({"num": "Maximum number of labels kept per sentence", "good": "Fraction of F1 when using original dataset"}, axis=1)
        sns.lineplot(df, x="Maximum number of labels kept per sentence", y="Fraction of F1 when using original dataset", hue='mode', errorbar='sd', hue_order=XX)
        plt.ylim(bottom=0)
        savefig(f'analysis/plots/compare/compare_local.png')
    elif a == 2:
        # df = df[np.logical_or(df['mode'] == 'local_cap_labels', df['mode'] == 'local_swap_labels_like_cap')]
        rename_thing('local_cap_labels', 'Capping LabelsLocal')
        rename_thing('local_swap_labels_like_cap', 'Swapping LabelsLocal')
        rename_thing('global_cap_sentences', 'Capping Sentences')
        rename_thing('global_cap_labels', 'Capping Labels')
        rename_thing('global_swap_labels', 'Swapping Labels')
        
        df = df.rename({"num": "Maximum number of labels kept per sentence", "good": "Fraction of F1 when using original dataset"}, axis=1)
        sns.lineplot(df, x="Maximum number of labels kept per sentence", y="Fraction of F1 when using original dataset", hue='mode', errorbar='sd', hue_order=XX)
        plt.ylim(bottom=0)
        savefig(f'analysis/plots/compare/compare_all_bad.png')

def plot_corrupted_stats():
    for corruption, v in DATA_DICT.items():
        xs = []
        ys = []
        for param, d in v.items():
            xs.append(param)
            this_param = []
            for lang, perc in d.items():
                this_param.append(perc)
            ys.append(this_param)
        xs = np.array(xs)
        ys = np.array(ys)
        mean, std = np.mean(ys, axis=1), np.std(ys, axis=1)
        plt.plot(xs, mean)
        plt.fill_between(xs, mean - std, mean + std, alpha=0.3)
        plt.title(CLEAN_MODES[corruption])
        plt.xlabel("Number of Sentences Kept")
        plt.ylabel("Percentage of Labels Remaining")
        # plt.show()
        # savefig()
        savefig(f'analysis/plots/corrupted_data/v2_{corruption}.png')
        plt.close()
        
           
def get_total_entities(X):
    total_unique = 0
    n_labels = []
    unique_ents =  defaultdict(lambda: 0) # set()
    unique_words = defaultdict(lambda: 0) #set()
    num_words = 0
    for ex in X:
        idxs = get_labels_position(ex.labels.copy())
        num_words += len(ex.labels)
        n_labels.extend(idxs)
        for start, end in idxs:
            if 1:
                w = ' '.join(ex.words[start:end+1])
                unique_ents[w] += 1
            if 1:
                for jj in range(start, end+1):
                    unique_words[(ex.words[jj])] += 1
    
    return len(n_labels), len(unique_ents), len(unique_words), unique_ents, unique_words, num_words

def get_propotion_entities():
    langs = ['amh','conll_2003_en','hau','ibo','kin','lug','luo','pcm','swa','wol','yor',]
    corruption_stats = ["local_swap_labels_like_cap", "local_cap_labels", "global_cap_sentences"]
    corruption_stats = ["global_cap_labels", "global_cap_sentences"]
    corruption_stats = ["global_cap_sentences", "global_cap_sentences_seed1", "global_cap_sentences_seed2", "global_cap_labels"]
    percentage = [i / 10 for i in range(1, 11)] + [0.01, 0.05]
    number = [i for i in range(1, 11)]
    
    params = {
        "local_swap_labels_like_cap": number,
        "global_cap_sentences": percentage,
        "global_cap_labels": [i / 10 for i in range(1, 11)],
        "global_cap_sentences_seed1": percentage,
        "global_cap_sentences_seed2": percentage,
    }
    entities_prop_unique = {}
    all = {
        'mode':        [],
        'lang':        [],
        'param':       [],
        'frac':        [],
        'value': []
    }
    for c in corruption_stats:
        if c in ["global_cap_sentences", "global_cap_sentences1", "global_cap_sentences2", "global_cap_labels"]:
            for l in langs:
                for p in params[c]:
                    examples = read_examples_from_file(f'{data_dir}/{c}/{p}/{l}', 'train')
                    orig_examples = read_examples_from_file(f'{data_dir}/original/1/{l}', 'train')
                    cnow, cnow_unique_ents, cnow_unique_words, now_ents, now_words, num_words_tot = get_total_entities(examples)
                    cog, cog_unique_ents, cog_unique_words, og_ents, og_words, og_words_tot = get_total_entities(orig_examples)
                    # print(l, p, c, cnow/cog, cnow_unique/cog_unique);#exit()
                    cc = c if not 'global_cap_sentences' in c else 'global_cap_sentences'
                    # cc = c
                    for _ in range(6):
                        all['mode'].append(cc);# all['mode'].append(cc); all['mode'].append(cc)
                        all['lang'].append(l);# all['lang'].append(l); all['lang'].append(l)
                        all['param'].append(p);# all['param'].append(p); all['param'].append(p)
                    
                    def mycounts(A, B):
                        t = []
                        for a, v in B.items():
                            # if a in A:
                            if cc == 'global_cap_sentences':
                                t.append(1)
                            else:
                                t.append(A.get(a, 0) / v)
                        return np.mean(t)
                    def mycountsv2(A, B):
                        t = []
                        for a, v in A.items():
                            t.append(v)
                        return np.mean(t) / og_words_tot
                    
                    all['value'].append(cnow / cog)
                    all['value'].append(cnow_unique_ents / cog_unique_ents)
                    all['value'].append(cnow_unique_words / cog_unique_words)
                    
                    all['value'].append(mycounts(now_ents, og_ents))
                    all['value'].append(mycounts(now_words, og_words))
                    
                    all['value'].append((cnow / num_words_tot))
                    
                    all['frac'].append('normal')
                    all['frac'].append('unique entities')
                    all['frac'].append('unique tokens')
                    all['frac'].append('Entities Ratio')
                    all['frac'].append('Words Ratio')
                    all['frac'].append('correct labels/total words')
                    # all['frac_unique'].append(cnow_unique / cog_unique)
    df = pd.DataFrame(all)
    print(df)
    
    # df = df[np.logical_or(df['frac'] == 'Entities Ratio', df['frac'] == 'Words Ratio')]
    df = df[(df['frac'] == 'correct labels/total words')]
    sns.lineplot(df, x='param', y='value', style='mode'        , errorbar='sd', hue='frac')#, hue='mode')
    # sns.lineplot(df, x='param', y='frac_unique' , errorbar='sd')#, hue='mode')
    plt.show()


def bad_get_things():
    def get_total_entities(X):
        
        n_labels = []
        
        for ex in X:
            n_labels.append(len(get_labels_position(ex.labels.copy())))
        return sum(n_labels)
    original_num_entities = {'amh': 2247,
        'conll_2003_en': 19484,
        'hau': 3989,
        'ibo': 3254,
        'kin': 3629,
        'lug': 3075,
        'luo': 1138,
        'pcm': 4632,
        'swa': 4570,
        'wol': 1387,
        'yor': 2757
    }
    
    langs = ['amh','conll_2003_en','hau','ibo','kin','lug','luo','pcm','swa','wol','yor',]
    corruption_stats = ["local_swap_labels_like_cap", "local_cap_labels", "global_cap_sentences"]
    corruption_stats = ["global_cap_sentences"]
    percentage = [i / 10 for i in range(1, 11)] + [0.01, 0.05]
    number = [i for i in range(1, 11)]
    
    params = {
        "local_swap_labels_like_cap": number,
        "global_cap_sentences": percentage,
        "local_cap_labels": number
    }
    data_dir = "data"
    
    entities_prop = {}
    
    for c in corruption_stats:
        cp = {}
        if c in ["global_cap_sentences", "local_cap_labels"]:
            # continue
            for p in params[c]:
                ld = {}
                for l in langs:
                    examples = read_examples_from_file(f'{data_dir}/{c}/{p}/{l}', 'train')
                    orig_examples = read_examples_from_file(f'{data_dir}/{c}/{p}/{l}', 'train')
                    ld[l] = get_total_entities(examples) / original_num_entities[l]
                    
                    print(p, l, c, ld[l])
                    exit()
                cp[p] = ld
            entities_prop[c] = cp
                    
        else:
            for p in params[c]:
                ld = {}
                for l in langs:
                    examples = read_examples_from_file(f'{data_dir}/{c}/{p}/{l}', 'train')
                    orig_examples = read_examples_from_file(f'{data_dir}/original/1/{l}', 'train')
                    ld[l] = get_total_entities_not_different(examples, orig_examples)  / original_num_entities[l]
                cp[p] = ld
            entities_prop[c] = cp
                
    pprint(entities_prop)

def get_f1_from_filename(f):
    lines = pathlib.Path(f).read_text().split("\n")                        
    line = [l for l in lines if 'f1 = ' in l]
    assert len(line) == 1
    return float(line[0].split("f1 = ")[-1])

def check_quality_and_quantity():
    def inner(LANG):
        D = 'results/afro_xlmr/'
        ps = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0][1:]#, 1]
        result = {
            'sent': [],
            'label': [],
            'seed': [],
            'f1': [],
        }
        
        result = {
            # 'seed': {f'Label {p}': [] for p in ps}
        }
        
        def sents(p):
            return f'{round(p*100)}%'
            return f'{round(p*100)}% Sentences'
        def labs(p):
            return f'{round(p*100)}%'
            return f'{round(p*100)}% Labels'
        for p in ps:
            # result[f'Sent {p}'] = {f'Label {p}': [] for p in ps}
            result[sents(p)] = {labs(p): [] for p in ps}
        for p_sent in ps:
            for p_label in ps:
                for seed in range(1, 4):
                    OG = os.path.join(D, 'original', f'1', LANG, str(seed), 'test_results.txt')
                    OG_PERF = get_f1_from_filename(OG)
                    # if p_sent == 1:
                    #     d = os.path.join(D, 'global_cap_labels', f'{p_label}', 'swa', str(seed), 'test_results.txt')
                    # elif p_label == 1:
                    #     d = os.path.join(D, 'global_cap_sentences', f'{p_sent}', 'swa', str(seed), 'test_results.txt')
                    # else:
                    if p_sent == 1.0 and p_label == 1.0:
                        f1 = 1.0
                    else:
                        d = os.path.join(D, 'global_cap_sentences_and_labels', f'cap_{p_sent}_{p_label}', LANG, str(seed), 'test_results.txt')
                        f1 = get_f1_from_filename(d) / OG_PERF
                    # lines = pathlib.Path(d).read_text().split("\n")                        
                    # line = [l for l in lines if 'f1 = ' in l]
                    # assert len(line) == 1
                    # f1 = float(line[0].split("f1 = ")[-1]) / 0.8845676458419529
                    # result[f'Sent {p_sent}'][f'Label {p_label}'].append(f1)
                    result[sents(p_sent)][labs(p_label)].append(f1)
        df = pd.DataFrame(result)
        def std(x):
            return np.std(x)
        def test(x):
            return np.mean(x)
            print(x)
            exit()
        
        # Clean rows and columns
        def clean_col(k):
            if "% Sentences" in k: return k
            p = round(float(k.split(" ")[-1]) * 100)
            return f"{p}% Sentences"
        # df = df.rename({k: clean_col(k) for k in df.columns}, axis=1)
        
        df_std = df.applymap(std)
        df = df.applymap(test)
        print(df)
        sns.heatmap(df, annot=True, cmap="YlGnBu", vmin=0, vmax=1)
        plt.xlabel("Sentences")
        plt.ylabel("Labels")
        savefig(f'analysis/plots/cap_sent_and_labels/heatmap_{LANG}.png')
        plt.close()
        # print(df_std)
        # exit()
        # df = df.T
        # df_std = df_std.T
        for c in df.columns:
            v = df[c]
            stds = df_std[c]
            fff = float(c.split('%')[0])
            LLL = 1
            if LANG == 'luo': LLL = 2700
            if LANG == 'swa': LLL = 7000
            if LANG == 'conll_2003_en': LLL = 30_000
            LLL /= 100
            LLL = 1
            plt.plot([i * fff * LLL for i in ps], v, label=c)
            plt.fill_between([i * fff for i in ps], v - stds, v + stds, alpha=0.3)
        plt.xlabel("Label fraction")
        # plt.xlabel("Sentence Corruption * Label Corruption")
        # plt.xlabel(r"Sentence Corruption $\times$ Label Corruption")
        plt.xlabel("Overall Percentage of Labels Remaining")
        plt.ylabel(YLABEL)
        plt.legend(title="Fraction of Sentences")
        savefig(f'analysis/plots/cap_sent_and_labels/lineplot_{LANG}.png')
        plt.close()
    inner('swa')
    inner('luo')
    inner('conll_2003_en')

if __name__ == '__main__':
    # main(True)
    # plot_dataset_stats()
    # main()
    # plot_entity_frequency()
    # comparison_plots(2)
    # plot_corrupted_stats()
    # main()
    # get_propotion_entities()
    check_quality_and_quantity()
    # bad_get_things()
    # plot_dataset_stats()
