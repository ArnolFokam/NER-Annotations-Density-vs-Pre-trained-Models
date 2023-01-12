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

from scripts.message import DATA_DICT
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
    
    X = pd.pivot_table(temp, columns='Model', index='Language', values='f1')
    X2 = ' (' + pd.pivot_table(temp, columns='Model', index='Language', values='f1_std', aggfunc=np.sum).round(2).astype(str) + ')'
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
        X[v] = X[v].apply(lambda x: f"{x:.2f}")
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
                    # "Number of Sentences": len(examples),
                    # "num_labels": num_labels,
                    # "num_words": num_words,
                    **stats,
                }
                if lang == 'conll_2003_en': continue # lang = 'en'
                df[lang] = stats
                print(f"{lang:<10} {li[0]} {stats}")
    df = pd.DataFrame(df)
    df = df.T
    df['Total Entities'] = df['LOC'] + df['ORG'] + df['DATE'].fillna(0) + df['PER']
    df = df.fillna(0).astype(np.int32)
    # df["Entity Density"] = df["Entity Density"].astype(np.float32)
    # df["Entity Density $(10^2)$"] = round((df["num_labels"] * 100)/df["num_words"], 2)
    # df = df.drop("num_labels", axis=1)
    # df = df.drop("num_words", axis=1)
    df = df.sort_values('Total Entities', ascending=False)
    # df /= df['Total Entities']
    # print(df)
    if 1:
        df['LOC']               /= df['Total Entities']
        df['ORG']               /= df['Total Entities']
        df['DATE']    /=             df['Total Entities']
        df['PER']               /= df['Total Entities']
    # exit()
    df.to_latex("analysis/number_entities.tex")
    # sns.barplot(df, x='lang', 
    df = df.drop('Total Entities', axis=1)
    df.plot(kind='bar', stacked=True)
    plt.xlabel('Language')
    plt.ylabel("Number of Entities")
    # plt.ylim(bottom=1)
    # plt.yscale('log')
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
        savefig(f'analysis/plots/corrupted_data/{corruption}.png')
        plt.close()
        

if __name__ == '__main__':
    # main(True)
    plot_dataset_stats()
    # main()
    # plot_entity_frequency()
    # comparison_plots(2)
    # plot_corrupted_stats()
    # main()