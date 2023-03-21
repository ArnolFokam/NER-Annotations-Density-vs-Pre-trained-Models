from matplotlib import pyplot as plt
import pandas as pd
from scripts.message import DATA_DICT

def main():
    og = pd.read_csv("analysis/main_results_v2.csv")
    MODEL = 'AfriBERTa'
    LANGS = ['amh', 'conll_2003_en', 'hau', 'ibo', 'kin', 'lug', 'luo', 'pcm', 'swa', 'wol', 'yor']
    
    fig, axs = plt.subplots(3, 4)
    for LANG, ax in zip(LANGS, axs.ravel()):
        corruption = 'local_swap_labels_like_cap'
        corruption = 'local_cap_labels'
        for corruption in ['local_swap_labels_like_cap', 'local_cap_labels', 'global_cap_sentences']:
            df = og.copy(deep=True)
            df = df[df['model'] == MODEL]
            df = df[df['lang'] == LANG]
            df2 = df.copy(deep=True)
            df3 = df.copy(deep=True)
            df2 = df2[df2['mode'] == 'global_cap_labels']
            df3 = df3[df3['mode'] == 'global_swap_labels']
            df = df[df['mode'] == corruption]
            print(df)
            
            xs = []
            ys = []
            for _, row in df.iterrows():
                N = int(row['num']) if 'local' in corruption else row['num']
                xs.append(DATA_DICT[corruption][N][LANG])
                ys.append(row['good'])
            ax.plot(xs, ys, label=corruption)
        ax.plot(df2['num'], df2['good'], label='global cap labels')
        ax.plot(df3['num'], df3['good'], label='global swap labels')
        ax.set_title(LANG)
        if LANG == LANGS[-1]:
            ax.legend()
    plt.show()
if __name__ == '__main__':
    main()