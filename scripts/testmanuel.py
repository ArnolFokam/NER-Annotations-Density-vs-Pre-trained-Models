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
                    "Num Sentences": len(examples),
                    "Num Entity Labels (except \myO)": num_labels,
                    "Num Tokens": num_words,
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
    df = df.drop(['LOC', 'ORG', 'DATE', 'PER'], axis=1)
    df["Entity Labels/Sentences"] = round((df["Num Entity Labels (except \myO)"])/df["Num Sentences"], 2)
    df["Entity Labels/Tokens $(10^2)$"] = round((df["Num Entity Labels (except \myO)"] * 100)/df["Num Tokens"], 2)
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