from collections import defaultdict
import os
import pathlib
import shutil
def do_thing(filepath):
    s = pathlib.Path(f'{filepath}/train.txt').read_text()
    sentences = s.split("\n\n")
    vals = defaultdict(lambda: defaultdict(lambda: 0))
    for sent in sentences:
        if sent.strip() == '': continue
        lines = sent.split('\n')
        for l in lines:
            token, label = l.split(' ')
            if label != 'O': label = label.split("-")[1]
            vals[token][label] += 1

    new_text = ''
    sentences = s.split("\n\n")
    for sent in sentences:
        if sent.strip() == '': 
            new_text += '\n'
            continue
        lines = sent.split('\n')
        for i, l in enumerate(lines):
            prevlabel = 'O'
            token, label = l.split(' ')
            ttt = sorted(vals[token].items(), key=lambda x: x[1])
            if label != 'O': newlabel = label
            else:
                if ttt[-1][0] == 'O':
                    newlabel = label
                else:
                    if '-' in label:
                        newlabel = label.split("-")[0] + '-' +  ttt[-1][0]
                    else:
                        first = 'B-'
                        if prevlabel.startswith('B-'):
                            first = 'I-'
                        newlabel = first +  ttt[-1][0]
            prevlabel = newlabel
            xx = '\n' if i != len(lines) -1 else '\n\n'
            new_text += f"{token} {newlabel}{xx}"
    
    D = f'{filepath}_majority/'
    os.makedirs(D, exist_ok=True)
    pathlib.Path(f'{filepath}_majority/train.txt').write_text(new_text[:-1])
    shutil.copyfile(f'{filepath}/dev.txt', f'{filepath}_majority/dev.txt')
    shutil.copyfile(f'{filepath}/test.txt', f'{filepath}_majority/test.txt')
    
if __name__ == '__main__':
    all_vals = [
        'data/global_cap_sentences_and_labels/cap_0.1_0.1',
        'data/global_cap_sentences_and_labels/cap_0.01_0.1',
        'data/global_cap_sentences_and_labels/cap_0.1_0.01',
        'data/global_cap_sentences_and_labels/cap_0.01_0.01',
        'data/global_cap_sentences_and_labels/cap_0.1_0.5',
        'data/global_cap_sentences_and_labels/cap_0.01_0.5',
        'data/global_cap_sentences_and_labels/cap_0.1_0.05',
        'data/global_cap_sentences_and_labels/cap_0.01_0.05',
        'data/global_cap_sentences_and_labels/cap_0.1_0.25',
        'data/global_cap_sentences_and_labels/cap_0.01_0.25',
        'data/global_cap_sentences_and_labels/cap_0.1_0.75',
        'data/global_cap_sentences_and_labels/cap_0.01_0.75',
        'data/global_cap_sentences_and_labels/cap_0.1_1.0',
        'data/global_cap_sentences_and_labels/cap_0.01_1.0',
        'data/global_cap_sentences_and_labels/cap_0.5_0.1',
        'data/global_cap_sentences_and_labels/cap_0.05_0.1',
        'data/global_cap_sentences_and_labels/cap_0.5_0.01',
        'data/global_cap_sentences_and_labels/cap_0.05_0.01',
        'data/global_cap_sentences_and_labels/cap_0.5_0.5',
        'data/global_cap_sentences_and_labels/cap_0.05_0.5',
        'data/global_cap_sentences_and_labels/cap_0.5_0.05',
        'data/global_cap_sentences_and_labels/cap_0.05_0.05',
        'data/global_cap_sentences_and_labels/cap_0.5_0.25',
        'data/global_cap_sentences_and_labels/cap_0.05_0.25',
        'data/global_cap_sentences_and_labels/cap_0.5_0.75',
        'data/global_cap_sentences_and_labels/cap_0.05_0.75',
        'data/global_cap_sentences_and_labels/cap_0.5_1.0',
        'data/global_cap_sentences_and_labels/cap_0.05_1.0',
        'data/global_cap_sentences_and_labels/cap_0.25_0.1',
        'data/global_cap_sentences_and_labels/cap_0.25_0.01',
        'data/global_cap_sentences_and_labels/cap_0.25_0.5',
        'data/global_cap_sentences_and_labels/cap_0.25_0.05',
        'data/global_cap_sentences_and_labels/cap_0.25_0.25',
        'data/global_cap_sentences_and_labels/cap_0.25_0.75',
        'data/global_cap_sentences_and_labels/cap_0.25_1.0',
        'data/global_cap_sentences_and_labels/cap_0.75_0.1',
        'data/global_cap_sentences_and_labels/cap_0.75_0.01',
        'data/global_cap_sentences_and_labels/cap_0.75_0.5',
        'data/global_cap_sentences_and_labels/cap_0.75_0.05',
        'data/global_cap_sentences_and_labels/cap_0.75_0.25',
        'data/global_cap_sentences_and_labels/cap_0.75_0.75',
        'data/global_cap_sentences_and_labels/cap_0.75_1.0',
        'data/global_cap_sentences_and_labels/cap_1.0_0.1',
        'data/global_cap_sentences_and_labels/cap_1.0_0.01',
        'data/global_cap_sentences_and_labels/cap_1.0_0.5',
        'data/global_cap_sentences_and_labels/cap_1.0_0.05',
        'data/global_cap_sentences_and_labels/cap_1.0_0.25',
        'data/global_cap_sentences_and_labels/cap_1.0_0.75',
        'data/global_cap_sentences_and_labels/cap_1.0_1.0',
    ]
    for lang in ['swa', 'luo', 'conll_2003_en']:
        for v in all_vals:
            do_thing(f'{v}/{lang}')
        