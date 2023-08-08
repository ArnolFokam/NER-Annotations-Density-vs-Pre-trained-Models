# Reviewer #1

## Core Review: What is this paper about, what contributions does it make, and what are the main strengths and weaknesses?
The paper conducts a systematic analysis to assess the impact of data quality and quantity in low-resourced languages through a case study on transformer-based NER for 10 low-resourced African languages and English.
The paper shows that operating with less data is more efficient than operating with missing labels, while noisy data lead to the worst performance, even with a sufficient amount of data. The conclusion holds when experimenting with different languages and transfomer-based LMs.

## Strengths:

- The paper is useful for work on low-resourced languages, where gathering data is usually an obstacle.
- The paper run the experiments systematically across different setups.
- The evaluation is thorough

## Weaknesses:

- The conclusion is somehow intuitive.
- The paper only considers one case study, NER. This is fine given the scope, but this should be clearly stated in the title and other pivotal sections in order to avoid making too strong conclusions.
- In sentence capping setting, I believe the reduction of the datasets should be relative to where the learning saturates instead of the original sizes of the datasets. Assume a training corpus of 100K units and the learning saturates at 60K. This makes a 40% reduction have no impact on performance. However, if the saturation takes place beyond 100K unites, then a 1% reduction would impact the performance. Accordingly, the results across the different setups are not that comparable.
- Label distortion should be informed given a confusion matrix of annotation mistakes. On the other hand, random distortion creates an unrealistic setup.
- The paper is full of redundant repetition that is unnecessary and somehow disturbing.

## Reasons to accept
- The paper is useful for research on low-resourced languages,
- The paper run the experiments systematically across different setups.
- The evaluation is thorough

## Reasons to reject
- Intuitive conclusions that are only supported for NER.
- The setups in the sentence capping setting are not so comparable as different setups have different saturation points.
- Data distortion is not informed.

## Typos, Grammar, Style, and Presentation Improvements
- Incorrect footnote numbering
- Missing footnote 1 referenced in line 459.

```
Experiment Results:	      3
Overall Recommendation:	  2
Reviewer Confidence:	  4
Ethical Concerns:	      No
```

## Response

We thank the review for the remark on the thoroughness of our empirical evaluation and the constructive critism.

Indeed, as noted by the reviewer, the study's conclusion is somewhat intuitive. The primary aim of the study is to propose a systematic approach for assessing how data quality affects model performance. We show that this can be achieved through the design of parameterized corruption strategies.

Also, we acknowledge the study's narrow focus on NER. This was mainly done to prevent a dilution of experimental results due to constraints on our limited computational budget. We anticipate that this study will serve as an inspiration for other researchers to undertake similar investigations for other natural language processing tasks. We will empahsis on our focus on NER notably in the abstract and other sections of the paper where necessary.

Furthermore, we agree with the reviewer's observation regarding the sentence capping setting. As depicted in Figure 2, it is evident that an incremental enhancement in data quality through different corruption strategies doesn't always lead to a proportional increase in model performance (capping sentences vs capping labels). This nuanced relationship is a central focus of our paper, emphasizing that the performance of NER models is more influenced by the density of  correct labeled entities rather than the sample count. We will emphasize this aspect further in the paper's final version.

Regarding the remarks on label distortion, we concur that sampling based on the entity distribution in the dataset is a more suitable method for this analysis. This approach prevents the complete elimination of under-represented entities from the dataset. We will make sure to highlight the limitation of our current approach in the paper's final version.

Finally, we will correct the footnote numbering.

# Reviewer #2

## Core Review: What is this paper about, what contributions does it make, and what are the main strengths and weaknesses?
This paper concerns the impact of noisy data / labels for named entity recognition in low-resource languages.
Specifically, the paper looks at 11 languages (10 African languages and English as a control) using the MasakhaNER dataset, and CoNLL data used for English.

They compare 4 pretrained language models (AfriBERTa, Afro-XLM-R, XLM Roberta, mBERT) with various degrees of coverage and quality of the languages addressed.

The experiments concern the impact of three key ways data can be corrupted and their impact on finetuning performance on NER: (1) number of annotated sentences available, (2) number of labeled entities available, and (3) label noise, controlled by swapping labels for different proportions of sentences, motivated by the fact that low-resource languages more commonly require crowd-sourced where it may be more challenging to quality check the data or find annotators familiar with the task.

The analysis of the results concerns how corruption affects model performance, the performance across models and languages, and the trade-off between data quality and quantity.

Experiments are systematic and clear; swapping and capping labels are more impactful than the number of sentences, with at the extreme end, 25% of sentences able to be used in place of the full dataset for 99% of the resulting performance, whereas corrupting labels has a much quicker drop-off, though of course at a certain point it is more beneficial to have more imperfect annotations than fewer perfect annotations to achieve reasonable performance. This may seem obvious, but demonstrating and quantifying this seems helpful for guiding future efforts.

Performance trends are relatively consistent in terms of the impact across models and languages. This is interesting as some languages have different amounts of labelled data, and the experiments are mostly done with percentages -- it would be nice to see that controlled for by adding an experiment artificially subsampling to the same amount of data across languages. It would be also interesting to see if including more distantly related languages changes trends in any way.

The paper is well-motivated and generally well-organized and clear, and the experiments well-considered, though there is space for some meaningful additions for the final version. It could be helpful to those looking to annotate data for NER in low-resource languages and/or do limited data cleaning to improve NER performance in this setting.

## Reasons to accept
- Quantifies the impact of data / label quality for NER in low-resource languages, which may potentially be taken for granted in other settings and may help guide future annotation efforts
- Clear and well-motivated experiments

## Reasons to reject
- Relatively limited scope (one task only, NER)

## Typos, Grammar, Style, and Presentation Improvements
- It may be helpful to revise the abstract to be more specific/targeted. Right now, the abstract is not clear which tasks/settings are addressed, and whether the paper concerns pretraining vs finetuning. The first two sentences can likely be significantly shorted to give more space for this.
- For Table 3, state in the caption what the parentheses signal
- It would be nice to see mention of the languages covered here or reference that section / Table 1 earlier than page 4
- Line 250 describes type "O" as "not relevant" rather than "non-entity" : using "non-entity" might make it clearer to a non-NER audience that this is removing the label fully / not finetuning the model for example with additional information about entities just without labels
- It might be helpful to rephrase 3.2 slightly so that it is clearer that CONLL data is used for English only in order to use English as a comparison, and all experiments use the MasakhaNER dataset for the 10 African languages, and CONLL for English.
- Perhaps something like 'data quality vs quantity' rather than 'data corruption' in the title? The insights seem to largely be about the trade-off between the amount of annotated data and its quality, so it might be good to more directly reflect that
- Are models finetuned for each language individually, or multilingually (together)?

## Additional Suggestions for the Author(s)
- In addition to percentages, given that some languages have more annotations than others, it may be helpful to comment more directly on performance as it relates to raw thresholds. For example, for Luo, there are only 644 sentences vs 2.2k for Igbo. This mentioned at a high level L410-430, but, it could be nice to directly experiment with this, by artificially subsampling, to isolate amount of data vs language effects and get a better sense for the generalizability of these results and insights for how much data is minimally worth annotating if undertaking such an effort in the future.
- It is not clear how prevalent label swap errors are in NER datasets; rather than randomly swapping labels would it be possible to use more meaningful or expected swaps or label errors observed in existing datasets or annotations?
- Could it be that the small differences in finetuning between pretrained models have something to do with the noisiness of their original pretraining data for those languages?
- This paper addresses primarily label noise, whereas there may be additional noise due to the quality of the data annotated itself (particularly if the data is crawled). It may be helpful to mention this briefly as something to be addressed in the future

```
Experiment Results:	      4
Overall Recommendation:	  3.5
Reviewer Confidence:	  4
Ethical Concerns:	      No
```

## Response


We appreciate the reviewer's insightful comments and particularly value the suggestions for improvement.

Regarding the experimental setups, we finetuned the pre-trained models for each language individually. We believe this isolation captures the distinct properties of each corpus (corrupted) and enhances the credibility of the experimental analysis.

A interesting point was raised concerning the quality of the original corpora. This factor indeed has the potential to influence the strength of our claim. For this reason, we chose to depict the relative performance (relative to the original corpora) in Figure 3, rather than absolute performance.

Regarding concerns about data noise, we acknowledge this as a valid concern. We attempted to address it by utilizing the MasakhaNER dataset, which underwent manual annotation by native speakers from the same regions as the news sources from which the data was obtained. However, we recognize that this solution isn't without flaws. Consequently, we consider this a genuine limitation of our study that should be included in the final version of the paper.

Furthermore, concerning the prevalence of label swap errors in NER datasets, the direct investigation of errors in existing datasets is an excellent idea. However, this task is more complex than the one outlined in our paper, as it would necessitate collaboration with various linguistic experts to identify corpus errors. We believe this study is a good trade-off between in-depth analysis and limited experimental budget.

Once again, we value the discourse initiated by the reviewer, and the suggestions provided will certainly be considered to enhance the paper's clarity.

# Reviewer #3

## Core Review: What is this paper about, what contributions does it make, and what are the main strengths and weaknesses?
This paper deals with the impact of data corruption on Named Entity Recognition for Low-resourced Languages. The MasakhaNER dataset (Adelani et al., 2021), was used and includes ten low-resourced African languages.

## Reasons to accept
- This study deals with the analysis and quantification of the impact of data corruption on the performance of pre-trained language models. A NER task was selected as an application. An interesting study and well written paper that focuses on African languages.

## Reasons to reject
- The authors finding is obvious and has been already proved in other studies :"having fewer completely-labelled sentences is significantly better than having more sentences with missing labels; and that models can perform remarkably well with few training data ".
- The corruption strategies should be extended to include more annotation errors.

```
Experiment Results:	      3
Overall Recommendation:	  3.5
Reviewer Confidence:	  3
Ethical Concerns:	      No

```

# Response

We value the reviewer's concerns regarding the novelty of the paper's findings. The primary objective of the study was to present a fresh approach for systematically evaluating the trade-off between data quality and quantity, involving diverse implemented corruption strategies. We recognize the study's limitations concerning task selection (NER) and the design of corruption strategies. We will make it a point to highlight these limitations in the final paper and propose potential avenues for expanding the study in future research.