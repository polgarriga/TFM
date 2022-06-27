# -*- coding: utf-8 -*-


### OPEN FILES ###

def read_files(myfile):
    with open(myfile, 'r', encoding="utf-8") as f:
        return f.readlines()

ref_flores_ca = read_files('../../Corpus/Flores/cat.devtest')
ref_flores_de = read_files('../../Corpus/Flores/deu.devtest')
ref_wmt_ca = read_files('../../Corpus/WMT13/wmt13_ca.txt')
ref_wmt_de = read_files('../../Corpus/WMT13/wmt13_de.txt')

hyp_flores_ca = read_files("hyp_google_flores_ca.txt")
hyp_flores_de = read_files("hyp_google_flores_de.txt")
hyp_wmt_ca = read_files("hyp_google_wmt_ca.txt")
hyp_wmt_de = read_files("hyp_google_wmt_de.txt")



### EVALUATION ###

from datasets import load_metric
import numpy as np

metric = load_metric("sacrebleu")

def evaluate(hyp, ref, name):
   scores = []
    for i, j in zip(hyp, ref):
        met = metric.compute(predictions=[i], references=[[j]])
        scores.append(met["score"])
    mean_scores = np.mean(scores)
    print(f"The score for {name} is {round(mean_scores, 3)})
    return scores, mean_scores

evaluate(hyp_flores_ca, ref_flores_ca, "google, flores, de>ca")
evaluate(hyp_flores_de, ref_flores_de, "google, flores, ca>de")
evaluate(hyp_wmt_ca, ref_wmt_ca, "google, wmt, de>ca")
evaluate(hyp_wmt_de, ref_wmt_de, "google, wmt, ca>de")