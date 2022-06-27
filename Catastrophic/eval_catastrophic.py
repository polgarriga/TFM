## OPEN ##

### OPEN FILES ###

def read_files(myfile):
    with open(myfile, 'r', encoding="utf-8") as f:
        sents = f.readlines()
        sents = [x[:-1] for x in sents]
        return sents

ref_ca = read_files('../Corpus/Flores/cat.devtest')
ref_de = read_files('../Corpus/Flores/deu.devtest')
ref_it = read_files('ita.devtest')
ref_fr = read_files('fra.devtest')

hyp_fft_ca_de = read_files('hyp_fft_ca_de.txt')
hyp_fft_de_it = read_files('hyp_fft_de_it.txt')
hyp_fft_it_ca = read_files('hyp_fft_it_ca.txt')
hyp_fft_it_fr = read_files('hyp_fft_it_fr.txt')

hyp_m2m_ca_de = read_files('../Models/M2M/hyp_m2m_flores_de.txt')
hyp_m2m_de_it = read_files('hyp_m2m_de_it.txt')
hyp_m2m_it_ca = read_files('hyp_m2m_it_ca.txt')
hyp_m2m_it_fr = read_files('hyp_m2m_it_fr.txt')


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
   print(f"The score for {name} is {round(mean_scores, 3)}")
   return scores, mean_scores

evaluate(hyp_fft_ca_de, ref_de, "fft, ca>de")
evaluate(hyp_fft_de_it, ref_it, "fft, de>it")
evaluate(hyp_fft_it_ca, ref_ca, "fft, it>ca")
evaluate(hyp_fft_it_fr, ref_fr, "fft, it>fr")

evaluate(hyp_m2m_ca_de, ref_de, "m2m, ca>de")
evaluate(hyp_m2m_de_it, ref_it, "m2m, de>it")
evaluate(hyp_m2m_it_ca, ref_ca, "m2m, it>ca")
evaluate(hyp_m2m_it_fr, ref_fr, "m2m, it>fr")