## IMPORT TESTSETS ##

with open('wmt13_de.txt', 'r', encoding="utf-8") as f:
    ref_de = f.readlines()


## TRANSLATE ##

import pyonmttok
import ctranslate2

tokenizer = pyonmttok.Tokenizer(mode="none", sp_model_path = "deu-cat-2021-10-23/tokenizer/sp_m.model")
translator = ctranslate2.Translator("deu-cat-2021-10-23/ctranslate2/")

import re

hyp_soft = []
for count, sentence in enumerate(ref_de):
    tokenized = tokenizer.tokenize(sentence)
    translated = translator.translate_batch([tokenized[0]])
    translation = tokenizer.detokenize(translated[0][0]['tokens'])
    translation = re.sub("<unk>", "", translation)
    hyp_soft.append(translation)
    print(count)


## WRITE RESULTS ##

with open('hyp_soft_wmt_ca.txt', 'w', encoding="utf-8") as f:
    for line in hyp_soft:
        f.write(line)
        f.write("\n")


## EVALUATE ##

from datasets import load_metric
import numpy as np

metric = load_metric("sacrebleu")

def evaluate(hyp, ref):
    scores = []
    for i, j in zip(hyp, ref):
        met = metric.compute(predictions=[i], references=[[j]])
        scores.append(met["score"])
    return scores

with open('wmt13_modificat.ca', 'r', encoding="utf-8") as f:
    ref_ca = f.readlines()

scores_ca = evaluate(ref_ca, hyp_soft)

mean_scores_ca = np.mean(scores_ca)

print(f"BLEU score for German to Catalan: {round(mean_scores_ca, 3)}")