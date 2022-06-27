### OPEN FILES ###

def read_files(myfile):
    with open(myfile, 'r', encoding="utf-8") as f:
        return f.readlines()

ref_flores_ca = read_files('../../Corpus/Flores/cat.devtest')
ref_flores_de = read_files('../../Corpus/Flores/deu.devtest')
ref_wmt_ca = read_files('../../Corpus/WMT13/wmt13_ca.txt')
ref_wmt_de = read_files('../../Corpus/WMT13/wmt13_de.txt')


## TRANSLATE ##

import pyonmttok
import ctranslate2
import re

def translate(ref, ref_code):
    if ref_code = "ca":
        checkpoint_1 = "cat-deu-2021-10-27/tokenizer/sp_m.model"
        checkpoint_2 = "cat-deu-2021-10-27/ctranslate2/"
    if ref_code = "de":
        checkpoint_1 = "deu-cat-2021-10-23/tokenizer/sp_m.model"
        checkpoint_2 = "deu-cat-2021-10-23/ctranslate2/" 

    tokenizer = pyonmttok.Tokenizer(mode="none", sp_model_path = checkpoint_1)
    translator = ctranslate2.Translator(checkpoint_2)

    hyp = []
    for count, sentence in enumerate(ref):
        tokenized = tokenizer.tokenize(sentence)
        translated = translator.translate_batch([tokenized[0]])
        translation = tokenizer.detokenize(translated[0][0]['tokens'])
        translation = re.sub("<unk>", "", translation)
        hyp.append(translation)
        print(count)
    return hyp

hyp_flores_de = translate(ref_flores_ca, "ca")
hyp_flores_ca = translate(ref_flores_de, "de")
hyp_wmt_de = translate(ref_wmt_ca, "ca")
hyp_wmt_ca = translate(ref_wmt_de, "de")


## WRITE RESULTS ##

def write_files(mylist, myfile):
    with open(myfile, 'w', encoding="utf8") as f:
        for line in mylist:
            f.write(line)
            f.write("\n")

write_files(hyp_flores_ca, "hyp_soft_flores_ca.txt")
write_files(hyp_flores_de, "hyp_soft_flores_de.txt")
write_files(hyp_wmt_ca, "hyp_soft_wmt_ca.txt")
write_files(hyp_wmt_de, "hyp_soft_wmt_de.txt")