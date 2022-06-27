# -*- coding: utf-8 -*-

### OPEN FILES ###

def read_files(myfile):
    with open(myfile, 'r', encoding="utf-8") as f:
        sents = f.readlines()
        return sents

ref_ca = read_files('../Corpus/Flores/cat.devtest')
ref_de = read_files('../Corpus/Flores/deu.devtest')
ref_it = read_files('ita.devtest')
ref_fr = read_files('fra.devtest')


## TRANSLATE ##

def write_files(mylist, myfile):
    with open(myfile, 'a', encoding="utf8") as f:
        for line in mylist:
            f.write(line)
            f.write("\n")

# from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")

# def translate(ref_testset, ref_lang, hyp_lang, myfile):
#     tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang=ref_lang, tgt_lang=hyp_lang)
#     hyp = []
#     for index, i in enumerate(ref_testset):
#         print(f"{index}/{len(ref_testset)}")
#         encoded_ref = tokenizer(i, return_tensors="pt")
#         generated_tokens = model.generate(**encoded_ref, forced_bos_token_id=tokenizer.get_lang_id(hyp_lang))
#         sentence = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
#         print(sentence)
#         hyp.append(sentence)
#         if index % 100 == 99:
#             write_files(hyp, myfile)
#         if index == len(ref_testset):
#             write_files(hyp, myfile)

# translate(ref_ca, "ca", "de", "hyp_m2m_ca_de.txt")
# translate(ref_de, "de", "it", "hyp_m2m_de_it.txt")
# translate(ref_it, "it", "ca", "hyp_m2m_it_ca.txt")
# translate(ref_it, "it", "fr", "hyp_m2m_it_fr.txt")


## TRANSLATE ##

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

checkpoint = "../Models/Full_finetuning/checkpoint-160000"

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def translate(ref_testset, ref_lang, hyp_lang, myfile):
    tokenizer.src_lang = ref_lang
    tokenizer.tgt_lang = hyp_lang
    hyp = []
    for index, i in enumerate(ref_testset):
        print(f"{index}/{len(ref_testset)}")
        encoded_ref = tokenizer(i, return_tensors="pt")
        generated_tokens = model.generate(**encoded_ref, forced_bos_token_id=tokenizer.get_lang_id(hyp_lang))
        sentence = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        hyp.append(sentence)
    write_files(hyp, myfile)

# translate(ref_ca, "ca", "de", "hyp_fft_ca_de.txt")
# translate(ref_de, "de", "it", "hyp_fft_de_it.txt")
translate(ref_it[202:1000], "it", "ca", "hyp_fft_it_ca.txt")
# translate(ref_it, "it", "fr", "hyp_fft_it_fr.txt")