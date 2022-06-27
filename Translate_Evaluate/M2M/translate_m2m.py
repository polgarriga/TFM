# -*- coding: utf-8 -*-



### OPEN FILES ###

def read_files(myfile):
    with open(myfile, 'r', encoding="utf-8") as f:
        return f.readlines()

ref_dev_ca = read_files('../../Corpus/Flores/cat.dev')
ref_dev_de = read_files('../../Corpus/Flores/deu.dev')
ref_flores_ca = read_files('../../Corpus/Flores/cat.devtest')
ref_flores_de = read_files('../../Corpus/Flores/deu.devtest')
ref_wmt_ca = read_files('../../Corpus/WMT13/wmt13_ca.txt')
ref_wmt_de = read_files('../../Corpus/WMT13/wmt13_de.txt')



### TRANSLATE ###

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer_ca_de = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang="ca", tgt_lang="de")
tokenizer_de_ca = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang="de", tgt_lang="ca")

def translate(ref, ref_code, hyp_code):
    if ref_code == "ca":
        tok = tokenizer_ca_de
    else:
        tok = tokenizer_de_ca
    hyp = []
    for i in ref:
        encoded = tok(i, return_tensors="pt")
        generated_tokens = model.generate(**encoded, forced_bos_token_id=tok.get_lang_id("ca"))
        hyp.append(tok.batch_decode(generated_tokens, skip_special_tokens=True))
    return hyp

hyp_dev_de = translate(ref_dev_ca, "ca", "de")
hyp_dev_ca = translate(ref_dev_de, "de", "ca")
hyp_flores_de = translate(ref_flores_ca, "ca", "de")
hyp_flores_ca = translate(ref_flores_de, "de", "ca")
hyp_wmt_de = translate(ref_wmt_ca, "ca", "de")
hyp_wmt_ca = translate(ref_wmt_de, "de", "ca")


### WRITE RESULTS TO FILE ###
## Be careful ##

def write_files(mylist, myfile):
    with open(myfile, 'w', encoding="utf8") as f:
        for line in mylist:
            f.write(line)
            f.write("\n")

write_files(hyp_dev_ca, "hyp_m2m_dev_ca.txt")
write_files(hyp_dev_de, "hyp_m2m_dev_de.txt")
write_files(hyp_flores_ca, "hyp_m2m_flores_ca.txt")
write_files(hyp_flores_de, "hyp_m2m_flores_de.txt")
write_files(hyp_wmt_ca, "hyp_m2m_wmt_ca.txt")
write_files(hyp_wmt_de, "hyp_m2m_wmt_de.txt")