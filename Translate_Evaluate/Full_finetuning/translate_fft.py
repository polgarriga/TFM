# -*- coding: utf-8 -*-



### OPEN FILES ###

def read_files(myfile):
    with open(myfile, 'r', encoding="utf-8") as f:
        return f.readlines()

ref_flores_ca = read_files('../../Corpus/Flores/cat.devtest')
ref_flores_de = read_files('../../Corpus/Flores/deu.devtest')
ref_wmt_ca = read_files('../../Corpus/WMT13/wmt13_ca.txt')
ref_wmt_de = read_files('../../Corpus/WMT13/wmt13_de.txt')


## TRANSLATE ##

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

checkpoint = "checkpoint-160000"

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def translate(ref_testset, ref_lang, hyp_lang):
    tokenizer.src_lang = ref_lang
    tokenizer.tgt_lang = hyp_lang
    hyp = []
    for index, i in enumerate(ref_testset):
        print(f"{index}/{len(ref_testset)}")
        encoded_ref = tokenizer(i, return_tensors="pt")
        generated_tokens = model.generate(**encoded_ref, forced_bos_token_id=tokenizer.get_lang_id(hyp_lang))
        hyp.append(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
    return hyp

hyp_flores_de = translate(ref_flores_ca, "ca", "de")
# hyp_flores_ca = translate(ref_flores_de, "de", "ca")
hyp_wmt_de = translate(ref_wmt_ca, "ca", "de")
# hyp_wmt_ca = translate(ref_wmt_de, "de", "ca")



## WRITE RESULTS ##

def write_files(mylist, myfile):
    with open(myfile, 'w', encoding="utf8") as f:
        for line in mylist:
            f.write(line)
            f.write("\n")

# write_files(hyp_flores_ca, "hyp_fft_flores_ca.txt")
write_files(hyp_flores_de, "hyp_fft_flores_de.txt")
# write_files(hyp_wmt_ca, "hyp_fft_wmt_ca.txt")
write_files(hyp_wmt_de, "hyp_fft_wmt_de.txt")