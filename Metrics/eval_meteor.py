import evaluate

metric = evaluate.load("meteor")

def read_files(myfile):
    with open(myfile, 'r', encoding="utf-8") as f:
        sents = f.readlines()
        sents = [x.replace('\n','') for x in sents]
        return sents

ref_flores_ca = read_files("../Corpus/Flores/cat.devtest")
ref_flores_de = read_files("../Corpus/Flores/deu.devtest")
ref_wmt_ca = read_files("../Corpus/WMT13/wmt13_ca.txt")
ref_wmt_de = read_files("../Corpus/WMT13/wmt13_de.txt")

# hyp_flores_ca_soft = read_files("../Models/Softcatala/hyp_soft_flores_ca.txt")
# hyp_flores_de_soft = read_files("../Models/Softcatala/hyp_soft_flores_de.txt")
# hyp_wmt_ca_soft = read_files("../Models/Softcatala/hyp_soft_wmt_ca.txt")
# hyp_wmt_de_soft = read_files("../Models/Softcatala/hyp_soft_wmt_de.txt")

# hyp_flores_ca_google = read_files("../Models/Google/hyp_google_flores_ca.txt")
# hyp_flores_de_google = read_files("../Models/Google/hyp_google_flores_de.txt")
# hyp_wmt_ca_google = read_files("../Models/Google/hyp_google_wmt_ca.txt")
# hyp_wmt_de_google = read_files("../Models/Google/hyp_google_wmt_de.txt")

# hyp_flores_ca_m2m = read_files("../Models/m2m/hyp_m2m_flores_ca.txt")
# hyp_flores_de_m2m = read_files("../Models/m2m/hyp_m2m_flores_de.txt")
# hyp_wmt_ca_m2m = read_files("../Models/m2m/hyp_m2m_wmt_ca.txt")
# hyp_wmt_de_m2m = read_files("../Models/m2m/hyp_m2m_wmt_de.txt")

# hyp_flores_ca_fft = read_files("../Models/Full_finetuning/hyp_fft_flores_ca.txt")
# hyp_wmt_ca_fft = read_files("../Models/Full_finetuning/hyp_fft_wmt_ca.txt")

hyp_flores_ca_para = read_files("../Models/Adapters/hyp_para_flores_ca.txt")
hyp_flores_de_para = read_files("../Models/Adapters/hyp_para_flores_de.txt")
hyp_wmt_ca_para = read_files("../Models/Adapters/hyp_para_wmt_ca.txt")
hyp_wmt_de_para = read_files("../Models/Adapters/hyp_para_wmt_de.txt")

hyp_flores_ca_seq = read_files("../Models/Sequential/hyp_seq_flores_ca.txt")
hyp_flores_de_seq = read_files("../Models/Sequential/hyp_seq_flores_de.txt")
hyp_wmt_ca_seq = read_files("../Models/Sequential/hyp_seq_wmt_ca.txt")
hyp_wmt_de_seq = read_files("../Models/Sequential/hyp_seq_wmt_de.txt")

import numpy as np

def evaluate(ref, hyp, name):
    score = []
    for i, j in zip(hyp, ref):
        results = metric.compute(predictions=[i], references=[j])['meteor']
        score.append(results)
    average = np.mean(score)
    print(f"METEOR score for {name} is {round(average, 3)}")

# evaluate(ref_flores_ca, hyp_flores_ca_soft, "Softcatalà, de->ca, FLoRes")
# evaluate(ref_flores_de, hyp_flores_de_soft, "Softcatalà, ca->de, FLoRes")
# evaluate(ref_wmt_ca, hyp_wmt_ca_soft, "Softcatalà, de->ca, wmt")
# evaluate(ref_wmt_de, hyp_wmt_de_soft, "Softcatalà, ca->de, wmt")

# evaluate(ref_flores_ca, hyp_flores_ca_google, "Google, de->ca, FLoRes")
# evaluate(ref_flores_de, hyp_flores_de_google, "Google, ca->de, FLoRes")
# evaluate(ref_wmt_ca, hyp_wmt_ca_google, "Google, de->ca, wmt")
# evaluate(ref_wmt_de, hyp_wmt_de_google, "Google, ca->de, wmt")

# evaluate(ref_flores_ca, hyp_flores_ca_m2m, "M2M100, de->ca, FLoRes")
# evaluate(ref_flores_de, hyp_flores_de_m2m, "M2M100, ca->de, FLoRes")
# evaluate(ref_wmt_ca, hyp_wmt_ca_m2m, "M2M100, de->ca, wmt")
# evaluate(ref_wmt_de, hyp_wmt_de_m2m, "M2M100, ca->de, wmt")

# evaluate(ref_flores_ca, hyp_flores_ca_fft, "FFT, de->ca, FLoRes")
# evaluate(ref_wmt_ca, hyp_wmt_ca_fft, "FFT, de->ca, wmt")

evaluate(ref_flores_ca, hyp_flores_ca_para, "para, de->ca, FLoRes")
evaluate(ref_flores_de, hyp_flores_de_para, "para, ca->de, FLoRes")
evaluate(ref_wmt_ca, hyp_wmt_ca_para, "para, de->ca, wmt")
evaluate(ref_wmt_de, hyp_wmt_de_para, "para, ca->de, wmt")

evaluate(ref_flores_ca, hyp_flores_ca_seq, "seq, de->ca, FLoRes")
evaluate(ref_flores_de, hyp_flores_de_seq, "seq, ca->de, FLoRes")
evaluate(ref_wmt_ca, hyp_wmt_ca_seq, "seq, de->ca, wmt")
evaluate(ref_wmt_de, hyp_wmt_de_seq, "seq, ca->de, wmt")
