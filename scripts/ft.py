import fastText.FastText as ft
import sys, pickle
import os
import spacy, numpy


def train_vectors(fname, dim):
    model = ft.train_unsupervised(path + "filtered_output/" + fname, model='skipgram', dim=dim)
    model.save_model(path + "model_output/" + fname.split(".")[0] + "_vectors.bin")


def get_labels(fname):
    en_nlp = spacy.load("en_core_web_sm")
    result = []
    with open(fname) as f:
        while True:
            sample = []
            line1 = f.readline().strip()
            line2 = f.readline().strip()
            if not line2: break
            l1 = [l.lemma_ for l in en_nlp(line1.decode("utf-8"))]
            l2 = en_nlp(line2.decode("utf-8"))
            for word in l2:
                if word.lemma_ in l1:
                    sample.append(1.0)
                else:
                    sample.append(0.0)
            result.append(numpy.array(sample))
    return result


if __name__ == '__main__':
    fname = sys.argv[1]

    mode = "train"
    if "nt" == os.name:
        path = "E:/Martin/PyCharm Projects/Summarization/"
    else:
        path = "/home/matulma4/summarization/"

    if mode == "train":
        train_vectors("summarization_" + fname + ".txt", 256)

    else:
        labels = get_labels(path + "filtered_output/summarization_" + fname +".txt")
        pickle.dump(labels, open(path + "model_output/" + fname + "_labels.pickle", "wb"))
