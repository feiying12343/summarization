import os
import pickle

import fastText.FastText as ft
import numpy as np
import spacy


def get_samples(fname, parent):
    maxlength = 120
    en_nlp = spacy.load("en_core_web_sm")
    y = []
    X = []
    ft_model = ft.load_model(path + "model_output/summarization_" + fname + "_vectors.bin")
    with open(path + "filtered_output/summarization_" + fname + ".txt") as f:
        while True:
            sample_label = []
            sample = []

            line1 = f.readline().strip()
            line2 = f.readline().strip()
            if not line2: break

            l1 = [l.lemma_ for l in en_nlp(line1.decode("utf-8"))]
            l2 = en_nlp(line2.decode("utf-8"))

            for word in l2:
                features = ft_model.get_word_vector(word.text)
                if parent:
                    features = np.concatenate(features, ft_model.get_word_vector(word.head.text))
                sample.append(features)
                if "GO" == word.text:
                    sample_label.append(np.array([0.0, 0.0, 1.0]))
                elif word.lemma_ in l1:
                    sample_label.append(np.array([0.0, 1.0, 0.0]))
                else:
                    sample_label.append(np.array([1.0, 0.0, 0.0]))

            if len(sample_label) > maxlength:
                continue
            elif len(sample_label) < maxlength:
                sample_label = np.concatenate((np.array(sample_label), np.concatenate(
                    (np.ones((maxlength - len(sample_label), 1)), np.zeros((maxlength - len(sample_label), 2))), axis=1)))

            padding = maxlength - len(sample)
            X.append(np.concatenate((np.zeros((padding, sample[0].shape[0])), sample[::-1], sample[1:],
                                     np.zeros((padding, sample[0].shape[0])))))
            y.append(np.array(sample_label))
    return np.array(X), np.array(y)


if __name__ == '__main__':
    fname = "gossip"
    parents = False

    if "nt" == os.name:
        path = "E:/Martin/PyCharm Projects/Summarization/"
    else:
        path = "/home/matulma4/summarization/"

    X, y = get_samples(fname, parents)
    pickle.dump((X, y), open(path + "model_output/samples/" + fname + "_samples.pickle", "wb"))
