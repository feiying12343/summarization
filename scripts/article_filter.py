import json
import os
from HTMLParser import HTMLParser

import nltk
import spacy
import sys


def load_data(fname, head_key, body_key):
    parser = HTMLParser()

    with open(fname) as f:
        # a = [line for line in f]
        try:
            l = [json.loads(line) for line in f]
        except IndexError:
            return []
        except ValueError:
            return []
        result = []
        for article in l:
            if body_key in article.keys() and head_key in article.keys():
                n = nltk.sent_tokenize(article[body_key])
                if len(n) == 0:
                    continue
                if len(n[0].split(" ")) > 100:
                    continue
                try:
                    result += [(parser.unescape(article[head_key] + ".").encode("utf-8"))] + [
                        parser.unescape(sent).encode("utf-8") for sent in n[:1]]
                except UnicodeEncodeError:
                    continue
        return result


def compare(h, s):
    hlen = len(h)
    slen = len(s)
    hdct = dict([(k, []) for k in d])
    sdct = dict([(k, []) for k in d])
    for hword in h:
        if hword.tag_[0] in d:
            hdct[hword.tag_[0]].append(hword.lemma_.lower())
    if len(hdct["V"]) == 0:
        return False
    for sword in s:
        if sword.tag_[0] in d:
            sdct[sword.tag_[0]].append(sword.lemma_.lower())
    counter = 0
    for k in d:
        hh = hdct[k]
        ss = sdct[k]
        for w in hh:
            if w not in ss:
                counter += 1
    if float(counter) / float(hlen) >= 0.3:
        return False
    if hlen < 5 or slen < 5:
        return False
    if float(slen) / float(hlen) < 1.5:
        return False
    if h[1].tag_ is None or h[1] is None:
        return False
    if h[1].tag_[0] == "V" or h[1].tag_[0] == "W":
        return False
    return True


if __name__ == '__main__':
    d = "NJVR"
    mode = sys.argv[1]

    en_nlp = spacy.load("en_core_web_sm")
    if "nt" == os.name:
        path = "E:/Martin/PyCharm Projects/Summarization/"
    else:
        path = "/home/matulma4/summarization/"

    if "gossip" != mode:
        hkey = "headline"
        bkey = "body"
    else:
        hkey = "title"
        bkey = "content"
    g = open(path + "filtered_output/summarization_" + mode + ".txt", "a")
    for fname in os.listdir(path + "data/" + mode):
        if "json" not in fname:
            continue
        print("Now processing " + fname)
        docs = load_data(path + "data/" + mode + "/" + fname, hkey, bkey)
        if mode != "compression":
            for i in range(0, len(docs), 2):
                try:
                    h = en_nlp(docs[i].decode("utf-8"))
                    s = en_nlp(docs[i + 1].decode("utf-8"))
                    if compare(h, s):
                        g.write("GO " + h.doc.text.encode("utf-8") + "\n")
                        g.write("GO " + s.doc.text.encode("utf-8") + "\n")
                except IndexError:
                    continue
        else:
            for i in range(0, len(docs)):
                g.write("GO " + docs[i] + "\n")
    g.close()
    print("Done.")
