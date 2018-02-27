import csv
import json
import os
import sys

fsize = 400


def transform_kaggle(d, out_d):
    csv.field_size_limit(sys.maxint)
    i = 0
    j = 0
    g = open(out_d + "kaggle_" + str(i) + ".json", "w")
    for fname in os.listdir(d):
        with open(d + fname, "rb") as f:
            c = csv.reader(f)
            for line in c:
                g.write(json.dumps({"headline": line[2].decode("utf-8"), "body": line[9].decode("utf-8")}) + "\n")
                j += 1
                if j == fsize:
                    g.close()
                    i += 1
                    g = open(out_d + "kaggle_" + str(i) + ".json", "w")
                    j = 0
    g.close()


def transform_compression(d, out_d):
    with open(d + "compression-data.json") as f:
        content = f.read().replace("\n", "").split("}{")
        data = [json.loads(content[0] + "}")] + [json.loads("{" + c + "}") for c in content[1:-1]] + [
            json.loads("{" + content[-1])]
        i = 0
        j = 0
        g = open(out_d + "compression_" + str(i) + ".json", "w")
        for d in data:
            g.write(json.dumps({"headline": d["headline"], "body": d['compression']["text"]}) + "\n")
            j += 1
            if j == fsize:
                g.close()
                i += 1
                g = open(out_d + "compression_" + str(i) + ".json", "w")
                j = 0

if __name__ == '__main__':

    mode = 1

    if "nt" == os.name:
        workdir = "E:/Martin/PyCharm Projects/Summarization"
    else:
        workdir = "/home/matulma4/summarization"
    if mode == 0:
        transform_kaggle(workdir + "/input/kaggle/", workdir + "/data/kaggle/")
    else:
        transform_compression(workdir + "/input/compression/", workdir + "/data/compression/")
