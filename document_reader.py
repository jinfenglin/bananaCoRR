# /usr/bin/python
# -*- coding: utf-8 -*-

"""
This program is to:

"""
import collections
import os
import pprint
import re
import sys

reload(sys)
sys.setdefaultencoding('utf8')

__author__ = 'krim'
__date__ = '3/5/15'
__email__ = 'krim@brandeis.edu'


import main

class reader(object):
    """
    reader is the main class for reading document files in project dataset
    """
    def __init__(self, filename):
        super(reader, self).__init__()
        self.filename = filename
        self.sentences = []
        self.stanford_dependency = self.load_dep_parse
        self.process_file(filename)

    def process_file(self, filename):
        with open(os.path.join(main.POS_DATA_PATH, filename + ".raw.pos")) as document:
            for line in document:
                if line != "\n":
                    self.sentences.append(self.tokenize(line))

    def get_all_sents(self):
        sents = []
        for sent in self.sentences:
            sents.append([w for w, _ in sent])
        return sents

    def write_raw_sents(self):
        with open(os.path.join(main.RAW_DATA_PATH, self.filename), "w") as outf:
            for sent in self.get_all_sents():
                outf.write(" ".join(sent))
                outf.write("\n")

    @property
    def load_dep_parse(self):
        sents = []
        with open(os.path.join(main.DEPPARSE_DATA_PATH,
                               self.filename + ".raw.depparse")) as parse:
            sent = {}
            for line in parse:
                if line == "\n":
                    sents.append(sent)
                    print len(sent)
                    sent = {}
                else:
                    m = re.match(r"^(.+)\((.+)-([0-9]+), (.+)-([0-9]+)\)", line)
                    rel = m.groups()[0]
                    gov = m.groups()[1]
                    gov_idx = int(m.groups()[2]) - 1
                    dep = m.groups()[3]
                    dep_idx = int(m.groups()[4]) - 1

                    try:
                        sent[gov_idx][1][rel].append((dep_idx, dep))
                    except KeyError:
                        sent[gov_idx] = (gov,
                                         collections.defaultdict(list),
                                         collections.defaultdict(list))
                        sent[gov_idx][1][rel].append((dep_idx, dep))

                    try:
                        sent[dep_idx][2][rel].append((gov_idx, gov))
                    except KeyError:
                        sent[dep_idx] = (dep,
                                         collections.defaultdict(list),
                                         collections.defaultdict(list))
                        sent[dep_idx][2][rel].append((gov_idx, gov))
        return sents

    @staticmethod
    def tokenize(line):
        """returns [(word, pos)]"""
        tokens = []
        for token in line.split():
            token = token.split("_")
            if len(token) > 2:
                token = ["".join(token[:-1]), token[-1]]
            tokens.append(token)
        # return [tuple(token.split("_")) for token in line.split()]
        return tokens

    def get_tokens(self, sent, start, end):
        if not isinstance(sent, int):
            sent = int(sent)
        if not isinstance(start, int):
            start = int(start)
        if not isinstance(end, int):
            end = int(end)
        return self.sentences[sent][start:end]

    def get_words(self, sent, start, end):
        return [w for w, _ in self.get_tokens(sent, start, end)]

    def get_pos(self, sent, start, end):
        return [p for _, p in self.get_tokens(sent, start, end)]

if __name__ == '__main__':
    r = reader("APW20001001.2021.0521.head.coref")
    pprint.pprint(r.stanford_dependency)

