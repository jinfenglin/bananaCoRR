# /usr/bin/python
# -*- coding: utf-8 -*-

"""
This program is to:

"""
import os
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
        self.sentences = []
        # TODO implement this
        self.stanford_dependency = []
        self.process_file(filename)

    def process_file(self, filename):
        with open(os.path.join(main.RAW_DATA_PATH, filename + ".raw.pos")) as document:
            for line in document:
                if line != "\n":
                    self.sentences.append(self.tokenize(line))

    @staticmethod
    def tokenize(line):
        """returns [(word, pos)]"""
        return [tuple(token.split("_")) for token in line.split()]

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

