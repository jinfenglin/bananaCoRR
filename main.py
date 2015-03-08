# /usr/bin/python
# -*- coding: utf-8 -*-

"""
This program is to:
Use various feature functions to train a MALLET MaxEnt model to classify whether
entities in a text are coreferent or not.
"""
import collections
import re
import subprocess
import sys
import gc
import cPickle as pickle
from nltk.stem.lancaster import LancasterStemmer

reload(sys)
sys.setdefaultencoding('utf8')

__author__ = ["Keigh Rim", "Todd Curcuru", "Yalin Liu"]
__date__ = "3/1/2015"
__email__ = ['krim@brandeis.edu', 'tcurcuru@brandeis.edu', 'yalin@brandeis.edu']

import os
import document_reader

PROJECT_PATH = os.getcwd()
DATA_PATH = os.path.join(PROJECT_PATH, "data")
POS_DATA_PATH = os.path.join(DATA_PATH, "postagged")
RAW_DATA_PATH = os.path.join(DATA_PATH, "rawtext")
DEPPARSE_DATA_PATH = os.path.join(DATA_PATH, "depparsed")

RES_PATH = os.path.join(PROJECT_PATH, "resources")
DICT_PATH = os.path.join(RES_PATH, "dicts")
FREQ_PATH = os.path.join(RES_PATH, "freqCounts")
CLUSTER_PATH = os.path.join(RES_PATH, "clusters")


class FeatureTagger():
    """FeatureTagger is a framework for tagging tokens from data file"""

    def __init__(self):
        # pairs is a list that looks like
        # [0:(words_i, poss_i, ner_i),
        #  1:(words_j, poss_j, ner_j),
        #  additional info, ...]
        """
        additional info
        2: corefer - bool (None for test data)
        3: in_same_sent - bool
        """
        self.pairs = None

        # dicts will store name dictionaries
        self.dicts = {}
        self.org_suffixes = []

        # all feature_functions should
        # 1. take no parameters
        # (use self.pairs)
        # 2. return a list or an iterable which has len of # number of tokens
        self.feature_functions = [  self.i_pronoun,
                                    self.j_pronoun,
                                    self.only_i_pronoun,
                                    self.only_j_pronoun,
                                    self.string_match_no_articles,
                                    self.string_contains_no_articles,
                                    self.str_stem_match,
                                    self.pro_str_match,
                                    self.pn_str_match,
                                    self.pn_str_contains,
                                    self.words_str_match
                                 ]

    def read_data(self, input_filename):
        """load sentences from data file"""
        self.pairs = []
        cur_filename = None
        with open(os.path.join(DATA_PATH, input_filename)) as in_file:
            for line in in_file:
                filename, i_line, i_start, i_end, i_ner, i_word, j_line, j_start,\
                j_end, j_ner, j_word, coref = line.split(" ")

                # split underscored words
                i_words = i_word.split("_")
                j_words = j_word.split("_")

                if cur_filename != filename:
                    r = document_reader.reader(filename)
                    cur_filename = filename
                i_pos = r.get_pos(i_line, i_start, i_end)
                j_pos = r.get_pos(j_line, j_start, j_end)
                pair = [
                    # info on i
                    (i_words, i_pos, i_ner),
                    # info on j
                    (j_words, j_pos, j_ner),
                    # additional info
                    coref.strip(), i_line == j_line]
                try:
                    assert j_words == r.get_words(j_line, j_start, j_end)
                except AssertionError:
                    print "mismatch of I at {} {} {}-{}, raw: {}, input: {}".format(
                        filename, j_line, j_start, j_end,
                        r.get_words(j_line, j_start, j_end), j_words)
                try:
                    assert i_words == r.get_words(i_line, i_start, i_end)
                except AssertionError:
                    print "mismatch of I at {} {} {}-{}, raw: {}, input: {}".format(
                        filename, i_line, i_start, i_end,
                        r.get_words(i_line, i_start, i_end), i_words)
                self.pairs.append(pair)

        print self.pairs[0]
        # self.populate_freq(300)
        # self.populate_dict()

    def is_coref(self):
        """return gold standard labels for each pairs"""
        coref = []
        return [p[2] for p in self.pairs]
        # for p in self.pairs:
        #     if p[2] :
        #         coref.append('yes')
        #     else:string_match_no_articles=false	string_contains_no_articles=true
        #         coref.append('no')
        # return coref

    def get_i_words(self):
        """Return list of i words"""
        return [p[0][0] for p in self.pairs]

    def get_j_words(self):
        """Return list of j words"""
        return [p[1][0] for p in self.pairs]

    def get_i_poss(self):
        """Return list of pos tags of i words"""
        return [p[0][1] for p in self.pairs]

    def get_j_poss(self):
        """Return list of pos tags of j words"""
        return [p[1][0] for p in self.pairs]
        # poss = []
        # for p in self.pairs:
        #     poss.extend(p[1][1])
        # return poss

    def get_i_ners(self):
        """Return list of ner tag of i words"""
        return [p[0][2] for p in self.pairs]

    def get_j_ners(self):
        """Return list of ner tag of j words"""
        return [p[1][2] for p in self.pairs]

    def get_i_j_words(self):
        return zip(self.get_i_words(), self.get_j_words())

    def feature_matrix(self, out_filename, train=True):
        """use this method to get all feature values and printed out as a file"""
        with open(out_filename, "w") as outf:
            features = self.get_features(train)
            for tok_index in range(len(features)):
                outf.write("\t".join(features[tok_index]) + "\n")
                # try:
                #     if features[tok_index+1][0] == "0":
                #         outf.write("\n")
                # except KeyError:
                #     pass

    def get_features(self, train=True):
        """traverse function list and get all values in a dictionary"""
        features = collections.defaultdict(list)
        # unigram is the only default feature
        # for i, (w_index, word) in enumerate(self.tokens()):
        #     features[i] = [w_index, word]

        # add gold bio tags while training
        if train:
            self.feature_functions.insert(0, self.is_coref)
        # traverse functions
        # note that all function should take no parameter and return an iterable
        # which has length of the number of total tokens
        for fn in self.feature_functions:
            for num, feature in enumerate(fn()):
                features[num].append(feature)

        # remove gold tags when it's done
        if train:
            self.feature_functions.remove(self.is_coref)
        return features

    def populate_dict(self):
        """Populate dictionaries using external files"""
        for filename in filter(lambda x: x.endswith(".dict"), os.listdir(DICT_PATH)):
            dict_type = filename.split(".")[0]
            self.dicts[dict_type] = []
            with open(os.path.join(DICT_PATH, filename)) as d:
                for line in d:
                    if line != "\n":
                        self.dicts[dict_type].append(line.strip().lower())
            # now load up useful suffix dict
            if dict_type == "org":  # only organization names have useful suffixes
                with open(os.path.join(DICT_PATH, dict_type + ".suff")) as suff:
                    for line in suff:
                        if line != "\n":
                            self.org_suffixes.append(line.strip().lower())

    def in_dict(self, typ):
        """See each token is in a certain dictionary"""
        tag = []
        s = "In_" + typ
        d = self.dicts[typ]
        for num, sent in enumerate(self.pairs):
            tokens = [w for w, _, _ in sent]
            i = 0
            if typ == "person":
                for token in tokens:
                    if token.lower() in d:
                        tag.append(s)
                    else:
                        tag.append("-" + s)
                pass
            else:
                while i < len(tokens):
                    for j in range(i + 1, len(tokens)):
                        if " ".join(map(lambda x: x.lower(), tokens[i:j])) in d:
                            tag.extend([s] * (j - i))
                            i = j
                            break
                    tag.append("-" + s)
                    i += 1
        return tag

    """""""""""""""""
    feature functions
    """""""""""""""""
    from nltk.stem.lancaster import LancasterStemmer
    st = LancasterStemmer()

    def i_pronoun(self):
        """Is the first entity a pronoun"""
        name, t, f = "i_pronoun=", "true", "false"
        values = []
        for poss in self.get_i_poss():
            for pos in poss:
                if pos.startswith("PRP"):
                    values.append(name + t)
                else:
                    values.append(name + f)
        return values

    def j_pronoun(self):
        """Is the second entity a pronoun"""
        name, t, f = "j_pronoun=", "true", "false"
        values = []
        for poss in self.get_j_poss():
            for pos in poss:
                if pos.startswith("PRP"):
                    values.append(name + t)
                else:
                    values.append(name + f)
        return values

    def only_j_pronoun(self):
        """Checks if only the second entity is a pronoun, and not the first"""
        name, t, f = "only_j_pronoun=", "true", "false"
        values = []
        for bools in zip(self.i_pronoun(), self.j_pronoun()):
            if bools[0].endswith("false") and bools[1].endswith("true"):
                values.append(name + t)
            else:
                values.append(name + f)
        return values
        
    def only_i_pronoun(self):
        """Checks if only the first entity is a pronoun, and not the second"""
        name, t, f = "only_i_pronoun=", "true", "false"
        values = []
        for bools in zip(self.i_pronoun(), self.j_pronoun()):
            if bools[1].endswith("false") and bools[0].endswith("true"):
                values.append(name + t)
            else:
                values.append(name + f)
        return values

    def remove_articles(self, words, tags):
        """Removes any articles from a list of words and a list of their tags"""
        return_string = ""
        for i in range(len(words)):
            if tags[i] != "DT":
                return_string += words[i]
        return return_string
        
    def string_match_no_articles(self):
        """Checks to see if two entities match exactly, without articles"""
        name, t, f = "string_match_no_articles=", "true", "false"
        values = []
        i_words = self.get_i_words()
        j_words = self.get_j_words()
        i_tags = self.get_i_poss()
        j_tags = self.get_j_poss()
        for i in range(len(i_words)):
            comparator_i = self.remove_articles(i_words[i], i_tags[i])
            comparator_j = self.remove_articles(j_words[i], j_tags[i])
            if comparator_i == comparator_j:
                values.append(name + t)
            else:
                values.append(name + f)
        return values
        
    def string_contains_no_articles(self):
        """Checks if one entities is contained in another, without articles"""
        name, t, f = "string_contains_no_articles=", "true", "false"
        values = []
        i_words = self.get_i_words()
        j_words = self.get_j_words()
        i_tags = self.get_i_poss()
        j_tags = self.get_j_poss()
        for i in range(len(i_words)):
            comparator_i = self.remove_articles(i_words[i], i_tags[i])
            comparator_j = self.remove_articles(j_words[i], j_tags[i])
            if comparator_i in comparator_j or \
               comparator_j in comparator_i:
                values.append(name + t)
            else:
                values.append(name + f)
        return values

    def score_similarity(self, set1, set2, numeric=False):
        """A simple similiarity metric between two sets"""
        total1 = len(set1)
        total2 = len(set2)
        total = total1 + total2
        similarity_count = 0
        for item in set1:
            if item in set2:
                similarity_count += 1
        if numeric:
            return similarity_count*2/total
        else:
            if similarity_count >= total/2:
                return "most"
            elif similarity_count >= total/2:
                return "some"
            elif similarity_count > 0:
                return "few"
            else:
                return "none"      

    def yago_ontology(self):
        """Uses the yago ontology to calculate the similarity between entities"""
        name = "yago_ontology="
        values = []
        i_words = self.get_i_words()
        j_words = self.get_j_words()
        
        for i in range(len(i_words)):
            search1 = i_words[i].join("_").lower()
            search2 = j_words[i].join("_").lower()
            result1 = None
            result2 = None
            found1 = False
            found2 = False
            i = 1

            while not (found1 and found2):
                if i > 8:
                    break
                entity_dict = pickle.load(open("entity_dict" + str(i) + ".p", "rb"))
                if search1 in entity_dict:
                    result1 = entity_dict[search1]
                    found1 = True
                if search2 in entity_dict:
                    result2 = entity_dict[search2]
                    found2 = True
                entity_dict = {}
                gc.collect()
                i += 1
                
            if result1 == None or result2 == None:
                values.append(name + "no data")
            else:
                values.append(name + str(self.score_similarity(result1, result2)))
                
        return values
        
    def str_stem_match(self):
        """Stem first, and then check string match"""
        name, t, f = "str_stem_match=", "true", "false"
        values = []
        st = LancasterStemmer()
        i_words = self.get_i_words()
        j_words = self.get_j_words()
        i_tags = self.get_i_poss()
        j_tags = self.get_j_poss()
        for i in range(len(i_words)):
            comparator_i = st.stem(self.remove_articles(i_words[i], i_tags[i]))
            comparator_j = st.stem(self.remove_articles(j_words[i], j_tags[i]))
            if comparator_i == comparator_j:
                values.append(name + t)
            else:
                values.append(name + f)
        return values
    
    def pro_str_match(self):
        """Check if both entities are pronouns and they both match"""
        name, t, f = "pro_str_match=", "true", "false"
        values = []
        i_words = self.get_i_words()
        j_words = self.get_j_words()
        pro_bools = zip(self.i_pronoun(), self.j_pronoun())
        for i in range(len(i_words)):
            if pro_bools[i][0].endswith("true")  \
            and pro_bools[i][1].endswith("true") \
            and i_words[i] == j_words[i]:
                values.append(name + t)
            else:
                values.append(name + f)
        return values
        
    def pn_str_match(self):
        """Check if both entities are proper nouns and they both match"""
        name, t, f = "pn_str_match=", "true", "false"
        values = []
        i_words = self.get_i_words()
        j_words = self.get_j_words()
        i_tags = self.get_i_poss()
        j_tags = self.get_j_poss()
        for i in range(len(i_words)):
            i_nnps = [tag for tag in i_tags[i] if tag.startswith("NNP")]
            j_nnps = [tag for tag in j_tags[i] if tag.startswith("NNP")]
            if len(i_nnps) > 0 and len(j_nnps) > 0 \
            and i_words[i].join() == j_words[i].join():
                values.append(name + t)
            else:
                values.append(name + f)
        return values
        
    def pn_str_contains(self):
        """Check if both entities are proper nouns and one contains the other"""
        name, t, f = "pn_str_contains=", "true", "false"
        values = []
        i_words = self.get_i_words()
        j_words = self.get_j_words()
        i_tags = self.get_i_poss()
        j_tags = self.get_j_poss()
        for i in range(len(i_words)):
            i_nnps = [tag for tag in i_tags[i] if tag.startswith("NNP")]
            j_nnps = [tag for tag in j_tags[i] if tag.startswith("NNP")]
            i_string = i_words[i].join()
            j_string = j_words[i].join()
            if len(i_nnps) > 0 and len(j_nnps) > 0 \
            and (i_string.contains(j_string) or j_string.contains(i_string)):
                values.append(name + t)
            else:
                values.append(name + f)
        return values
    
    def words_str_match(self):
        """Check if both entities are not pronouns and they both match"""
        name, t, f = "words_str_match=", "true", "false"
        values = []
        i_words = self.get_i_words()
        j_words = self.get_j_words()
        pro_bools = zip(self.i_pronoun(), self.j_pronoun())
        for i in range(len(i_words)):
            if pro_bools[i][0].endswith("false")  \
            and pro_bools[i][1].endswith("false") \
            and i_words[i].join() == j_words[i].join():
                values.append(name + t)
            else:
                values.append(name + f)
        return values
    
'''
    # TODO which one?
    def PN_STR(tuple1, tuple2):
        if get_pos(tuple1) != 'PRP*' or get_pos(tuple2) != 'PRP*':
            return False
        return str_match(get_word(tuple1), get_word(tuple2))


    def both_are(pron1, pron2, str):
        if pron1 == str and pron2 == str:
            return True
        else:
            return False


    def word_overlap(tuple1, tuple2):
        s1 = set(get_word(tuple1).split())
        s2 = set(get_word(tuple2).split())
        if set(s1).intersection(s2):
            return True
        else:
            return False


    def pn_substr(tuple1, tuple2):
        if both_are(get_pos(tuple1), get_pos(tuple2), 'PPN*'):
            if get_word(tuple1) in get_word(tuple2) or get_word(tuple2) in get_word(
                    tuple1):
                return True
        else:
            return False


    def j_definite(tuple1, tuple2):
        pass


    def j_indefinite(tuple1, tuple2):
        pass


    def j_demonstrative(tuple1, tuple2):
        pass


    def num_agr(tuple1, tuple2):
        pass


    def gen_agr(tuple1, tuple2):
        pass


    def both_proper(tuple1, tuple2):
        pass


    def both_diff_proper(tuple1, tuple2):
        pass


    def alias_date(tuple1, tuple2):
        pass


    def alias_person(tuple1, tuple2):
        pass


    def alias_org(tuple1, tuple2):
        pass


    def words_substr(tuple1, tuple2):
        pass


    def both_pronouns(tuple1, tuple2):
        pass


    def gen_num_agr(tuple1, tuple2):
        pass


    def span(tuple1, tuple2):
        pass


    def contains_pn(tuple1, tuple2):
        pass


    class feature_functions:
        def __init__(self, list):
            self.tuple_1 = list[0]
            self.tuple_2 = list[1]
            self.feature_name_list = list[2:]
            for feature_fn in self.feature_name_list:
                feature_fn(self.tuple_1, self.tuple_2)


if __name__ == '__main__':
    ff = feature_functions(
        [('foo', 'vb', 'foo'), ('bar', 'np', 'bar'), i_pronoun, j_pronoun])

'''


class CoreferenceResolver(object):
    """
    NER class is a classifier to detect named entities
    using TaggerFrame as feature extractor
    and CRF++ as classification algorithm
    """

    def __init__(self):
        super(CoreferenceResolver, self).__init__()
        os.chdir(PROJECT_PATH)
        self.ft = FeatureTagger()
        self.trainfile \
            = os.path.join('result', 'trainFeatureVector.txt')
        self.targetfile \
            = os.path.join('result', 'targetFeatureVector.txt')
        self.windows = sys.platform.startswith("win")
        # TODO make a way for windows system
        self.me_script \
            = os.path.join(".", "mallet-maxent-classifier.sh")
        self.modelfile = os.path.join("result", "model")

    def train(self, train_filename):
        """train crf++ module with a given data file"""
        self.ft.read_data(train_filename)
        self.ft.feature_matrix(self.trainfile)
        subprocess.check_call(
            [self.me_script, "-train",
             "-model=" + self.modelfile,
             "-gold=" + self.trainfile])

    def classify(self, target_filename):
        """Run crfpp classifier to classify target file"""
        self.ft.read_data(target_filename)
        self.ft.feature_matrix(self.targetfile, train=False)
        if not os.path.isfile(self.modelfile):
            raise Exception("Model not found.")
        resultfile = os.path.join("result", "result.txt")

        crfppc = subprocess.Popen(
            [self.me_script, "-classify",
             "-model=" + self.modelfile,
             "-input=" + self.targetfile],
            stdout=subprocess.PIPE)
        with open(resultfile, "w") as outf:
            for line in crfppc.communicate()[0]:
                outf.write(line)

        # evaluate the result
        target_name = target_filename.split("/")[-1].split(".")[0]
        subprocess.check_call(
            ["python", "coref-evaluator.py",
             "data/%s.gold" % target_name, resultfile])

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        help="name of train set file",
        default=os.path.join(PROJECT_PATH, 'data', 'coref-trainset.gold')
    )
    parser.add_argument(
        "-t",
        help=
        "name of target file, if not given, program will ask users after training",
        default=None
    )
    args = parser.parse_args()

    ft = FeatureTagger()
    # ft.read_data(args.i)
    # '''
    cor = CoreferenceResolver()
    cor.train(args.i)
    if args.t is None:
        try:
            target = input(
                "enter a test file name with its path\n"
                + "(relative or full, default: data/coref-testset.notag): ")
        # if imput is empty
        except SyntaxError:
            target = "coref-testset.notag"
    else:
        target = args.t
    cor.classify(target)
