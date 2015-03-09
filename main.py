# /usr/bin/python
# -*- coding: utf-8 -*-

"""
This program is to:
Use various feature functions to train a MALLET MaxEnt model to classify whether
entities in a text are coreferent or not.
"""
import collections
import subprocess
import sys
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
YAGO_PATH = os.path.join(RES_PATH, "yago")


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
        self.T = 'true'
        self.F = 'false'
        self.pairs = None

        # dicts will store name dictionaries
        self.dicts = {}
        self.org_suffixes = []
        self.populate_dict()

        # all feature_functions should
        # 1. take no parameters
        # (use self.pairs)
        # 2. return a list or an iterable which has len of # number of tokens
        self.feature_functions = [ self.i_pronoun,
                                   self.j_pronoun,
                                   self.only_i_pronoun,
                                   self.only_j_pronoun,
                                   self.string_match_no_articles,
                                   self.string_contains_no_articles,
                                   self.str_stem_match,                     # WOW
                                   self.pro_str_match,
                                   self.pn_str_match,
                                   self.pn_str_contains,
                                   self.words_str_match,
                                   # self.yago_ontology,
                                   self.j_definite,
                                   self.j_demonstrative,
                                   self.word_overlap,
                                   self.i_proper_noun,
                                   self.j_proper_noun,
                                   # self.i_proper_j_pronoun,               # hurts
                                   self.both_proper,
                                   self.both_diff_proper,
                                   self.ner_tag_match
        ]

    def read_data(self, input_filename):
        """load sentences from data file"""
        self.pairs = []
        cur_filename = None
        with open(os.path.join(DATA_PATH, input_filename)) as in_file:
            for line in in_file:
                filename, i_line, i_start, i_end, i_ner, i_word, j_line, j_start, \
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

    def get_features(self, train=True):
        """traverse function list and get all values in a dictionary"""
        features = collections.defaultdict(list)

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
        # currently yago data only

        with open(os.path.join("resources", "yago", "yago_entries.p"), "rb") as pjar:
            self.dicts["yago"] = pickle.load(pjar)

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
        name = "i_pronoun="
        values = []
        poss = self.get_i_poss()
        for pos in poss:
            if len(pos) == 1 and pos[0].startswith("PRP"):
                values.append(name + self.T)
            else:
                values.append(name + self.F)

        return values

    def j_pronoun(self):
        """Is the second entity a pronoun"""
        name = "j_pronoun="
        values = []
        poss = self.get_j_poss()
        for pos in poss:
            if len(pos) == 1 and pos[0].startswith("PRP"):
                values.append(name + self.T)
            else:
                values.append(name + self.F)

        return values


    def only_j_pronoun(self):
        """Checks if only the second entity is a pronoun, and not the first"""
        name = "only_j_pronoun="
        values = []
        for bools in zip(self.i_pronoun(), self.j_pronoun()):
            if bools[0].endswith("false") and bools[1].endswith("true"):
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def only_i_pronoun(self):
        """Checks if only the first entity is a pronoun, and not the second"""
        name = "only_i_pronoun="
        values = []
        for bools in zip(self.i_pronoun(), self.j_pronoun()):
            if bools[1].endswith("false") and bools[0].endswith("true"):
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values
    
    def ner_tag_match(self):
        """true if two mentions share same ber tag"""
        name = "ner_tag_match="
        values = []
        for tags in zip(self.get_i_ners(), self.get_j_ners()):
            if tags[0] == tags[1]:
                values.append(name + self.T)
            else:
                values.append(name + self.F)
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
        name = "string_match_no_articles="
        values = []
        i_words = self.get_i_words()
        j_words = self.get_j_words()
        i_tags = self.get_i_poss()
        j_tags = self.get_j_poss()
        for i in range(len(i_words)):
            comparator_i = self.remove_articles(i_words[i], i_tags[i])
            comparator_j = self.remove_articles(j_words[i], j_tags[i])
            if comparator_i == comparator_j:
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def string_contains_no_articles(self):
        """Checks if one entities is contained in another, without articles"""
        name = "string_contains_no_articles="
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
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def score_jaccard(self, set1, set2, numeric=False):
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

        def yago_query(words, person):
            # print "QUERIED: ", words
            # if person:
            #     if len(words) > 1:
            #         result = retrieve_yago(words + ["_".join(words)], person)
            #     else:
            #         result = retrieve_yago(words, person)
            # else:
            result = retrieve_yago("_".join(words).lower(), person)

            return result

        def retrieve_yago(query, person):
            """get yago attribute list"""
            # print "querying: ", query
            attribs = set()
            if query in results.keys():
                # print "MEMO!"
                attribs = results[query]
            else:
                # print "CUR_SET: ", attribs
                # if querying person name, also query for each token
                # (eg> Barack Obama should be retrieved by querying Obama only)
                # if person:
                #     yago_name = self.dicts['yago'].get(query)
                #     print "FULL_NAMES from CUR_Q {}: {}".format(query, yago_name)
                # otherwise, just use a singleton of full name
                # else:
                yago_name = [query.lower()]
                if yago_name is not None:
                    for name in yago_name:
                        # print "HASHED: ", yago_hash(name)
                        yago_d = load_yago(yago_hash(name))
                        try:
                            # print "ATTRIBS_TO_ADD: ", yago_d[name]
                            attribs.update(yago_d[name])
                        except KeyError:
                            pass
                results[query] = attribs
            # print "result: ", attribs
            return attribs

        def load_yago(filename):
            """Load up a relevant yago file, given a hashed name"""
            yago_path = os.path.join(PROJECT_PATH, "resources", "yago")
            yago_dict = {}
            with open(os.path.join(yago_path, filename)) as yago_file:
                for line in yago_file:
                    entry, attribs = line.split(": ")
                    attribs = attribs[2:-2].split("', '")
                    try:
                        yago_dict[entry].extend(attribs)
                    except KeyError:
                        yago_dict[entry] = attribs
            return yago_dict

        def yago_hash(string):
            """Take the first two alphabetic character of a string"""
            hashed = ""
            for c in string.lower():
                if len(hashed) > 1:
                    return hashed
                else:
                    if c.isalpha():
                        hashed += c
            while len(hashed) < 2:
                hashed += "_"
            return hashed

        # main part of this feature function
        name = "yago_ontology="
        values = []

        # YAGO dict is sooo huge, we need to memoize
        results = {}

        i_words = self.get_i_words()
        j_words = self.get_j_words()
        # refer to yago dict for only proper nouns
        i_pns = self.i_proper_noun()
        j_pns = self.j_proper_noun()
        i_ners = self.get_i_ners()
        j_ners = self.get_j_ners()
        for instance in range(len(i_words)):
            # print "TARGETS: ", i_words[instance], i_pronouns[instance], \
            #     j_words[instance], j_pronouns[instance]
            if i_pns[instance].endswith("true"):
                # print "QUERYING I: ", i_words[instance]
                result1 = yago_query(i_words[instance], i_ners[instance] == "PER")
            else:
                result1 = []
            if j_pns[instance].endswith("true"):
                # print "QUERYING J: ", j_words[instance]
                result2 = yago_query(j_words[instance], j_ners[instance] == "PER")
            else:
                result2 = []

            # get queries for i word, check if i mention i is person
            # if i_ners[instance] == "PER":
            #     i_queries = tuple(i_words[instance])
            #     result1 = retrieve_yago(i_queries, person=True)
            # else:
            #     i_queries = ("_".join(i_words[instance]))
            #     result1 = retrieve_yago(i_queries, person=False)
            #
            # get queries for j word, also check if it's person
            # if j_ners[instance] == "PER":
            #     j_queries = tuple(j_words[instance])
            #     result2 = retrieve_yago(j_queries, person=True)
            # else:
            #     j_queries = ["_".join(j_words[instance])]
            #     result2 = retrieve_yago(j_queries, person=False)

            # print "YAGO doing ", instance
            # return jaccard score
            if result1 == [] or result2 == []:
                values.append(name + "no data")
            else:
                values.append(name + str(self.score_jaccard(result1, result2)))
        return values

    def str_stem_match(self):
        """Stem first, and then check string match"""
        name = "str_stem_match="
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
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def pro_str_match(self):
        """Check if both entities are pronouns and they both match"""
        name = "pro_str_match="
        values = []
        i_words = self.get_i_words()
        j_words = self.get_j_words()
        pro_bools = zip(self.i_pronoun(), self.j_pronoun())
        for i in range(len(i_words)):
            if pro_bools[i][0].endswith("true") \
                    and pro_bools[i][1].endswith("true") \
                    and i_words[i] == j_words[i]:
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def pn_str_match(self):
        """Check if both entities are proper nouns and they both match"""
        name = "pn_str_match="
        values = []
        i_words = self.get_i_words()
        j_words = self.get_j_words()
        i_tags = self.get_i_poss()
        j_tags = self.get_j_poss()
        for i in range(len(i_words)):
            i_nnps = [tag for tag in i_tags[i] if tag.startswith("NNP")]
            j_nnps = [tag for tag in j_tags[i] if tag.startswith("NNP")]
            if len(i_nnps) > 0 and len(j_nnps) > 0 \
                    and " ".join(i_words[i]) == " ".join(j_words[i]):
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def pn_str_contains(self):
        """Check if both entities are proper nouns and one contains the other"""
        name = "pn_str_contains="
        values = []
        i_words = self.get_i_words()
        j_words = self.get_j_words()
        i_tags = self.get_i_poss()
        j_tags = self.get_j_poss()
        for i in range(len(i_words)):
            i_nnps = [tag for tag in i_tags[i] if tag.startswith("NNP")]
            j_nnps = [tag for tag in j_tags[i] if tag.startswith("NNP")]
            i_string = " ".join(i_words[i])
            j_string = " ".join(j_words[i])
            if len(i_nnps) > 0 and len(j_nnps) > 0 \
                    and (j_string in i_string or i_string in j_string):
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def words_str_match(self):
        """Check if both entities are not pronouns and they both match"""
        name = "words_str_match="
        values = []
        i_words = self.get_i_words()
        j_words = self.get_j_words()
        pro_bools = zip(self.i_pronoun(), self.j_pronoun())
        for i in range(len(i_words)):
            if pro_bools[i][0].endswith("false") \
                    and pro_bools[i][1].endswith("false") \
                    and " ".join(i_words[i]) == " ".join(j_words[i]):
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def j_definite(self):
        """Check if second entity is a definite NP"""
        name = "j_definite="
        values = []
        for words in self.get_j_words():
            if words[0].lower() == "the":
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def j_indefinite(self):
        """Check if second entity is an indefinite NP.
        Without apositive???"""
        pass

    def j_demonstrative(self):
        """Check if second entity is a demonstrative NP"""
        name = "j_demonstrative="
        values = []
        demons = {"these", "those", "this", "that"}
        for words in self.get_j_words():
            if words[0].lower() in demons:
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def word_overlap(self):
        """Check if entities have any words in common"""
        name = "word_overlap="
        values = []
        i_words = self.get_i_words()
        j_words = self.get_j_words()
        for i in range(len(i_words)):
            i_set = set(word.lower() for word in i_words[i])
            j_set = set(word.lower() for word in j_words[i])
            if len(i_set.intersection(j_set)) > 0:
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def i_proper_noun(self):
        """Check if the first mention is a proper noun"""
        name = "i_proper="
        values = []
        tags = self.get_i_poss()
        for i in range(len(tags)):
            nnps = [tag for tag in tags[i] if tag.startswith("NNP")]
            if len(nnps) == len(tags[i]):
                # print i, " found NNP at first loc"
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def j_proper_noun(self):
        """Check if the first mention is a proper noun"""
        name = "j_proper="
        values = []
        tags = self.get_j_poss()
        for i in range(len(tags)):
            nnps = [tag for tag in tags[i] if tag.startswith("NNP")]
            if len(nnps) == len(tags[i]):
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def i_proper_j_pronoun(self):
        name = "i_pn_i_pro="
        values = []
        for bools in zip(self.i_proper_noun(), self.j_pronoun()):
            if bools[0].endswith("true") and bools[1].endswith("true"):
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values


    def both_proper(self):
        """Check if both entities are proper nouns"""
        name = "both_proper="
        values = []

        for bools in zip(self.i_proper_noun(), self.j_proper_noun()):
            if bools[0].endswith("true") and bools[1].endswith("true"):
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def both_diff_proper(self):
        """Check if both entities are proper nouns and no words match"""
        name = "both_diff_proper="
        values = []
        i_words = self.get_i_words()
        j_words = self.get_j_words()
        i_tags = self.get_i_poss()
        j_tags = self.get_j_poss()
        for i in range(len(i_words)):
            i_nnps = [tag for tag in i_tags[i] if tag.startswith("NNP")]
            j_nnps = [tag for tag in j_tags[i] if tag.startswith("NNP")]
            i_set = set(word.lower() for word in i_words[i])
            j_set = set(word.lower() for word in j_words[i])
            if len(i_nnps) > 0 and len(j_nnps) > 0 and \
                            len(i_set.intersection(j_set)) == 0:
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values

    def acronym_match(self):
        """Check lexically if one entity is an acronym of the other"""
        name = "acronym_match="
        values = []
        i_words = self.get_i_words()
        j_words = self.get_j_words()
        for i in range(len(i_words)):
            i_string = "".join([word[0] for word in i_words])
            j_string = "".join([word[0] for word in j_words])
            if i_string == j_words[0] or j_string == i_words[0]:
                values.append(name + self.T)
            else:
                values.append(name + self.F)
        return values


'''
    def num_agr(tuple1, tuple2):
        pass


    def gen_agr(tuple1, tuple2):
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
