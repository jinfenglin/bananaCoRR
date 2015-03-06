from nltk.stem.lancaster import LancasterStemmer

st = LancasterStemmer()


def get_word(tup):
    return tup[0]


def get_pos(tup):
    return tup[1]


def get_ner(tup):
    return tup[2]


def i_pronoun(tuple1, tuple2):
    if get_pos(tuple1) == 'PRP':
        return True
    else:
        return False


def j_pronoun(tuple1, tuple2):
    if get_pos(tuple2) == 'PRP':
        return True
    else:
        return False


def only_j_pronoun(tuple1, tuple2):
    if get_pos(tuple2) == 'PRP' and not get_pos(tuple1) == 'PRP':
        return True
    else:
        return False


def only_i_pronoun(tuple1, tuple2):
    if get_pos(tuple1) == 'PRP' and not get_pos(tuple2) == 'PRP':
        return True
    else:
        return False


def remove_articulate(str, ban_list=("a", "an", "the", "this", "that", "those",
                                     "these")):
    for word in ban_list:
        str.replace(word, "")
    return str.trim()


def str_match(str1, str2):
    return remove_articulate(str1) == remove_articulate(str2)


def str_stem_match(str1, str2):
    st = LancasterStemmer()
    str1 = st.stem(remove_articulate(str1))
    str2 = st.stem(remove_articulate(str2))
    return str1 == str2


def pro_str(tuple1, tuple2):
    if get_pos(tuple1) != 'PRP*' or get_pos(tuple2) != 'PRP*':
        return False
    return str_match(get_word(tuple1), get_word(tuple2))


def pn_str(tuple1, tuple2):
    if get_pos(tuple1) != 'NNP*' or get_pos(tuple2) != 'NNP*':
        return False
    return str_match(get_word(tuple1), get_word(tuple2))


def words_str(tuple1, tuple2):
    pass


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
