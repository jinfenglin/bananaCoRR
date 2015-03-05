from nltk.stem.lancaster import LancasterStemmer
st = LancasterStemmer()
def get_word(tuple):
    return tuple[0]
def get_pos(tuple):
    return tuple[1]
def get_ner(tuple):
    return tuple[2]
def I_PRONOUN(tuple1,tuple2):
     if get_pos(tuple1) == 'PRP':
         return True
     else:
         return False
def J_PRONOUN(tuple1,tuple2):
     if get_pos(tuple2) == 'PRP':
         return True
     else:
         return False
def ONLY_J_PRONOUN(tuple1,tuple2):
    if get_pos(tuple2) == 'PRP' and not get_pos(tuple1) == 'PRP':
        return True
    else:
        return False
def ONLY_I_PRONOUN(tuple1,tuple2):
    if get_pos(tuple1) == 'PRP' and not get_pos(tuple2) == 'PRP':
        return True
    else:
        return False
def remove_articulate(str,ban_list=["a","an","the","this","that","those","these"]):
     for word in ban_list:
        str.replace(word,"")
     return str.trim()


def STR_MATCH(str1,str2):
   return remove_articulate(str1)==remove_articulate(str2)

def STR_STEM_MATCH(str1,str2):
    st = LancasterStemmer()
    str1=st.stem(remove_articulate(str1))
    str2=st.stem(remove_articulate(str2))
    return str1==str2


def PRO_STR(tuple1,tuple2):
    if get_pos(tuple1) != 'PRP*' or get_pos(tuple2) !='PRP*':
        return False
    return STR_MATCH(get_word(tuple1,tuple2))
def PN_STR(tuple1,tuple2):
     if get_pos(tuple1) != 'NNP*' or get_pos(tuple2) !='NNP*':
        return False
     return STR_MATCH(get_word(tuple1,tuple2))

def WORDS_STR(tuple1,tuple2):
    pass
def PN_STR(tuple1,tuple2):
     if get_pos(tuple1) != '!PRP*' or get_pos(tuple2) != '!PRP*':
        return False
     return STR_MATCH(get_word(tuple1,tuple2))

def both_are(pron1,pron2,str):
    if pron1==str and pron2==str:
        return True
    else:
        return False

def WORD_OVERLAP(tuple1,tuple2):
    s1=set(get_word(tuple1).split())
    s2=set(get_word(tuple2).split())
    if set(s1).intersection(s2):
        return True
    else:
        return False
def PN_SUBSTR(tuple1,tuple2):
    if both_are(get_pos(tuple1),get_pos(tuple2),'PPN*'):
        if get_word(tuple1) in get_word(tuple2) or get_word(tuple2) in get_word(tuple1):
            return True
    else:
        return False
def J_DEFINITE(tuple1,tuple2):
    pass
def J_INDEFINITE(tuple1,tuple2):
    pass
def J_DEMONSTRATIVE(tuple1,tuple2):
    pass
def NUM_AGR(tuple1,tuple2):
    pass
def GEN_AGR(tuple1,tuple2):
    pass
def BOTH_PROPER(tuple1,tuple2):
    pass
def BOTH_DIFF_PROPER(tuple1,tuple2):
    pass
def ALIAS_DATE(tuple1,tuple2):
    pass
def ALIAS_PERSON(tuple1,tuple2):
    pass
def ALIAS_ORG(tuple1,tuple2):
    pass
def WORDS_SUBSTR(tuple1,tuple2):
    pass
def BOTH_PRONOUNS(tuple1,tuple2):
    pass
def GEN_NUM_AGR(tuple1,tuple2):
    pass
def SPAN(tuple1,tuple2):
    pass
def CONTAINS_PN(tuple1,tuple2):
    pass
class feature_functions:
    def __init__(self,list):
        self.tuple_1=list[0]
        self.tuple_2=list[1]
        self.feature_name_list=list[2:]
        for feature_fn in self.feature_name_list:
            feature_fn(self.tuple_1,self.tuple_2)



if __name__=='__main__':
    ff=feature_functions([('foo','vb','foo'),('bar','np','bar'),I_PRONOUN,J_PRONOUN])
