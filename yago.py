# yago.py
# TJC
# Make dictionaries out of YAGO info for searching later

import os
import gc
import cPickle as pickle

PROJECT_PATH = os.getcwd()
YAGO_PATH = os.path.join(PROJECT_PATH, "data", "yagoTransitiveType.tsv")

entity_dict = {}
count = 1

with open( YAGO_PATH, "r") as f:
    for line in f:
        name, info, subtypeOf = line.split()
        entry = name[1:-1].lower()
        if entry not in entity_dict:
            if len(entity_dict) >= 350000:
                pickle.dump( entity_dict, open("entity_dict" + str(count) + ".p", "wb"))
                count += 1
                entity_dict = {}
                gc.collect()
            #print len(entity_dict)
            entity_dict[entry] = set()
        entity_dict[entry].add(subtypeOf)

print len(entity_dict)
print entity_dict.keys()[0]
