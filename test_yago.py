# test_yago.py
# TJC

import gc
import cPickle as pickle

#for i in range(8):
#    entity_dict = pickle.load(open("entity_dict" + str(i+1) + ".p", "rb"))
#    print entity_dict.items()[:2]
#    print len(entity_dict)
#    entity_dict = {}
#    gc.collect()

#entity_dict = pickle.load(open("entity_dict1.p", "rb"))
#print entity_dict["Feel_the_Music"]

search1 = "Elizabeth_Fretwell".lower()
search2 = "Feel_the_Music".lower()
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
    
print search1, result1
print search2, result2
