#!/usr/bin/python
#compute the accuracy of an NE tagger

#usage: coref-evaluator.py [gold_file][output_file]

import sys, re

if len(sys.argv) != 3:
    sys.exit("usage: coref-evaluator.py [gold_file][output_file]")

#gold standard file
goldfh = open(sys.argv[1], 'r')
#system output
testfh = open(sys.argv[2], 'r')

gold_tag_list = []
#gold_word_list = []
test_tag_list = []

emptyline_pattern = re.compile(r'^\s*$')

for gline in goldfh.readlines():
    if not emptyline_pattern.match(gline):
        parts = gline.split()
        # print parts[-1]
        gold_tag_list.append(parts[-1])


for tline in testfh.readlines():
    if not emptyline_pattern.match(tline):
        parts = tline.split("\t")
        # print parts[0]
        test_tag_list.append(parts[0])

test_total = 0
gold_total = 0
correct = 0


for i in range(len(gold_tag_list)):
    if gold_tag_list[i] != 'no':
        gold_total += 1
    if test_tag_list[i] != 'no':
        test_total += 1
    if gold_tag_list[i] != 'no' and gold_tag_list[i] == test_tag_list[i]:
        correct += 1

print gold_total
print test_total
print correct


precision = float(correct) / test_total
recall = float(correct) / gold_total
f = precision * recall * 2 / (precision + recall)

#print correct, gold_total, test_total
print 'precision =', precision, 'recall =', recall, 'f1 =', f
            
    
