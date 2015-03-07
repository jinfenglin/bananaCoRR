#! /bin/bash

rm depparsed/*
cd rawtext
for f in *; do
    ~/dev/stanfordNLP/stanford-parser-full-2015-01-30/lexparser_collapsed.sh \
        "$f" > ../depparsed/"$f".depparse
    echo "$f"
done

