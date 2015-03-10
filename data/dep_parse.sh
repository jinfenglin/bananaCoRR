#! /bin/bash

rm depparsed/*
cd rawtext
for f in *; do
    ../../lib/stanford-parser/lexparser_collapsed.sh \
        "$f" > ../depparsed/"$f".depparse
    echo "$f"
done

