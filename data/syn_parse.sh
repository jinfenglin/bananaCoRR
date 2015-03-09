#! /bin/bash

rm depparsed/*
cd rawtext
for f in *; do
    ././../lib/stanford-parser/lexparser_collapsed.sh \
        "$f" > ../synparsed/"$f".psparse
    echo "$f"
done

