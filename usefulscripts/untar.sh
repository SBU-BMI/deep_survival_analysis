#!/bin/bash

source_dir=$1
target_dir=$2

echo $source_dir
echo $target_dir

N=`ls $source_dir | wc -l`

i=0
for file in $(ls $source_dir); do
    echo "$source_dir/$file"
    tar -xvf "$source_dir/$file" -C $target_dir
    i=$((i+1))
    echo "$i of $N done!"
done
