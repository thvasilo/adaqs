#!/bin/sh

DIR="$( cd "$( dirname "$0" )" && pwd )"

cd $DIR && cd ../..

block_size_try=(10, 50, 100, 200, 500, 1000, 2000, 5000)
batch_size_try=(1, 5, 10, 20, 50, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000)

for block_size in "${block_size_try[@]}"
do
    for batch_size in "${batch_size_try[@]}"
    do
        # The sample size is 10,000. Hence repeat 2 times implies 20,000 samples.
        ./release/app/qs_eval_parameters $block_size $batch_size 10
    done
done

