#!/bin/bash

for i in "$@"
do
case $i in
    -d=*|--dataset=*)
    DATASET="${i#*=}"
    shift
    ;;
    -e=*|--estimator=*)
    ESTIMATOR="${i#*=}"
    shift
    ;;
    *)
    ;;
esac
done
python experiment.py with dataset.name=$DATASET estimator.name=$ESTIMATOR --file_storage=../.results/$DATASET-$ESTIMATOR
