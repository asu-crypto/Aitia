#!/bin/bash

mkdir log

for BUDGETRATE in $(seq 0.1 0.1 1.0)
do
	COMMAND="python ./simulate.py --alg bsgd --dataset $1 --feature $2 --target $3 --budget_rate ${BUDGETRATE} > log/$1_$2_$3_${BUDGETRATE}.log;"
	echo $COMMAND
        python ./simulate.py --alg bsgd --dataset $1 --feature $2 --target $3 --budget_rate ${BUDGETRATE} > log/$1_$2_$3_${BUDGETRATE}.log 
done
