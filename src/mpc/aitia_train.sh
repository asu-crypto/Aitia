#!/bin/bash

mkdir log
numparty=2
while getopts ":n:" opt; do
        case ${opt} in
        n )
            numparty=$OPTARG
            ;;
        esac
done

for TARGET in {0..4}
do
	COMMAND="python simulate.py --alg bsgd --dataset liver_disorder --feature 5 --target ${TARGET} > log/sgd_bupa_5_${TARGET}.log & python simulate.py --alg bsgd --dataset liver_disorder --feature ${TARGET} --target 5 > log/sgd_bupa_${TARGET}_5.log"
	echo $COMMAND
        python simulate.py --alg bsgd --dataset liver_disorder --num-party $numparty --feature 5 --target ${TARGET} > log/bsgd_bupa_5_${TARGET}.${numparty}party.log & python simulate.py --alg bsgd --dataset liver_disorder --num-party $numparty --feature ${TARGET} --target 5 > log/bsgd_bupa_${TARGET}_5.${numparty}party.log
done

for TARGET in {0..6}
do
        COMMAND="python simulate.py --alg bsgd --dataset abalone --feature 7 --target ${TARGET} --test-direction f2t > log/bsgd_abalone_7_${TARGET}.log & python simulate.py --alg bsgd --dataset abalone --target  7 --feature ${TARGET} --test-direction t2f > log/bsgd_abalone_${TARGET}_7.log"
        echo $COMMAND
        python simulate.py --alg bsgd --dataset abalone --num-party $numparty --feature 7 --target ${TARGET} --test-direction f2t > log/bsgd_abalone_7_${TARGET}.${numparty}party.log & python simulate.py --alg bsgd --dataset abalone --num-party $numparty --target  7 --feature ${TARGET} --test-direction t2f > log/bsgd_abalone_${TARGET}_7.${numparty}party.log
done

for TARGET in {1..2}
do
	COMMAND="python simulate.py --alg bsgd --dataset income --feature 0 --target ${TARGET} > log/sgd_income_0_${TARGET}.log & python simulate.py --alg bsgd --dataset income --feature ${TARGET} --target 0 > log/sgd_income_${TARGET}_0.log"
	echo $COMMAND
        python simulate.py --alg bsgd --dataset income --num-party $numparty --feature 0 --target ${TARGET} > log/sgd_income_0_${TARGET}.${numparty}party.log & python simulate.py --alg bsgd --dataset income --num-party $numparty --feature ${TARGET} --target 0 > log/sgd_income_${TARGET}_0.${numparty}party.log
done


for TARGET in {1..3}
do
        COMMAND="python simulate.py --alg bsgd --dataset arrhythmia --feature 0 --target ${TARGET} > log/bsgd_arrhythmia_0_${TARGET}.log & python simulate.py --alg bsgd --dataset arrhythmia --target 0 --feature ${TARGET} > log/bsgd_arrhythmia_${TARGET}_0.log"
        echo $COMMAND
        python simulate.py --alg bsgd --dataset arrhythmia --num-party $numparty --feature 0 --target ${TARGET} > log/bsgd_arrhythmia_0_${TARGET}.${numparty}party.log & python simulate.py --alg bsgd --dataset arrhythmia --num-party $numparty --target 0 --feature ${TARGET} > log/bsgd_arrhythmia_${TARGET}_0.${numparty}party.log
done

COMMAND="python simulate.py --alg bsgd --dataset ncep --num-party $numparty --feature 0 --target 1 > log/bsgd_ncep_0_1.${numparty}party.log & python simulate.py --alg bsgd --dataset ncep --num-party $numparty --feature 1 --target 0 > log/bsgd_ncep_1_0.log"
echo $COMMAND
python simulate.py --alg bsgd --dataset ncep --num-party $numparty --feature 0 --target 1 > log/bsgd_ncep_0_1.${numparty}party.log & python simulate.py --alg bsgd --dataset ncep --num-party $numparty --feature 1 --target 0 > log/bsgd_ncep_1_0.${numparty}party.log

COMMAND="python simulate.py --alg bsgd --dataset ncep --num-party $numparty --feature 2 --target 3 > log/bsgd_ncep_2_3.log & python simulate.py --alg bsgd --dataset ncep --num-party $numparty --feature 3 --target 2 > log/bsgd_ncep_3_2.log"
echo $COMMAND
python simulate.py --alg bsgd --dataset ncep --num-party $numparty --feature 2 --target 3 > log/bsgd_ncep_2_3.${numparty}party.log & python simulate.py --alg bsgd --dataset ncep --num-party $numparty --feature 3 --target 2 > log/bsgd_ncep_3_2.${numparty}party.log


COMMAND="python simulate.py --alg bsgd --dataset ncep --num-party $numparty --feature 4 --target 5 > log/bsgd_ncep_4_5.log & python simulate.py --alg bsgd --dataset ncep --num-party $numparty --feature 5 --target 4 > log/bsgd_ncep_5_4.log"
echo $COMMAND
python simulate.py --alg bsgd --dataset ncep --num-party $numparty --feature 4 --target 5 > log/bsgd_ncep_4_5.${numparty}party.log & python simulate.py --alg bsgd --dataset ncep --num-party $numparty --feature 5 --target 4 > log/bsgd_ncep_5_4.${numparty}party.log

COMMAND="python simulate.py --alg bsgd --dataset ncep --num-party $numparty --feature 6 --target 7 > log/bsgd_ncep_6_7.log & python simulate.py --alg bsgd --dataset ncep --num-party $numparty --feature 7 --target 6 > log/bsgd_ncep_7_6.log"
echo $COMMAND
python simulate.py --alg bsgd --dataset ncep --num-party $numparty --feature 6 --target 7 > log/bsgd_ncep_6_7.${numparty}party.log & python simulate.py --alg bsgd --dataset ncep --num-party $numparty --feature 7 --target 6 > log/bsgd_ncep_7_6.${numparty}party.log
