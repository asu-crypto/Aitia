#!/bin/bash

# mkdir log

for TARGET in {0..4}
do
	COMMAND="python simulate.py --alg bsgd --dataset liver_disorder --feature 5 --target ${TARGET} > log/bsgd_liver_disorder_5_${TARGET}.log;"
	echo $COMMAND
        python simulate.py --alg bsgd --dataset liver_disorder --feature 5 --target ${TARGET} > log/bsgd_liver_disorder_5_${TARGET}.log 
done

for TARGET in {0..6}
do
        COMMAND="python simulate.py --alg bsgd --dataset abalone --feature 7 --target ${TARGET} > log/bsgd_abalone_7_${TARGET}.log;"
        echo $COMMAND
        python simulate.py --alg bsgd --dataset abalone --feature 7 --target ${TARGET} > log/bsgd_abalone_7_${TARGET}.log 
done

for TARGET in {1..2}
do
        COMMAND="python simulate.py --alg bsgd --dataset income --feature 0 --target ${TARGET} > log/bsgd_income_0_${TARGET}.log;"
        echo $COMMAND
        python simulate.py --alg bsgd --dataset income --feature 0 --target ${TARGET} > log/bsgd_income_0_${TARGET}.log 
done

for TARGET in {1..3}
do
        COMMAND="python simulate.py --alg bsgd --dataset arrhythmia --feature 0 --target ${TARGET} > log/bsgd_arrhythmia_0_${TARGET}.log;"
        echo $COMMAND
        python simulate.py --alg bsgd --dataset arrhythmia --feature 0 --target ${TARGET} > log/bsgd_arrhythmia_0_${TARGET}.log 
done


COMMAND="python simulate.py --alg bsgd --dataset ncep --feature 0 --target 1 > log/bsgd_ncep_0_1.log;"
echo $COMMAND
python simulate.py --alg bsgd --dataset ncep --feature 0 --target 1 > log/bsgd_ncep_0_1.log 

COMMAND="python simulate.py --alg bsgd --dataset ncep --feature 2 --target 3 > log/bsgd_ncep_2_3.log;"
echo $COMMAND
python simulate.py --alg bsgd --dataset ncep --feature 2 --target 3 > log/bsgd_ncep_2_3.log 

COMMAND="python simulate.py --alg bsgd --dataset ncep --feature 4 --target 5 > log/bsgd_ncep_4_5.log;"
echo $COMMAND
python simulate.py --alg bsgd --dataset ncep --feature 4 --target 5 > log/bsgd_ncep_4_5.log 

COMMAND="python simulate.py --alg bsgd --dataset ncep --feature 6 --target 7 > log/bsgd_ncep_6_7.log;"
echo $COMMAND
python simulate.py --alg bsgd --dataset ncep --feature 6 --target 7 > log/bsgd_ncep_6_7.log 

