#!/bin/bash

mkdir log

for TARGET in {0..4}
do
	COMMAND="python simulate.py --alg gp --dataset liver_disorder --feature 5 --target ${TARGET} > log/gp_bupa_5_${TARGET}.log & python simulate.py --alg gp --dataset liver_disorder --feature ${TARGET} --target 5 > log/gp_bupa_${TARGET}_5.log"
	echo $COMMAND
        python simulate.py --alg gp --dataset liver_disorder --feature 5 --target ${TARGET} > log/gp_bupa_5_${TARGET}.log & python simulate.py --alg gp --dataset liver_disorder --feature ${TARGET} --target 5 > log/gp_bupa_${TARGET}_5.log
done

# python simulate.py --alg gp --dataset liver_disorder --feature 5 --target 0 > log_5_0.txt & python simulate.py --alg gp --dataset liver_disorder --feature 0 --target 5 > log_0_5.txt


for TARGET in {0..4}
do
        COMMAND="python simulate.py --alg gp --dataset abalone --feature 7 --target ${TARGET} > log/gp_abalone_7_${TARGET}.log & python simulate.py --alg gp --dataset abalone --target 7 --feature ${TARGET} > log/gp_abalone_${TARGET}_7.log;"
        echo $COMMAND
        python simulate.py --alg gp --dataset abalone --feature 7 --target ${TARGET} > log/gp_abalone_7_${TARGET}.log & python simulate.py --alg gp --dataset abalone --target 7 --feature ${TARGET} > log/gp_abalone_${TARGET}_7.log
done

for TARGET in {1..2}
do
        COMMAND="python simulate.py --alg gp --dataset income --feature 0 --target ${TARGET} > log/gp_income_0_${TARGET}.log & python simulate.py --alg gp --dataset income --target 0 --feature ${TARGET} > log/gp_income_${TARGET}_0.log;"
        echo $COMMAND
        python simulate.py --alg gp --dataset income --feature 0 --target ${TARGET} > log/gp_income_0_${TARGET}.log & python simulate.py --alg gp --dataset income --target 0 --feature ${TARGET} > log/gp_income_${TARGET}_0.log
done

for TARGET in {1..3}
do
        COMMAND="python simulate.py --alg gp --dataset arrhythmia --feature 0 --target ${TARGET} > log/gp_arrhythmia_0_${TARGET}.log;"
        echo $COMMAND
        python simulate.py --alg gp --dataset arrhythmia --feature 0 --target ${TARGET} > log/gp_arrhythmia_0_${TARGET}.log & python simulate.py --alg gp --dataset arrhythmia --target 0 --feature ${TARGET} > log/gp_arrhythmia_${TARGET}_0.log 
done


COMMAND="python simulate.py --alg gp --dataset ncep --feature 0 --target 1 > log/gp_ncep_0_1.log;"
echo $COMMAND
python simulate.py --alg gp --dataset ncep --feature 0 --target 1 > log/gp_ncep_0_1.log & python simulate.py --alg gp --dataset ncep --target 0 --feature 1 > log/gp_ncep_1_0.log 

COMMAND="python simulate.py --alg gp --dataset ncep --feature 2 --target 3 > log/gp_ncep_2_3.log;"
echo $COMMAND
python simulate.py --alg gp --dataset ncep --feature 2 --target 3 > log/gp_ncep_2_3.log & python simulate.py --alg gp --dataset ncep --target 2 --feature 3 > log/gp_ncep_3_2.log

COMMAND="python simulate.py --alg gp --dataset ncep --feature 4 --target 5 > log/gp_ncep_4_5.log;"
echo $COMMAND
python simulate.py --alg gp --dataset ncep --feature 4 --target 5 > log/gp_ncep_4_5.log & python simulate.py --alg gp --dataset ncep --target 4 --feature 5 > log/gp_ncep_5_4.log 

COMMAND="python simulate.py --alg gp --dataset ncep --feature 6 --target 7 > log/gp_ncep_6_7.log;"
echo $COMMAND
python simulate.py --alg gp --dataset ncep --feature 6 --target 7 > log/gp_ncep_6_7.log & python simulate.py --alg gp --dataset ncep --target 6 --feature 7 > log/gp_ncep_7_6.log 

