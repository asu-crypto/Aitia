#!/bin/bash

# mkdir log

for TARGET in {0..4}
do
	COMMAND="python simulate.py --alg smo --dataset liver_disorder --feature 5 --target ${TARGET} > log/smo_bupa_5_${TARGET}.log & python simulate.py --alg smo --dataset liver_disorder --feature ${TARGET} --target 5 > log/smo_bupa_${TARGET}_5.log"
	echo $COMMAND
        python simulate.py --alg smo --dataset liver_disorder --feature 5 --target ${TARGET} > log/smo_bupa_5_${TARGET}.log & python simulate.py --alg smo --dataset liver_disorder --feature ${TARGET} --target 5 > log/smo_bupa_${TARGET}_5.log
done

for TARGET in {0..6}
do
        COMMAND="python simulate.py --alg smo --dataset abalone --feature 7 --target ${TARGET} --test-direction f2t > log/smo_abalone_7_${TARGET}.log & python simulate.py --alg smo --dataset abalone --target  7 --feature ${TARGET} --test-direction t2f > log/smo_abalone_${TARGET}_7.log"
        echo $COMMAND
        python simulate.py --alg smo --dataset abalone --feature 7 --target ${TARGET} --test-direction f2t > log/smo_abalone_7_${TARGET}.log & python simulate.py --alg smo --dataset abalone --target  7 --feature ${TARGET} --test-direction t2f > log/smo_abalone_${TARGET}_7.log
done

for TARGET in {1..2}
do
	COMMAND="python simulate.py --alg smo --dataset income --feature 0 --target ${TARGET} > log/smo_income_0_${TARGET}.log & python simulate.py --alg smo --dataset income --feature ${TARGET} --target 0 > log/smo_income_${TARGET}_0.log"
	echo $COMMAND
        python simulate.py --alg smo --dataset income --feature 0 --target ${TARGET} > log/smo_income_0_${TARGET}.log & python simulate.py --alg smo --dataset income --feature ${TARGET} --target 0 > log/smo_income_${TARGET}_0.log
done


for TARGET in {1..3}
do
        COMMAND="python simulate.py --alg smo --dataset arrhythmia --feature 0 --target ${TARGET} > log/smo_arrhythmia_0_${TARGET}.log & python simulate.py --alg smo --dataset arrhythmia --target 0 --feature ${TARGET} > log/smo_arrhythmia_${TARGET}_0.log"
        echo $COMMAND
        python simulate.py --alg smo --dataset arrhythmia --feature 0 --target ${TARGET} > log/smo_arrhythmia_0_${TARGET}.log & python simulate.py --alg smo --dataset arrhythmia --target 0 --feature ${TARGET} > log/smo_arrhythmia_${TARGET}_0.log
done

COMMAND="python simulate.py --alg smo --dataset ncep --feature 0 --target 1 > log/smo_ncep_0_1.log & python simulate.py --alg smo --dataset ncep --feature 1 --target 0 > log/smo_ncep_1_0.log"
echo $COMMAND
python simulate.py --alg smo --dataset ncep --feature 0 --target 1 > log/smo_ncep_0_1.log & python simulate.py --alg smo --dataset ncep --feature 1 --target 0 > log/smo_ncep_1_0.log

COMMAND="python simulate.py --alg smo --dataset ncep --feature 2 --target 3 > log/smo_ncep_2_3.log & python simulate.py --alg smo --dataset ncep --feature 3 --target 2 > log/smo_ncep_3_2.log"
echo $COMMAND
python simulate.py --alg smo --dataset ncep --feature 2 --target 3 > log/smo_ncep_2_3.log & python simulate.py --alg smo --dataset ncep --feature 3 --target 2 > log/smo_ncep_3_2.log


COMMAND="python simulate.py --alg smo --dataset ncep --feature 4 --target 5 > log/smo_ncep_4_5.log & python simulate.py --alg smo --dataset ncep --feature 5 --target 4 > log/smo_ncep_5_4.log"
echo $COMMAND
python simulate.py --alg smo --dataset ncep --feature 4 --target 5 > log/smo_ncep_4_5.log & python simulate.py --alg smo --dataset ncep --feature 5 --target 4 > log/smo_ncep_5_4.log

COMMAND="python simulate.py --alg smo --dataset ncep --feature 6 --target 7 > log/smo_ncep_6_7.log & python simulate.py --alg smo --dataset ncep --feature 7 --target 6 > log/smo_ncep_7_6.log"
echo $COMMAND
python simulate.py --alg smo --dataset ncep --feature 6 --target 7 > log/smo_ncep_6_7.log & python simulate.py --alg smo --dataset ncep --feature 7 --target 6 > log/smo_ncep_7_6.log
