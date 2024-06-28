# Aitia: Efficient Secure Computation for Causal Discovery

## Install
```
pip install -r requirements.txt
```

## Download dataset
Dataset can be found at https://drive.google.com/file/d/1m8RU9i4cZDj-DOIEzSuvMjzjqxBCab1e/view?usp=sharing

Download and unzip the file to the folder `data`

## Run plaintext models
```
cd src/plain
./aitia_train.sh
```

## Run MPC models

### Option 1: 2pc
```
cd src/mpc
./aitia_train.sh
```

### Option 2: MPC

For 3pc (You can replace 3 with other number to have different number of servers):
```
cd src/mpc
./aitia_train.sh -n 3
```
