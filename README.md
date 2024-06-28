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
```
cd src/mpc
./aitia_train.sh
```

TODO: Add support to customizable number of servers to bash file
