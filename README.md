# P-ROCKET: Pruning Random Convolution Kernels for Time Series Classification

Hello there, welcome to our Repo! 

Feel free to contact me (shaowu-chen@foxmail.com) if you have questions about the paper. 


The implementaion is based on [ROCKET](https://github.com/angus924/rocket) and [S-ROCKET](https://github.com/salehinejad/srocket)

Many thanks to Angus Dempster and Hojjat Salehinejad for their kindly help.


# 1. Environment:
python3.6.12 ; Sklearn 0.24.2.

# 2. Description for files:

```
  ├── ROCKET-PPV-MAX: contains code for ROCKET-PPV-MAX
  ├────test
  ├── ROCKET-PPV: contains code for ROCKET-PPV
  ├── MiniROCKET: contains code for MiniROCKET
  ├── demo.txt: write down the name of datasets for imeplementation here
```

# 3. Dataset:
Find and download UCR 2018 on [UCR2018](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)

# 4. Demo/How to run:
- download and save the dataset archieve on the root path
- run the follow command

```bash
cd ./ROCKET-PPV-MAX 
python reproduce_experiments_ucr.py -o save_path -n 10 -e 50 
```

