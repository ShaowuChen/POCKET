# POCKET: Pruning Random Convolution Kernels for Time Series Classification
![image](https://github.com/ShaowuChen/POCKET/assets/78587515/7cff47b7-df40-46c8-80b1-d2ef6ed88bca)


Feel free to contact me (shaowu-chen@foxmail.com) if you have questions about the paper. 

# Acknowledgement
- The implementation is based on [ROCKET](https://github.com/angus924/rocket) and [S-ROCKET](https://github.com/salehinejad/srocket)

- Many thanks to Angus Dempster and Hojjat Salehinejad for their kind help.

# Idea
- Pruning random kernels via feature selection in the classifier
- Propose an ADMM-based Algorithm
- Propose an accelerated Algorithm: `POCKET`
  - two stages; introduce relatively invariant penalties
  - Prune up to `60%` kernels
  - `11`$\times$ faster than compared methods
  

# Requirements:
- python (3.6.12)
- Sklearn (0.24.2)

# Code Description 
**Note that we have rename our algorithm in the manuscript; ```P-ROCKET``` in the code corresponds to  ```POCKET```**
```
  ├── ROCKET-PPV-MAX                  : contains code for ROCKET-PPV-MAX
      ├── reproduce_experiments_ucr.py: Main code 
      ├── ADMM_pruner.py              : Our ADMM-based Algorithm
      ├── PROCKET_pruner.py           : Our POCKET Algorithm
      ├── rocket_functions.py         : Generate kernels
      ├── utils.py                    : Process results
  ├── ROCKET-PPV                      : contains code for ROCKET-PPV
  ├── MiniROCKET                      : contains code for MiniROCKET
  ├── demo.txt                        : write down the name of datasets for implementation here
```

# Dataset:
Find and download UCR 2018 on [UCR2018](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)

(We simply fill Nulls with zeros)

# Demo/Training:
- download and save the dataset archive on the root path
- run the following command

```bash
cd ./ROCKET-PPV-MAX 
python reproduce_experiments_ucr.py -o save_path -n 10 -e 50 
```

# Results

## Pruning ROCKET-PPV-MAX on 85 `bake off' datasets
![image](https://github.com/ShaowuChen/POCKET/assets/78587515/8c4fc351-8be8-4c7e-b4d8-e920a711df29)
![image](https://github.com/ShaowuChen/POCKET/assets/78587515/dfc23ac4-208a-48e7-bb4a-be694217f933)


## Pruning ROCKET-PPV-MAX on 43 `extra' datasets
![image](https://github.com/ShaowuChen/POCKET/assets/78587515/32c707ee-330a-45ab-8834-a78d1917a408)
