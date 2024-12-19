# POCKET: Pruning Random Convolution Kernels for Time Series Classification
(previous name: `P-ROCKET: Pruning Random Convolution Kernels for Time Series Classification`)
- Welcome to visit our Repo! Don't hesitate to contact me (shaowu-chen@foxmail.com) if you have questions about the paper. 
- Download the paper here [arXiv](https://arxiv.org/abs/2309.08499)
- If the code help, please cite our work:
```
 @article{chen2024pocket,
  title={POCKET: Pruning random convolution kernels for time series classification from a feature selection perspective},
  author={Chen, Shaowu and Sun, Weize and Huang, Lei and Li, Xiao Peng and Wang, Qingyuan and John, Deepu},
  journal={Knowledge-Based Systems},
  volume={300},
  pages={112253},
  year={2024},
  publisher={Elsevier}
}
```

![image](https://github.com/ShaowuChen/POCKET/assets/78587515/b585c1f5-5d3c-45d0-bfd4-f1dee8594a1c)




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
**Note that we have rename our algorithm in the manuscript; ```P-ROCKET``` or ```PROCKET``` in the code corresponds to  ```POCKET```**
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
- Find and download UCR 2018 on [UCR2018](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)
- Note that there are NaN data or missing elements in the datasets, and we simply fill Nulls with zeros

# Demo/Training:
- download and save the dataset archive on the root path
- add expected name of datasets in `demo.txt` (in which 
- run the following command

```bash
cd ./ROCKET-PPV-MAX 
python reproduce_experiments_ucr.py -o save_path -n 10 -e 50 
```

# Results
(Please find the editable tables in LaTex format in [arXiv](https://arxiv.org/abs/2309.08499), downlod the `TeX Source` file)

## Pruning ROCKET-PPV-MAX on 85 `bake off' datasets
![](./results.html)

![image](https://github.com/ShaowuChen/POCKET/assets/78587515/c5625d56-e7ce-4fcb-977e-095f82f6c4f6)
![image](https://github.com/ShaowuChen/POCKET/assets/78587515/e4c9dc04-92b4-4ecc-8d68-7897ea293bf8)



## Pruning ROCKET-PPV-MAX on 43 `extra' datasets
![image](https://github.com/ShaowuChen/POCKET/assets/78587515/2e2dd9de-4154-4ee6-ae75-27684449c448)

