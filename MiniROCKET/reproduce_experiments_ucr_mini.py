# Developed by Shaowu Chen, for the paper "P-ROCKET: Pruning Random Convolution Kernels for Time Series Classification"
# Based on https://github.com/angus924/rocket and https://github.com/salehinejad/srocket
# Many thanks to Angus Dempster et al (ROCKET/MiniROCKET) and Hojjat Salehinejad et al (S-ROCEKT)

# Hope everyone have a great and happy day

import argparse
import numpy as np
import pandas as pd
import time

from sklearn.linear_model import RidgeClassifierCV
from minirocket import fit, transform

import os
from utils import process_result
from sklearn.model_selection import GridSearchCV
from PROCKET_pruner import PROCKETPruner
import sys
# sys.path.insert(0, "srocket")
# from Srocket_main import Srocket
import pickle
import shutil
from sklearn.metrics import make_scorer

def score(y_true, y_predict):

    acc = np.mean(y_predict == y_true)

    return acc
    
my_scorer = make_scorer(score, greater_is_better=True)

# from rocket_functions import generate_kernels, apply_kernels


# == notes =====================================================================

# Reproduce the experiments on the UCR archive.
#
# For dataset: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/.
#
# Arguments:
# -d --dataset_names : txt file of dataset names
# -i --input_path    : parent directory for datasets
# -o --output_path   : path for results
# -n --num_runs      : number of runs (optional, default 10)
# -k --num_kernels   : number of kernels (optional, default 10,000)
#
# -e --num_epochs    : number of iteraions for AMDD/PROCKET algorighm
# -s --stop_thr      : to detect in which step iteration a given threshold is reached
# 
# *dataset_names* should be a txt file of dataset names, each on a new line.
#
# If *input_path* is, e.g., ".../Univariate_arff/", then each dataset should be
# located at "{input_path}/{dataset_name}/{dataset_name}_TRAIN.txt", etc.


# == parse arguments ===========================================================



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset_names", default="../demo.txt")
    parser.add_argument("-i", "--input_path", default="../UCRArchive_2018")
    parser.add_argument("-o", "--output_path", required = True)
    parser.add_argument("-n", "--num_runs", type = int, default = 10)
    parser.add_argument("-k", "--num_kernels", type = int, default = 10000)

    parser.add_argument("-e", "--num_epochs", type = int, default = 50)
    parser.add_argument("-s", "--stop_thr", type = float, default = 0.001)

    arguments = parser.parse_args()

 
    os.makedirs(arguments.output_path, exist_ok=True)
    shutil.copyfile('./reproduce_experiments_ucr_mini.py', './' + arguments.output_path + '/reproduce_experiments_ucr_mini.py')
    shutil.copyfile('./PROCKET_pruner.py', './' + arguments.output_path + '/PROCKET_pruner.py')



    pruning_remain_rates = [0.10, 0.35, 0.10, 0.30, 0.19, 0.49, 0.01, 0.39, 0.30, 0.33, 0.34, 0.19, 0.20, 0.62, 0.54,
                            0.27, 0.33, 0.73, 0.40, 0.01, 0.01, 0.20, 0.39, 0.21, 0.29, 0.37, 0.72, 0.67, 0.80, 0.78]
    pruning_remain_numbers = [int(10000*item) for item in [0.10, 0.35, 0.10, 0.30, 0.19, 0.49, 0.01, 0.39, 0.30, 0.33, 0.34, 0.19, 0.20, 0.62, 0.54,
                            0.27, 0.33, 0.73, 0.40, 0.01, 0.01, 0.20, 0.39, 0.21, 0.29, 0.37, 0.72, 0.67, 0.80, 0.78]]
    # pruning_remain_numbers = [924,3444,924,2940,1848,4872,84,3864,2940,3276,3360,1848,1932,6132,5376,2688,3276,7224,3948,84,84,1932,3864,2100,2856,3696,7140,6636,7980,7728]

    # == run =======================================================================

    dataset_names = np.loadtxt(arguments.dataset_names, "str")


    print(f"RUNNING".center(80, "="))
    # 'modify here'
    for dth, dataset_name in enumerate(dataset_names):


        # remain_num = int(arguments.num_kernels * pruning_remain_rates[dth])
        remain_num = pruning_remain_numbers[dth]

        print(f"{dataset_name}".center(80, "-"))

        # -- read data -------------------------------------------------------------

        print(f"Loading data".ljust(80 - 5, "."), end = "", flush = True)

        training_data = np.loadtxt(f"{arguments.input_path}/{dataset_name}/{dataset_name}_TRAIN.tsv")
        Y_training, X_training = training_data[:, 0].astype(np.int32), training_data[:, 1:].astype(np.float32)
        X_training[np.isnan(X_training)] = 0  # fill missing data with 0

        test_data = np.loadtxt(f"{arguments.input_path}/{dataset_name}/{dataset_name}_TEST.tsv")
        Y_test, X_test = test_data[:, 0].astype(np.int32), test_data[:, 1:].astype(np.float32)
        X_test[np.isnan(X_test)] = 0  # fill missing data with 0

        # make the labels start from 0
        assert (Y_training.min() == Y_test.min())
        Y_training = Y_training - Y_training.min()
        Y_test = Y_test - Y_test.min()
        print("Done.")

        # -- run -------------------------------------------------------------------

        print(f"Performing runs".ljust(80 - 5, "."), end = "", flush = True)

        my_results, descriptions = [], []
        for run in range(arguments.num_runs):

            print('\n\n','-'*40, dataset_name, ' ', run, '-'*40,'\n\n')

            # -- transform training ------------------------------------------------
            parameters = fit(X_training)
            time_a = time.perf_counter()
            X_training_transform_orig = transform(X_training, parameters)
            time_b = time.perf_counter()
            transform_time_training = time_b - time_a
            X_training_transform = X_training_transform_orig.copy()

            # -- transform test ----------------------------------------------------
            time_a = time.perf_counter()
            X_test_transform_orig = transform(X_test, parameters)
            time_b = time.perf_counter()
            transform_time_test = time_b - time_a
            X_test_transform = X_test_transform_orig.copy()


            '''
            ======================Original miniRocket RidgeClassifier======================
            '''
            # -- training ----------------------------------------------------------

            time_a = time.perf_counter()
            classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
            classifier.fit(X_training_transform.copy(), Y_training.copy())
            time_b = time.perf_counter()
            training_time = time_b - time_a

            training_acc = classifier.score(X_training_transform.copy(), Y_training.copy())


            # -- test --------------------------------------------------------------

            time_a = time.perf_counter()
            test_acc = classifier.score(X_test_transform.copy(), Y_test.copy())
            time_b = time.perf_counter()
            test_time = time_b - time_a

            results_index = 0
            my_results, descriptions = process_result(my_results, descriptions,
                                        [classifier.alpha_, training_acc*100, np.nan, test_acc*100, np.nan, training_time, test_time],
                                        ['Rocket_' + item for item in
                                         ['alpha', 'train_acc', 'train_acc_std', 'test_acc','test_acc_std', 'train_time', 'test_time']],
                                        run, results_index)
            print('\n','='*20,'ridge','='*20)
            # print(training_acc)
            print(test_acc)
            # print(np.sort(np.linalg.norm(classifier.coef_, axis=0))[::-1])
            # print(classifier.intercept_)


            'Too time-consuming, we do not do mini sorkcet'
            '''
            ======================Compared methods: EA SRocket, random, norm-based======================
            '''

            # remain_num, test_acc_SRocket, test_acc_random, test_acc_l1, test_acc_l2, EA_time, EA_post_train_time = \
            #      Srocket(weights_norm1, X_training_transform, Y_training, X_test_transform, Y_test)
            #
            # results_index += 1
            # my_results, descriptions = process_result(my_results, descriptions,
            #                             [remain_num, test_acc_SRocket*100, np.nan,
            #                              test_acc_random*100, np.nan,
            #                              EA_time, EA_post_train_time,
            #                              test_acc_l1*100, np.nan,
            #                              test_acc_l2*100, np.nan],
            #                             ['remain_num', 'SRocket_test_acc', 'SRocket_test_acc_std',
            #                              'test_acc_random', 'test_acc_random_std',
            #                              'EA_time', 'EA_post_train_time',
            #                              'test_acc_l1', 'test_acc_l1_std',
            #                              'test_acc_l2', 'test_acc_l2_std'],
            #                             run, results_index)



            '''
            ======================Proposed  P-ROCKET Pruner======================
            '''
            print('='*20+'doing ADMM_Pruner, please wait'+'='*20)

            X_training_transform_copy = X_training_transform_orig.copy()
            Y_training_copy = Y_training.copy()
            X_test_transform_copy = X_test_transform_orig.copy()
            Y_test_copy = Y_test.copy()

            # data normalization
            mean = np.mean(X_training_transform, axis=0)

            X_training_transform -= mean
            norm = np.linalg.norm(X_training_transform, axis=0)
            norm[norm == 0] = 1
            X_training_transform /= norm

            X_test_transform -= mean
            X_test_transform /= norm

            # Y +-1 coded
            n_class = int(np.max(Y_training)) + 1
            Y_training_coded = np.ones([n_class, n_class]) * -1
            for _ in range(Y_training_coded.shape[0]):
                Y_training_coded[_, _] = 1
            Y_training_coded = Y_training_coded[Y_training]


            gs = GridSearchCV(
                PROCKETPruner(n_class, Y_training, X_test_transform, Y_test, remain_num=remain_num, stop_thr=arguments.stop_thr, epoch=arguments.num_epochs),
                {'k': [10,1,0.1]+[0.01,100,1000]}, cv=5, n_jobs=-1,return_train_score=True,scoring=my_scorer)


            start_time = time.time()
            gs.fit(X_training_transform, Y_training.copy())
            ADMM_fit_time = time.time() - start_time


            estimator = gs.best_estimator_
            
            scores = estimator.scores(X_training_transform, Y_training, X_test_transform, Y_test)
            
            print('\n','*'*10)
            print('k=',estimator.k)
            ADMM_k = estimator.k
            print(gs.cv_results_)

            '''post training'''
            retrain_time,  retrain_scores_thr_based, retrain_alpha_thr_based=\
             estimator.retrain(X_training_transform_copy,Y_training_copy,X_test_transform_copy, Y_test_copy, model='PPV')


            scores.insert(0, remain_num)
            # scores.insert(1, 0)

            scores.insert(1, gs.refit_time_)
            scores.insert(2, ADMM_fit_time)
            scores.insert(3, ADMM_k)
            '''here'''
            scores.insert(4, estimator.iter)
            scores.insert(5, estimator.stop_thr)

            scores += [retrain_time, retrain_scores_thr_based,np.nan,retrain_alpha_thr_based]


            results_index += 1
            my_results, descriptions = process_result(my_results, descriptions, scores,
                                                      ['cv'+item for item in ['`my_remain_num`', 'refit_time', 'all_fit_time','ADMM_k','iter_epoch','final_thr','training_acc_W', 'training_acc_W_std', 'test_acc_W', 'test_acc_W_std', 'sparse_rate_W', \
                                                       'training_acc_sparse_W',  'training_acc_sparse_W_std', 'test_acc_sparse_W',  'test_acc_sparse_W_std','sparse_rate_sparse_W', \
                                                       'training_acc_Theta', 'training_acc_Theta_std', 'test_acc_Theta', 'test_acc_Theta_std', 'sparse_rate_Theta','training_acc_SparseW_thr_based', 'training_acc_SparseW_thr_based_std', 'test_acc_SparseW_thr_based', 'test_acc_SparseW_thr_based_std','retrain_time', 'retrain_acc','retrain_acc_std','retrain_alpha']],
                                                        run, results_index)


            if 'results' not in locals():
                writer = pd.ExcelWriter(arguments.output_path + '/MiniRocketResults_PPV.xlsx')

                results = pd.DataFrame(index=dataset_names,
                                       columns=sum(descriptions,[]),
                                       data=0)
                results.index.name = "dataset"
            else:
                pass

        for index1, name1 in enumerate(descriptions):
            for index2, name2 in enumerate(name1):
                if 'acc' in name2 and 'std' in name2:
                    results.loc[dataset_name, name2] = np.round(np.array(my_results[index1])[:, index2-1].std(),
                                                                           decimals=2)
                else:
                    results.loc[dataset_name, name2] = np.round(np.array(my_results[index1])[:, index2].mean(), decimals=2)

        results.to_excel(writer)
        writer.save()


        print("Done.")

        with open(arguments.output_path+'/'+ dataset_name +'_descriptions', 'wb') as fp:    # pickling
            pickle.dump(descriptions,fp)
        # with open("descriptions", "rb") as fp:  # Unpickling
        #     descriptions = pickle.load(fp)

        with open(arguments.output_path+'/'+ dataset_name +'_results', 'wb') as fp:    # pickling
            pickle.dump(my_results,fp)
        # with open("results", "rb") as fp:  # Unpickling
        #     results = pickle.load(fp)


    print(f"FINISHED".center(80, "="))

