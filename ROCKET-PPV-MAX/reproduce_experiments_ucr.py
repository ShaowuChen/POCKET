# Developed by Shaowu Chen, for the paper "P-ROCKET: Pruning Random Convolution Kernels for Time Series Classification"
# Based on https://github.com/angus924/rocket and https://github.com/salehinejad/srocket
# Many thanks to Angus Dempster et al (ROCKET/MiniROCKET) and Hojjat Salehinejad et al (S-ROCEKT)

# Hope everyone have a great and happy day

import argparse
import numpy as np
import pandas as pd
import time

from sklearn.linear_model import RidgeClassifierCV
from rocket_functions import generate_kernels, apply_kernels

import os
from utils import process_result
from sklearn.model_selection import GridSearchCV

# proposed methods
from PROCKET_pruner import PROCKETPruner
from ADMM_pruner import ADMMPruner

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
    parser.add_argument("-i", "--input_path", default="../../../UCRArchive_2018")
    parser.add_argument("-o", "--output_path", required = True)
    parser.add_argument("-n", "--num_runs", type = int, default = 10)
    parser.add_argument("-k", "--num_kernels", type = int, default = 10000)

    parser.add_argument("-e", "--num_epochs", type = int, default = 50)
    parser.add_argument("-s", "--stop_thr", type = float, default = 0.001)

    arguments = parser.parse_args()


    os.makedirs(arguments.output_path, exist_ok=True)
    shutil.copyfile('./reproduce_experiments_ucr.py', './'+ arguments.output_path  + '/reproduce_experiments_ucr.py')
    shutil.copyfile('./PROCKET_pruner.py', './'+ arguments.output_path  + '/PROCKET_pruner.py')

    remain_rates = [0.5,0.2447,0.1946,0.2108,0.2430,0.3428,0.1798,0.2444,0.4067,0.1806,0.2675,0.7295,0.5250,0.7040,0.2423,0.3285,0.3271,0.5020,0.1843,0.1827,0.3264,0.1088,0.3391,0.2218,0.6433,0.5156,0.4661,0.1830,0.4843,0.5]

    # == run =======================================================================

    dataset_names = np.loadtxt(arguments.dataset_names, "str")


    print(f"RUNNING".center(80, "="))
    for dth, dataset_name in enumerate(dataset_names):

        remain_num = int(10000*remain_rates[dth])
        print(f"{dataset_name}".center(80, "-"))

        # -- read data -------------------------------------------------------------

        print(f"Loading data".ljust(80 - 5, "."), end = "", flush = True)

        training_data = np.loadtxt(f"{arguments.input_path}/{dataset_name}/{dataset_name}_TRAIN.tsv")
        Y_training, X_training = training_data[:, 0].astype(np.int32), training_data[:, 1:]
        X_training[np.isnan(X_training)] = 0  # fill missing data with 0

        test_data = np.loadtxt(f"{arguments.input_path}/{dataset_name}/{dataset_name}_TEST.tsv")
        Y_test, X_test = test_data[:, 0].astype(np.int32), test_data[:, 1:]
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

            input_length = X_training.shape[-1]
            # the generate_kernels function is modified for comparing EA-SROCKET
            kernels = generate_kernels(input_length, arguments.num_kernels)
            weights_norm1 = kernels[-1]
            kernels = kernels[:-1]

            # -- transform training ------------------------------------------------

            time_a = time.perf_counter()
            X_training_transform = apply_kernels(X_training.copy(), kernels)
            time_b = time.perf_counter()
            transform_time_training = time_b - time_a

            # -- transform test ----------------------------------------------------

            time_a = time.perf_counter()
            X_test_transform = apply_kernels(X_test.copy(), kernels)
            time_b = time.perf_counter()
            transform_time_test = time_b - time_a


            '''
            ======================Original Rocket======================
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
            print(test_acc)
 

            '''
            ======================Compared method: EA SRocket======================
            '''

            # remain_num, test_acc_SRocket, test_acc_random, test_acc_l1, test_acc_l2, EA_time, EA_post_train_time = \
            #      Srocket(weights_norm1, X_training_transform, Y_training, X_test_transform, Y_test)

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
            ======================Proposed Pruner======================
            '''

            X_training_transform_copy = X_training_transform.copy()
            Y_training_copy = Y_training.copy()
            X_test_transform_copy = X_test_transform.copy()
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

            # in case no kernels pruned in S-Rocket
            if remain_num==arguments.num_kernels:
                remain_num = arguments.num_kernels//2



            '''
             =======Proposed ADMM Pruner========
            '''
            print('=' * 20 + 'doing ADMM_Pruner, please wait' + '=' * 20)

            gs = GridSearchCV(
                ADMMPruner(n_class, Y_training, None, Y_test, remain_num=remain_num, epoch=arguments.num_epochs,
                          stop_thr=arguments.stop_thr, _dataset_name=dataset_name),
                {'rho_1': [1e-3, 1e-1, 1, 10, 100, 1000], 'rho_2': [1e-3, 1e-1, 1, 10, 100, 1000]}, cv=5,
                n_jobs=25, return_train_score=True, scoring=my_scorer)

            start_time = time.time()
            gs.fit(X_training_transform, Y_training.copy())
            ADMM_fit_time = time.time() - start_time

            estimator = gs.best_estimator_
            scores = estimator.scores(X_training_transform, Y_training, X_test_transform, Y_test)

            print('\n', '*' * 10)
            print('rho_1=', estimator.rho_1)
            print('rho_2=', estimator.rho_2)
            rho_1 = estimator.rho_1
            rho_2 = estimator.rho_2
            print(gs.cv_results_)
            print('*' * 10, '\n')

            scores.insert(0, remain_num)
            scores.insert(1, gs.refit_time_)
            scores.insert(2, ADMM_fit_time)
            scores.insert(3, rho_1)
            scores.insert(4, rho_2)
            scores.insert(5, estimator.iter)
            scores.insert(6, estimator.stop_thr)

            results_index += 1
            my_results, descriptions = process_result(my_results, descriptions, scores,
                                                      ['adm' + item for item in
                                                       ['my_remain_num', 'refit_time', 'all_fit_time', 'rho_1', 'rho_2',
                                                        'iter_epoch', 'final_thr', 'training_acc_W',
                                                        'training_acc_W_std', 'test_acc_W', 'test_acc_W_std',
                                                        'sparse_rate_W', \
                                                        'training_acc_sparse_W', 'training_acc_sparse_W_std',
                                                        'test_acc_sparse_W', 'test_acc_sparse_W_std',
                                                        'sparse_rate_sparse_W', \
                                                        'training_acc_Theta', 'training_acc_Theta_std',
                                                        'test_acc_Theta', 'test_acc_Theta_std', 'sparse_rate_Theta',
                                                        'training_acc_SparseW_thr_based',
                                                        'training_acc_SparseW_thr_based_std',
                                                        'test_acc_SparseW_thr_based',
                                                        'test_acc_SparseW_thr_based_std']],
                                                      run, results_index)


            '''
            =======Proposed P-ROCKET Pruner========
            '''
            print('='*20+'doing P-ROCKET pruner, please wait'+'='*20)

            gs = GridSearchCV(
                PROCKETPruner(n_class, Y_training, X_test_transform, Y_test, remain_num=remain_num, epoch=arguments.num_epochs,stop_thr=arguments.stop_thr, _dataset_name=dataset_name),
                {'k':  [1e-3,1e-1,1,10,100,1000]}, cv=5, n_jobs=25,return_train_score=True,scoring=my_scorer)

            start_time = time.time()
            gs.fit(X_training_transform, Y_training.copy())
            ADMM_fit_time = time.time() - start_time

            estimator = gs.best_estimator_
            scores = estimator.scores(X_training_transform, Y_training, X_test_transform, Y_test)
           
            print('\n','*'*10)
            print('k=',estimator.k)
            ADMM_k = estimator.k
            print(gs.cv_results_)
            print( '*'*10,'\n')

            '''post training'''
            retrain_time,  retrain_scores_thr_based, retrain_alpha_thr_based=\
             estimator.retrain(X_training_transform_copy,Y_training_copy,X_test_transform_copy, Y_test_copy, model='PPVMAX')


            scores.insert(0, remain_num)
            scores.insert(1, gs.refit_time_)
            scores.insert(2, ADMM_fit_time)
            scores.insert(3, ADMM_k)
            scores.insert(4, estimator.iter)
            scores.insert(5, estimator.stop_thr)
            scores += [retrain_time, retrain_scores_thr_based,np.nan,retrain_alpha_thr_based]

            results_index += 1
            my_results, descriptions = process_result(my_results, descriptions, scores,
                                                      ['my_remain_num', 'refit_time', 'all_fit_time','ADMM_k','iter_epoch','final_thr', 'training_acc_W', 'training_acc_W_std', 'test_acc_W', 'test_acc_W_std', 'sparse_rate_W', \
                                                       'training_acc_sparse_W',  'training_acc_sparse_W_std', 'test_acc_sparse_W',  'test_acc_sparse_W_std','sparse_rate_sparse_W', \
                                                       'training_acc_Theta', 'training_acc_Theta_std', 'test_acc_Theta', 'test_acc_Theta_std', 'sparse_rate_Theta','training_acc_SparseW_thr_based', 'training_acc_SparseW_thr_based_std', 'test_acc_SparseW_thr_based', 'test_acc_SparseW_thr_based_std','retrain_time', 'retrain_acc_thr','retrain_acc_std_thr','retrain_alpha_thr'],
                                                        run, results_index)

            if 'results' not in locals():
                writer = pd.ExcelWriter(arguments.output_path + '/RocketPMResults.xlsx')

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

