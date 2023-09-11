import numpy as np
import time
from sklearn.base import BaseEstimator

class ADMMPruner(BaseEstimator):

    '''
    y_training: plain labels such as [0,1,2,3]
    Y_training: binary coded matrix
    '''

    def __init__(self, n_class, y_training, X_test_transform, y_test, rho_1=None, rho_2=None, k=None, stop_thr=None, remain_num=None,  epoch=100, sigma_thr=1e-4, finetune=False, if_print=True, reweight=False, epsilon=None, one_dual=True, _dataset_name=None):
        self.k = k
        assert (remain_num == int(remain_num))
        self.remain_num = int(remain_num)

        self.n_class = n_class
        # self.remain_rate = remain_rate
        self.epoch = epoch
        self.sigma_thr = sigma_thr
        self.finetune = finetune
        self.reweight = reweight
        self.if_print = if_print
        self.epsilon = epsilon
        self.one_dual = one_dual

        self.rho_1 = rho_1
        self.rho_2 = rho_2

        self.y_training = y_training
        self.X_test_transform = X_test_transform
        self.y_test = y_test
        self._dataset_name = _dataset_name

        self.stop_thr = stop_thr
        self.iter = epoch
    
    def transform_Y_coded(self, y):
        Y_coded = np.ones([self.n_class, self.n_class]) * -1
        for _ in range(Y_coded.shape[0]):
            Y_coded[_, _] = 1
        Y_coded = Y_coded[y]

        return Y_coded

    def fit(self, X_training_transform, y_training):
        Y_training = self.transform_Y_coded(y_training)

        Y_mean = np.mean(Y_training,axis=0)
        self.Y_mean = Y_mean
        Y_demean = Y_training - Y_mean

        assert (self.k != 0)
        n_sample, n_feature = X_training_transform.shape
        n_filter = n_feature // 2
        n_class = Y_demean.shape[1]

        Theta = np.zeros([n_feature, n_class])
        U = np.zeros([n_feature, n_class])
        V = np.zeros(Y_demean.shape)

         # Woodbury formula /Sherman-Morrison-Woodbury formula
        # P = np.eye(n_feature)/self.k - 1/self.k*X_training_transform.T @ np.linalg.inv(
            # self.k * np.eye(n_sample) + X_training_transform @ X_training_transform.T) @ X_training_transform
        # print(np.linalg.norm(P))

        # ADM
        P = X_training_transform @ X_training_transform.T

        record = True
        time_start = time.time()

        
        # for image
        self.norm_diff = []
        self.thres = []

        for iter in range(self.epoch):
            # primal update: W
            if iter==0:
                threshold=0
                W_old = 0
            else:
                W_old = W.copy()

            # rho1: ridge
            # rho2: classification
            # rho3: 1/threshold
            self.rho_3 = 1/threshold if threshold!=0 else 1e10
            coef = self.rho_1+self.rho_3

            # INV = np.linalg.inv( (coef * np.eye(n_feature)+self.rho_2*X_training_transform.T @ X_training_transform) )
            # INV = np.eye(n_feature)/coef - 1/coef*(rho_2*X_training_transform.T) @ np.linalg.inv(coef*np.eye(n_sample) + X_training_transform@X_training_transform.T) @ X_training_transform

            INV = coef * np.eye(n_sample) + P
            INV = np.linalg.inv(INV)
            INV =   -1/coef * self.rho_2 * X_training_transform.T @ INV @ X_training_transform
            for ele in range(INV.shape[0]): #save memory
                INV[ele,ele]+=1/coef

            W = INV @ (self.rho_3*(Theta + U) + self.rho_2 * X_training_transform.T @ Y_demean)

            W_tilde = W - U
            W_tilde_group = np.reshape(W_tilde.copy(), [n_filter, -1])
            group_norm = np.linalg.norm(W_tilde_group, axis=1)
            assert (np.inf not in group_norm)
            assert (True not in np.isnan(group_norm))

            sorted_group_norm = np.sort(group_norm)[::-1]
            threshold = sorted_group_norm[self.remain_num]

            # not applied; reweight==None
            if self.reweight:
                if self.epsilon==None:
                    shrinkage = np.max([np.zeros(n_filter), 1 - np.square((threshold) / (group_norm))], axis=0)
                else:
                    shrinkage = np.max([np.zeros(n_filter), 1 - np.square((threshold+self.epsilon) / (group_norm+self.epsilon))], axis=0)
            else:
                if self.epsilon == None:
                    shrinkage = np.max([np.zeros(n_filter), 1 - (threshold) / (group_norm)], axis=0)
                    assert (np.inf not in abs(shrinkage))
                else:
                    shrinkage = np.max([np.zeros(n_filter), 1 - (threshold+self.epsilon) / (group_norm+self.epsilon)], axis=0)
                    assert (np.inf not in abs(shrinkage))

            assert (shrinkage.shape == (n_filter,))
            assert (np.inf not in shrinkage)  # there are only -inf
            if np.sum(np.isnan(shrinkage)) !=0 or threshold==0:
                with open('zero_threshold_at_iter0.log', 'a+') as f:
                    f.write(self._dataset_name+'\n')

            shrinkage[abs(shrinkage) == np.inf] = 0

            Theta = np.reshape(np.diag(shrinkage) @ W_tilde_group, [n_feature, -1])

            # dual update: U
            U = U + Theta - W
            if self.one_dual==False:
                V = V + X_training_transform @ W - Y_demean

            if record==True and self.stop_thr!=None and threshold/sorted_group_norm[0]<self.stop_thr:
                record == False
                self.iter = iter+1
            if iter==self.epoch-1:
                self.stop_thr = threshold/sorted_group_norm[0]

            self.norm_diff.append(np.linalg.norm(W-Theta))
            self.thres.append(threshold)

        self.count_time = time.time() - time_start

        self.W = W.copy()
        self.Theta = Theta.copy()

        sparse_group_W = np.reshape(W.copy(), [n_filter, -1])
        W_group_norm = np.linalg.norm(sparse_group_W, axis=1)
        self.sparse_rate_W = np.mean(W_group_norm == 0) * 100

        W_group_sparse_index = np.argsort(W_group_norm)[:(n_filter - self.remain_num)]

        sparse_W_norm_based = sparse_group_W.copy()
        sparse_W_norm_based[W_group_sparse_index, :] = 0
        sparse_W_norm_based = np.reshape(sparse_W_norm_based, [n_feature, -1])
        self.SparseW = sparse_W_norm_based.copy()
        self.sparse_index = W_group_sparse_index.copy()
        self.pruned_W = np.reshape(np.delete(sparse_group_W, W_group_sparse_index, axis=0), [-1, n_class])

        self.Theta_zero_index = np.squeeze(np.argwhere(shrinkage == 0))
        sparse_W_thr_based = sparse_group_W.copy()
        sparse_W_thr_based[self.Theta_zero_index, :] = 0
        sparse_W_thr_based = np.reshape(sparse_W_thr_based, [n_feature, -1])
        self.SparseW_thr_based = sparse_W_thr_based.copy()

        # not applied
        if self.finetune:
            del_X_training_transform = np.delete(X_training_transform, self.Theta_zero_index, axis=1)
            del_Theta = np.delete(self.Theta, self.Theta_zero_index, axis=0)
            del_U = np.delete(U,self. Theta_zero_index, axis=0)
            del_P = np.linalg.inv(self.k  * np.eye(del_X_training_transform.shape[1]) + del_X_training_transform.T @ del_X_training_transform)
            finetuneW = del_P @ (self.k*(del_Theta + del_U) - del_X_training_transform.T @ (V - Y_demean))
            self.finetune_pruned_W = finetuneW



    def predict(self, X):

        y_pred = X @ self.SparseW + self.Y_mean
        y_pred = np.argmax(y_pred, axis=1)

        return y_pred



    def retrain(self, X_training_transform, Y_training, X_test_transform,Y_test, model='PPVMAX'):
        from sklearn.linear_model import RidgeClassifierCV

        if model=='PPVMAX':
            pruned_X_training_transform = np.delete(X_training_transform.copy(), np.hstack((2 * self.sparse_index, 2 * self.sparse_index + 1)), axis=1)
            pruned_X_test_transform = np.delete(X_test_transform.copy(), np.hstack((2 * self.sparse_index, 2 * self.sparse_index + 1)), axis=1)
        else:
            pruned_X_training_transform = np.delete(X_training_transform.copy(),  self.sparse_index, axis=1)
            pruned_X_test_transform = np.delete(X_test_transform.copy(), self.sparse_index, axis=1)
        print('\nretrain remain_feature', pruned_X_training_transform.shape[1])
        time_a = time.perf_counter()
        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
        classifier.fit(pruned_X_training_transform, Y_training.copy())
        time_b = time.perf_counter()
        training_time = time_b - time_a

        test_acc = classifier.score(pruned_X_test_transform, Y_test)

        pruned_X_training_transform = np.delete(X_training_transform.copy(),
                                                np.hstack((2 * self.Theta_zero_index, 2 * self.Theta_zero_index + 1)), axis=1)
        pruned_X_test_transform = np.delete(X_test_transform.copy(),
                                            np.hstack((2 * self.Theta_zero_index, 2 * self.Theta_zero_index + 1)), axis=1)

        time_a = time.perf_counter()
        classifier_thr = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
        classifier_thr.fit(pruned_X_training_transform, Y_training.copy())
        time_b = time.perf_counter()
        training_time = time_b - time_a

        test_acc_thr = classifier_thr.score(pruned_X_test_transform, Y_test)        

        print('retrain ridge acc:', test_acc)
        return training_time, test_acc*100, classifier.alpha_, test_acc_thr*100, classifier_thr.alpha_



    def scores(self, X_training_transform, y_training, X_test_transform, y_test, iter=None):
        print('self.if_print',self.if_print)
        training_acc_W, test_acc_W = self.single_score(self.W, X_training_transform, y_training, X_test_transform, y_test)
        if self.if_print:
            print('\niter=', iter, '\nW sparse_rate=', self.sparse_rate_W, ', training_acc=', training_acc_W,
                  ', test_acc=', test_acc_W)

        sparse_rate_sparse_W = np.mean(np.linalg.norm(self.SparseW, axis=1) == 0) * 100
        training_acc_sparse_W, test_acc_sparse_W = self.single_score(self.SparseW, X_training_transform, y_training, X_test_transform, y_test)
        if self.if_print:
            print('Sparse_W sparse rate', sparse_rate_sparse_W, ', training_acc=', training_acc_sparse_W, ', test_acc=',
                  test_acc_sparse_W)


        training_acc_SparseW_thr_based, test_acc_SparseW_thr_based = self.single_score(self.SparseW_thr_based, X_training_transform, y_training, X_test_transform, y_test)
        if self.if_print:
            print('SparseW_thr_based sparse rate', sparse_rate_sparse_W, ', training_acc=', training_acc_SparseW_thr_based, ', test_acc=',
                  test_acc_SparseW_thr_based)

        sparse_rate_Theta = np.mean(np.linalg.norm(np.reshape(self.Theta, [self.Theta.shape[0]//2, -1]), axis=1) == 0) * 100
        training_acc_Theta, test_acc_Theta = self.single_score(self.Theta, X_training_transform, y_training, X_test_transform, y_test)
        if self.if_print:
            print('Theta sparse rate', sparse_rate_Theta, ', training_acc=',
                  training_acc_Theta, ', test_acc=', test_acc_Theta)

        if self.finetune:
            del_X_training_transform = np.delete(X_training_transform, self.Theta_zero_index, axis=1)
            del_X_test_transform = np.delete(X_test_transform, self.Theta_zero_index, axis=1)

            training_acc_finetune_pruned_W, test_acc_finetune_pruned_W = self.single_score(self.finetune_pruned_W, del_X_training_transform, y_training,
                                                                   del_X_test_transform, y_test)
            if self.if_print:
                print('Theta sparse rate', sparse_rate_Theta, ', training_acc=',
                   training_acc_finetune_pruned_W, ', test_acc=', test_acc_finetune_pruned_W)

        self.score_return = [training_acc_W, np.nan, test_acc_W,  np.nan, self.sparse_rate_W, \
               training_acc_sparse_W, np.nan, test_acc_sparse_W, np.nan, sparse_rate_sparse_W, \
               training_acc_Theta, np.nan, test_acc_Theta, np.nan, sparse_rate_Theta, training_acc_SparseW_thr_based, np.nan, test_acc_SparseW_thr_based, np.nan]

        return self.score_return

    def single_score(self, weight, X_training_transform, y_training, X_test_transform, y_test):

        output = X_training_transform @ weight + self.Y_mean
        output = np.argmax(output, axis=1)
        training_acc = np.mean(output == y_training)*100

        output = X_test_transform @ weight + self.Y_mean
        output = np.argmax(output, axis=1)
        test_acc = np.mean(output == y_test)*100

        return training_acc, test_acc