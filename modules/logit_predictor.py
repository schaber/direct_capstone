import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

class PlastPredictor():
    """
    This class builds a logistic regression model to predict likelihood of
    being a plasticizer
    """
    def __init__(self, reg_param=1.0, reg_type='l1'):
        self.verbose = False
        self.reg_param = reg_param
        self.reg_type = reg_type

        self.W = None
        self.b = None

    def train_test_split(data, split=0.8, return_n=False):
        """
        This function splits the data into test and train sets at the desired
        split
        """
        n_samples = data.shape[0]
        n_train = int(n_samples * split)
        n_test = n_samples - n_train
        rand_idxs = np.random.choice(np.arange(n_samples), size=n_samples)
        train_idxs = rand_idxs[:n_train]
        test_idxs = rand_idxs[n_train:]
        data_train = data[train_idxs,:]
        data_test = data[test_idxs,:]

        if return_n:
            return data_train, data_test, n_train, n_test
        else:
            return data_train, data_test

    def train(pl_data, org_data, num_iter=100):
        """
        This function takes the set of plasticizers and a set of molecules
        unlikely to be plasticizers and builds a logistic regression model
        """
        n_pl_samples = pl_data.shape[0]
        n_org_samples = org_data.shape[0]
        n_features = pl_data.shape[1]
        assert n_features == org_data.shape[1], "Both datasets should have the \
                                                 the same number of features"

        # Save models to average and accuracy data
        w_avg = np.zeros((num_iter, n_features))
        b_avg = np.zeros((num_iter,))
        pl_train_accs = []
        pl_test_accs = []
        org_train_accs = []
        org_test_accs = []

        for i in range(num_iter):
            # Randomly select subset of negative dataset equal to number of plasts
            pl_data = np.concatenate([pl_data, np.ones((n_pl_samples,1))], axis=1)
            org_select = np.random.choice(np.arange(n_org_samples), size=n_pl_samples)
            org_data = org_data[org_select,:]
            org_data = np.concatenate([org_data, np.zeros((n_pl_samples,1))], axis=1)

            # Train/test split
            pl_train, pl_test, n_train, n_test = train_test_split(pl_data, return_n=True)
            org_train, org_test = train_test_split(org_data)
            train_data = np.concatenate([pl_train, org_train])
            test_data = np.concatenate([pl_test, org_test])
            np.random.shuffle(train_data)
            np.random.shuffle(test_data)
            X_train = train_data[:,:-1]
            y_train = train_data[:,-1]
            X_test = test_data[:,:-1]
            y_test = test_data[:,-1]

            # Scale and fit regressor
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            clf = LogisticRegression(solver='liblinear', penalty=reg_type, C=reg_param)
            clf.fit(X_train, y_train)

            # Calculate accuracies
            pl_train = scaler.transform(pl_train[:,:-1])
            pl_test = scaler.transform(pl_test[:,:-1])
            org_train = scaler.transform(org_train[:,:-1])
            org_test = scaler.transform(org_test[:,:-1])
            pl_train_accs.append(clf.score(pl_train, np.ones((n_train,1))))
            pl_test_accs.append(clf.score(pl_test, np.ones((n_test,1))))
            org_train_accs.append(clf.score(org_train, np.zeros((n_train,1))))
            org_test_accs.append(clf.score(org_test, np.zeros((n_test,1))))
            w_avg[i,:] = clf.coef_[0,:]
            b_avg[i] = clf.intercept_[0]

        self.W = np.mean(w_avg, axis=0)
        self.b = np.mean(b_avg, axis=0)
        self.pl_train_acc = np.mean(pl_train_accs)
        self.pl_test_acc = np.mean(pl_test_accs)
        self.org_train_acc = np.mean(org_train_accs)
        self.org_test_acc = np.mean(org_test_accs)

    def predict(X, type='prob', class='pos'):
        assert self.W is not None, "ERROR: You must train the model before predicting"
        assert self.W.shape[0] == self.X.shape[1], "ERROR: Input data must have same \
                                                   number of features as training sets"

        if class == 'pos':
            y = 1 / (1 + np.exp(-X@w+b))
        elif class = 'neg':
            y = 1 - (1 / (1 + np.exp(-X@w+b)))
        if type == 'prob':
            return y
        elif type == 'binary':
            return np.where(y > 0.5).sum()
