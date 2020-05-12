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

        self.clf = None
        self.scaler = None

    def train_test_split(self, data, upsample_ratio=1.0, split=0.8, return_n=False):
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
        data_train = np.tile(data_train, (upsample_ratio, 1))
        data_test = data[test_idxs,:]
        n_train *= upsample_ratio

        if return_n:
            return data_train, data_test, int(n_train), int(n_test)
        else:
            return data_train, data_test

    def fit_model(self, pl_data, org_data, upsample=True, downsample=False):
        """
        This function takes the set of plasticizers and a set of molecules
        unlikely to be plasticizers and builds a logistic regression model
        """
        n_pl_samples = pl_data.shape[0]
        n_org_samples = org_data.shape[0]
        if upsample:
            upsample_ratio = int(n_org_samples / n_pl_samples)
        else:
            upsample_ratio = 1.0
        n_features = pl_data.shape[1]
        assert n_features == org_data.shape[1], "Both datasets should have the \
                                                 the same number of features"

        # Save models to average and accuracy data
        pl_train_accs = []
        pl_test_accs = []
        org_train_accs = []
        org_test_accs = []

        # Copy input arrays so we can modify them multiple times
        org_data_init = org_data.copy()
        pl_data_init = pl_data.copy()


        # Randomly select subset of negative dataset equal to number of plasts
        pl_data = np.concatenate([pl_data_init, np.ones((n_pl_samples,1))], axis=1)
        org_select = np.random.choice(np.arange(n_org_samples), size=n_org_samples)
        org_data = org_data_init[org_select,:]
        org_data = np.concatenate([org_data, np.zeros((n_org_samples,1))], axis=1)

        # Train/test split
        pl_train, pl_test, n_train_pl, n_test_pl = self.train_test_split(pl_data, upsample_ratio=upsample_ratio, return_n=True)
        org_train, org_test, n_train_org, n_test_org = self.train_test_split(org_data, return_n=True)
        train_data = np.concatenate([pl_train, org_train])
        test_data = np.concatenate([pl_test, org_test])
        np.random.shuffle(train_data)
        np.random.shuffle(test_data)
        X_train = train_data[:,:-1]
        y_train = train_data[:,-1]
        X_test = test_data[:,:-1]
        y_test = test_data[:,-1]

        # Scale and fit regressor
        self.scaler = MinMaxScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        self.clf = LogisticRegression(solver='liblinear', penalty=self.reg_type, \
                                 C=self.reg_param)
        self.clf.fit(X_train, y_train)

        # Calculate accuracies
        pl_train = self.scaler.transform(pl_train[:,:-1])
        pl_test = self.scaler.transform(pl_test[:,:-1])
        org_train = self.scaler.transform(org_train[:,:-1])
        org_test = self.scaler.transform(org_test[:,:-1])
        pl_train_accs.append(self.clf.score(pl_train, np.ones((n_train_pl,1))))
        pl_test_accs.append(self.clf.score(pl_test, np.ones((n_test_pl,1))))
        org_train_accs.append(self.clf.score(org_train, np.zeros((n_train_org,1))))
        org_test_accs.append(self.clf.score(org_test, np.zeros((n_test_org,1))))

        # Save model and data
        self.pl_train_acc = np.mean(pl_train_accs)
        self.pl_test_acc = np.mean(pl_test_accs)
        self.org_train_acc = np.mean(org_train_accs)
        self.org_test_acc = np.mean(org_test_accs)

    def predict(self, X, type='prob', class_id='pos'):
        assert self.clf is not None, "ERROR: You must train the model before predicting"
        assert self.clf.coef_.shape[1] == X.shape[1], "ERROR: Input data must have same number of features as training sets"

        X = self.scaler.transform(X)

        if class_id == 'pos':
            y = self.clf.predict_proba(X)[:,1]
        elif class_id == 'neg':
            y = self.clf.predict_proba(X)[:,0]
        if type == 'prob':
            return y
        elif type == 'binary':
            return np.where(y > 0.5)[0].shape[0] / X.shape[0]
