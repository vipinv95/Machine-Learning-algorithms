import numpy as np


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids or means, membership, number_of_updates )
            Note: Number of iterations is the number of time you update means other than initialization
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership untill convergence or untill you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        k_means = x[list(map(int,np.around(np.random.rand(self.n_cluster)*N)))]
        J = None
        for iter in range(self.max_iter):
            R = []
            for xi in range(len(x)):
                argmin_k = np.argmin(np.sum(np.square(k_means-x[xi]),axis=1))
                R.append(argmin_k)
            J_new = (1/N)*np.sum(np.sum(np.square(k_means[R]-x),axis=1))
            if J is not None:
                if abs(J-J_new) <= self.e:
                    R = np.array(R)
                    return (k_means,R,iter+1)
            J = J_new
            for k in range(self.n_cluster):
                R_arr = np.array(R)
                k_means[k] = np.sum(x[np.where(R_arr==k)],axis=0)/R.count(k) if R.count(k) > 0 else k_means[k]
        R = np.array(R)
        return (k_means,R,self.max_iter)


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - N size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering
                self.centroid_labels : labels of each centroid obtained by
                    majority voting
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape

        # - assign means to centroids
        # - assign labels to centroid_labels

        k_means = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)
        centroids, membership, i = k_means.fit(x)
        centroid_labels = []
        for ci in range(len(centroids)):
            lbl_list = list(y[np.where(membership==ci)])
            if len(np.where(membership==ci)) > 0:
                centroid_labels.append(max(set(lbl_list), key=lbl_list.count))
            else:
                centroid_labels.append(0)
        centroid_labels = np.array(centroid_labels)


        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a vector of shape {}'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape

        # - return labels

        lbls = []
        for xi in range(len(x)):
            argmin_k_lbl = self.centroid_labels[int(np.argmin(np.sum(np.square(self.centroids-x[xi]),axis=1)))]
            lbls.append(argmin_k_lbl)
        return np.array(lbls)

