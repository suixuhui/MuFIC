from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist, euclidean
import numpy as np
from tqdm import tqdm

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

class Inverse_JumpsMethod(object):

    def __init__(self, data, k_list):
        self.data = data
        self.cluster_list = list(k_list)
        # dimension of 'data'; data.shape[0] would be size of 'data'
        self.p = len(data[0])

    def Distortions(self, random_state=0):
        # cluster_range = range(1, len(cluster_list) + 1)
        cluster_range = range(0, len(self.cluster_list) + 1)
        """ returns a vector of calculated distortions for each cluster number.
            If the number of clusters is 0, distortion is 0 (SJ, p. 2)
            'cluster_range' -- range of numbers of clusters for KMeans;
            'data' -- n by p array """
        # dummy vector for Distortions
        self.distortions = np.repeat(0, len(cluster_range)).astype(np.float32)
        self.K_list = []
        # for each k in cluster range implement
        for i in cluster_range:
            if i == cluster_range[-1]:
                parameter = self.cluster_list[-1] + (self.cluster_list[1] - self.cluster_list[0])
            else:
                parameter = self.cluster_list[i]
            KM = KMeans(n_clusters=parameter, random_state=random_state)
            KM.fit(self.data)
            centers = KM.cluster_centers_  # calculate centers of suggested k clusters

            K = parameter
            self.K_list.append(K)
            # since we need to calculate the mean of mins create dummy vec
            for_mean = np.repeat(0, len(self.data)).astype(np.float32)

            # for each observation (i) in data implement
            for j in range(len(self.data)):
                # dummy for vec of distances between i-th obs and k-center
                dists = np.repeat(0, K).astype(np.float32)

                # for each cluster in KMean clusters implement
                for cluster in range(K):
                    euclidean_d = euclidean(normalize(self.data[j]), normalize(centers[cluster]))
                    dists[cluster] = euclidean_d * euclidean_d / 2
                for_mean[j] = min(dists)

            # take the mean for mins for each observation
            self.distortions[i] = np.mean(for_mean) / self.p
        return self.distortions

    def Jumps(self, distortions=None):
        self.distortions = distortions  # change
        """ returns a vector of jumps for each cluster """

        self.jumps = []
        self.jumps += [np.log(self.distortions[k]) - np.log(self.distortions[k - 1]) \
                       for k in range(1, len(self.distortions))]  # argmax

        # calculate recommended number of clusters
        recommended_index = int(np.argmax(np.array(self.jumps)))

        if recommended_index > 0:
            self.recommended_cluster_number = self.cluster_list[recommended_index-1]
        else:
            self.recommended_cluster_number = int(self.cluster_list[0] - (self.cluster_list[1] - self.cluster_list[0]))
        return self.jumps