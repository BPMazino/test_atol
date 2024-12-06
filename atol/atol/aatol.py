# coding: utf-8
"""
@author: Martin Royer
@modified_by: OpenAI Assistant
@copyright: INRIA 2019-2020
"""

import numpy as np
from sklearn.metrics import pairwise
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans  # Ensure KMeans is imported

def _lapl_contrast(measure, centers, inertias, eps=1e-8):
    return np.exp(-np.sqrt(pairwise.pairwise_distances(measure, Y=centers) / (inertias + eps)))

def _gaus_contrast(measure, centers, inertias, eps=1e-8):
    return np.exp(-pairwise.pairwise_distances(measure, Y=centers) / (inertias + eps))

def _indicator_contrast(measure, centers, inertias, eps=1e-8):
    pair_dist = pairwise.pairwise_distances(measure, Y=centers)
    flat_circ = (pair_dist < (inertias + eps)).astype(int)
    robe_curve = np.clip((2 - pair_dist / (inertias + eps)), a_min=0, a_max=None)
    return flat_circ + robe_curve

def _cloud_weighting(measure):
    return np.ones(shape=measure.shape[0])

def _iidproba_weighting(measure):
    return np.ones(shape=measure.shape[0]) / measure.shape[0]

class Atol(BaseEstimator, TransformerMixin):
    """
    This class allows vectorization of measures (e.g., point clouds, persistence diagrams) after a quantization step.

    ATOL paper: https://arxiv.org/abs/1909.13472
    """

    def __init__(self, quantiser, weighting_method="cloud", contrast="gaus", n_codebooks=1):
        """
        Constructor for the Atol measure vectorization class.

        Parameters:
            quantiser (Object): An object with `fit` (consistent with sklearn API) and `cluster_centers_` and `n_clusters`
                attributes (e.g., KMeans()). This object will be cloned for each codebook.
            weighting_method (str): Method for weighting the measure points ("cloud" or "iidproba").
                Default is "cloud" (uniform weighting).
            contrast (str): Contrast function to use ("gaus", "lapl", or "indi").
                Default is "gaus" (Gaussian contrast function).
            n_codebooks (int): Number of codebooks to create for adaptive vectorization.
                Default is 1 (standard ATOL without adaptation).
        """
        self.quantiser = quantiser
        self.contrast = {
            "gaus": _gaus_contrast,
            "lapl": _lapl_contrast,
            "indi": _indicator_contrast,
        }.get(contrast, _gaus_contrast)
        self.weighting_method = {
            "cloud": _cloud_weighting,
            "iidproba": _iidproba_weighting,
        }.get(weighting_method, _cloud_weighting)
        self.n_codebooks = n_codebooks
        # Initialize containers for centers and inertias
        self.centers_list = []
        self.inertias_list = []
        self.cluster_model = None

    def fit(self, X, y=None, sample_weight=None):
        """
        Calibration step: fit centers to the sample measures and derive inertias between centers.

        Parameters:
            X (list of numpy arrays): Input measures in R^d (each measure can have different number of points).
            y: Ignored, present for API consistency by convention.
            sample_weight (list of numpy arrays): Weights for each measure point in X, optional.
                If None, the object's weighting_method will be used.

        Returns:
            self
        """
        if not hasattr(self.quantiser, 'fit'):
            raise TypeError("quantiser %s has no `fit` attribute." % (self.quantiser))

        # Flatten the list of measures into a single array
        measures_concat = np.concatenate(X)
        if sample_weight is None:
            sample_weight = np.concatenate([self.weighting_method(measure) for measure in X])

        # Cluster the measures into n_codebooks clusters
        self.cluster_model = KMeans(n_clusters=self.n_codebooks)
        measure_reps = [np.mean(measure, axis=0) for measure in X]  # Represent each measure by its centroid
        measure_reps = np.vstack(measure_reps)
        self.cluster_model.fit(measure_reps)

        # Assign each measure to a cluster
        measure_labels = self.cluster_model.labels_

        # Fit a quantiser for each cluster to create codebooks
        for cluster_idx in range(self.n_codebooks):
            # Get measures belonging to the current cluster
            indices = np.where(measure_labels == cluster_idx)[0]
            if len(indices) == 0:
                # If no measures are assigned to this cluster, skip it
                continue
            X_cluster = [X[i] for i in indices]
            if sample_weight is None:
                sample_weight_cluster = np.concatenate([self.weighting_method(measure) for measure in X_cluster])
            else:
                sample_weight_cluster = np.concatenate([sample_weight[i] for i in indices])

            measures_concat_cluster = np.concatenate(X_cluster)
            quantiser_cluster = self._clone_quantiser()
            quantiser_cluster.fit(X=measures_concat_cluster, sample_weight=sample_weight_cluster)
            centers = quantiser_cluster.cluster_centers_

            # Compute inertias for the cluster-specific codebook
            dist_centers = pairwise.pairwise_distances(centers)
            np.fill_diagonal(dist_centers, np.inf)
            inertias = np.min(dist_centers, axis=0) / 2

            self.centers_list.append(centers)
            self.inertias_list.append(inertias)

        return self

    def _clone_quantiser(self):
        """
        Helper method to clone the quantiser for use in cluster-specific codebooks.

        Returns:
            A new instance of the quantiser with the same parameters.
        """
        from sklearn.base import clone
        return clone(self.quantiser)

    def __call__(self, measure, sample_weight=None):
        """
        Apply measure vectorization on a single measure.

        Parameters:
            measure (numpy array): Input measure in R^d.

        Returns:
            numpy array: Feature vector representation of the measure.
        """
        if sample_weight is None:
            sample_weight = self.weighting_method(measure)

        # Determine the relevance of each codebook to the measure
        measure_rep = np.mean(measure, axis=0).reshape(1, -1)
        codebook_relevances = self.cluster_model.predict(measure_rep)

        feature_vectors = []
        for idx, (centers, inertias) in enumerate(zip(self.centers_list, self.inertias_list)):
            features = np.sum(
                sample_weight * self.contrast(measure, centers, inertias.T).T,
                axis=1
            )
            # Optionally, you can weight the features by the relevance
            # For now, we can simply concatenate the features
            feature_vectors.append(features)

        return np.concatenate(feature_vectors)

    def transform(self, X, sample_weight=None):
        """
        Apply measure vectorization on a list of measures.

        Parameters:
            X (list of numpy arrays): Input measures in R^d.
            sample_weight (list of numpy arrays): Weights for each measure point in X, optional.
                If None, the object's weighting_method will be used.

        Returns:
            numpy array: Matrix of feature vectors for each measure.
        """
        if sample_weight is None:
            sample_weight = [self.weighting_method(measure) for measure in X]
        return np.stack([self(measure, weight) for measure, weight in zip(X, sample_weight)])

    def fit_transform(self, X, y=None, sample_weight=None):
        """
        Fit the model and transform the data in one step.

        Parameters:
            X (list of numpy arrays): Input measures in R^d.
            y: Ignored, present for API consistency by convention.
            sample_weight (list of numpy arrays): Weights for each measure point in X, optional.

        Returns:
            numpy array: Matrix of feature vectors for each measure.
        """
        self.fit(X, sample_weight=sample_weight)
        return self.transform(X, sample_weight=sample_weight)
