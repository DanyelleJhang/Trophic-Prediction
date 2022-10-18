#!/usr/bin/env python
# coding: utf-8
# 2022/1/16 add BernoulliRBM 
import sys, os
call_module_for_bash = os.path.dirname(str(os.getcwd())) + "/Python"
sys.path.insert(1, call_module_for_bash)
from utility.usage_measurement import hardware_usage
#################### Unsupervise Method ####################
########## Matrix Decompositio ##########
# decomposition.DictionaryLearning
# decomposition.FactorAnalysis
# decomposition.FastICA
# decomposition.IncrementalPCA
# decomposition.KernelPCA
# decomposition.LatentDirichletAllocation
# decomposition.MiniBatchDictionaryLearning
# decomposition.MiniBatchSparsePCA
# decomposition.NMF
# decomposition.PCA
# decomposition.SparsePCA 
# decomposition.SparseCoder
# decomposition.TruncatedSVD



########## Manifold Learning ########## 
# manifold.Isomap(*[, n_neighbors, …])
# manifold.LocallyLinearEmbedding(*[, …])
# manifold.MDS([n_components, metric, n_init, …])
# manifold.SpectralEmbedding([n_components, …])
# manifold.TSNE([n_components, perplexity, …])
# manifold.locally_linear_embedding(X, *, …)
# manifold.smacof(dissimilarities, *[, …])
# manifold.spectral_embedding(adjacency, *[, …])
# manifold.trustworthiness(X, X_embedded, *[, …])



########### DO NOT CONSIDER TEMPORARILY ###########
###### manifold.smacof(dissimilarities, *[, …])
#from sklearn.manifold import smacof
###### manifold.spectral_embedding(adjacency, *[, …])
#from sklearn.manifold import spectral_embedding
###### manifold.trustworthiness(X, X_embedded, *[, …])
#from sklearn.manifold import trustworthiness


################ Common parameters Input ################
# n_components ; default=2
# random_state ; default=0
# n_neighbors ; default=5
# fit_algorithm: {'lars', 'cd'}; default='lars'
# transform_algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'}; default='omp'
# eigen_solver: {'auto', 'arpack', 'dense'}, default='auto'

##### decomposition.DictionaryLearning #####
@hardware_usage
def DictionaryLearning_fun(input_high_dimention_features,
						   input_n_components,
						   input_fit_algorithm,input_transform_algorithm,input_random_state):
	from sklearn.decomposition import DictionaryLearning
	dict_learner = DictionaryLearning(
		n_components=input_n_components,
		fit_algorithm = input_fit_algorithm,
		transform_algorithm=input_transform_algorithm,
		random_state=input_random_state
	)
	feature_transformed = dict_learner.fit_transform(input_high_dimention_features)
	return feature_transformed


###### decomposition.FactorAnalysis ######
# svd_method = ['lapack','randomized'], default='randomized'
# rotation = ['varimax','quartimax',eval('None')] , default=eval('None')
@hardware_usage
def FactorAnalysis_fun(input_high_dimention_features,
					   input_n_components,
					   input_rotation,input_svd_method,input_random_state):
	from sklearn.decomposition import FactorAnalysis
	transformer = FactorAnalysis(
		n_components=input_n_components,
		rotation =input_rotation,
		svd_method=input_svd_method,
		random_state=input_random_state
	)
	feature_transformed = transformer.fit_transform(input_high_dimention_features)
	return feature_transformed


###### decomposition.FastICA ######
# algorithm = ['parallel','deflation'], default='parallel'
# fun = ['logcosh','exp','cube'] or callable, default='logcosh'
@hardware_usage
def FastICA_fun(input_high_dimention_features,
				input_n_components,
				input_algorithm,input_fun,input_random_state):
	from sklearn.decomposition import FastICA

	transformer = FastICA(
		n_components=input_n_components,
		algorithm = input_algorithm,
		fun=input_fun,
		random_state=input_random_state
	)
	feature_transformed = transformer.fit_transform(input_high_dimention_features)
	return feature_transformed


###### decomposition.IncrementalPCA ######
@hardware_usage
def IncrementalPCA_fun(input_high_dimention_features,
					   input_n_components):
	from sklearn.decomposition import IncrementalPCA

	transformer = IncrementalPCA(
		n_components=input_n_components
		)
	feature_transformed = transformer.fit_transform(input_high_dimention_features)
	return feature_transformed


###### decomposition.KernelPCA ######
# kernel = ['linear','poly','rbf','sigmoid','cosine','precomputed'], default='linear'

@hardware_usage
def KernelPCA_fun(input_high_dimention_features,
				  input_n_components,
				  input_kernel,input_eigen_solver,input_random_state):
	from sklearn.decomposition import KernelPCA

	transformer = KernelPCA(
		n_components=input_n_components,
		kernel = input_kernel,
		eigen_solver=input_eigen_solver,
		random_state=input_random_state
	)
	feature_transformed = transformer.fit_transform(input_high_dimention_features)
	return feature_transformed


###### decomposition.LatentDirichletAllocation ######
# learning_method= ['batch','online'], default=’batch’
@hardware_usage
def LatentDirichletAllocation_fun(input_high_dimention_features,
				  input_n_components,
				  input_learning_method,input_random_state):
	from sklearn.decomposition import LatentDirichletAllocation

	transformer = LatentDirichletAllocation(
		n_components=input_n_components,
		learning_method = input_learning_method,
		random_state=input_random_state
	)
	feature_transformed = transformer.fit_transform(input_high_dimention_features)
	return feature_transformed


###### decomposition.MiniBatchDictionaryLearning ######
# [lars, cd], default=lars
@hardware_usage
def MiniBatchDictionaryLearning_fun(input_high_dimention_features,
				  input_n_components,
				  input_fit_algorithm,input_transform_algorithm,
				  input_random_state):
	from sklearn.decomposition import MiniBatchDictionaryLearning

	transformer = MiniBatchDictionaryLearning(
		n_components=input_n_components,
		fit_algorithm = input_fit_algorithm,
		transform_algorithm= input_transform_algorithm,
		random_state=input_random_state
	)
	feature_transformed = transformer.fit_transform(input_high_dimention_features)
	return feature_transformed


###### decomposition.MiniBatchSparsePCA ######
# method = ['lars','lars'], default='lars'

@hardware_usage
def MiniBatchSparsePCA_fun(input_high_dimention_features,
				  input_n_components,
				  input_method,input_random_state):
	from sklearn.decomposition import MiniBatchSparsePCA
	
	transformer = MiniBatchSparsePCA(
		n_components=input_n_components,
		method = input_method,
		random_state=input_random_state
	)
	
	feature_transformed = transformer.fit_transform(input_high_dimention_features)
	return feature_transformed


###### decomposition.NMF ######
# init: {‘random’, ‘nndsvd’, ‘nndsvda’, ‘nndsvdar’, ‘custom’}, default=None
# solver: {‘cd’, ‘mu’}, default=’cd’
# beta_loss: float or {‘frobenius’, ‘kullback-leibler’, ‘itakura-saito’}, default=’frobenius’

@hardware_usage
def NMF_fun(input_high_dimention_features,
				  input_n_components,
				  input_init,
				  input_solver,
				  input_beta_loss,
				  input_random_state):
	from sklearn.decomposition import NMF
	
	transformer = NMF(
		n_components=input_n_components,
		init = input_init,
		solver = input_solver,
		beta_loss = input_beta_loss,
		random_state=input_random_state
	)
	
	feature_transformed = transformer.fit_transform(input_high_dimention_features)
	return feature_transformed


###### decomposition.PCA
# svd_solver: {‘auto’, ‘full’, ‘arpack’, ‘randomized’}

@hardware_usage
def PCA_fun(input_high_dimention_features,
				  input_n_components,
				  input_svd_solver,
				  input_random_state):
	from sklearn.decomposition import PCA
	
	transformer = PCA(
		n_components=input_n_components,
		svd_solver = input_svd_solver,
		random_state=input_random_state
	)
	
	feature_transformed = transformer.fit_transform(input_high_dimention_features)
	return feature_transformed


###### decomposition.SparsePCA ######
# method: {‘lars’, ‘cd’}

@hardware_usage
def SparsePCA_fun(input_high_dimention_features,
				  input_n_components,
				  input_method,input_random_state):
	from sklearn.decomposition import SparsePCA
	
	transformer = SparsePCA(
		n_components=input_n_components,
		method = input_method,
		random_state=input_random_state
	)
	
	feature_transformed = transformer.fit_transform(input_high_dimention_features)
	return feature_transformed

###### decomposition.TruncatedSVD ######
# algorithm: {‘arpack’, ‘randomized’}, default=’randomized’

@hardware_usage
def TruncatedSVD_fun(input_high_dimention_features,
				  input_n_components,
				  input_algorithm,input_random_state):
	from sklearn.decomposition import TruncatedSVD
	
	transformer = TruncatedSVD(
		n_components=input_n_components,
		algorithm = input_algorithm,
		random_state=input_random_state
	)
	
	feature_transformed = transformer.fit_transform(input_high_dimention_features)
	return feature_transformed


###### Isomap ######
# path_method {‘auto’, ‘FW’, ‘D’}, default=’auto’
# neighbors_algorithm{‘auto’, ‘brute’, ‘kd_tree’, ‘ball_tree’}, default=’auto’

@hardware_usage
def Isomap_fun(input_high_dimention_features,
				  input_n_components,
				  input_n_neighbors,
				  input_eigen_solver,
				  input_path_method,
				  input_neighbors_algorithm):
	from sklearn.manifold import Isomap
	
	transformer = Isomap(
		n_components=input_n_components,
		n_neighbors = input_n_neighbors,
		eigen_solver = input_eigen_solver,
		path_method = input_path_method,
		neighbors_algorithm = input_neighbors_algorithm
	)
	
	feature_transformed = transformer.fit_transform(input_high_dimention_features)
	return feature_transformed


###### LocallyLinearEmbedding ######
# LLE_method = ['standard','hessian','modified','ltsa'], default=’standard’
@hardware_usage
def LocallyLinearEmbedding_fun(input_high_dimention_features,
				  input_n_components,
				  input_n_neighbors,
				  input_eigen_solver,
				  input_method,
				  input_random_state):
	from sklearn.manifold import LocallyLinearEmbedding
	
	transformer = LocallyLinearEmbedding(
		n_components=input_n_components,
		n_neighbors = input_n_neighbors,
		eigen_solver = input_eigen_solver,
		method = input_method,
		random_state = input_random_state
	)
	
	feature_transformed = transformer.fit_transform(input_high_dimention_features)

	return feature_transformed


###### manifold.MDS([n_components, metric, n_init, …])
# dissimilarity: {‘euclidean’, ‘precomputed’}, default=’euclidean’
@hardware_usage
def MDS_fun(input_high_dimention_features,
				  input_n_components,
				  input_dissimilarity):
	from sklearn.manifold import MDS
	
	transformer = MDS(
		n_components = input_n_components,
		dissimilarity = input_dissimilarity
	)
	
	feature_transformed = transformer.fit_transform(input_high_dimention_features)
	return feature_transformed


###### manifold.SpectralEmbedding ######
# SE_affinity; affinity: {‘nearest_neighbors’, ‘rbf’, ‘precomputed’, ‘precomputed_nearest_neighbors’} or callable, default=’nearest_neighbors’
# SE_eigen_solver;  eigen_solver: {‘arpack’, ‘lobpcg’, ‘amg’}
@hardware_usage
def SpectralEmbedding_fun(input_high_dimention_features,
				  input_n_components,
				  input_affinity,
				  input_eigen_solver,
				  input_random_state):
	from sklearn.manifold import SpectralEmbedding
	
	transformer = SpectralEmbedding(
		n_components = input_n_components,
		affinity = input_affinity,
		eigen_solver = input_eigen_solver,
		random_state = input_random_state
	)
	
	feature_transformed = transformer.fit_transform(input_high_dimention_features)
	return feature_transformed

###### manifold.TSNE ###### 
@hardware_usage
def TSNE_fun(input_high_dimention_features,
				  input_n_components,input_random_state):
	from sklearn.manifold import TSNE
	
	transformer = TSNE(
		n_components = input_n_components,
		random_state = input_random_state
	)
	
	feature_transformed = transformer.fit_transform(input_high_dimention_features)
	return feature_transformed

@hardware_usage
def BernoulliRBM_fun(input_high_dimention_features,
	input_n_components,
	input_random_state):
	from sklearn.neural_network import BernoulliRBM
	transformer = BernoulliRBM(
		n_components=input_n_components,
		random_state = input_random_state)
	feature_transformed=transformer.fit_transform(input_high_dimention_features)
	return feature_transformed


def DR_scatter_plot(input_y_array,input_label_name,input_label_position,DR_2D_array,input_color_name,title_name):
    import matplotlib.pyplot as plt
    import numpy as np
    fig = plt.figure()

    not_exist_array = DR_2D_array[np.where(input_y_array[:,input_label_position]==0)[0],:]
    exist_array = DR_2D_array[np.where(input_y_array[:,input_label_position]==1)[0],:]

    plt.scatter(not_exist_array[:,0], not_exist_array[:,1],c="black", alpha=0.1,label="not "+input_label_name)
    plt.scatter(exist_array[:,0], exist_array[:,1],c=input_color_name, alpha=0.1,label=input_label_name)

    plt.legend(bbox_to_anchor=(1.2, 0.7),fontsize="x-large")

    fig.set_figheight(12)
    fig.set_figwidth(16)
    plt.title("PCA plot",fontsize=20) 
    return plt