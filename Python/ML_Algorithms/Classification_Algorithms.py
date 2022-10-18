import numpy as np
# sklean version == 1.1.0
# xgboost version == 1.5.2
# lightgbm == 3.3.2
# numpy == 1.20.3
# scipy == 1.7.1
# pandas == 1.3.4
# Original Data: (687, n_feature>19000)
"""
********************************************* 		The Popular Classifier 		*****************************************************
[SupportVectorMachine]
	C:
		-- the tolerance of error
		-- if C came larger, it indicated lower tolerance
		-- if C came smaller, it might be overfited(hard to generalization)
		-- Regularization parameter. 
		-- The strength of the regularization is inversely proportional to C. 
		-- Must be strictly positive. The penalty is a squared l2 penalty.
	kernel:
		-- Specifies the kernel type to be used in the algorithm. 
			- linear  -> 
			- rbf 	  -> 
			- sigmoid -> 
	gamma:
		-- Kernel coefficient for rbf, poly and sigmoid
			- scale -> 1 / (n_features * X.var())
			- auto  -> 1 / n_features
[LogisticRegression]
	C:
		-- Inverse of regularization strength
		-- must be a positive float
		-- Like in support vector machines, smaller values specify stronger regularization.
	solver:
		-- Algorithm to use in the optimization problem.
			- newton-cg -> For multiclass problems; the penalty chosen: [l2, none]
			- lbfgs		-> For multiclass problems; the penalty chosen: [l2, none]
			- liblinear -> For small datasets; limited to one-versus-rest schemes; the penalty chosen: [l1, l2]
			- sag 		-> For large datasets; multiclass problems; the penalty chosen: [l2, none]
			- saga		-> For large datasets; multiclass problems; the penalty chosen: [elasticnet, l1, l2, none]
[NeuralNetworkClassifier]
	hidden_layer_sizes:
		-- The ith element represents the number of neurons in the ith hidden layer.
	activation:
		-- Activation function for the hidden layer
			- identity -> no-op activation, useful to implement linear bottleneck, returns f(x) = x
			- logistic -> the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
			- tanh -> the hyperbolic tan function, returns f(x) = tanh(x).
			- relu -> the rectified linear unit function, returns f(x) = max(0, x)
	solver:
		-- The solver for weight optimization.
		-- The default solver ‘adam’ works pretty well on relatively large datasets (with thousands of training samples or more) 
		-- in terms of both training time and validation score. 
		-- For small datasets, however, "lbfgs" can converge faster and perform better.
			- lbfgs -> is an optimizer in the family of quasi-Newton methods.
			- sgd -> refers to stochastic gradient descent.
			- adam -> refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba
	alpha:
		-- Strength of the L2 regularization term. 
		-- The L2 regularization term is divided by the sample size when added to the loss.
	learning_rate_init:
		-- The initial learning rate used. 
		-- It controls the step-size in updating the weights. 
		-- Only used when solver="sgd" or "adam".
	early_stopping:
		-- Whether to use early stopping to terminate training when validation score is not improving. 
		-- If set to true, 
		-- it will automatically set aside 10% of training data as validation and terminate training when validation score is not improving by at least tol for n_iter_no_change consecutive epochs. The split is stratified, except in a multilabel setting. 
		-- If early stopping is False, 
		-- then the training stops when the training loss does not improve by more than tol for n_iter_no_change consecutive passes over the training set.
		-- Only effective when solver="sgd" or "adam".



********************************************* KNeighborsClassifier and RadiusNeighborsClassifier ***********************************
[KNeighborsClassifier]
		n_neighbors(int): Number of neighbors to use by default for kneighbors queries.
[RadiusNeighborsClassifier]
		radius(float): Range of parameter space to use by default for radius_neighbors queries
	[common parameters]
		weights: Weight function used in prediction.
			- uniform  ->  uniform weights. All points in each neighborhood are weighted equally.
			- distance ->  weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.
		metric: Metric to use for distance computation.
			- cosine    -> cosine_distances
			- haversine -> haversine_distances
			- l1        -> manhattan_distances
			- l2        -> euclidean_distances



********************************************* 		Tree Based Classifier 		*****************************************************
[DecisionTreeClassifier]
[RandomForestClassifier]
[ExtraTreesClassifier]
	[common parameters]
		n_estimators(int): 
			-- The number of trees in the forest.
		criterion: The function to measure the quality of a split.
			- gini    -> Gini impurity
			- entropy -> Shannon information gain
		splitter: 
			-- The strategy used to choose the split at each node.
			- best   -> choose the best split
			- random -> choose the best random split.
		max_depth(int): 
			-- The maximum depth of the tree.
		max_features(colsample_bynode in XGBClassifier; feature_fraction_bynode in LightGBM): 
			-- The number of features to consider when looking for the best split
			-- constraint values must be in the range (0.0, 1.0]
			-- the features considered at each split will be max(1, int(max_features * n_features_in_)).
			-- is the subsample ratio of columns for each node (split). 
			-- Subsampling occurs once every time a new split is evaluated. 
			-- Columns are subsampled from the set of columns chosen for the current level.
			-- Randomly select a subset of features on each tree node if max_features is smaller than 1.0. 
			-- For example, if you set it to 0.8, it will select 80% of features at each tree node can be used to deal with over-fitting
[AdaBoostClassifier]
	base_estimator:
		-- The base estimator from which the boosted ensemble is built.
	algorithm:
		-- The SAMME.R algorithm typically converges faster than SAMME, achieving a lower test error with fewer boosting iterations.
			- SAMME.R -> SAMME.R real boosting algorithm; base_estimator must support calculation of class probabilities. 
			- SAMME -> then use the SAMME discrete boosting algorithm. 
	learning_rate:
		-- Weight applied to each classifier at each boosting iteration. 
		-- A higher learning rate increases the contribution of each classifier. 
		-- There is a trade-off between the learning_rate and n_estimators parameters. 
[GradientBoostingClassifier]
		loss: 
			-- The loss function to be optimized.
			- log_loss -> refers to binomial and multinomial deviance, the same as used in logistic regression. It is a good choice for classification with probabilistic outputs. 
			- exponential -> gradient boosting recovers the AdaBoost algorithm.
		learning_rate(0.0, inf): 
			-- Learning rate shrinks the contribution of each tree by learning_rate.
			-- There is a trade-off between learning_rate and n_estimators.
[XGBClassifier]
		learning_rate (alias "eta"):
			-- Step size shrinkage used in update to prevents overfitting. 
			-- After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative.
		objective: Specify the learning task and the corresponding learning objective or a custom objective function to be used
			- binary:logistic ->  logistic regression for binary classification, output probability
			- binary:logitraw -> logistic regression for binary classification, output score before logistic transformation
			- binary:hinge -> hinge loss for binary classification. This makes predictions of 0 or 1, rather than producing probabilities.
		booster: Specify which booster to use: gbtree, gblinear or dart
			- gbtree
			- gblinear
			- dart
		verbosity:
			- 0 -> silent
			- 1 -> warning
			- 2 -> info
			- 3 -> debug
[LGBMClassifier]
		boosting_type:
			- gbdt -> traditional Gradient Boosting Decision Tree
			- dart -> Dropouts meet Multiple Additive Regression Trees. 
			- goss -> Gradient-based One-Side Sampling. 
			- rf   -> Random Forest.
		verbosity
			- < 0 -> Fatal
			-   0 -> Error (Warning)
			-   1 -> Info
			- > 1 -> Debug
	[common parameters amongst XGBClassifier and LGBMClassifier]
		gamma(min_split_loss) ; min_gain_to_split(min_split_gain): 
			-- https://blog.csdn.net/weiyongle1996/article/details/78446244
			-- "gamma(min_split_loss; default:0.0)" in XGBClassifier is same as "min_gain_to_split(min_split_gain; default:0.0)" in LGBMClassifier
			-- The range of that parameter is (0, Infinite)
			-- The higher Gamma is, the higher the regularization. 
			-- Default value is 0 (no regularization).
			-- due to lightGBM research didnt concentrate on optimization, such regularization and objective function is based on XGBoost, not mentioned in paper, according to OPEN SOURCE CODE
			-- Minimum loss reduction required to make a further partition on a leaf node of the tree
			-- The larger gamma is, the more conservative the algorithm will be.
			-- A node is split only when the resulting split gives a positive reduction in the loss function. 
			-- Gamma specifies the minimum loss reduction required to make a split.
			-- Makes the algorithm conservative. 
			-- The values can vary depending on the loss function and should be tuned.
			-- Finding a "good" gamma is very dependent on both your data set and the other parameters you are using. 
			-- There is no optimal gamma for a data set, there is only an optimal (real-valued) gamma depending on both the training set + the other parameters you are using.
			-- Gamma is dependent on both the training set and the other parameters you use.
			-- There is no "good Gamma" for any data set alone
			-- It is a pseudo-regularization hyperparameter in gradient boosting.
			-- Mathematically you call “Gamma” the “Lagrangian multiplier” (complexity control).
			-- Gamma values around 20 are extremely high, and should be used only when you are using high depth 
			-- (i.e overfitting blazing fast, not letting the variance/bias tradeoff stabilize for a local optimum) 
			-- or if you want to control the directly the features which are dominating in the data set 
			-- (i.e too strong feature engineering).
			-- constraints: min_gain_to_split >= 0.0
			-- the minimal gain to perform split
			-- can be used to speed up training
			-- This parameter will describe the minimum gain to make a split. 
			-- It can used to control number of useful splits in tree.
			-- To deal with over-fitting



********************************************* 	 Classifier won't be implemented		*****************************************************
[SGDClassifier]
	loss:
		--The loss function to be used.
			- hinge 						-> gives a linear SVM.
			- log_loss 						-> gives logistic regression, a probabilistic classifier.
			- modified_huber  				-> is another smooth loss that brings tolerance to outliers as well as probability estimates.
			- squared_hinge 				-> is like hinge but is quadratically penalized.
			- perceptron 					-> is the linear loss used by the perceptron algorithm.
			- squared_error 				-> are designed for regression but can be useful in classification as well
			- huber 						-> are designed for regression but can be useful in classification as well
			- epsilon_insensitive 			-> are designed for regression but can be useful in classification as well
			- squared_epsilon_insensitive 	-> are designed for regression but can be useful in classification as well
	penalty:
		-- The penalty (aka regularization term) to be used.
			- l2 		 -> which is the standard regularizer for linear SVM models.
			- l1 		 -> might bring sparsity to the model (feature selection)
			- elasticnet -> might bring sparsity to the model (feature selection)
[BernoulliNB]
[GaussianNB]
[ComplementNB]
"""
All_Name_list=["RandomForestClassifier",
		   "AdaBoostClassifier",
		   "BernoulliNB",
		   "DecisionTreeClassifier",
		   "ExtraTreesClassifier",
		   "GaussianNB",
		   "GradientBoostingClassifier",
		   "KNeighborsClassifier",
		   "LogisticRegression",
		   "XGBClassifier",
		   "LGBMClassifier",
		   "NeuralNetworkClassifier",
		   "SGDClassifier",
		   "ComplementNB",
		   "RadiusNeighborsClassifier",
		   "SupportVectorMachine"]
def hyper_parameters(Learning_Type,n_iter=30,seed_num=1):
	n_iteration = n_iter
	from sklearn.naive_bayes import BernoulliNB
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.naive_bayes import GaussianNB
	from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
	from sklearn.linear_model import LogisticRegression
	from sklearn.svm import SVC
	from sklearn.neural_network import MLPClassifier
	np.random.seed(seed_num)
	All_Name_list=["RandomForestClassifier",
			   "AdaBoostClassifier",
			   "BernoulliNB",
			   "DecisionTreeClassifier",
			   "ExtraTreesClassifier",
			   "GaussianNB",
			   "GradientBoostingClassifier",
			   "KNeighborsClassifier",
			   "LogisticRegression",
			   "XGBClassifier",
			   "LGBMClassifier",
			   "NeuralNetworkClassifier",
			   "SGDClassifier",
			   "ComplementNB",
			   "RadiusNeighborsClassifier",
			   "SupportVectorMachine"]
	Tuning_Parameters={
	"GridSearch_parameters":{"RandomForestClassifier":{"n_estimators ":[i for  i in np.arange(100,1000,100)],
														"max_depth":[i for  i in np.arange(5,100,5)],
														"max_features":[i for  i in np.arange(0.01, 1., 0.05)],
														"criterion": ["gini","entropy"]},
							"AdaBoostClassifier":{"n_estimators ":[i for  i in np.arange(100,1000,100)],
												  "base_estimator":[DecisionTreeClassifier(),
																	KNeighborsClassifier(n_jobs=15),
																	LogisticRegression(n_jobs=15),
																	SVC(probability=True),
																	RadiusNeighborsClassifier(n_jobs=15)],
													"algorithm": ["SAMME","SAMME.R"],
													"learning_rate": np.arange(0.00001, 0.1, 0.005)},
							"BernoulliNB":{"alpha":np.arange(0.0, 1.0, 0.05),
											"binarize":np.arange(90.0, 95.0, 0.02)},
							"DecisionTreeClassifier":{"criterion":["gini","entropy"],
													  "splitter":["best","random"],
													  "max_depth":[i for  i in np.arange(5,100,5)],
													  "max_features":[i for  i in np.arange(0.01, 1., 0.05)]},
							"ExtraTreesClassifier":{"n_estimators ":[i for  i in np.arange(100,1000,100)],
													"max_depth":[i for  i in np.arange(5,100,5)],
													"max_features":[i for  i in np.arange(0.01, 1., 0.05)],
													"criterion":["gini","entropy"]},
							"GaussianNB":{"var_smoothing":np.arange(1e-9,1e+2,0.001)},
							"GradientBoostingClassifier":{"n_estimators ":[i for  i in np.arange(100,1000,100)],
														  "max_depth":[i for  i in np.arange(5,100,5)],
														  "max_features":[i for  i in np.arange(0.01, 1., 0.05)],
														  "learning_rate":np.arange(0.00001, 0.1, 0.005)},
							"KNeighborsClassifier":{"n_neighbors":[i for  i in np.arange(3,100,8)], # guild mode upto 100 instance; pick odd number avoid even issue
													"weights":["uniform","distance"],
													"metric":["cosine","haversine","l1","l2"]},
							"LogisticRegression":{"C":np.arange(10**-5, 10**1, 0.05),
												  "solver":["newton-cg","lbfgs","liblinear","sag","saga"]},
							"XGBClassifier":{"n_estimators ":[i for  i in np.arange(100,1000,100)],
											 "max_depth":[i for  i in np.arange(5,100,5)],
											 "colsample_bynode":[i for  i in np.arange(0.01, 1., 0.05)],
											 "learning_rate":np.arange(0.00001, 0.1, 0.005),
											 "verbosity":[0],
											 "objective":["binary:logistic","binary:logitraw","binary:hinge"],
											 "booster":["gbtree","gblinear","dart"],
											 "gamma":np.append([0.0], np.arange(0.00001, 10.0, 0.05))},
							"LGBMClassifier": {"n_estimators": [i for  i in np.arange(100,1000,100)],
											   "max_depth":[i for  i in np.arange(5,100,5)],
											   "feature_fraction_bynode":[i for  i in np.arange(0.01, 1., 0.05)],
											   "boosting_type":["gbdt","dart", "goss", "rf"],
											   "learning_rate":np.arange(0.00001, 0.1, 0.005),
											   "min_gain_to_split":np.append([0.0], np.arange(0.00001, 10.0, 0.05)),
											   "verbosity":[-1]},
							"NeuralNetworkClassifier":{"hidden_layer_sizes":[tuple(np.random.randint(low = 2, high=100,size=np.random.randint(low = 2,high=100))) for i in range(n_iteration)],
													   "activation": ["identity", "logistic",'tanh','relu'],
													   "solver": ["lbfgs", "sgd",'adam'],
													   "alpha": np.arange(0.0, 1.0, 0.05), # alpha = 0:Ridg; alpha = 1: lasso; 0 ≤ alpha ≤ 1:elastic net
													   "learning_rate_init": np.arange(0.00001, 0.1, 0.005),
													   "early_stopping": [True,False]},
							"SGDClassifier":{"loss":['hinge','log_loss','modified_huber','huber'],
											"penalty":["l2","l1","elasticnet"]},
							"ComplementNB":{"alpha":np.arange(0.0, 1.0, 0.05),
											"norm":[True,False]},
							"RadiusNeighborsClassifier":{"radius":np.arange(0.1, 15.0, 0.1),
														 "weights":["uniform","distance"],
														 "metric":["cosine","haversine","l1","l2"],
														 "outlier_label":[0,"most_frequent"]},
							"SupportVectorMachine":{"C":np.arange(10**-5, 10**1, 0.05),
													"kernel":["linear","rbf","sigmoid"],
													"gamma":["scale","auto"],
													"probability":[True]}

							},
	"Randomized_parameters":{"RandomForestClassifier":{"n_estimators":np.random.randint(low=100, high=1000, size=n_iteration, dtype=np.int64),
													   "max_depth": np.random.randint(low=5, high=100, size=n_iteration, dtype=np.int64),
													   "max_features": np.random.uniform(0.01, 1.0, size=n_iteration),
													   "criterion" : ["gini","entropy"]},
							"AdaBoostClassifier":{"n_estimators":np.random.randint(low=100, high=1000, size=n_iteration, dtype=np.int64),
												  "base_estimator":[DecisionTreeClassifier(),
																	KNeighborsClassifier(n_jobs=15),
																	LogisticRegression(n_jobs=15),
																	SVC(probability=True),
																	RadiusNeighborsClassifier(n_jobs=15)],
												 "algorithm": ["SAMME","SAMME.R"],
												 "learning_rate": np.random.uniform(0.00001, 0.1, size=n_iteration)},
							"BernoulliNB":{"alpha":np.random.uniform(0.0, 1.0,size=n_iteration),
										   "binarize":np.random.uniform(90.0,95.0,size=n_iteration)},
							"DecisionTreeClassifier":{"criterion":["gini","entropy"],
													"splitter":["best","random"],
													"max_depth":np.random.randint(low=2, high=100, size=n_iteration, dtype=np.int64),
													"max_features":np.random.uniform(0.01, 1.0, size=n_iteration)},
							"ExtraTreesClassifier":{"n_estimators":np.random.randint(low=100, high=1000, size=n_iteration, dtype=np.int64),
													"max_depth": np.random.randint(low=5, high=100, size=n_iteration, dtype=np.int64),
													"max_features": np.random.uniform(0.01, 1.0, size=n_iteration),
													"criterion":["gini","entropy"]},
							"GaussianNB":{"var_smoothing":np.random.uniform(1e-9,1e+2,size=n_iteration)},
							"GradientBoostingClassifier":{"n_estimators":np.random.randint(low=100, high=1000, size=n_iteration, dtype=np.int64),
														  "max_depth": np.random.randint(low=5, high=100, size=n_iteration, dtype=np.int64),
														  "max_features": np.random.uniform(0.01, 1.0, size=n_iteration),
														  "learning_rate":np.random.uniform(0.00001, 0.1, size=n_iteration)},
							"KNeighborsClassifier":{"n_neighbors":list(filter(lambda x: x if (x%2==1) else None ,np.random.randint(low=3, high=100, size=n_iteration, dtype=np.int64))), # guild mode upto 100 instance; pick odd number avoid even issue
													"weights":["uniform","distance"],
													"metric":["cosine","haversine","l1","l2"]},
							"LogisticRegression":{"C":np.random.uniform(10**-5, 10**1, size=n_iteration),
												  "solver":["newton-cg","lbfgs","liblinear","sag","saga"]},
							"XGBClassifier":{"n_estimators":np.random.randint(low=100, high=1000, size=n_iteration, dtype=np.int64),
											 "max_depth":np.random.randint(low=5, high=100, size=n_iteration, dtype=np.int64),
											 "colsample_bynode":np.random.uniform(0.01, 1.0, size=n_iteration),
											 "learning_rate":np.random.uniform(0.00001, 0.1, size=n_iteration),
											 "verbosity":[0],
											 "objective":["binary:logistic","binary:logitraw","binary:hinge"],
											 "booster":["gbtree","gblinear","dart"],
											 "gamma":np.append([0.0], np.random.uniform(0.00001, 10.0, size=n_iteration))},
							"LGBMClassifier": {"n_estimators":np.random.randint(low=100, high=1000, size=n_iteration, dtype=np.int64),
											   "max_depth":np.random.randint(low=5, high=100, size=n_iteration, dtype=np.int64),
											   "feature_fraction_bynode":np.random.uniform(0.01, 1.0, size=n_iteration),
											   "boosting_type":["gbdt","dart", "goss", "rf"],
											   "learning_rate":np.random.uniform(0.00001, 0.1, size=n_iteration),
											   "min_gain_to_split":np.append([0.0], np.random.uniform(0.00001, 10.0, size=n_iteration)),
											   "verbosity":[-1]},
							"NeuralNetworkClassifier":{"hidden_layer_sizes":[tuple(np.random.randint(low = 2, high=100,size=np.random.randint(low = 2,high=100))) for i in range(n_iteration)],
														"activation": ["identity", "logistic",'tanh','relu'],
														"solver": ["lbfgs", "sgd",'adam'],
														"alpha": np.append([1.0,0.0], np.random.uniform(0.0001, 0.9, size=n_iteration)), # alpha = 0:Ridg; alpha = 1: lasso; 0 ≤ alpha ≤ 1:elastic net
														"learning_rate_init": np.random.uniform(0.00001, 0.1, size=n_iteration),
														"early_stopping": [True,False]},
							"SGDClassifier":{"loss":['hinge','log_loss','modified_huber','huber'],
											 "penalty":["l2","l1"] # alpha = 0:Ridg; alpha = 1: lasso; 0 ≤ alpha ≤ 1:elastic net
											 },
							"ComplementNB":{"alpha":np.append([1.0], np.random.uniform(0.0001, 0.1, size=10)),
											"norm":[True,False]},
							"RadiusNeighborsClassifier":{"radius":np.random.uniform(0.1, 15.0, size=n_iteration),
														"weights":["uniform","distance"],
														"metric":["cosine","haversine","l1","l2"],
														"outlier_label":[0,"most_frequent"]},
							"SupportVectorMachine":{"C":np.random.uniform(10**-5, 10**1, size=n_iteration),
													"kernel":["linear","rbf","sigmoid"],
													"gamma":["scale","auto"],
													"probability":[True]
													}
							}
					}

	
	if Learning_Type == "SelfTraining":
		SelfTraining_Parameters = {}
		for i in ["GridSearch_parameters","Randomized_parameters"]:
			d_2 = {}
			for c in All_Name_list:
				d_1={}
				for k, v in Tuning_Parameters[i][c].items():
					d_1["base_estimator__"+k]=v
				d_2[c]=d_1
			SelfTraining_Parameters[i]=d_2
		return SelfTraining_Parameters
	else:
		Supervised_Parameters = {}
		for i in ["GridSearch_parameters","Randomized_parameters"]:
			d_2 = {}
			for c in All_Name_list:
				d_1={}
				for k, v in Tuning_Parameters[i][c].items():
					d_1[k]=v
				d_2[c]=d_1
			Supervised_Parameters[i]=d_2
		return Supervised_Parameters



## 2022/3/7 revise
## 2022/5/2 add NN
## 2022/5/18 add SGDClassifier, ComplementNB, RadiusNeighborsClassifier, SupportVectorMachine
## 2022/5/18 update Adaboost parameters
## 2022/5/18 update to sklearn 1.1.0
def Estimator(Algorithm_Name,cpu_core):
	####### Algorithm
	if Algorithm_Name == "RandomForestClassifier":
		from sklearn.ensemble import RandomForestClassifier
		estimate_ = RandomForestClassifier(n_jobs=cpu_core)
	if Algorithm_Name == "AdaBoostClassifier":
		from sklearn.ensemble import AdaBoostClassifier
		estimate_ = AdaBoostClassifier()
	if Algorithm_Name == "BernoulliNB":
		from sklearn.naive_bayes import BernoulliNB
		estimate_ = BernoulliNB()
	if Algorithm_Name == "DecisionTreeClassifier":
		from sklearn.tree import DecisionTreeClassifier
		estimate_ = DecisionTreeClassifier()
	if Algorithm_Name == "ExtraTreesClassifier":
		from sklearn.ensemble import ExtraTreesClassifier
		estimate_ = ExtraTreesClassifier(n_jobs=cpu_core)
	if Algorithm_Name == "GaussianNB":
		from sklearn.naive_bayes import GaussianNB
		estimate_ = GaussianNB()
	if Algorithm_Name == "GradientBoostingClassifier":
		from sklearn.ensemble import GradientBoostingClassifier
		estimate_ = GradientBoostingClassifier()
	if Algorithm_Name == "KNeighborsClassifier":
		from sklearn.neighbors import KNeighborsClassifier
		estimate_ = KNeighborsClassifier(n_jobs=cpu_core)
	if Algorithm_Name == "LogisticRegression":
		from sklearn.linear_model import LogisticRegression
		estimate_ = LogisticRegression(n_jobs=cpu_core)
	if Algorithm_Name == "XGBClassifier":
		from xgboost import XGBClassifier
		estimate_ = XGBClassifier(n_jobs=cpu_core)
	if Algorithm_Name == "LGBMClassifier":
		from lightgbm import LGBMClassifier
		estimate_ = LGBMClassifier(n_jobs=cpu_core)
	if Algorithm_Name =="NeuralNetworkClassifier":
		from sklearn.neural_network import MLPClassifier
		estimate_ = MLPClassifier() # No GPU
	if Algorithm_Name =="LabelPropagation":
		from sklearn.semi_supervised import LabelPropagation
		estimate_ = LabelPropagation(n_jobs=cpu_core) # semi supervise
	if Algorithm_Name =="LabelSpreading":
		from sklearn.semi_supervised import LabelSpreading
		estimate_ = LabelSpreading(n_jobs=cpu_core)  # semi supervise
	if Algorithm_Name =="SGDClassifier":
		from sklearn.linear_model import SGDClassifier
		estimate_ = SGDClassifier(n_jobs=cpu_core)
	if Algorithm_Name =="ComplementNB":  
		from sklearn.naive_bayes import ComplementNB
		estimate_ = ComplementNB()
	if Algorithm_Name =="RadiusNeighborsClassifier":  
		from sklearn.neighbors import RadiusNeighborsClassifier
		estimate_ = RadiusNeighborsClassifier(n_jobs=cpu_core)
	if Algorithm_Name =="SupportVectorMachine":        
		from sklearn.svm import SVC
		estimate_ = SVC()
	return estimate_

## 2022/5/18 revise
def integratSingleModel(Algorithm_Name,Learning_Type,Hyperparameter_Method,cpu_core,inner_cv,best_score,parameters,SelfTrain_threshold=0.75,SelfTrain_max_iter=100):
	import warnings
	warnings.simplefilter("ignore")
	Base_Estimator = Estimator(Algorithm_Name,cpu_core)
	if Learning_Type != "SelfTraining":
		# this inculde Supervised learning
		# and Semi supervised learning
		_clf_ = Base_Estimator
	else:
		from sklearn.semi_supervised import SelfTrainingClassifier
		_clf_ = SelfTrainingClassifier(Base_Estimator,threshold=SelfTrain_threshold,max_iter=SelfTrain_max_iter) # semi supervise
	####### Hyperparameter
	if Hyperparameter_Method == "No":
		CLF = _clf_
	if Hyperparameter_Method == "GridSearch":
		from sklearn.model_selection import GridSearchCV
		par = parameters['GridSearch_parameters']
		CLF = GridSearchCV(_clf_,par[Algorithm_Name], cv=inner_cv,n_jobs=cpu_core,scoring=best_score)
	if Hyperparameter_Method == "RandomizedSearch":
		from sklearn.model_selection import RandomizedSearchCV
		par = parameters['Randomized_parameters']
		CLF = RandomizedSearchCV(_clf_,par[Algorithm_Name], cv=inner_cv,n_jobs=cpu_core,scoring=best_score)
	return CLF


