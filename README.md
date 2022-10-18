# **Appling Machine Learning Model to predict biological role of fungi on the basis of genomic profile (Developing)**

## Prediction Pipeline
![image](https://github.com/DanyelleJhang/Appling-Machine-Learning-Model-to-predict-biological-role-of-fungi-on-the-basis-of-genomic-profile/blob/main/Pic/Main_Pipeline.png)


### Motivation

- This study aims to develop a computational model searching for potential biological function and assist experimental study in the future

![image](https://github.com/DanyelleJhang/Appling-Machine-Learning-Model-to-predict-biological-role-of-fungi-on-the-basis-of-genomic-profile/blob/main/WorkLog/Pic/Main_Pipeline.png)

### Research Purpose

- Is there any possible way to inspect **biological function of** **fungi** through their **peptide** **profile** ?

- How to predict biological function of sequenced metagenome shotgun data from our lab ?
![image](https://github.com/DanyelleJhang/Appling-Machine-Learning-Model-to-predict-biological-role-of-fungi-on-the-basis-of-genomic-profile/blob/main/WorkLog/Pic/Research_Purpose.png)

### Conceptual Framework

| Dataset                      | Source                                       |
| ---------------------------- | -------------------------------------------- |
| Annotation Dataset (AN)      | EnsemblFungi linked to FUNGulid              |
| Unannotation Dataset 1 (UN1) | EnsemblFungi failed linking to FUNGuild      |
| Unannotation Dataset 2 (UN2) | Sequenced metagenome shotgun data in HCL lab |

<img src="C:\Users\fabia\Local_Work\Open_Repository\Pic\Conceptual_Framework.png" style="zoom: 50%;" />

###  Feature Extraction

- Due to shut-gun sequencing exists incomplete genome, a proportion of gene will be dropped

![image](https://github.com/DanyelleJhang/Appling-Machine-Learning-Model-to-predict-biological-role-of-fungi-on-the-basis-of-genomic-profile/blob/main/WorkLog/Pic/Feature_Extraction_1.png)
![image](https://github.com/DanyelleJhang/Appling-Machine-Learning-Model-to-predict-biological-role-of-fungi-on-the-basis-of-genomic-profile/blob/main/WorkLog/Pic/Feature_Extraction_2.png)


###  Feature Selection Methods

- The ability of Single Feature Discriminates Labels

  - Permutation test approach 

  -  Heuristic approach based on non-parametric test

- Which is Informative Feature to Labels

- Features require to be independent with each other

#### 1. Heuristic approach based on non-parametric test

- If different sampled distributions of Single Label are all significant difference, such Feature will be remained

![image](https://github.com/DanyelleJhang/Appling-Machine-Learning-Model-to-predict-biological-role-of-fungi-on-the-basis-of-genomic-profile/blob/main/WorkLog/Pic/Discrimination.png)

#### 2. Permutation test approach 

- We acquire to comprehend their median of positive feature and negative feature is significantly difference or not instead of greater or less

- If such feature significantly difference, we will remain this feature.

![image](https://github.com/DanyelleJhang/Appling-Machine-Learning-Model-to-predict-biological-role-of-fungi-on-the-basis-of-genomic-profile/blob/main/WorkLog/Pic/Permutation_Test.png)

#### 3. Which is Informative Feature to Labels

- Information Theory is useful metric for measuring the correlation among random variables
- We will only remain those features which Changed Ratio of Information are greater than 1

![image](https://github.com/DanyelleJhang/Appling-Machine-Learning-Model-to-predict-biological-role-of-fungi-on-the-basis-of-genomic-profile/blob/main/WorkLog/Pic/Information.png)

#### 4. Features require to be independent with each other

- Different features but value render same impact on prediction

- We will only keep one of feature to utilize

- Those remaining features are weakly relevance to each other

- Those absolute correlation coefficient value amongst different features which are smaller than 0.3 will be remained.

- P value will not be considered

![image](https://github.com/DanyelleJhang/Appling-Machine-Learning-Model-to-predict-biological-role-of-fungi-on-the-basis-of-genomic-profile/blob/main/WorkLog/Pic/Independent.png)

### Ensemble Feature Selection Method
- In general, we selected union opinion from different feature selection methods and filtered out dependent features

![image](https://github.com/DanyelleJhang/Appling-Machine-Learning-Model-to-predict-biological-role-of-fungi-on-the-basis-of-genomic-profile/blob/main/WorkLog/Pic/ESMBLE_FS.png)

### Ensemble Hierarchical Classification
- In order to decrease tuning time, we introduce three types of voting style and such prediction value based on discrete value and probability value

  - Type 1: same algorithms in N round but pick up the best to vote

  - Type 2: the best model at Nth round (maybe XGB in 2nd round is best but Logistic in 3rd is the best) to vote

  - Type 3: vote by all model
![image](https://github.com/DanyelleJhang/Appling-Machine-Learning-Model-to-predict-biological-role-of-fungi-on-the-basis-of-genomic-profile/blob/main/WorkLog/Pic/Hierarchical_Classification.png)
