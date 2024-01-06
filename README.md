# ****Machine learning for predicting the biological role of fungi from genomic profiles****

## Prediction Pipeline
![image](https://github.com/DanyelleJhang/Appling-Machine-Learning-Model-to-predict-biological-role-of-fungi-on-the-basis-of-genomic-profile/blob/main/WorkLog/Pic/Main_Pipeline.png)


### Motivation

- This study aims to develop a computational model searching for potential biological function and assist experimental study in the future

### Research Purpose

- Is there any possible way to inspect **biological function of** **fungi** through their **peptide** **profile** ?

- How to predict biological function of sequenced metagenome shotgun data from our lab ?



### Original Data

| Dataset                      | Source                                       |
| ---------------------------- | -------------------------------------------- |
| Annotation Dataset (AN)      | EnsemblFungi linked to FUNGulid              |
| Unannotation Dataset 1 (UN1) | EnsemblFungi failed linking to FUNGuild      |
| Unannotation Dataset 2 (UN2) | Sequenced metagenome shotgun data in HCL lab |



###  Feature Extraction

```bash
bash ./Count_and_Combination.sh
```

- Due to shut-gun sequencing, it revealed incomplete genome and a proportion of gene will be dropped

###  Feature Selection Methods

- The ability of Single Feature Discriminates Labels

  - Permutation test approach 

  - Heuristic approach based on non-parametric test

- Which is Informative Feature to Labels

- Features require to be independent with each other

#### 1. Heuristic approach based on non-parametric test

- If different sampled distributions of Single Label are all significant difference, such Feature will be remained

#### 2. Permutation test approach 

- We acquire to comprehend their median of positive feature and negative feature are significantly difference or not instead of greater or less

- If such feature significantly difference, we will remain this feature.

#### 3. Which is Informative Feature to Labels

- Information Theory is useful metric for measuring the correlation among random variables
- We will only remain those features which Changed Ratio of Information are greater than 1

#### 4. Features require to be independent with each other

- Different features but value render same impact on prediction

- We will only keep one of feature to utilize

- Those rest of features are weakly relevance to each other

- Those absolute correlation coefficient value amongst different features which are smaller than 0.3 will be remained.

- P value will not be considered



### Ensemble Feature Selection Method

```bash
bash ./FS_TrophicMode.sh
bash ./FS_guildMode.sh
```

- In general, we selected union opinion from different feature selection methods and filtered out dependent features



### Ensemble Hierarchical Classification

```bash
bash ./ML_guildMode.sh
bash ./ML_TrophicMode.sh
```

- In order to increase prediction performance, we introduce weight-average ensemble model
