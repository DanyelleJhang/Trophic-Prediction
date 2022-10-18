## Python (3.9.7)

- pandas=='1.3.4'
- numpy=='1.20.3' 
- scikit-learn=='1.1.0'
- scipy=='1.8.1'
- imbalanced-learn=='0.9.1'
- xgboost=='1.5.2'
- lightgbm=='3.3.2'
- shap=='0.40.0'
- joblib==''1.1.0''

## R (4.0.5)

install R==4.0.5 on server without ROOT privilege

```bash
conda create --name r4-base
conda activate r4-base
conda install -c conda-forge r-base
conda install -c conda-forge/label/gcc7 r-base
```

install required R package

```bash
conda activate r4-base
conda install -c conda-forge r-factominer
conda install -c conda-forge r-dendextend
conda install -c r r-tidyverse
conda install -c conda-forge r-argparser
```



