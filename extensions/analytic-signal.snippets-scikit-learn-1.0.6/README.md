# Scikit-learn Snippets

This extension is for data scientists writing Python code to create, fit, and evaluate machine learning models using the `scikit-learn` package. It provides code snippets that work with Python (`.py`) files or (`.ipynb`) notebooks to ease the friction of working with `scikit-learn`. Snippets boost productivity by reducing the amount of Python syntax and function arguments that you need to remember, eliminating keystrokes, and helping to free the data scientist to focus on building useful models from their data. 

## 1. Usage  

### 1.1 Overview 

All snippets provided by this extension have triggers prefixed `sk`, which activate the IntelliSense pop up or allow filtering in the Command Palette. Trigger prefixes are further organised as follows:  

| Prefix                               | Description                                                                |
| -------------                        | -----------                                                                |
| <nobr>`sk-setup`</nobr>              | Starting point for importing commonly used modules and setting defaults.   |
| <nobr>`sk-read`</nobr>               | Read input training data or existing models from file.                     |
| <nobr>`sk-prep`</nobr>               | Preprocess input training data for model fitting.                          |
| <nobr>`sk-regress`</nobr>            | Create and fit regression models.                                          |
| <nobr>`sk-classify`</nobr>           | Create and fit classification models.                                      |
| <nobr>`sk-cluster`</nobr>            | Create and fit clustering models.                                          |
| <nobr>`sk-density`</nobr>            | Create and fit density estimation models.                                  |
| <nobr>`sk-embed`</nobr>              | Create and fit dimensionality reduction (embedding) models.                |
| <nobr>`sk-anomaly`</nobr>            | Create and fit anomaly detection models.                                   |
| <nobr>`sk-validation`</nobr>         | Model validation.                                                          |
| <nobr>`sk-inspect`</nobr>            | Model inspection and explainability.                                       |
| <nobr>`sk-io`</nobr>                 | Save and restore models on disk.                                           |
| <nobr>`sk-args`</nobr>               | Select and adjust model parameters from lists of valid options.            |

> For more information about the organization of the snippets provided by this extension see [Section 4](#4-snippet-reference) for a visual reference of the complete hierarchy.

### 1.2 Snippets for machine learning workflows
A typical workflow is to:  

- import commonly used modules with `sk-setup`,  
- read training data from file with `sk-read`,  
- preprocess training data with `sk-prep`,  
- create and train models across a range of machine learning tasks such as:

	- regression (<nobr>`sk-regress`</nobr>),  
	- classification (<nobr>`sk-classify`</nobr>),  
	- clustering (<nobr>`sk-cluster`</nobr>),
	- density estimation (<nobr>`sk-density`</nobr>), 
	- dimensionality reduction (<nobr>`sk-embed`</nobr>),   
	- anomaly detection (<nobr>`sk-anomaly`</nobr>)    

- evaluate fitted models by cross-validation (`sk-validation`) and inspection (`sk-inspect`),  
- save and restore models on disk (`sk-io`),
- optionally adjust the parameters that control model fitting with <nobr>`sk-args`</nobr>.

See the Features section below for a full list of the available snippets and their prefix triggers.

### 1.3 Inserting snippets 
Inserting machine learning code snippets into your Python code is easy. Use either of these methods:

**Command Palette**

1. Click inside a Python notebook cell or editor and choose *Insert Snippet* from the Command Palette.

2. A list of snippets appears. You can type to filter the list; start typing `sk` to filter the list for snippets provided by this extension. You can further filter the list by continuing to type the desired prefix.

3. Choose a snippet from the list and it is inserted into your Python code.

4. Placeholders indicate the minimum arguments required to train a model. Use the tab key to step through the placeholders.

**IntelliSense**

1. Start typing `sk` in a Python notebook cell or editor.

2. The IntelliSense pop up will appear. You can further filter the pop up list by continuing to type the desired prefix.

3. Choose a snippet from the pop up and it is inserted into your Python code.

4. Placeholders indicate the minimum arguments required to train a model. Use the tab key to step through the placeholders.

You can also trigger IntelliSense by typing  *Ctrl+Space*.

## 2. Features

### 2.1 Setup

The following snippets are triggered by `sk-setup` and provide the starting point for creating models. Usually inserted near the beginning of a code file or notebook, `sk-setup` is the key snippet for importing modules that are commonly used in the machine learning workflow, and setting defaults that apply to data visualizations. 

| Snippet              | Placeholders           | Description |
| ---                  | ---                    | ---         |
| <nobr>`sk-setup`</nobr> | `pio.renderers.default`<br>`pio.templates.default` | Provides the initial starting point for creating models. Imports commonly used modules (`pandas`, `numpy`, `json`, `pickle`, `plotly.express`, and `plotly.io`), and sets the default figure renderer and template. |
| | | |

### 2.2 Read training data

The following snippets are triggered by `sk-read` and provide features for creating `pandas` data frames from Comma Separated Value (`.csv`), Microsoft Excel (`.xlsx`), Feather (`.feather`), and Parquet (`.parquet`) format files. Tabular data stored in `pandas` data frames is a common source of training data required for fitting `scikit-learn` models.

| Snippet              | Placeholders           | Description |
| ---                  | ---                    | ---             |
| <nobr>`sk-read-csv`</nobr>        | `df`,<br>`file`        | Read tabular training data from CSV (`.csv`) file (`file`) into pandas data frame (`df`) and report info.     |
| <nobr>`sk-read-excel`</nobr>      | `df`,<br>`file`        | Read tabular training data from Excel (`.xlsx`) file (`file`) into pandas data frame (`df`) and report info.    |
| <nobr>`sk-read-feather`</nobr>    | `df`,<br>`file`        | Read tabular training data from Feather (`.feather`) file (`file`) into pandas data frame (`df`) and report info. |
| <nobr>`sk-read-parquet`</nobr>    | `df`,<br>`file`        | Read tabular training data from Parquet (`.parquet`) file (`file`) into pandas data frame (`df`) and report info. |
| | | |

### 2.3 Preprocessing for supervised learning  

The following snippets are triggered by `sk-prep-target` and provide features for preparing tabular training data for supervised learning. The data preparation process involves extracting one or more named features (`X`) and a named target variable (`y`) from an input data frame (`df`), and creating an input training dataset. The data arrays are used to train models for a range of supervised machine learning tasks.

Optionally, the process can also include extracting one or more secondary variables (`Z`) that are used to better understand the results of model fitting. It is important to note that these secondary variables do not play any role in the model fitting itself, but they can be useful for interpreting the results.

| Snippet              | Placeholders           | Description |
| ---                  | ---                    | ---             |
| <nobr>`sk-prep-target-features`</nobr>  | `X1`, `X2`, `X3`,<br>`Y`<br>`df`,<br>`X`, `y` | Prepare training data (`X`, `y`) for supervised learning. Training data is identified by a sequence of feature names (`X1`, `X2`, `X3, ...`), and a target name (`Y`), sourced from a data frame (`df`). 
| <nobr>`sk-prep-target-features-secondary`</nobr>  | `X1`, `X2`, `X3`,<br>`Y`<br>`Z1`, `Z2`, `Z3`,<br>`df`,<br>`X`, `y`, `Z` | Prepare training data (`X`, `y`) for supervised learning, and prepare secondary data (`Z`) for model evaluation. Training data is identified by a sequence of feature names (`X1`, `X2`, `X3, ...`), and a target name (`Y`), sourced from a data frame (`df`). Secondary data is also identified by a sequence of feature names (`Z1`, `Z2`, `Z3, ...`) sourced from the same data frame (`df`). <br><br>**Note:** Secondary data is used only for interpreting model output and plays no role in model training.|
| <nobr>`sk-prep-train_test_split`</nobr> | `X`, `y`,<br>&#x2699; `train_size`,<br><nobr>&#x2699; `random_state`</nobr> | Randomly split input data (`X`, `y`) into training and test sets using the [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function with the supplied parameters (&#x2699;). Holding out a test set from the training process helps to evaluate the performance of supervised learning models. |
| | | |

### 2.4 Preprocessing for unsupervised learning  

The following snippets are triggered by `sk-prep-features` and provide features for preparing tabular training data for unsupervised learning. The data preparation process involves extracting one or more named features (`X`) from an input data frame (`df`), and creating an input training dataset. The data array is used to train models for a range of unsupervised machine learning tasks. In contrast to supervised learning, there is no target variable (`y`) in unsupervised learning.

Similar to the supervised learning case, the process can optionally include extracting one or more secondary variables (`Z`) that are used to better understand the results of model fitting. It is important to note that these secondary variables do not play any role in the model fitting itself, but they can be useful for interpreting the results.

| Snippet              | Placeholders           | Description |
| ---                  | ---                    | ---             |
| <nobr>`sk-prep-features`</nobr>         | `X1`, `X2`, `X3`<br>`df`,<br>`X`| Prepare training data features (`X`) for unsupervised learning. Training data is identified by a sequence of feature names (`X1`, `X2`, `X3, ...`) sourced from a data frame (`df`). |
| <nobr>`sk-prep-features-secondary`</nobr>  | `X1`, `X2`, `X3`,<br>`Z1`, `Z2`, `Z3`,<br>`df`,<br>`X`, `Z` | Prepare training data features (`X`) for unsupervised learning, and prepare secondary data (`Z`) for model evaluation. Training data is identified by a sequence of feature names (`X1`, `X2`, `X3, ...`) sourced from a data frame (`df`). Secondary data is also identified by a sequence of feature names (`Z1`, `Z2`, `Z3, ...`) sourced from the same data frame (`df`). <br><br>**Note:** Secondary data is used only for interpreting model output and plays no role in model training. |
| | | |

### 2.5 Regression

#### 2.5.1 Linear regression

The following snippets are triggered by `sk-regress-linear` and provide features for various types of linear regression ranging from simple ordinary least squares (`LinearRegression`), regression with a transformed target (`TransformedTargetRegressor`), regression with transformed features (`FunctionTransformer`, `PolynomialFeatures`, `SplineTransformer`), and regularized models such as ridge regression (`Ridge`), lasso regression (`Lasso`), and elastic net regression (`ElasticNet`). 

| Snippet | Placeholders | Description |
| ---     | ---          | ---         |
| <nobr>`sk-regress-linear`</nobr> | `estimator_linear`,<br>&#x2699; `fit_intercept`,<br>&#x2699; `positive`,<br> `X`, `y` | **Linear regression:** Create and fit a [`LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) regression model (`estimator_linear`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). |
| <nobr>`sk-regress-linear-transform-target`</nobr> | `estimator_transform_target`,<br>&#x2699; `func`, <br>&#x2699; `func_inverse`,<br>&#x2699; `fit_intercept`,<br>&#x2699; `positive`,<br>&#x2699; `check_inverse`,<br>`X`, `y` | **Linear regression with transformed target:** Create and fit a [`TransformedTargetRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html) regression model (`estimator_transform_target`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). |
| <nobr>`sk-regress-linear-transform`</nobr>             | `estimator_transform`,<br>&#x2699; `func`,<br>&#x2699; `fit_intercept`,<br>&#x2699; `positive`,<br>`X`, `y` | **Linear regression with transformed features:** Create and fit a [`LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) with [`FunctionTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html) regression model (`estimator_transform`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). |
| <nobr>`sk-regress-linear-polynomial`</nobr>            | `estimator_polynomial`,<br>&#x2699; `degree`,<br>&#x2699; `fit_intercept`,<br>&#x2699; `positive`,<br>`X`, `y` | **Polynomial regression:** Create and fit a [`LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) with [`PolynomialFeatures`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html) regression model (`estimator_polynomial`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). |
| <nobr>`sk-regress-linear-spline`</nobr>                | `estimator_spline`,<br>&#x2699; `n_knots`,<br>&#x2699; `degree`,<br>&#x2699; `knots`,<br>&#x2699; `extrapolation`,<br>&#x2699; `fit_intercept`,<br>&#x2699; `positive`,<br>`X`, `y` | **Spline regression:** Create and fit a [`LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) with [`SplineTransformer`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html) regression model (`estimator_spline`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). |
| <nobr>`sk-regress-linear-pcr`</nobr>   | `estimator_pcr`,<br>&#x2699; `n_components`,<br>&#x2699; `whiten`,<br>&#x2699; `fit_intercept`,<br>&#x2699; `positive`,<br>`X`, `y` | **Principal component regression (PCR):** Create and fit a [`LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) with [`PCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) regression model (`estimator_pcr`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). |
| <nobr>`sk-regress-linear-pls`</nobr> | `estimator_pls`,<br>&#x2699; `n_components`,<br>&#x2699; `scale`,<br>&#x2699; `max_iter`,<br>`X`, `y` | **Partial least squares regression (PLS):** Create and fit a [`PLSRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html) regression model (`estimator_pls`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). |
| <nobr>`sk-regress-linear-ridge`</nobr>   | `estimator_ridge`,<br>&#x2699; `alpha`,<br>&#x2699; `fit_intercept`,<br>&#x2699; `positive`,<br>`X`, `y` | **Ridge regression:** Create and fit a [`Ridge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) regression model (`estimator_ridge`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). |
| <nobr>`sk-regress-linear-ridgecv`</nobr> | `estimator_ridgecv`,<br>&#x2699; `alphas`,<br>&#x2699; `cv`,<br>&#x2699; `fit_intercept`,<br>`X`, `y` | **Ridge regression with cross-validation:** Create and fit a [`RidgeCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html) regression model (`estimator_ridgecv`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). |
| <nobr>`sk-regress-linear-lasso`</nobr>   | `estimator_lasso`,<br>&#x2699; `alpha`,<br>&#x2699; `fit_intercept`<br>&#x2699; `positive`,<br>&#x2699; `selection`,<br>&#x2699; `random_state`,<br>`X`, `y` | **Lasso regression:** Create and fit a [`Lasso`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html) regression model (`estimator_lasso`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). |
| <nobr>`sk-regress-linear-lassocv`</nobr> | `estimator_lassocv`,<br>&#x2699; `alphas`,<br>&#x2699; `cv`,<br>&#x2699; `fit_intercept`,<br>&#x2699; `positive`,<br>&#x2699; `selection`,<br>&#x2699; `random_state`,<br>`X`, `y` | **Lasso regression with cross-validation:** Create and fit a [`LassoCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html) regression model (`estimator_lassocv`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). |
| <nobr>`sk-regress-linear-elasticnet`</nobr>   | `estimator_elasticnet`,<br>&#x2699; `alpha`,<br>&#x2699; `l1_ratio`,<br>&#x2699; `fit_intercept`,<br>&#x2699; `positive`,<br>&#x2699; `selection`,<br>&#x2699; `random_state`,<br>`X`, `y` | **ElasticNet regression:** Create and fit an [`ElasticNet`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html) regression model (`estimator_elasticnet`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). |
| <nobr>`sk-regress-linear-elasticnetcv`</nobr>   | `estimator_elasticnetcv`,<br>&#x2699; `l1_ratio`,<br>&#x2699; `alphas`,<br>&#x2699; `fit_intercept`,<br>&#x2699; `positive`,<br>&#x2699; `selection`,<br>&#x2699; `random_state`,<br>`X`, `y` | **ElasticNet regression with cross-validation:** Create and fit an [`ElasticNetCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html) regression model (`estimator_elasticnetcv`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). |
| <nobr>`sk-regress-linear-quantile`</nobr>   | `estimator_quantile`,<br>&#x2699; `quantile`,<br>&#x2699; `alpha`,<br>&#x2699; `fit_intercept`,<br>&#x2699; `solver`,<br>`X`, `y` | **Quantile regression:** Create and fit a [`QuantileRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html) regression model (`estimator_quantile`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). |
||||

#### 2.5.2 Nearest neighbour regression

The following snippets are triggered by `sk-regress-neighbors` and provide features for nearest neighbour based regression. These non-parametric methods make predictions from the local neighbourhood of each query sample in the feature space, allowing the model to adapt naturally to complex relationships without assuming a specific functional form. 

| Snippet            | Placeholders | Description |
| ---                | ---          | ---          |
| <nobr>`sk-regress-neighbors-k`</nobr> | `estimator_knn`,<br>&#x2699; `n_neighbors`,<br>&#x2699; `weights`,<br>&#x2699; `algorithm`,<br>&#x2699; `leaf_size`,<br>&#x2699; `p`,<br>&#x2699; `metric`,<br>`X`, `y` | **K-nearest neighbors regression:** Create and fit a [`KNeighborsRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html) model (`estimator_knn`) with preprocessing pipeline using the supplied parameters (&#x2699;) and training data (`X`, `y`). Uses `make_pipeline` with `StandardScaler` for feature scaling. | 
| <nobr>`sk-regress-neighbors-radius`</nobr> | `estimator_radius`,<br>&#x2699; `radius`,<br>&#x2699; `weights`,<br>&#x2699; `algorithm`,<br>&#x2699; `leaf_size`,<br>&#x2699; `p`,<br>&#x2699; `metric`,<br>`X`, `y` | **Radius neighbors regression:** Create and fit a [`RadiusNeighborsRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsRegressor.html) model (`estimator_radius`) with preprocessing pipeline using the supplied parameters (&#x2699;) and training data (`X`, `y`). Uses `make_pipeline` with `StandardScaler`  for feature scaling. |
| | | |

#### 2.5.3 Gaussian process regression

The following snippets are triggered by `sk-regress-gaussian` and provide features for Gaussian Process Regression (GPR). GPR is a non-parametric Bayesian approach that models uncertainty in predictions and uses  kernels to capture complex structure.

| Snippet | Placeholders | Description |
| ---     | ---          | ---         |
| <nobr>`sk-regress-gaussian-process`</nobr> | `estimator_gp`,<br>&#x2699; `kernel`,<br>&#x2699; `alpha`,<br>&#x2699; `n_restarts_optimizer`,<br>&#x2699; `random_state`,<br>`X`, `y` | **Gaussian Process Regression:** Create and fit a [`GaussianProcessRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html) model (`estimator_gp`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). |
| <nobr>`sk-regress-gaussian-process-kernel`</nobr> | `estimator_gp_kernel`,<br>&#x2699; `kernel`,<br>&#x2699; `alpha`,<br>&#x2699; `n_restarts_optimizer`,<br>&#x2699; `random_state`,<br>`X`, `y` | **Gaussian Process Regression with custom kernel:** Create and fit a [`GaussianProcessRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html) model (`estimator_gp_kernel`) with a custom kernel, the supplied parameters (&#x2699;) and training data (`X`, `y`).<br><br>Build kernels from components in [`sklearn.gaussian_process.kernels`](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.gaussian_process.kernels) (e.g., [`RBF`](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html), [`Matern`](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html), [`RationalQuadratic`](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RationalQuadratic.html), [`DotProduct`](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.DotProduct.html), [`ExpSineSquared`](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.ExpSineSquared.html), [`WhiteKernel`](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.WhiteKernel.html), [`ConstantKernel`](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.ConstantKernel.html)) and combine them with `+` (additive) and `*` (multiplicative) operators to encode structure. Example: `ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3)) + WhiteKernel(1e-5, (1e-8, 1e1))`. Set hyperparameter bounds (e.g., `length_scale_bounds`) to enable automatic tuning via log-marginal-likelihood optimization.<br><br>See [Kernels for Gaussian Processes](https://scikit-learn.org/stable/modules/gaussian_process.html#kernels-for-gaussian-processes) for more details. |. |
| <nobr>`sk-regress-gaussian-transform-target`</nobr> | `estimator_gp_transform_target`,<br>&#x2699; `func`,<br>&#x2699; `func_inverse`,<br>&#x2699; `kernel`,<br>&#x2699; `alpha`,<br>&#x2699; `n_restarts_optimizer`,<br>&#x2699; `random_state`,<br>`X`, `y` | **Gaussian Process Regression with transformed target:** Create and fit a [`TransformedTargetRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.compose.TransformedTargetRegressor.html) wrapping a [`GaussianProcessRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html) (`estimator_gp_transform_target`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). <br><br>Transforming the target (`y`) to approximate a Normal distribution honours a key assumption of Gaussian process regression and can improve predictive performance. |
| | | |

#### 2.5.4 Ensemble regression

The following snippets are triggered by `sk-regress-ensemble` and provide features for various types of ensemble regression. Ensemble regression models combine multiple base regression models to create a model that has superior performance compared to a single base model. Rather than relying on a single model's prediction, ensemble methods aggregate predictions from several models to produce a final result that is typically more accurate.

| Snippet | Placeholders | Description |  
| ---     | ---          | ---         |
| <nobr>`sk-regress-ensemble-random-forest`</nobr> | `estimator_random_forest`,<br>&#x2699; `n_estimators`,<br>&#x2699; `criterion`,<br>&#x2699; `min_samples_leaf`,<br>&#x2699; `random_state`,<br>`X`, `y` | **Random forest regression:** Create and fit a [`RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) regression model (`estimator_random_forest`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). | 
| <nobr>`sk-regress-ensemble-extra-trees`</nobr> | `estimator_extra_trees`,<br>&#x2699; `n_estimators`,<br>&#x2699; `criterion`,<br>&#x2699; `min_samples_leaf`,<br>&#x2699; `random_state`,<br>`X`, `y` | **Extremely randomized trees (extra-trees) regression:** Create and fit an [`ExtraTreesRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html) regression model (`estimator_extra_trees`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). | 
| <nobr>`sk-regress-ensemble-gradient-boosting`</nobr> | `estimator_gradient_boosting`,<br>&#x2699; `n_estimators`,<br>&#x2699; `loss`,<br>&#x2699; `learning_rate`,<br>&#x2699; `min_samples_leaf`,<br>&#x2699; `random_state`,<br>`X`, `y` | **Gradient boosting regression:** Create and fit a [`GradientBoostingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) regression model (`estimator_gradient_boosting`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). |
| <nobr>`sk-regress-ensemble-hist-gradient-boosting`</nobr> | `estimator_hist_gradient_boosting`,<br>&#x2699; `loss`,<br>&#x2699; `learning_rate`,<br>&#x2699; `min_samples_leaf`,<br>&#x2699; `random_state`,<br>`X`, `y` | **Histogram-based gradient boosting regression:** Create and fit a [`HistGradientBoostingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html) regression model (`estimator_hist_gradient_boosting`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). |
| <nobr>`sk-regress-ensemble-stacking`</nobr> | `estimator_stacking`,<br>&#x2699; `estimators`,<br>&#x2699; `final_estimator`,<br>&#x2699; `cv`,<br>&#x2699; `passthrough`,<br>`X`, `y` | **Stack of estimators with a final regressor:** Create and fit a [`StackingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html) regression model (`estimator_stacking`) with the supplied parameters (&#x2699;) and training data (`X`, `y`).<br><br>**Note:** By default, all regression estimators in the current scope are collected for stacking. |
| <nobr>`sk-regress-ensemble-voting`</nobr> | `estimator_voting`,<br>&#x2699; `estimators`,<br>&#x2699; `weights`,<br>`X`, `y` | **Voting regression:** Create and fit a [`VotingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html) regression model (`estimator_voting`) with the supplied parameters (&#x2699;) and training data (`X`, `y`).<br><br>**Note:** By default, all regression estimators in the current scope are collected for voting. |
| | | |

#### 2.5.4 Regression helpers

The following snippets are triggered by `sk-regress` and provide features for regression helpers.

| Snippet | Placeholders | Description |  
| ---     | ---          | ---         |
| <nobr>`sk-regress-dummy`</nobr> | `estimator_dummy`,<br>&#x2699; `strategy`,<br>&#x2699; `constant`,<br>&#x2699; `quantile`,<br>`X`, `y` | **Dummy regression:** Create and fit a [`DummyRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html) regression model (`estimator_dummy`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). |
| <nobr>`sk-regress-report`</nobr> | `estimator`,<br>`X`,<br> `y_true`,<br> `y_pred`<br> | **Regression report:** Apply a regression model (`estimator`) to an input dataset (`X`) to predict values (`y_pred`) using the model's `predict()` function and compare the results with the true values (`y_true`). Print diagnostic information with the [`mean_squared_error()`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) and [`r2_score()`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html) functions to evaluate the performance of the regressor on the supplied data. |
||||

### 2.6 Classification

The following snippets are triggered by `sk-classify` and provide features for various types of classification models. Classification is a supervised learning task where the goal is to predict discrete class labels for input data. These models learn from labeled training data and can be used for binary classification (two classes) or multi-class classification problems.  

#### 2.6.1 Linear classification

The following snippets are triggered by `sk-classify-linear` and provide features for various types of linear classification models. Linear classifiers separate classes using linear decision boundaries in the feature space. These models are often computationally efficient, interpretable, and serve as good baseline models for many classification problems. Common approaches include linear discriminant analysis, logistic regression, and linear support vector machines.

| Snippet | Placeholders | Description |
| ---     | ---          | ---          |
| <nobr>`sk-classify-linear-lda`</nobr>  | `estimator_lda`,<br>&#x2699; `solver`,<br>&#x2699; `shrinkage`,<br>&#x2699; `n_components`,<br>&#x2699; `store_covariance`,<br>`X`, `y` | **Linear Discriminant Analysis (LDA):** Create and fit a [`LinearDiscriminantAnalysis`](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html) classification model (`estimator_lda`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). |
| <nobr>`sk-classify-linear-qda`</nobr>  | `estimator_qda`,<br>&#x2699; `priors`,<br>&#x2699; `reg_param`,<br>&#x2699; `store_covariance`,<br>`X`, `y`  | **Quadratic Discriminant Analysis (QDA):** Create and fit a [`QuadraticDiscriminantAnalysis`](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html) classification model (`estimator_qda`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). |
| <nobr>`sk-classify-linear-logistic`</nobr>  | `estimator_logistic`,<br>&#x2699; `C`,<br>&#x2699; `penalty`,<br>&#x2699; `solver`,<br>&#x2699; `multi_class`,<br>&#x2699; `max_iter`,<br>&#x2699; `random_state`,<br>`X`, `y` | **Logistic Regression:** Create and fit a [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) classification model (`estimator_logistic`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). |
| <nobr>`sk-classify-linear-svm`</nobr>       | `estimator_svm`,<br>&#x2699; `C`,<br>&#x2699; `penalty`,<br>&#x2699; `loss`,<br>&#x2699; `dual`,<br>&#x2699; `max_iter`,<br>&#x2699; `random_state`,<br>`X`, `y` | **Linear Support Vector Machine:** Create and fit a [`LinearSVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) classification model (`estimator_svm`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). |
| <nobr>`sk-classify-linear-svm-sgd`</nobr>   | `estimator_sgd`,<br>&#x2699; `loss`,<br>&#x2699; `penalty`,<br>&#x2699; `alpha`,<br>&#x2699; `learning_rate`,<br>&#x2699; `eta0`,<br>&#x2699; `max_iter`,<br>&#x2699; `random_state`,<br>`X`, `y` | **SGD Classifier:** Create and fit a [`SGDClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) classification model (`estimator_sgd`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). Uses stochastic gradient descent for fast training on large datasets. |
| <nobr>`sk-classify-linear-perceptron`</nobr> | `estimator_perceptron`,<br>&#x2699; `penalty`,<br>&#x2699; `alpha`,<br>&#x2699; `max_iter`,<br>&#x2699; `random_state`,<br>`X`, `y` | **Perceptron:** Create and fit a [`Perceptron`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html) classification model (`estimator_perceptron`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). |
| <nobr>`sk-classify-linear-ridge`</nobr>     | `estimator_ridge`,<br>&#x2699; `alpha`,<br>&#x2699; `solver`,<br>&#x2699; `max_iter`,<br>&#x2699; `random_state`,<br>`X`, `y` | **Ridge Classifier:** Create and fit a [`RidgeClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html) classification model (`estimator_ridge`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). |
| | | |

#### 2.6.2 Nearest neighbour classification

The following snippets are triggered by `sk-classify-neighbors` and provide features for nearest neighbour based classification. These non-parametric methods make predictions from the local neighbourhood of each query sample in the feature space, allowing the model to adapt naturally to complex relationships without assuming a specific functional form. 

| Snippet            | Placeholders | Description |
| ---                | ---          | ---          |
| <nobr>`sk-classify-neighbors-k`</nobr> | `estimator_knn`,<br>&#x2699; `n_neighbors`,<br>&#x2699; `weights`,<br>&#x2699; `algorithm`,<br>&#x2699; `leaf_size`,<br>&#x2699; `p`,<br>&#x2699; `metric`,<br>`X`, `y` | **K-nearest neighbors classification:** Create and fit a [`KNeighborsClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) model with preprocessing pipeline using the supplied parameters (&#x2699;) and training data (`X`, `y`). Uses `make_pipeline` with `StandardScaler` for proper feature scaling. | 
| <nobr>`sk-classify-neighbors-radius`</nobr> | `estimator_radius`,<br>&#x2699; `radius`,<br>&#x2699; `weights`,<br>&#x2699; `algorithm`,<br>&#x2699; `leaf_size`,<br>&#x2699; `p`,<br>&#x2699; `metric`,<br>&#x2699; `outlier_label`,<br>`X`, `y` | **Radius neighbors classification:** Create and fit a [`RadiusNeighborsClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier.html) model with preprocessing pipeline using the supplied parameters (&#x2699;) and training data (`X`, `y`). Uses `make_pipeline` with `StandardScaler` for proper feature scaling. | 
| <nobr>`sk-classify-neighbors-centroid`</nobr> | `estimator_centroid`,<br>&#x2699; `metric`,<br>&#x2699; `shrink_threshold`,<br>&#x2699; `priors`<br>`X`, `y` | **Nearest centroid classification:** Create and fit a [`NearestCentroid`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html) model with preprocessing pipeline using the supplied parameters (&#x2699;) and training data (`X`, `y`). Uses `make_pipeline` with `StandardScaler` for proper feature scaling.  
| | | |

#### 2.6.3 Naive Bayes classification

The following snippets are triggered by `sk-classify-bayes` and provide features for Naive Bayes classification algorithms. These are fast, simple generative classifiers that assume conditional independence of features given the class. Each variant has specific assumptions about the feature distribution and format; choose preprocessing that matches those assumptions rather than generic feature scaling.

| Snippet | Placeholders | Description |
| ---     | ---          | ---          |
| <nobr>`sk-classify-bayes-gaussian`</nobr> | `estimator_gaussian_nb`,<br>&#x2699; `priors`,<br>&#x2699; `var_smoothing`,<br>`X`, `y` | **Gaussian Naive Bayes:** Create and fit a [`GaussianNB`](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html) classification model (`estimator_gaussian_nb`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). Intended for continuous features; standardization is not required. For skewed/heavy‑tailed features consider `PowerTransformer` or `QuantileTransformer` rather than `StandardScaler`. |
| <nobr>`sk-classify-bayes-multinomial`</nobr> | `estimator_multinomial_nb`,<br>&#x2699; `alpha`,<br>&#x2699; `fit_prior`,<br>&#x2699; `class_prior`,<br>`X`, `y` | **Multinomial Naive Bayes:** Create and fit a [`MultinomialNB`](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html) model (`estimator_multinomial_nb`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). Expects non‑negative features (e.g., counts or tf‑idf). For text, precede with `CountVectorizer`/`TfidfVectorizer` instead of `StandardScaler`. |
| <nobr>`sk-classify-bayes-bernoulli`</nobr> | `estimator_bernoulli_nb`,<br>&#x2699; `alpha`,<br>&#x2699; `binarize`,<br>&#x2699; `fit_prior`,<br>&#x2699; `class_prior`,<br>`X`, `y` | **Bernoulli Naive Bayes:** Create and fit a [`BernoulliNB`](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html) model (`estimator_bernoulli_nb`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). Designed for binary (0/1) features. If inputs are continuous, set `binarize` or add a `Binarizer` step—do not scale. |
| <nobr>`sk-classify-bayes-complement`</nobr> | `estimator_complement_nb`,<br>&#x2699; `alpha`,<br>&#x2699; `fit_prior`,<br>&#x2699; `class_prior`,<br>&#x2699; `norm`,<br>`X`, `y` | **Complement Naive Bayes:** Create and fit a [`ComplementNB`](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html) model (`estimator_complement_nb`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). More robust for imbalanced text; requires non‑negative features (counts/tf‑idf). |
| <nobr>`sk-classify-bayes-categorical`</nobr> | `estimator_categorical_nb`,<br>&#x2699; `alpha`,<br>&#x2699; `fit_prior`,<br>&#x2699; `class_prior`,<br>&#x2699; `min_categories`,<br>`X`, `y` | **Categorical Naive Bayes:** Create and fit a [`CategoricalNB`](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html) model (`estimator_categorical_nb`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). Expects integer‑encoded categorical features; use `OrdinalEncoder` with appropriate handling of unknowns. Do not scale. |
| | | |

#### 2.6.4 Ensemble classification

The following snippets are triggered by `sk-classify-ensemble` and provide features for various types of ensemble classification. Ensemble classification models combine multiple base classification models to create a more robust and accurate classifier. These methods typically improve classification performance by aggregating predictions from several models, reducing overfitting, and increasing the model's ability to generalize to new data. Common ensemble techniques include bagging, boosting, and stacking approaches.

| Snippet            | Placeholders | Description |
| ---                | ---          | ---          |
| <nobr>`sk-classify-ensemble-random-forest`</nobr> | `estimator_random_forest`,<br>&#x2699; `n_estimators`,<br>&#x2699; `criterion`,<br>&#x2699; `min_samples_leaf`,<br>&#x2699; `random_state`,<br>`X`, `y` | **Random forest classification:** Create and fit a [`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) classification model (`estimator_random_forest`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). | 
| <nobr>`sk-classify-ensemble-extra-trees`</nobr> | `estimator_extra_trees`,<br>&#x2699; `n_estimators`,<br>&#x2699; `criterion`,<br>&#x2699; `min_samples_leaf`,<br>&#x2699; `random_state`,<br>`X`, `y` | **Extremely randomized trees (extra-trees) classification:** Create and fit an [`ExtraTreesClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html) classification model (`estimator_extra_trees`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). | 
| <nobr>`sk-classify-ensemble-gradient-boosting`</nobr> | `estimator_gradient_boosting`,<br>&#x2699; `n_estimators`,<br>&#x2699; `loss`,<br>&#x2699; `learning_rate`,<br>&#x2699; `min_samples_leaf`,<br>&#x2699; `random_state`,<br>`X`, `y` | **Gradient boosting classification:** Create and fit a [`GradientBoostingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) classification model (`estimator_gradient_boosting`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). |
| <nobr>`sk-classify-ensemble-hist-gradient-boosting`</nobr> | `estimator_hist_gradient_boosting`,<br>&#x2699; `loss`,<br>&#x2699; `learning_rate`,<br>&#x2699; `min_samples_leaf`,<br>&#x2699; `random_state`,<br>`X`, `y` | **Histogram-based gradient boosting classification:** Create and fit a [`HistGradientBoostingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html) classification model (`estimator_hist_gradient_boosting`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). |
| <nobr>`sk-classify-ensemble-stacking`</nobr> | `estimator_stacking`,<br>&#x2699; `estimators`,<br>&#x2699; `final_estimator`,<br>&#x2699; `cv`,<br>&#x2699; `stack_method`,<br>&#x2699; `passthrough`,<br>`X`, `y` | **Stack of estimators with a final classifier:** Create and fit a [`StackingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html) classification model (`estimator_stacking`) with the supplied parameters (&#x2699;) and training data (`X`, `y`).<br><br>**Note:** By default, all classification estimators in the current scope are collected for stacking. |
| <nobr>`sk-classify-ensemble-voting`</nobr> | `estimator_voting`,<br>&#x2699; `estimators`,<br>&#x2699; `voting`,<br>&#x2699; `weights`,<br>`X`, `y` | **Voting classification:** Create and fit a [`VotingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html) classification model (`estimator_voting`) with the supplied parameters (&#x2699;) and training data (`X`, `y`).<br><br>**Note:** By default, all classification estimators in the current scope are collected for voting. |
| | | |

#### 2.6.5 Classification helpers

The following snippets are triggered by `sk-classify` and provide features for classification helpers.

| Snippet | Placeholders | Description |  
| ---     | ---          | ---         |
| <nobr>`sk-classify-dummy`</nobr> | `estimator_dummy`,<br>&#x2699; `strategy`,<br>&#x2699; `random_state`,<br>&#x2699; `constant`,<br>`X`, `y` | **Dummy classification:** Create and fit a [`DummyClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html) classification model (`estimator_dummy`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). |
| <nobr>`sk-classify-report`</nobr> | `estimator`,<br>`X`,<br> `y_true`,<br> `y_pred`<br> | **Classification report:** Apply a classification model (`estimator`) to an input dataset (`X`) to predict labels (`y_pred`) using the model's `predict()` function and compare the results with the true labels (`y_true`). Print diagnostic information with the [`classification_report()`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) and [`confusion_matrix()`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) functions to evaluate the performance of the classifier on the supplied data. |
||||

### 2.7 Clustering

The following snippets are triggered by `sk-cluster` and provide features for various types of clustering models. Clustering is an unsupervised learning task that groups similar data points together based on their features. These models identify natural groupings within a dataset without requiring labeled examples, making them useful for discovering patterns, segmenting data, and identifying structure in unlabeled datasets.  

| Snippet       | Placeholders | Description |
| --- | --- | --- |
| <nobr>`sk-cluster-kmeans`</nobr> | `estimator_kmeans`,<br>&#x2699; `n_clusters`,<br>&#x2699; `init`,<br>&#x2699; `n_init`,<br>&#x2699; `max_iter`,<br>&#x2699; `tol`,<br>&#x2699; `random_state`,<br>&#x2699; `algorithm`,<br>`X` | **K-means clustering:** Create and fit a [`KMeans`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) clustering model (`estimator_kmeans`) with the supplied parameters (&#x2699;) and training data (`X`). Includes StandardScaler in pipeline for proper feature scaling. |
| <nobr>`sk-cluster-kmeans-minibatch`</nobr> | `estimator_kmeans_minibatch`,<br>&#x2699; `n_clusters`,<br>&#x2699; `init`,<br>&#x2699; `batch_size`,<br>&#x2699; `max_iter`,<br>&#x2699; `max_no_improvement`,<br>&#x2699; `tol`,<br>&#x2699; `random_state`,<br>&#x2699; `reassignment_ratio`,<br>`X` | **MiniBatch K-means clustering:** Create and fit a [`MiniBatchKMeans`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html) clustering model (`estimator_kmeans_minibatch`) with the supplied parameters (&#x2699;) and training data (`X`). Optimized for large datasets with StandardScaler in pipeline. |
| <nobr>`sk-cluster-meanshift`</nobr> | `estimator_meanshift`,<br>&#x2699; `bandwidth`,<br>&#x2699; `seeds`,<br>&#x2699; `bin_seeding`,<br>&#x2699; `min_bin_freq`,<br>&#x2699; `cluster_all`,<br>&#x2699; `max_iter`,<br>`X` | **Mean shift clustering:** Create and fit a [`MeanShift`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html) clustering model (`estimator_meanshift`) with the supplied parameters (&#x2699;) and training data (`X`). Includes StandardScaler in pipeline for proper feature scaling. |
| <nobr>`sk-cluster-dbscan`</nobr> | `estimator_dbscan`,<br>&#x2699; `eps`,<br>&#x2699; `min_samples`,<br>&#x2699; `metric`,<br>&#x2699; `algorithm`,<br>&#x2699; `leaf_size`,<br>`X` | **DBSCAN clustering:** Create and fit a [`DBSCAN`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) clustering model (`estimator_dbscan`) with the supplied parameters (&#x2699;) and training data (`X`). Includes StandardScaler in pipeline which is critical for distance-based algorithms. |
| <nobr>`sk-cluster-hdbscan`</nobr> | `estimator_hdbscan`,<br>&#x2699; `min_cluster_size`,<br>&#x2699; `min_samples`,<br><nobr>&#x2699; `cluster_selection_epsilon`,</nobr><br>&#x2699; `cluster_selection_method`,<br>&#x2699; `alpha`,<br>&#x2699; `metric`,<br>&#x2699; `alpha`,<br>&#x2699; `algorithm`,<br>&#x2699; `leaf_size`,<br>`X` | **HDBSCAN clustering:** Create and fit an [`HDBSCAN`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html) clustering model (`estimator_hdbscan`) with the supplied parameters (&#x2699;) and training data (`X`). |
| <nobr>`sk-cluster-predict`</nobr> | `estimator`,<br>`X` | **Cluster prediction:** Apply a clustering model (`estimator`) to an input dataset (`X`) to predict cluster labels using the model's `predict()` function. Cluster labels are output to a new dataset (`X_estimator_cluster`). Only available for models that support prediction on new data. |
| | | |

### 2.8 Density estimation

The following snippets are triggered by `sk-density` and provide features for various types of density estimation models. Density estimation is an unsupervised learning task that creates a model of the probability distribution from which the observed data is drawn. These models can be used to generate new samples, detect outliers, and estimate the likelihood of data points.

| Snippet       | Placeholders | Description |
| --- | --- | --- |
| <nobr>`sk-density-kernel`</nobr> | `estimator_kernel_density`,<br>&#x2699; `bandwidth`,<br>&#x2699; `kernel`,<br>&#x2699; `metric`,<br>`X` | **Kernel Density Estimation:** Create and fit a [`KernelDensity`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KernelDensity.html) density estimation model (`estimator_kernel_density`) with the supplied parameters (&#x2699;) and training data (`X`). |
| <nobr>`sk-density-gaussian-mixture`</nobr> | `estimator_gaussian_mixture`,<br>&#x2699; `n_components`,<br>&#x2699; `covariance_type`,<br>&#x2699; `init_params`,<br>&#x2699; `random_state`,<br>&#x2699; `max_iter`,<br>`X` | **Gaussian Mixture Model:** Create and fit a [`GaussianMixture`](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html) density estimation model (`estimator_gaussian_mixture`) with the supplied parameters (&#x2699;) and training data (`X`). |
| <nobr>`sk-density-sample-kernel`</nobr> | `estimator`,<br>&#x2699; `n_samples`,<br>&#x2699; `random_state`, | **Sample from Kernel Density model:** Generate random samples from a fitted kernel density model (`estimator`) using the `sample()` function with the supplied parameters (&#x2699;). Samples are output to a new dataset (`estimator_samples`). |  
| <nobr>`sk-density-sample-gaussian-mixture`</nobr> | `estimator`,<br>&#x2699; `n_samples`, | **Sample from Gaussian Mixture model:** Generate random samples from a fitted Gaussian Mixture density model (`estimator`) using the `sample()` function with the supplied parameters (&#x2699;). Samples are output to new datasets (`estimator_samples`, and `estimator_components`). |  
| <nobr>`sk-density-score-samples`</nobr>     | `estimator`,<br>`X` | **Density score of each sample:** Apply a density estimation model (`estimator`) to an input dataset (`X`) to evaluate the log-likelihood of each sample using the model's `score_samples()` function. The log-likelihood is output to a new dataset (`X_estimator_density`), and is normalized to be a probability density, so the value will be low for high-dimensional data. |
| <nobr>`sk-density-score`</nobr>     | `estimator`,<br>`X` | **Density score:** Apply a density estimation model (`estimator`) to an input dataset (`X`) to evaluate the total log-likelihood of the data in `X` using the model's `score()` function. This is normalized to be a probability density, so the value will be low for high-dimensional data. |  
| | | |

### 2.9 Dimensionality reduction 

The following snippets are triggered by `sk-embed` and provide features for various types of dimensionality reduction or embedding. Dimensionality reduction algorithms transform data from a high-dimensional space into a lower-dimensional representation while preserving the most important structure or information. These techniques are useful for visualization, computational efficiency, and removing redundant features.  

> **Note**: The `umap` package may need to be installed separately using `pip install umap-learn`.

| Snippet | Placeholders | Description |
| ---     | ---          | ---          |
| <nobr>`sk-embed-pca`          | `estimator_pca`,<br>&#x2699; `n_components`,<br>&#x2699; `whiten`<br>`X` | **Principal Component Analysis:** Create and fit a [`PCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) dimensionality reduction model (`estimator_pca`) with the supplied parameters (&#x2699;) and  training data (`X`). |
| <nobr>`sk-embed-kpca`         | `estimator_kpca`,<br>&#x2699; `n_components`,<br>&#x2699; `kernel`,<br>&#x2699; `gamma`,<br>&#x2699; `degree`,<br>&#x2699; `coef0`<br>`X` | **Kernel PCA:** Create and fit a [`KernelPCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html) dimensionality reduction model (`estimator_kpca`) with the supplied parameters (&#x2699;) and  training data (`X`). |
| <nobr>`sk-embed-lle`</nobr>          | `estimator_lle`,<br>&#x2699; `n_components`,<br>&#x2699; `n_neighbors`,<br>&#x2699; `method`<br>`X` | **Locally Linear Embedding:** Create and fit a [`LocallyLinearEmbedding`](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html) dimensionality reduction model (`estimator_lle`) with the supplied parameters (&#x2699;) and  training data (`X`). |
| <nobr>`sk-embed-isomap`</nobr>       | `estimator_isomap`,<br>&#x2699; `n_components`,<br>&#x2699; `n_neighbors`,<br>&#x2699; `radius`,<br>&#x2699; `p`<br>`X` | **Isometric Mapping:** Create and fit an [`Isomap`](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html) dimensionality reduction model (`estimator_isomap`) with the supplied parameters (&#x2699;) and  training data (`X`). |
| <nobr>`sk-embed-mds`</nobr>          | `estimator_mds`,<br>&#x2699; `n_components`,<br>&#x2699; `metric`,<br>&#x2699; `n_init`,<br>&#x2699; `random_state`,<br><nobr>&#x2699; `normalized_stress`</nobr><br>`X` | **Multidimensional Scaling:** Create and fit an [`MDS`](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html) dimensionality reduction model (`estimator_mds`) with the supplied parameters (&#x2699;) and  training data (`X`). |
| <nobr>`sk-embed-spectral`</nobr>     | `estimator_spectral`,<br>&#x2699; `n_components`,<br>&#x2699; `affinity`,<br>&#x2699; `gamma`,<br>&#x2699; `random_state`,<br>&#x2699; `n_neighbors`<br>`X` | **Spectral Embedding:** Create and fit a [`SpectralEmbedding`](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html) dimensionality reduction model (`estimator_spectral`) with the supplied parameters (&#x2699;) and  training data (`X`). |
| <nobr>`sk-embed-tsne`</nobr>         | `estimator_tsne`,<br>&#x2699; `n_components`,<br>&#x2699; `perplexity`,<br>&#x2699; `random_state`,<br>&#x2699; `n_iter`<br>`X` | **t-Distributed Stochastic Neighbour Embedding:** Create and fit a [`TSNE`](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) dimensionality reduction model (`estimator_tsne`) with the supplied parameters (&#x2699;) and  training data (`X`). Embedding is output to `X_estimator_tsne`. | 
| <nobr>`sk-embed-nca`</nobr> | `estimator_nca`,<br>&#x2699; `n_components`,<br>&#x2699; `init`,<br>&#x2699; `random_state`<br>`X`, `y` | **Neighborhood Components Analysis:** Create and fit a [`NeighborhoodComponentsAnalysis`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NeighborhoodComponentsAnalysis.html) dimensionality reduction model (`estimator_nca`) with the supplied parameters (&#x2699;) and training data (`X`, `y`). |
| <nobr>`sk-embed-umap`</nobr> | `estimator_umap`,<br>&#x2699; `n_components`,<br>&#x2699; `metric`,<br>&#x2699; `n_neighbors`,<br>&#x2699; `min_dist`,<br>&#x2699; `random_state`,<br>`X` | **Uniform Manifold Approximation and Projection:** Create and fit a [`UMAP`](https://umap-learn.readthedocs.io/en/latest/api.html#umap.UMAP) dimensionality reduction model (`estimator_umap`) with the supplied parameters (&#x2699;) and training data (`X`). |
| <nobr>`sk-embed-transform`</nobr>     | `estimator`,<br>`X` | **Embedding transform:** Apply a dimensionality reduction model (`estimator`) to an input dataset (`X`) using the model's `transform()` function. Embedding is output to a new dataset (`X_estimator`). |
| | | |

### 2.10 Anomaly detection

The following snippets are triggered by `sk-anomaly` and provide features for various types of anomaly detection models. Anomaly detection is the task of identifying rare items, events, or observations that differ significantly from the majority of the data. These models learn patterns in the data and can detect unusual instances that do not conform to expected behavior. 

| Snippet       | Placeholders | Description |
| --- | --- | --- |
| <nobr>`sk-anomaly-one-class-svm`</nobr> | `estimator_one_class_svm`<br>&#x2699; `kernel`,<br>&#x2699; `gamma`,<br>&#x2699; `nu`,<br>&#x2699; `shrinking`,<br>`X` | **One-Class Support Vector Machine:** Create and fit a [`OneClassSVM`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html) anomaly detection model (`estimator_one_class_svm`) with the supplied parameters (&#x2699;) and  training data (`X`). |
| <nobr>`sk-anomaly-one-class-svm-sgd`</nobr> | `estimator_one_class_svm_sgd`<br>&#x2699; `kernel`,<br>&#x2699; `gamma`,<br>&#x2699; `n_components`,<br>&#x2699; `random_state`,<br>&#x2699; `nu`,<br>&#x2699; `fit_intercept`,<br>&#x2699; `max_iter`,<br>&#x2699; `tol`,<br>&#x2699; `shuffle`,<br>&#x2699; `learning_rate`,<br>&#x2699; `eta0`,<br>`X` | **One-Class SVM with Stochastic Gradient Descent:** Create and fit a [`SGDOneClassSVM`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDOneClassSVM.html) with [`Nystroem`](https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html) kernel anomaly detection model (`estimator_one_class_svm_sgd`) with the supplied parameters (&#x2699;) and  training data (`X`). |
| <nobr>`sk-anomaly-local-outlier-factor`</nobr> | `estimator_lof`<br>&#x2699; `n_neighbors`,<br>&#x2699; `algorithm`,<br>&#x2699; `leaf_size`,<br>&#x2699; `metric`,<br>&#x2699; `contamination`,<br>&#x2699; `novelty`,<br>`X` | **Local Outlier Factor:** Create and fit a [`LocalOutlierFactor`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html) anomaly detection model (`estimator_lof`) with the supplied parameters (&#x2699;) and  training data (`X`). |
| <nobr>`sk-anomaly-isolation-forest`</nobr> | `estimator_isolation_forest`<br>&#x2699; `n_estimators`,<br>&#x2699; `max_samples`,<br>&#x2699; `contamination`,<br>&#x2699; `max_features`,<br>&#x2699; `bootstrap`,<br>&#x2699; `n_jobs`,<br>&#x2699; `random_state`,<br>`X` | **Isolation Forest:** Create and fit an [`IsolationForest`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) anomaly detection model (`estimator_isolation_forest`) with the supplied parameters (&#x2699;) and  training data (`X`). |
| <nobr>`sk-anomaly-elliptic-envelope`</nobr> | `estimator_elliptic_envelope`<br>&#x2699; `store_precision`,<br>&#x2699; `assume_centered`,<br>&#x2699; `support_fraction`,<br>&#x2699; `contamination`,<br>&#x2699; `random_state`,<br>`X` | **Elliptic Envelope (Robust Covariance):** Create and fit an [`EllipticEnvelope`](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EllipticEnvelope.html) anomaly detection model (`estimator_elliptic_envelope`) with the supplied parameters (&#x2699;) and  training data (`X`).|
| <nobr>`sk-anomaly-dbscan`</nobr> | `estimator_dbscan`<br>&#x2699; `eps`,<br>&#x2699; `min_samples`,<br>&#x2699; `metric`,<br>&#x2699; `algorithm`,<br>&#x2699; `leaf_size`,<br>`X` | **DBSCAN:** Create and fit a [`DBSCAN`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) anomaly detection model (`estimator_dbscan`) with the supplied parameters (&#x2699;) and  training data (`X`). |
| <nobr>`sk-anomaly-predict`</nobr>     | `estimator`,<br>`X` | **Anomaly prediction:** Apply an anomaly detection model (`estimator`) to an input dataset (`X`) to perform outlier classification using the model's `predict()` function. Outlier class is output to a new dataset (`X_estimator_class`) where -1 indicates outliers, and +1 indicates inliers. |
| <nobr>`sk-anomaly-score`</nobr>     | `estimator`,<br>`X` | **Anomaly score:** Apply an anomaly detection model (`estimator`) to an input dataset (`X`) to evaluate the outlier score of each sample using the model's `descision_function()` function. Outlier score is output to a new dataset (`X_estimator_score`) where negative scores indicate outliers, and positive scores indicate inliers. |
| | | |

### 2.11 Model inspection

The following snippets are triggered by `sk-inspect` and provide features for inspecting and understanding fitted models. Model inspection tools help data scientists gain insights into how their models make predictions, which features are most important, and how changes in input features affect model outputs. These techniques are essential for explaining model behavior, debugging models, and ensuring that models are behaving as expected.

| Snippet       | Placeholders | Description |
| --- | --- | --- |
| <nobr>`sk-inspect-partial_dependence`</nobr> | `estimator`,<br>`X`,<br>&#x2699; `features`,<br>&#x2699; `percentiles`,<br>&#x2699; `grid_resolution`,<br>&#x2699; `kind`| **Partial dependence:** Compute the partial dependence of a model (`estimator`) against an input feature dataset (`X`) using the [`partial_dependence`](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.partial_dependence.html) function with the supplied parameters (&#x2699;). Output partial dependence curves are returned as a dictionary (`estimator_partial`). |
| <nobr>`sk-inspect-permutation_importance`</nobr> | `estimator`,<br>`X`, `y`,<br>&#x2699; `scoring`,<br>&#x2699; `n_repeats`,<br>&#x2699; `random_state`,<br>&#x2699; `sample_weight`,<br>&#x2699; `max_samples` | **Permutation importance:** Compute the permutation importance of a model (`estimator`) against an input dataset (`X`,`y`) using the [`permutation_importance`](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html) function with the supplied parameters (&#x2699;). Output feature importance metrics are returned as a dictionary (`estimator_permutation`). | 
| | | |


### 2.12 Model persistence

#### 2.12.1 Pickle

The following snippets are triggered by `sk-io-pickle` and provide features for reading and writing <nobr>`scikit-learn`</nobr> models in `pickle` format. Python's built-in `pickle` module enables saving and loading trained models in binary format. Loading pickle files from untrusted or unknown sources poses significant security risks, as malicious pickle files could execute arbitrary code. As a result it is strongly recommended to only load `pickle` files from trusted sources.

| Snippet              | Placeholders           | Description |
| ---                  | ---                    | ---             |
| <nobr>`sk-io-pickle-read`</nobr>     | `file`,<br>`estimator` | Read an existing model (`estimator`) from a (`.pickle`) format file (`file`). Report the model type and any fitted parameters. |
| <nobr>`sk-io-pickle-write`</nobr>    | `file`,<br>`estimator` | Write a model (`estimator`) to a (`.pickle`) format file (`file`).     |
| | | |


#### 2.12.2 Skops

The following snippets are triggered by `sk-io-skops` and provide features for reading and writing <nobr>`scikit-learn`</nobr> models in `skops` format. The `skops` package provides a more secure alternative to `pickle` for saving and loading models. It includes additional security features like model verification and safer deserialization to mitigate potential security risks associated with loading models from untrusted sources.  

> **Note**: The `skops` package may need to be installed separately using `pip install skops`.


| Snippet              | Placeholders           | Description |
| ---                  | ---                    | ---             |
| <nobr>`sk-io-skops-read`</nobr>     | `file`,<br>`estimator` | Read an existing model (`estimator`) from a (`.skops`) format file (`file`). Report the model type and any fitted parameters. |
| <nobr>`sk-io-skops-write`</nobr>    | `file`,<br>`estimator` | Write a model (`estimator`) to a (`.skops`) format file (`file`).     |
| | | |

### 2.13 Argument Snippets

Many `scikit-learn` function arguments take their values from extensive option lists. The following snippets are triggered with `sk-args` and provide features allowing you to easily select valid argument values from lists of available options. 

| Snippet       | Placeholders   | Description |
| ---           | ---            | ---         |
| <nobr>`sk-args-random_state`</nobr> | `random_state` | Set the `random_state` argument for reproducibility in randomized algorithms. This argument is the integer seed number used to initialize random number generation; a randomly chosen value is provided by default. |
| <nobr>`sk-args-alphas`</nobr> | `logspace` or `linspace`,<br>`start`, `stop`, `num` | Set the `alphas` argument (a logarithmic or linear sequence of regularization parameters) for cross-validation in `RidgeCV`, `LassoCV`, and `ElasticNetCV` regression. |
| <nobr>`sk-args-func`</nobr> | `func` | Set the `func` argument to a `FunctionTransformer` from a list of common transformations. For use in regression models with transformed features (`X`), see <nobr>`sk-regress-linear-transform`</nobr>. | 
| <nobr>`sk-args-func-inverse`</nobr> | `func`, `inverse_func` | Set the `func` and `inverse_func` arguments to a `FunctionTransformer` from a list of common forward and inverse transformation function pairs. For use in regression models with transformed target (`y`), see <nobr>`sk-regress-linear-transform-target`</nobr>. |
| <nobr>`sk-args-spline-extrapolation`</nobr> | `extrapolation` | Set the `SplineTransformer` extrapolation behavior beyond the minimum and maximum of the training data, see <nobr>`sk-regress-linear-spline`</nobr>. |
| <nobr>`sk-args-kernel`</nobr> | `kernel` | Set the `kernel` type for kernel density estimation from an option list. You can provide this argument when creating a `KernelDensity` model, see <nobr>`sk-density-kernel`</nobr>. |
| | |

## 3. Release Notes

### 3.1 Python packages

#### Scikit-learn

The snippets provided by this extension were developed using `scikit-learn` version 1.7 but will also produce working Python code for earlier and later versions. 

See the [scikit-learn documentation](https://scikit-learn.org/stable/index.html) for details.  

#### UMAP

UMAP embedding (`sk-embed-umap`) depends on the external `umap-learn` package, which is not part of `scikit-learn`. Install it separately before use, for example:

> `pip install umap-learn`

See the [umap-learn documentation](https://umap-learn.readthedocs.io/) for details.

#### Skops

The `sk-io-skops` snippets depend on the external `skops` package, which is not part of `scikit-learn`. Install it separately before use, for example:

> `pip install skops`

See the [skops](https://skops.readthedocs.io/) documentation for details. 

### 3.2 Using snippets

#### Editor Support for Snippets
 
Snippets for producing Python code, including those provided by this extension, are supported in the Python file (`.py`) editor and in the notebook (`.ipynb`) editor.  

#### Snippets and IntelliSense  

When triggered, the default behaviour of IntelliSense is to show snippets along with other context dependent suggestions. This may result in a long list of suggestions in the IntelliSense pop-up, particularly if the snippet trigger provided by this extension (`sk`) also matches other symbols in your editor.

It's easy to modify this behaviour using your Visual Studio Code settings. To access the relevant settings go to *Preferences > Settings* and type `snippets` in the *Search settings* field as shown below:

![Snippets and IntelliSense settings](https://www.analyticsignal.com/images/vscode-snippets-intellisense.png)

You can control whether snippets are shown with other suggestions and how they are sorted using the *Editor: **Snippet Suggestions*** dropdown. Choose one of the options to control how snippet suggestions are shown in the IntelliSense popup:

| Option   | IntelliSense |  
| ------   | ------------ |
| `top`    | Show snippet suggestions on top of other suggestions. |
| `bottom` | Show snippet suggestions below other suggestions. |
| `inline` | Show snippet suggestions with other suggestions (default). | 
| `none`   | Do not show snippet suggestions. |
|||

You can also use the *Editor > Suggest: **Show Snippets*** checkbox to enable or disable snippets in IntelliSense suggestions. When snippets are disabled in IntelliSense they are still accessible through the Command Palette *Insert Snippet* command.

## 4. Snippet reference

Snippet prefix triggers are organized in a hierarchical tree structure rooted at `sk` as shown in the figure below. The snippet hierarchy is designed to ease the user's cognitive load when developing models with this large and complex machine learning package. The branches at the top of the tree outline the main steps in a machine learning workflow, branches at lower levels outline a taxonomy of algorithms for specific tasks, whereas leaf nodes represent particular algorithms. The process of inserting a snippet amounts to navigating the tree and selecting the desired leaf node by either of the methods described in [Section 1.3](#13-inserting-snippets). 

```
sk
├── sk-setup
├── sk-read
│   ├── sk-read-csv
│   ├── sk-read-excel
│   ├── sk-read-feather
│   └── sk-read-parquet
├── sk-prep
│   ├── sk-prep-target-features
│   ├── sk-prep-target-features-secondary
│   ├── sk-prep-train_test_split
│   ├── sk-prep-features
│   └── sk-prep-features-secondary
├── sk-regress
│   ├── sk-regress-linear
│   │   ├── sk-regress-linear
│   │   ├── sk-regress-linear-transform-target
│   │   ├── sk-regress-linear-transform
│   │   ├── sk-regress-linear-polynomial
│   │   ├── sk-regress-linear-spline
│   │   ├── sk-regress-linear-pcr
│   │   ├── sk-regress-linear-pls
│   │   ├── sk-regress-linear-ridge
│   │   ├── sk-regress-linear-ridgecv
│   │   ├── sk-regress-linear-lasso
│   │   ├── sk-regress-linear-lassocv
│   │   ├── sk-regress-linear-elasticnet
│   │   ├── sk-regress-linear-elasticnetcv
│   │   └── sk-regress-linear-quantile
│   ├── sk-regress-neighbors
│   │   ├── sk-regress-neighbors-k 
│   │   └── sk-regress-neighbors-radius 
│   ├── sk-regress-gaussian
│   │   ├── sk-regress-gaussian-process
│   │   ├── sk-regress-gaussian-process-kernel
│   │   └── sk-regress-gaussian-transform-target 
│   ├── sk-regress-ensemble
│   │   ├── sk-regress-ensemble-random-forest
│   │   ├── sk-regress-ensemble-extra-trees
│   │   ├── sk-regress-ensemble-gradient-boosting
│   │   ├── sk-regress-ensemble-hist-gradient-boosting
│   │   ├── sk-regress-ensemble-stacking
│   │   └── sk-regress-ensemble-voting
│   ├── sk-regress-dummy
│   └── sk-regress-report
├── sk-classify
│   ├── sk-classify-linear
│   │   ├── sk-classify-linear-lda
│   │   ├── sk-classify-linear-qda
│   │   ├── sk-classify-linear-logistic
│   │   ├── sk-classify-linear-svm
│   │   ├── sk-classify-linear-svm-sgd
│   │   ├── sk-classify-linear-perceptron
│   │   └── sk-classify-linear-ridge
│   ├── sk-classify-neighbors
│   │   ├── sk-classify-neighbors-k 
│   │   ├── sk-classify-neighbors-radius 
│   │   └── sk-classify-neighbors-centroid 
│   ├── sk-classify-bayes
│   │   ├── sk-classify-bayes-gaussian
│   │   ├── sk-classify-bayes-multinomial
│   │   ├── sk-classify-bayes-bernoulli
│   │   ├── sk-classify-bayes-complement
│   │   └── sk-classify-bayes-categorical
│   ├── sk-classify-ensemble
│   │   ├── sk-classify-ensemble-random-forest
│   │   ├── sk-classify-ensemble-extra-trees
│   │   ├── sk-classify-ensemble-gradient-boosting
│   │   ├── sk-classify-ensemble-hist-gradient-boosting
│   │   ├── sk-classify-ensemble-stacking
│   │   └── sk-classify-ensemble-voting
│   ├── sk-classify-dummy
│   └── sk-classify-report
├── sk-cluster
│   ├── sk-cluster-kmeans
│   ├── sk-cluster-kmeans-minibatch
│   ├── sk-cluster-meanshift
│   ├── sk-cluster-dbscan
│   ├── sk-cluster-hdbscan
│   └── sk-cluster-predict
├── sk-density
│   ├── sk-density-kernel
│   ├── sk-density-gaussian-mixture
│   ├── sk-density-sample-kernel
│   ├── sk-density-sample-gaussian-mixture
│   ├── sk-density-score-samples
│   └── sk-density-score
├── sk-embed
│   ├── sk-embed-pca
│   ├── sk-embed-kpca
│   ├── sk-embed-lle
│   ├── sk-embed-isomap
│   ├── sk-embed-mds
│   ├── sk-embed-spectral
│   ├── sk-embed-tsne
│   ├── sk-embed-nca
│   ├── sk-embed-umap
│   └── sk-embed-transform
├── sk-anomaly
│   ├── sk-anomaly-one-class-svm
│   ├── sk-anomaly-one-class-svm-sgd
│   ├── sk-anomaly-local-outlier-factor
│   ├── sk-anomaly-isolation-forest
│   ├── sk-anomaly-elliptic-envelope
│   ├── sk-anomaly-dbscan
│   ├── sk-anomaly-predict
│   └── sk-anomaly-score
├── sk-inspect
│   ├── sk-inspect-partial_dependence
│   └── sk-inspect-permutation_importance
├── sk-io
│   ├── sk-io-pickle  
│   │   ├── sk-io-pickle-read
│   │   └── sk-io-pickle-write
│   └── sk-io-skops
│       ├── sk-io-skops-read
│       └── sk-io-skops-write
└── sk-args
    ├── sk-args-random_state
    ├── sk-args-alphas
    ├── sk-args-func
    ├── sk-args-func-inverse
    ├── sk-args-spline-extrapolation
    └── sk-args-kernel
```

---  
Copyright &copy; 2024-2025 Analytic Signal Limited, all rights reserved
