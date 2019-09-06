# feature-selector
Uses p-correlation to select and eliminate features for a  [SciKit-Learn](https://scikit-learn.org/stable) ML pipeline

The primary module in this repo is src/ennuipy/FeatureSelector.

FeatureSelector implements the SKLearn [BaseEstimator](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html), [SelectorMixin](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_selection/base.py) interfaces, ie the fit, transform, etc methods so that it can be used in piplines.

The FeatureSelector works by ranking the Pearson Correlation between the target and a large number of candidate features. The highest ranked feature is selected.

Then all features that correlate strongly with the selected feature are then eliminated.

Further features are selected using the above steps recursively.

I believe that the above algorithm is original.

The method is validated using the [World Bank Development Index](https://databank.worldbank.org/source/world-development-indicators) data set as features to predict happiness scores from the [United Nations World Happiness Report](https://worldhappiness.report/).

In all cases the feature selection improves learning and fit scores, although only marinally for Naive Bayes.

## Usage
The following hyper-parameters are provided:
* feature_corr_min=0.45 - only features with higher correlations to the target are selectable
* feature_x_corr_max=0.5 - all features that have a higher correlation with a previously selected feature are eliminated

The WBDI time series data should be converted to simple instance data using the [WBDI Excel Cleaner](https://github.com/KoosieDeMoer/wbdi-excel-cleaner)

This repo also contains sklearn pipelines that use the selector.

## Interesting Results

    Fit=0.5375763907411308
    Score=0.6731481565060642
    ['SE.SEC.ENRR', 'SH.XPD.GHED.GE.ZS', 'TX.VAL.MRCH.HI.ZS', 'SH.XPD.EHEX.PP.CD']
    {'alpha_1': 1e-06, 'alpha_2': 1e-06, 'compute_score': False, 'copy_X': True, 'fit_intercept': True, 'lambda_1': 1e-06, 'lambda_2': 1e-06, 'n_iter': 300, 'normalize': False, 'tol': 0.001, 'verbose': False}
    
The above indicates that of the 995 available features the the following four are best suited to learning from:
1. School enrollment, secondary (% gross)
1. Domestic general government health expenditure (% of general government expenditure)
1. Merchandise exports to high-income economies (% of total merchandise exports)
1. External health expenditure per capita, PPP (current international $)

![Fit results](results/Figure_1.png?raw=true "Fit results")




