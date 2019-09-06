# feature-selector
Uses p-correlation to select and eliminate features for a SciKit-Learn ML pipeline

The primary module in this repo is src/ennuipy/FeatureSelector.

The FeatureSelector works by ranking the p-correlation between the target and a large number of candidate features. THe highest ranked feature is selected.

Then all features that correlate strongly with the selected feature are then eliminated.

Further features are selected using the above steps recursively.

I believe that the above algorithm is original.

The method is validated using the WBDI data set.

In all cases the feature selection improves learning and fit scores, although only marinally for Naive Bayes.

## Usage
The following hyper-parameters are provided:
* feature_corr_min=0.45 - only features with higher correlations to the target are selectable
* feature_x_corr_max=0.5 - all features that have a higher correlation with a previously selected feature are eliminated

This repo also contains sklearn pipelines that use the selector.
