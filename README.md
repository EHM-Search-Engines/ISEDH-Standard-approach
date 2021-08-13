# EHM-BEP-2021

This repository contains four datasets which were used throughout our thesis.
- `small` was used to do preliminary testing on algorithms.
- `medium-ds` was used to decide on the specific combination of extractors and matchers. The result was SIFT extractors and a BF matcher
- `grande` was used to thoroughly test our algorithms and to test our parameters optimised in the medium-ds dataset
- Finally `ehm_dataset` was used to generate comparison precision recall curves against the Deep Learning approach group
# Usage
In every dataset there is an `adjust.csv` file which contains for every image if it matches all the other images. The idea is that this file will be used to estimate the correct threshold to obtain the highest possible performance.

Every script has a variable at the top: `TEST_DATA` which should point to the directory where the dataset is located (also where the `adjust.csv` file is located).

1. First the `extract_features_mthreaded.py` should be run. This script extracts all the sift features of the images in the dataset and saves them in a `data-keypoints.csv` file.
2. Then the features will be matched against eachother using the `write_threshold_csv_mthreaded.py` script. The output is the file `threshold.csv`.

...This file contains all the mathes between all image comparisons, thus it contains all the bad and the good matches. In the next step the thresholds will be calculated to filter the bad matches out, leaving only good matches.

3. Using the previously generated `threshold.csv` file, the ideal threshold parameters are calculated using the `calculate_threshold.py` script.

4. Now that the ideal thresholds have been calculated, the confusion matrix can be generated using `compare_confusion_matrix.py`, or one could generate the precision recall curves by varying every threshold individually using `calculate_PRC_per_param.py`.