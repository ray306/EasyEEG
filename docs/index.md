# Overview

[![PyPI Version][pypi-v-image]][pypi-v-link]

[pypi-v-image]: https://img.shields.io/pypi/v/easyEEG.png
[pypi-v-link]: https://pypi.python.org/pypi/easyEEG

EasyEEG provides simple, flexible and powerful methods that can be used to directly test neural and psychological hypotheses based on topographic responses. These multivariate methods can investigate effects in the dimensions of response magnitude and topographic patterns separately using data in the sensor space, therefore enable assessing neural sources and its dynamics without sophisticated localization. Python based algorithms provide concise and extendable features of Cafe. Users of all levels can benefit from Cafe and obtain a straightforward solution to efficiently handle and process EEG data and a complete pipeline from raw data to publication.  

**Highlights**:
Simplifying the analysis
- For beginners: clear logics for the steps (least concepts);
- For data players (who want to check various aspects of data or to try various analysis algorithms): one-line code for each idea;
- For data players (who want to apply DIY analysis methods): plenty of low/high level functions for quickly building your own algorithm.

Advantages:
- Don't need to extract the data in a messy way when you want to do new analysis. You can just use a string or a dictionary to describe the target now.
- Applying EEG analysis method by one-line code. (ERP, topography, specturm, time-frequency, etc.)
- Applying advanced EEG analysis methods (clustering, classification, etc.)
- Beautiful ploting
- Plenty of basic and advanced APIs which can help to build your own analysis algorithm and to visualize the result.

todo

**limitation**:

todo

---
## Documentation
See http://easyeeg.readthedocs.io/en/latest/ for introduction, tutorials, and reference manual.

---
# Installation instructions

The simplest way to install EasyEEG is through the Python Package Index (PyPI), which ensures that all required dependencies are established. This can be achieved by executing the following command:

```
pip install easyEEG
```
or:
```
sudo pip install easyEEG
```

The command of getting update:
```
pip install --upgrade easyEEG --no-deps
```
or:
```
sudo pip install --upgrade easyEEG --no-deps
```

### *Required Dependencies*

- numpy
- pandas
- scipy
- matplotlib
- statsmodels
- seaborn
- mne
- permute
- tqdm
- ipdb
