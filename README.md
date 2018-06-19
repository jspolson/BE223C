# BE223C Final Project
------

Welcome to the 223C Final Project by Jennifer Polson. Here's an overview of the code components and how they all work.

* `Image_Processing.ipynb`: a notebook that takes the files provided through the course and stores the half brains for the relevant modalities (Tmax, DWI, and FLAIR). This outputs a pickled dictionary (dictionary not included in the folder due to size and privacy constraints).
* `ML_Models.ipynb`: notebook that uses the dictionary outputted from `Image_Processing`, runs the two feature generation methods outlined in the project summary, and runs models. The graphs show the ROC-AUC curve for each model/fold. Relies on code from the following scripts:
    + `feature_utils.py`: contains the functions used to generate features, including the feature generation for the cluster and non-cluster method.
    + `model_utils.py`: contains functions to generate the models outlined in the project summary, run the models, and evaluate them for important metrics. 
