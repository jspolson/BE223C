from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


def generate_models(trainArr, trainRes, testArr, class_weights, scorer = 'roc_auc'):
    '''
    Generate Random Forest, Logistic Regression, and Multilayer Perceptron models with and without nested cross-validation for parameter tuning.
    Inputs:
    - trainArr: training features
    - trainRes: training labels
    - testArr: testing features (only used for feature scaling, not model fitting)
    - class_weights: the weights of each class
    - scorer: scoring metric used for parameter tuning. Default: 'roc_auc'
    
    Outputs:
    - models: dictionary with every stored model
    - trainArr_s: scaled training features
    - testArr_s: scaled testing features
    '''
    models = {}
    ##### RANDOM FOREST
    rf = RandomForestClassifier(n_estimators=600, 
                                max_features = 'sqrt',
                                min_samples_leaf = 34,
                                class_weight = class_weights,
                                criterion = 'gini',
                                warm_start = False,
                                random_state = 11)
    rf.fit(trainArr, trainRes) # fit the data to the algorithm
    models['rfc'] = rf
    
    test_params = {'max_features':np.arange(0.1,1,0.1).tolist(), 
                   'min_samples_leaf':np.arange(15,40,2).tolist()}

    search = GridSearchCV(estimator = RandomForestClassifier(random_state=11, n_estimators = 40), 
                          param_grid = test_params, scoring=scorer, iid=False, cv=5)
    search.fit(trainArr, trainRes)
    par = search.best_params_
    print(par)

    rf_t = RandomForestClassifier(n_estimators=600, 
                                max_features = par['max_features'],
                                min_samples_leaf = par['min_samples_leaf'],
                                class_weight = class_weights,
                                warm_start = False,
                                random_state = 11)
    rf_t.fit(trainArr, trainRes) # fit the data to the algorithm

    models['rfc_t'] = rf_t
    
    
    ##### LOGISTIC REGRESSION
    lr = LogisticRegression(class_weight = class_weights)
    lr.fit(trainArr, trainRes)
    
    models['lr'] = lr
    
    test_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
    
    search = LogisticRegression(class_weight = class_weights)
    search.fit(trainArr, trainRes)
    
    models['lr_t'] = search
    
    ##### MLP NEURAL NETWORK
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, activation = 'relu',
                    hidden_layer_sizes=(5, 2), random_state=11)
 
    scaler = StandardScaler()  
    scaler.fit(trainArr)  
    trainArr_s = scaler.transform(trainArr)  
    # apply same transformation to test data
    testArr_s = scaler.transform(testArr)
    clf.fit(trainArr_s, trainRes)

    models['mlp'] = clf
    
    test_params = {'learning_rate': ["constant", "invscaling", "adaptive"],
                   'hidden_layer_sizes': [(50,2), (5,2), (5,5,2)],
                   'activation': ["logistic", "relu", "tanh"]
                    }
    
    
    search = GridSearchCV(estimator = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state = 11), 
                          param_grid = test_params, scoring=scorer, iid=False, cv=5)
    search.fit(trainArr_s, trainRes)
    print(search.best_params_)
    
    models['mlp_t'] = search
    
    return models, trainArr_s, testArr_s


def binary_metrics (labels, preds):
    '''
    Generates binary metrics given a set of labels and predictions.
    Inputs:
    - labels, preds
    
    Outputs:
    - metrics: array detailing the accuracy, recall, precision, and F-1 score given the labels
    '''
    preds = preds > 0.75

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    #Accuracy
    acc  = round((tp + tn)/(tp + tn + fp + fn)*100 , 3)

    #Recall
    rec  = round(tp/(tp + fn)*100, 3)

    #Precision
    prec = round(tp/(tp + fp)*100,3)

    #F1 Score
    f1   = round(2*tp/((2*tp) + fp + fn)*100,3)
    
    metrics = [acc, rec, prec, f1]
    return metrics

from sklearn.model_selection import StratifiedKFold


def run_models (featArr, tss_label, num_folds = 10):
    '''
    Runs the specified models as called by generate_models function
    Inputs:
    - featArr: features
    - tss_label: labels
    - num_folds: number of cross-validation folds to use. Default: 10
    
    Outputs:
    - plots for each fold, detailing the ROC-AUC curve for each model
    - fold_results: dictionary containing the roc_auc score, other evaluation metrics, binary labels for each test case, and probabilities
    '''
    
    unique, counts = np.unique(tss_label, return_counts=True)
    class_weights = {unique[i]:counts[i] for i in np.arange(0,len(counts),1)}

    fold_results = {}

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=100)
    fold = 0
    for train_index, test_index in skf.split(featArr,tss_label):
        fold += 1
        trainArr = np.nan_to_num(np.asarray([featArr[i] for i in train_index]), copy = True)
        trainRes = np.nan_to_num(np.asarray([tss_label[i] for i in train_index]), copy = True)
        testArr = np.nan_to_num(np.asarray([featArr[i] for i in test_index]), copy = True)
        testRes = np.nan_to_num(np.asarray([tss_label[i] for i in test_index]), copy = True)


        models, trainArr_s, testArr_s = generate_models(trainArr, trainRes, testArr, class_weights)

        plt.figure(1)
        plt.figure(figsize=(20,10))
        plt.plot([0, 1], [0, 1], 'k--')

        results = {}
        probabilities = {}
        fpr = {}
        tpr = {}
        roc_auc = {}
        metrics = {}

        for key, model in models.items():
            if 'mlp' in key:
                results[key] = model.predict(testArr_s)
                probabilities[key] = model.predict_proba(testArr_s)[:,1]    
            else:
                results[key] = model.predict(testArr)
                probabilities[key] = model.predict_proba(testArr)[:,1]
            fpr[key], tpr[key], _ = roc_curve(testRes, probabilities[key])
            roc_auc[key] = auc(fpr[key], tpr[key])
            metrics[key] = binary_metrics(testRes, probabilities[key])
            plt.plot(fpr[key], tpr[key], label = str(key) +": "+ str(round(roc_auc[key], 3)))

        fold_results[fold] = {'roc_auc': roc_auc, 
                              'metrics': metrics,
                              'probabilities':probabilities,
                              'labels': testRes}


        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.show()

    return fold_results
