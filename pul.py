#!/usr/bin/env python
# coding: utf-8

from sklearn.datasets import make_classification,make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, recall_score,f1_score,make_scorer
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from sklearn.datasets import load_breast_cancer
from ucimlrepo import fetch_ucirepo 


PUs=[]
SUPs=[]
NEGRATEs=[]
class_seps=[0.001,0.001,0.001,0.01,0.01,0.01,0.1,0.1,0.1,0.5,0.5,0.5,1.0,1.0,1.0,'breast_cancer','breast_cancer','breast_cancer','pima','pima','pima','CDC_diabetes','CDC_diabetes','CDC_diabetes','tuandromd','tuandromd','tuandromd']

for class_sep in class_seps:
    if class_sep == 'moons':
        X, y = make_moons(n_samples=40000, shuffle=True, noise=5, random_state=None)
        input_df=pd.DataFrame(X)
        input_df['label']=y
        input_df=pd.concat([input_df[input_df.label == 0].sample(n=16000),input_df[input_df.label == 1].sample(n=4000)]).sample(frac=1).reset_index(drop=True)
        X=input_df.iloc[:,:-1].values
        y=input_df.iloc[:,-1].values
        del input_df
    elif class_sep == 'breast_cancer':
        X,y=load_breast_cancer(return_X_y=True)
        y=1-y #the dataset currently has benign as 1 and malignant as 0.  Let's switch it
    elif class_sep == 'pima':
        X=pd.read_csv('pima_diabetes.csv',header=None).iloc[:,:-1].values
        y=pd.read_csv('pima_diabetes.csv',header=None).iloc[:,-1].values
    elif type(class_sep) == type(-2.0): #create a toy binary classification dataset
        X, y = make_classification(n_samples=20000, flip_y=0, n_features=20, n_informative=4, 
                            n_redundant=5,shuffle=False, weights=[0.8,0.2], class_sep=class_sep)
    
    #create the data frame
    input_df=pd.DataFrame(X,columns=[f'feat_{n}' for n in range(X.shape[1])])
    input_df['label']=y
    
    if class_sep == 'CDC_diabetes':
        # fetch dataset 
        cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 
        
        # data (as pandas dataframes) 
        X = cdc_diabetes_health_indicators.data.features 
        y = cdc_diabetes_health_indicators.data.targets
        
        input_df=pd.concat([X,y],axis=1)
        input_df=input_df.rename(columns={'Diabetes_binary':'label'})
    
    if class_sep == 'tuandromd':
        # fetch dataset 
        input_df=pd.read_csv('TUANDROMD.csv')
        input_df=input_df.rename(columns={'Label':'label'})
    
    #drop NaN values
    df = input_df.dropna().copy()
    
    #create relabel column for tracking
    df['relabel']=df.label.values
    features=df.shape[1]-2 # -2 for the label and relabel columns
    
    ratio=df[df.label == 0].shape[0]/df[df.label == 1].shape[0]
    #split unlabeled cases and positives for train, spies, and test using 60/20/10 split of positives
    positives_train=int(df[df.label == 1].shape[0]*0.6)
    positives_spies=int(df[df.label == 1].shape[0]*0.2)
    positives_test=int(df[df.label == 1].shape[0]*0.1)
    negatives_test=int(positives_test*ratio)
    unlabeled=df[df.label == 0].shape[0] - negatives_test
    
    #use the remaining 10% to mislabel to practically test their recovery
    mislabeled=df[df.label == 1].shape[0] - positives_train - positives_spies - positives_test
    
    positives_test=df[df.label == 1].sample(n=positives_test)
    df=df.drop(index=positives_test.index)
    negatives_test=df[df.label==0].sample(n=negatives_test)
    df=df.drop(index=negatives_test.index)
    test=pd.concat([positives_test,negatives_test])
    
    #create dataframes
    unlabeled=df[df.label == 0].sample(n=unlabeled)
    df=df.drop(index=unlabeled.index)
    positives_train=df[df.label == 1].sample(n=positives_train)
    df=df.drop(index=positives_train.index)
    positives_spies=df[df.label == 1].sample(n=positives_spies)
    df=df.drop(index=positives_spies.index)
    
    #take the rest of the positives to mislabel them into the unlabeled pool to measure our recovery
    mislabeled=df[df.label == 1].sample(n=mislabeled)
    df=df.drop(index=mislabeled.index)
    mislabeled.loc[mislabeled.label==1,'relabel']=0
    
    #define the entire dataset used in training for later
    data=pd.concat([positives_train,positives_spies,unlabeled,mislabeled])
    
    max_depth= 3 if int(features**0.5) < 3 else int(features**0.5) #depth of each tree in random forest model
    n_estimators=100 # number of trees in random forest model
    k=1.5 #factor for tukey fence.  1.5 is typical.  3.0 is "way out"
    
    #initiliaze the negatives and their probability
    negatives=pd.concat([unlabeled,mislabeled])
    negatives['prob']=0
    old_idx=set([-1])
    
    #initialize some lists for recording
    rounds=[0]
    limits=[0.5]
    neg_samples=[0]
    recoveries=[0]
    sims=[0]
    min_rounds=rounds[-1]+5
    
    #iteratively remove likely positives from the negative pool
    for n in range(rounds[-1]+1,rounds[-1]+101):
        #construct the training set
        #positives=pd.concat([positives_train,positives_spies])
        #positives_train=positives.sample(n=len(positives_train))
        #positives_spies=positives[positives.index.isin(positives_train.index) == False]
        if negatives.shape[0] > 0:
            train=pd.concat([positives_train,negatives.sample(frac=0.5,replace=True)])
        else:
            negatives=pd.concat([negatives,recovered])
            train=pd.concat([positives_train,negatives.sort_values(by=['prob']).head(len(positives_train))])
        
        X_train=train.iloc[:,0:features].values
        y_train=train['relabel'].values
        
        if sims[-1] < 0.99:    
            #hyperparameter tune by grid search
            param_grid = {
                'max_depth': [3] if int(features**0.5) < 3 else list(range(3,2*int(features**0.5))),
            }
            clf=RandomForestClassifier(class_weight="balanced",n_jobs=-1)
            grid_search = GridSearchCV(estimator = clf, param_grid = param_grid, cv = 3, n_jobs = -1, scoring='precision')
            grid_search.fit(X_train,y_train)
            max_depth=grid_search.best_params_['max_depth']
        
        #train the classifier
        clf=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,class_weight="balanced",n_jobs=-1)
        clf.fit(X_train, y_train)
        
        #predict on everything
        X_data=data.iloc[:,0:features].values
        y_data=data['relabel'].values
        y_data_prob=clf.predict_proba(X_data)#[:,1]
        y_data_pred=clf.predict(X_data)#np.where(y_spies_prob >= 0.5, 1,0)
        
        #if all the predictions are the same, clf.predict will return an nx1 matrix, so probabilities need adjusted in the case of all negatives
        if y_data_pred.mean() == 0:
            y_data_prob=1-y_data_prob
        else:
            y_data_prob=y_data_prob[:,-1]
        
        data['prob']=y_data_prob
        
        #define each pool
        positives_train=data[data.index.isin(positives_train.index)]
        positives_spies=data[data.index.isin(positives_spies.index)]
        y_spies=positives_spies.label.values
        
        #use a Tukey fence to set the lower limit from known positives
        IQR=positives_spies.prob.quantile(0.75)-positives_spies.prob.quantile(0.25)
        pos_limit=positives_spies.prob.quantile(0.25)-k*IQR
            
        #assign the prediction labels based on the pos_limit
        y_data_pred=np.where(y_data_prob >= pos_limit, 1,0)
        data['pred']=y_data_pred
        
        #define each pool, just to get the spies predictions
        positives_train=data[data.index.isin(positives_train.index)]
        positives_spies=data[data.index.isin(positives_spies.index)]
        y_spies=positives_spies.label.values
        y_spies_pred=positives_spies.pred.values
        
        #separate recovered positives and negatives based on the tukey fence
        recovered=data[(data.relabel == 0) & (data.prob >= pos_limit)]
        negatives=data[(data.relabel == 0) & (data.prob < pos_limit)]
        
        #calculate the recovery of mislabeled positives
        recovery=recovered[recovered.label == 1].shape[0]/mislabeled.shape[0]
        recoveries.append(recovery)
        
        # record a few things
        neg_samples.append(negatives.shape[0])
        rounds.append(n)
        limits.append(pos_limit)
        
        #we can add in similarity to negative indices from the previous round and the current round
        #this would ensure we are converging not just on the same number of negatives, but the same negatives themselves
        #https://www.geeksforgeeks.org/python-percentage-similarity-of-lists/
        new_idx=set(negatives.index.tolist())
        common_elements=new_idx.intersection(old_idx)
        num_common_elements = len(common_elements)
        total_elements = set(old_idx).union(new_idx)
        num_total_elements = len(total_elements)
        neg_set_sim = (num_common_elements / num_total_elements)
        sims.append(neg_set_sim)
        old_idx=set(negatives.index.tolist())
        
        # we need to iterate at least three rounds to calculate derivatives by difference formulas
        if n < 3:
            continue
    
        #break if a couple metrics converge according to the relative slope at the last point
        #a very small number is added to the denominator to prevent division by zero
        negs_change=abs((neg_samples[-3]-4*neg_samples[-2]+3*neg_samples[-1])/2/(neg_samples[-1]+0.0000001))
        sim_change=abs((sims[-3]-4*sims[-2]+3*sims[-1])/2/(sims[-1]+0.0000001))
        recovery_change=abs((recoveries[-3]-4*recoveries[-2]+3*recoveries[-1])/2/(recoveries[-1]+0.0000001))
        limits_change=abs((limits[-3]-4*limits[-2]+3*limits[-1])/2/(limits[-1]+0.0000001))
        #print(negs_change,sim_change,limits_change)
        
        if negs_change < 0.01 and sim_change < 0.05 and n >= min_rounds:# and recovery_change < 0.05 and limits_change < 0.025
            break
    
    #calculate an enrichment factor for mislabeled positives in the spies set compared to all the mislabeled positives in all the unlabeled data
    negrate=negatives.shape[0]/unlabeled.shape[0] #data[data.relabel == 0].shape[0]
    NEGRATEs.append(negrate)
    
    train=pd.concat([positives_train,positives_spies,negatives])
    X_train=train.iloc[:,0:features].values
    y_train=train['relabel'].values #since all unlabeled samples were originally assigned a 0 for their labe, we use the label value here
    
    #hyperparameter tune by grid search
    param_grid = {
        'max_depth': [3] if int(features**0.5) < 3 else list(range(3,2*int(features**0.5))),
    }
    clf=RandomForestClassifier(class_weight="balanced",n_jobs=-1)
    grid_search = GridSearchCV(estimator = clf, param_grid = param_grid, cv = 3, n_jobs = -1,scoring='precision')
    grid_search.fit(X_train,y_train)
    max_depth=grid_search.best_params_['max_depth']
    
    #fit the final classifier
    clf=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,class_weight="balanced",n_jobs=-1)
    clf.fit(X_train, y_train)
        
    #run the classifier on the final test set
    X_test=test.iloc[:,0:features].values
    y_test=test['label'].values
    y_test_prob=clf.predict_proba(X_test)#[:,1]
    y_test_pred=clf.predict(X_test)#np.where(y_test_prob >= 0.5, 1,0)
    
    #if all the predictions are the same, clf.predict will return an nx1 matrix, so probabilities need adjusted in the case of all negatives
    if y_test_pred.mean() == 0:
        y_test_prob=1-y_test_prob
    else:
        y_test_prob=y_test_prob[:,-1]
    
    test['prob']=y_test_prob
    y_test_pred=np.where(y_test_prob >= limits[-1], 1,0)
    
    #summarize findings
    PU=recall_score(y_test, y_test_pred,pos_label=0)
    
    #get a baseline on traditional supervised learning
    #remake the training dataset only with 
    train_baseline=pd.concat([positives_train,positives_spies,mislabeled,unlabeled])
    X_train_baseline=train_baseline.iloc[:,0:features].values
    y_train_baseline=train_baseline['label'].values
    param_grid = {
                'max_depth': [3] if int(features**0.5) < 3 else list(range(3,2*int(features**0.5))),
            }
    clf3=RandomForestClassifier(class_weight="balanced",n_jobs=-1)
    grid_search = GridSearchCV(estimator = clf3, param_grid = param_grid, cv = 3, n_jobs = -1, scoring='precision')
    grid_search.fit(X_train,y_train)
    max_depth=grid_search.best_params_['max_depth']
    clf3=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,class_weight="balanced",n_jobs=-1)
    clf3.fit(X_train_baseline, y_train_baseline)
    y_train_pred=clf3.predict(X_train_baseline)
    X_test=test.iloc[:,0:features].values
    y_test=test['label'].values
    y_test_prob=clf3.predict(X_test)
    y_test_pred=clf3.predict(X_test)
    
    SUP=recall_score(y_test, y_test_pred,pos_label=0)
    PUs.append(PU)
    SUPs.append(SUP)


df=pd.DataFrame({'PU_Model':PUs,'Baseline_Model':SUPs,'Class_Separation':class_seps,'Neg_Sample_Ratio':NEGRATEs})

from sklearn.metrics import r2_score
r2score=r2_score(df.PU_Model,df.Neg_Sample_Ratio)
sns.scatterplot(df,x='PU_Model',y='Neg_Sample_Ratio',hue='Class_Separation',palette=sns.color_palette())
plt.xlabel('Negative Recall of Final PU Model on Test Set')
plt.ylabel('Assigned Negatives Ratio')
plt.title(f'Estimation of Negative Recall Without Test Set Negatives\n R-Squared={np.round(r2score,2)}')
plt.legend(loc='lower right',title='class_sep value')
plt.plot([(0,0),(1,1)],color='gray', linestyle='dashed')
plt.ylim(0,)
plt.xlim(0,)
plt.savefig('negrecall_in_PU.png')
plt.show()

sns.scatterplot(df[df["Class_Separation"].apply(lambda x: isinstance(x, float))],x='Class_Separation',y='Baseline_Model',label='Baseline_Negative_Recall')
sns.scatterplot(df[df["Class_Separation"].apply(lambda x: isinstance(x, float))],x='Class_Separation',y='PU_Model',label='PU_Negative_Recall')
sns.scatterplot(df[df["Class_Separation"].apply(lambda x: isinstance(x, float))],x='Class_Separation',y='Neg_Sample_Ratio',label='PU_ANR')

plt.xlabel('Class Separation Value')
plt.ylabel('Negative Recall')
plt.title('Evaluation on Final Test Set')
plt.xlim(0,)
plt.ylim(0,1)
plt.legend(loc='lower right',title='Model Type')
plt.savefig('Baseline_Comparison.png')
plt.show()

