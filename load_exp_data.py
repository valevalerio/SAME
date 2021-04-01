import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.tree import DecisionTreeClassifier


#RMEP
###
# S is a column vector from a bigger dataset, columns are considered as value of a feature for say 100 observations
# y denotes the CLASS of an observation which can be from 0 to N
###
#Y should be sorted according to the feature S which is being split over.
def recursive_partition(y):
    T_list = []
    y = np.asarray(y)
    #entroy along_subsection of y:
    def ent(p,t): return array_entropy(y[p:t])
    #k values along subsection of y:
    def k_fn(p,t): return len(np.unique(y[p:t]))
    
    def partition_subset(p,t):
        #n: |S| being split
        #p beginning of S
        #q place to try a split
        #t end of S
        
        n = t-p
        partial_entropy = [(ent(p,q)*(q-p)+ent(q,t)*(t-q))/n for q in range(p+1,t)]
        dq = np.argmin(partial_entropy)
        split_ent = partial_entropy[dq]
        q = p+dq+1
        gain = ent(p,t) - split_ent
        delta = (np.log2(3.**k_fn(p,t) - 2) -
                 (k_fn(p,t)*ent(p,t) - k_fn(p,q)*ent(p,q) - k_fn(q,t)*ent(q,t)))
        thresh = (np.log2(n-1) + delta)/n
        
        if gain > thresh:
            T_list.append(q)
            if q-p > 1: partition_subset(p,q)
            if t-q > 1: partition_subset(q,t)
                
        return
            
    partition_subset(0,len(y))
    return T_list
   

def array_entropy(y):
    unique,unique_counts = np.unique(y,return_counts=True)
    probabilities = unique_counts/len(y)
    return -sum(probabilities*np.log2(probabilities))

#discretization helper
def discretizeRMEP(numeric_x,cuts):
    for l_b,u_b in zip([-np.inf]+list(cuts),list(cuts)+[np.inf]):
        val = np.mean(numeric_x[np.where((numeric_x>l_b) & (numeric_x<=u_b))])
        numeric_x[np.where((numeric_x>l_b) & (numeric_x<=u_b))] = val

def prototypeFeatures(numeric_features,targets):
    for j in range(numeric_features.shape[1]):
        sortedDSet = np.array(sorted(zip(numeric_features[:,j],targets),key = lambda x:x[0]))
        #print(recursive_partition(sortedDSet[:,1]))
        thresholds = np.unique(np.array(sortedDSet[recursive_partition(sortedDSet[:,1]),0]))
        recEntrDT = DecisionTreeClassifier(max_leaf_nodes=max(2,len(thresholds)+1),criterion='entropy')
        recEntrDT.fit(numeric_features[:,j].reshape(-1,1),targets)
        ffeat = recEntrDT.tree_.feature
        vals = np.sort(recEntrDT.tree_.threshold[np.where(ffeat!=-2)])
        discretizeRMEP(numeric_features[:,j],vals)
    


#Functions to load and clean the datasets
def load_glass(return_X_y=True,prototipy = True):
    dt = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data',header=None)

    X = np.array(dt.iloc[:,1:-1])
    y = np.array(dt.iloc[:,-1])
    X = (X - np.mean(X,axis=0)) / np.std(X,axis=0)
    y[y==5]=4
    y[y==6]=5
    y[y==7]=6
    if prototipy:
        prototypeFeatures(X,y)
    return X,y

def load_employee(return_X_y=True,dummy=False,targetEnc=True, prototipy = True):
    dt = pd.read_csv('./Datasets/employee.csv',header=None)
    names = dt.iloc[0,:]
    numericFeaturesIdx = [i for i in  [0,5,12,18,19,20,23,28,29,31,32,33,34]]
    X_float = np.array(dt.iloc[1:,numericFeaturesIdx]).astype(np.float32)
    X_float = (X_float - np.mean(X_float,axis=0)) / np.std(X_float,axis=0)
    y = np.array(dt.iloc[1:,1])#np.array(dt.iloc[1:,1])

    y[y=='Yes'] = 1
    y[y=='No'] = 0
    y = y.astype(np.intc)


    categoricalFeaturesIdx = [2,4,6,7,10,11,13,14,15,16,17,22,24,25,27,30]
    X_cat = np.array(dt.iloc[1:,categoricalFeaturesIdx])
    #for i in categoricalFeaturesIdx:
    #    print(i,names[i],len(set(dt.iloc[1:,i])),([] if i == 3 else set(dt.iloc[1:,i])))
    xs = np.array([])
    for catFeatIdx in categoricalFeaturesIdx:
        x_c = dt.iloc[1:,catFeatIdx]
        x_c_values = np.unique(x_c)
        if targetEnc==True:
            str_data = [str(e) for e in x_c]
            tarEnc = ce.TargetEncoder()
            tarEnc.fit(str_data,y)
            xtargenc = tarEnc.transform([str(e) for e in x_c]).to_numpy()
            xs = np.hstack([xs,xtargenc]) if xs.size else xtargenc
        else:
            if dummy==True:
                cat_dict = {c_val:i for i,c_val in enumerate(x_c_values)}
                dummy_vars = np.array([cat_dict[e] for e in x_c]).reshape(len(x_c),1)
                
                xs = np.hstack([xs,dummy_vars]) if xs.size else dummy_vars
                
                
            else:
                onehot = np.zeros((x_c.shape[0],len(set(x_c))))


                onehot.shape
                catFeat2idx = {k:i for i,k in enumerate(x_c_values)}
                catFeat2idx
                for i,e in enumerate(x_c):
                    onehot[i,catFeat2idx[e]]=1
                xs = np.hstack([xs ,onehot]) if xs.size else onehot
    if prototipy:
        prototypeFeatures(X_float,y)
    X = np.hstack([X_float,xs])
    if return_X_y==True:
        return X,y
    else:
        return X,y,names

def load_compas(return_X_y=True,dummy=False,targetEnc=True, prototipy = True):
    df = pd.read_csv('./Datasets/compas-scores-two-years.csv')
    columns = ['age', #'age_cat', 
            'sex', 'race',  'priors_count', 'days_b_screening_arrest', 'c_jail_in', 'c_jail_out',
            'c_charge_degree', 'is_recid', 'is_violent_recid', 'two_year_recid', 'decile_score', 'score_text']
    

    df = df[columns]
    df['days_b_screening_arrest'] = np.abs(df['days_b_screening_arrest'])

    df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
    df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
    df['length_of_stay'] = (df['c_jail_out'] - df['c_jail_in']).dt.days
    df['length_of_stay'] = np.abs(df['length_of_stay'])

    df['length_of_stay'].fillna(df['length_of_stay'].value_counts().index[0], inplace=True)
    df['days_b_screening_arrest'].fillna(df['days_b_screening_arrest'].value_counts().index[0], inplace=True)

    df['length_of_stay'] = df['length_of_stay'].astype(int)
    df['days_b_screening_arrest'] = df['days_b_screening_arrest'].astype(int)

    df['class'] = df['score_text']

    del df['c_jail_in']
    del df['c_jail_out']
    del df['decile_score']
    del df['score_text']

    df.replace("Male",0,inplace=True)
    df.replace("Female",1,inplace=True)

    df.replace("Low",0,inplace=True)
    df.replace("Medium",1,inplace=True)
    df.replace("High",2,inplace=True)

    df.replace("M",0,inplace=True)
    df.replace("F",1,inplace=True)
    
    y = np.array(df["class"])
    dt = df
    
    numericFeaturesIdx = ["age","sex","priors_count","days_b_screening_arrest","c_charge_degree","is_recid","is_violent_recid","two_year_recid"]
    X_float = np.array(dt.loc[:,
                    numericFeaturesIdx
                    ])
    X_float = (X_float - np.mean(X_float,axis=0)) / np.std(X_float,axis=0)
    categoricalFeaturesIdx = ["race"]
    X_cat = np.array(dt.loc[:,categoricalFeaturesIdx])
    #for i in categoricalFeaturesIdx:
    #    print(i,names[i],len(set(dt.iloc[:,i])),([] if i == 3 else set(dt.iloc[:,i])))
    xs = np.array([])
    for catFeatIdx in categoricalFeaturesIdx:
        x_c = dt.loc[:,catFeatIdx]
        x_c_values = set(x_c)

        if targetEnc==True:
            str_data = [str(e) for e in x_c]
            tarEnc = ce.TargetEncoder()
            tarEnc.fit(str_data,y)
            xtargenc = tarEnc.transform([str(e) for e in x_c]).to_numpy()
            xs = np.hstack([xs,xtargenc]) if xs.size else xtargenc
            
        else:
            if dummy==True:
                cat_dict = {c_val:i for i,c_val in enumerate(x_c_values)}
                dummy_vars = np.array([cat_dict[e] for e in x_c]).reshape(len(x_c),1)   
                xs = np.hstack([xs,dummy_vars]) if xs.size else dummy_vars
    
            else:
                onehot = np.zeros((x_c.shape[0],len(set(x_c))))
                onehot.shape
                catFeat2idx = {k:i for i,k in enumerate(x_c_values)}
                catFeat2idx
                for i,e in enumerate(x_c):
                    onehot[i,catFeat2idx[e]]=1
                xs = np.hstack([xs ,onehot]) if xs.size else onehot
    if prototipy:
        prototypeFeatures(X_float,y)
    X = np.hstack([X_float,xs])
    
    
    assert y.shape[0]==X.shape[0],"not same X-y!"

    return X,y

def load_german(return_X_y=True,dummy=False,targetEnc=True,prototipy = True):
    dt = pd.read_csv('./Datasets/german-credit.csv')
    numericFeaturesIdx = ["duration_in_month","credit_amount","age","credits_this_bank","people_under_maintenance","installment_as_income_perc","present_res_since"]
    categoricalFeaturesIdx = ["personal_status_sex",
                            "present_emp_since","account_check_status",
                            "credit_history","purpose","savings","other_debtors","property",
                            "other_installment_plans","housing","job","foreign_worker"]

    y = np.array(dt['default'])

    X_float = np.array(dt.loc[:,
                    numericFeaturesIdx
                    ])
    X_cat = np.array(dt.loc[:,categoricalFeaturesIdx])
    #for i in categoricalFeaturesIdx:
    #    print(i,names[i],len(set(dt.iloc[:,i])),([] if i == 3 else set(dt.iloc[:,i])))
    xs = np.array([])
    for catFeatIdx in categoricalFeaturesIdx:
        x_c = dt.loc[:,catFeatIdx]
        x_c_values = set(x_c)
        if targetEnc==True:
            str_data = [str(e) for e in x_c]
            tarEnc = ce.TargetEncoder()
            tarEnc.fit(str_data,y)
            xtargenc = tarEnc.transform([str(e) for e in x_c]).to_numpy()
            xs = np.hstack([xs,xtargenc]) if xs.size else xtargenc
        else:
            if dummy==True:
                cat_dict = {c_val:i for i,c_val in enumerate(x_c_values)}
                dummy_vars = np.array([cat_dict[e] for e in x_c]).reshape(len(x_c),1)
                
                xs = np.hstack([xs,dummy_vars]) if xs.size else dummy_vars
        
            else:
            
                onehot = np.zeros((x_c.shape[0],len(set(x_c))))
                onehot.shape
                catFeat2idx = {k:i for i,k in enumerate(x_c_values)}
                catFeat2idx
                for i,e in enumerate(x_c):
                    onehot[i,catFeat2idx[e]]=1
                xs = np.hstack([xs ,onehot]) if xs.size else onehot

    X_float = (X_float - np.mean(X_float,axis=0)) / np.std(X_float,axis=0)
    if prototipy:
        prototypeFeatures(X_float,y)
    X = np.hstack([X_float,xs])
    X.shape,y.shape
    return X,y
def load_adult(return_X_y=True,dummy=False,targetEnc=True,prototipy = True):
    dt = pd.read_csv('./Datasets/adult.csv',header=None)
    names = dt.iloc[0,:]
    numericFeaturesIdx = [i for i in  [0,2,10,11,12]]
    X_float = np.array(dt.iloc[1:,numericFeaturesIdx]).astype(np.float32)
    y = np.array(dt.iloc[1:,14])#np.array(dt.iloc[1:,1])
    #Normalization of data
    X_float = (X_float - np.mean(X_float,axis=0)) / np.std(X_float,axis=0)
    y[y==' >50K'] = 1
    y[y==' <=50K'] = 0
    y = y.astype(np.intc)
    #for i in range(len(names)):
    #    print(i,names[i],len(set(dt[i])) if len(set(dt[i]))>10 else set(dt[i]))



    categoricalFeaturesIdx = [1,3,5,6,7,8,9]
    X_cat = np.array(dt.iloc[1:,categoricalFeaturesIdx])
    #for i in categoricalFeaturesIdx:
    #    print(i,names[i],len(set(dt.iloc[1:,i])),([] if i == 3 else set(dt.iloc[1:,i])))


    xs = np.array([])
    for catFeatIdx in categoricalFeaturesIdx:
        x_c = dt.iloc[1:,catFeatIdx]
        x_c_values = set(x_c)
        if targetEnc==True:
            str_data = [str(e) for e in x_c]
            tarEnc = ce.TargetEncoder()
            tarEnc.fit(str_data,y)
            xtargenc = tarEnc.transform([str(e) for e in x_c]).to_numpy()
            xs = np.hstack([xs,xtargenc]) if xs.size else xtargenc
        else:
            if dummy==True:
                cat_dict = {c_val:i for i,c_val in enumerate(x_c_values)}
                dummy_vars = np.array([cat_dict[e] for e in x_c]).reshape(len(x_c),1)
                
                xs = np.hstack([xs,dummy_vars]) if xs.size else dummy_vars
                
                
            else:
                onehot = np.zeros((x_c.shape[0],len(set(x_c))))
                catFeat2idx = {k:i for i,k in enumerate(x_c_values)}
                catFeat2idx
                for i,e in enumerate(x_c):
                    onehot[i,catFeat2idx[e]]=1
                xs = np.hstack([xs ,onehot]) if xs.size else onehot
    if prototipy:
        prototypeFeatures(X_float,y)
    X = np.hstack([X_float,xs])

    return X,y

def load_fico(return_X_y=True,prototipy = True):
    dt = pd.read_csv('./Datasets/fico.csv',header=None)
    numericFeaturesIdx = [i for i in  range(1,dt.shape[1])]
    X_float = np.array(dt.iloc[1:,numericFeaturesIdx]).astype(np.float32)
    X = X_float
    X = (X - np.mean(X,axis=0)) / np.std(X,axis=0)
    y = np.array(dt.iloc[1:,0])
    y[np.where(y=="Bad")]=0
    y[np.where(y=="Good")]=1
    y = y.astype(np.int32)
    if prototipy:
        prototypeFeatures(X,y)
    return X,y


def load_AOM(num_samples=1000,random_state=0,prototipy = True):
    rng = np.random.RandomState(random_state)
    X = 5*rng.random_sample(size=(num_samples,2))

    aom = lambda x: (0 if x[1]+x[0]<=6 and ((x[1]-x[0])<=2 and (x[1]-x[0])>-2) else
                (2 if ((x[1]+x[0])>6 and ((x[1]-x[0])<=2 and (x[1]-x[0])>-2)) else
                 (1)))
    y = np.array([aom(el) for el in X])
    if prototipy:
        prototypeFeatures(X,y)
    return X,y

def make_meshgrid(x1, x2, h=.02,x1_min=0,x1_max=0,x2_min=0,x2_max=0):
    if x1_min==0 and x1_max==0 and x2_min==0 and x2_max==0:
        x1_min, x1_max = x1.min() - 0.1, x1.max() + 0.1
        x2_min, x2_max = x2.min() - 0.1, x2.max() + 0.1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
    return np.vstack((xx1.ravel(), xx2.ravel())).T
