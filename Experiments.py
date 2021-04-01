import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap

from sklearn.metrics import accuracy_score,f1_score

class_map = lambda n : LinearSegmentedColormap.from_list('gy',[(.8,.0,0),(.0,.0,.8)], N=n) 

from MergeOTrees import mergeDecisionTrees,rec_buildTree,buildTreefromOblique
from SuperTree import *
#check_size, pruneRedundant,SuperNode,complexitiSuperTree,complexityDecisionTree,complexityOblique

import MergeTrees
from load_exp_data import *

from Oblique_Trees.HHCart import *

from Oblique_Trees.RandCart import *
from Oblique_Trees.segmentor import *
from pylab import rcParams

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import load_breast_cancer,load_iris,load_wine
from sklearn import ensemble
from sklearn.model_selection import RandomizedSearchCV

import time 


def complexity(mode):
    #depth,	nodes internals,	leaves,	avg path length,	avg nbr features
    if type(mode)==DecisionTreeClassifier:
        return complexityDecisionTree(mode)
    if type(mode) in [HouseHolderCART,Rand_CART]:
        return complexityOblique(mode._root)
    if type(mode)==SuperNode:
        return complexitiSuperTree(mode)
def getExplainer(mode,originalData=None,oblique_is_explainer=True,agnostic=True,case_six=False):
    if type(mode)==DecisionTreeClassifier:
        return mode,0
    if type(mode) in [HouseHolderCART,Rand_CART]:
        if not originalData is None:
            if oblique_is_explainer:
                return mode,0
            else:
                
                return getAgnosticExplainerWithX(mode,originalData)
        else:
            return mode,0
        
    if type(mode)==ensemble.BaggingClassifier:
        forest = mode
        if type(forest.estimators_[0])==DecisionTreeClassifier:
            print("bb is an orthogonal forest with",len(forest.estimators_))
            roots = np.array([rec_buildTree(t,FI_used) for t,FI_used in zip(forest.estimators_,forest.estimators_features_)])


            startTime = time.time()
            superT = MergeTrees.mergeDecisionTrees(roots, num_classes = np.unique(y).shape[0])

            return superT,time.time()- startTime
        elif case_six==False:
            print("not orthogonal forest")
            if agnostic:
                return getAgnosticExplainerWithX(mode,originalData)
            else:
                
                n_class = len(set(y))
                roots = np.array([buildTreefromOblique(t._root,n_class,FI_used) for t,FI_used in zip(forest.estimators_,forest.estimators_features_)])
                startTime = time.time()
                superT = mergeDecisionTrees(roots,
                                            num_classes = np.unique(y).shape[0],
                                            OriginalPoints = originalData,
                                           maximum_depth=10)
                
                return superT,time.time()- startTime
        else: #Case six
            print("A FOREST")
            roots=[]
            for r in forest.estimators_:
                y_tr = r.predict(originalData)
                f_j = getForest(md=[4,5,6],n_est=2,x_tr=originalData,y_tr=y_tr)
                for t,FI_used in zip(f_j.estimators_,f_j.estimators_features_):
                    prune_duplicate_leaves(t)
                    roots.append(rec_buildTree(t,FI_used))

            print("CASE SIX",len(forest.estimators),"estimators"," merging",len(roots),"trees")
            startTime = time.time()
            superT = MergeTrees.mergeDecisionTrees(roots, num_classes = np.unique(y).shape[0])

            return superT,time.time()- startTime
            

    else:
        return getAgnosticExplainerWithX(mode,originalData)
def getForest(x_tr,y_tr,md=[4,5],n_est=2):
    param_dict =  {
    'base_estimator__max_depth': m_d,
    'base_estimator__class_weight': [None, 'balanced'],
    }
    Itermediate_forest = ensemble.BaggingClassifier(base_estimator=DecisionTreeClassifier())
    random_search =  RandomizedSearchCV(Itermediate_forest,param_dict,iid=True,cv=3,refit=bestRFRefit)
    result = random_search.fit(x_tr,y_tr)
    forest = result.best_estimator_
    return forest
def getAgnosticExplainerWithX(black_box,originalData):
    
    X_trainO = originalData[np.random.choice(originalData.shape[0],int(originalData.shape[0]/5))]
    yBB = black_box.predict(X_trainO)
    RF = ensemble.BaggingClassifier(base_estimator=DecisionTreeClassifier())
    param_dict =  {
    'n_estimators': [20],
    'base_estimator__min_samples_split': [2, 0.002, 0.01, 0.05, 0.1, 0.2],
    'base_estimator__min_samples_leaf': [1, 0.001, 0.01, 0.05, 0.1, 0.2],
    'base_estimator__max_depth': [4, 5,6,7],
    'base_estimator__class_weight': [None, 'balanced'],
    'base_estimator__random_state': [0],
    }

    random_search =  RandomizedSearchCV(RF,param_dict,iid=True,cv=3,refit=bestRFRefit)
    result = random_search.fit(X_trainO,yBB)
    
    forest = result.best_estimator_
    print('I am an EXPLAINER')
    for w in ["param_n_estimators","param_base_estimator__max_depth"]:
        print(w,random_search.cv_results_[w][random_search.best_index_])
    for r in forest.estimators_:
        prune_duplicate_leaves(r)
    roots = np.array([rec_buildTree(t,FI_used) for t,FI_used in zip(forest.estimators_,forest.estimators_features_)])
    startTime = time.time()
    superT = MergeTrees.mergeDecisionTrees(roots, num_classes = np.unique(yBB).shape[0])
    
    

    return superT,time.time() - startTime


def report(results, n_top=1):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def bestTreeRefit(cv_results):
    res = pd.DataFrame(cv_results)
    ooo = res.sort_values(["rank_test_score","param_max_depth"],ascending=[True,True]).index[0]
    return ooo
def bestRFRefit (cv_results):
    res = pd.DataFrame(cv_results)
    return res.sort_values(["rank_test_score","param_n_estimators","param_base_estimator__max_depth"],ascending=[True,True,True]).index[0]
import time
def trepan(mode,originalData,md=0):
    
    param_dict =  {
    'max_depth': [2,3,4,6,7,8,9,10,11,12,14,15,17,20],
    }
    x_tr = originalData
    y_tr = mode.predict(originalData)
    if md == 0:
        random_search =  RandomizedSearchCV(DecisionTreeClassifier(),param_dict,iid=True,cv=3,refit=bestTreeRefit)
        result = random_search.fit(x_tr,y_tr)
        clf = result.best_estimator_
        elapsed = result.refit_time_
    else:
        clf = DecisionTreeClassifier(max_depth=md)
        start = time.time()
        clf.fit(x_tr,y_tr)
        elapsed = start-time.time()
    return clf,elapsed




#glass = load_glass()
#fico = load_fico(prototipy=True)
#adult = load_adult(targetEnc=True,prototipy = True)
#emp = load_employee(targetEnc=True,prototipy = True)
fico = load_german(targetEnc=True,prototipy = True)
#compas = load_compas(targetEnc=True)


#arm = load_AOM(num_samples=2000,prototipy=True)

#iris = load_iris(return_X_y=True)
#prototypeFeatures(iris[0],iris[1])

#cancer = load_breast_cancer(return_X_y=True)
#prototypeFeatures(cancer[0],cancer[1])


datasets  = [fico]
datasetsNames = ['fico']

for i,dn in enumerate(datasetsNames):
    print(i,dn)

datasetChoosen=0

X,y = datasets[datasetChoosen]
X_dev, X_test, y_dev, y_test = train_test_split(X, y,stratify=y, random_state = 0,test_size= .2)

X_bb, X_ap, y_bb, y_ap = train_test_split(X_dev, y_dev,stratify=y_dev, random_state = 0,test_size= .30)

X_train = X_bb
y_train = y_bb

param_dict =  {
    'max_depth': [5,6,7,8,10, 12, 16,None],
    'class_weight': [None, 'balanced'],
    'random_state': [1],
}
best_models = []
best_params = []
train_times = []
Models = [DecisionTreeClassifier,HouseHolderCART]
models_type = list(Models)

for model in Models:
    random_search =  RandomizedSearchCV(model(),param_dict,iid=True,cv=5,refit=bestTreeRefit)
    result = random_search.fit(X_train,y_train)
    dt_best_dept = random_search.best_params_["max_depth"]
    print("FOR THE MODEL "* 4)
    print(model)
    report(random_search.cv_results_)
    print("PARAMS ARE",result.best_params_)
    best_models.append(random_search.best_estimator_)
    best_params.append(random_search.best_params_)
    train_times.append(random_search.refit_time_)


best_params

param_dict =  {
    'n_estimators': [20],
    'base_estimator__min_samples_split': [2, 5,10,15],
    'base_estimator__min_samples_leaf': [1, 0.001, 0.01, 0.05, 0.1, 0.2],
    'base_estimator__max_depth':  [4,5],
    'base_estimator__class_weight': [None, 'balanced'],
    "bootstrap_features":[False],
    "random_state":[0],
    'base_estimator__random_state': [0],
}

RF = ensemble.BaggingClassifier(base_estimator=DecisionTreeClassifier())
#param_dict = {"n_estimators":[i for i in range(5,15)],"bootstrap_features":[True,False]}
random_search =  RandomizedSearchCV(RF,param_distributions = param_dict,iid=True,cv=3,refit=bestRFRefit)
result = random_search.fit(X_train,y_train)
#report(random_search.cv_results_)

best_models.append(random_search.best_estimator_)
best_params.append(random_search.best_params_)
train_times.append(random_search.refit_time_)


ORF = ensemble.BaggingClassifier()

param_dict["base_estimator"]=[Rand_CART()]

random_search =  RandomizedSearchCV(ORF,param_dict,iid=True,cv=3,refit=bestRFRefit)
result = random_search.fit(X_train,y_train)
#report(random_search.cv_results_)


m = random_search.best_estimator_
best_models.append(m)
best_params.append(random_search.best_params_)
train_times.append(random_search.refit_time_)
#2 times
train_times.append(random_search.refit_time_)



for i, model in enumerate(best_models):
    print(type(model))
best_models.append("")
best_models.append("")

experiments = {
    "explainer_costruction_time":np.zeros(len(best_models)),
    "explainer_predict_time":np.zeros(len(best_models)),
    "accuracy":np.zeros(len(best_models)),
    "fidelity":np.zeros(len(best_models)),
    "f1":np.zeros(len(best_models)),
    "depth":np.zeros(len(best_models)),
    "nodes_internals":np.zeros(len(best_models)),
    "leaves":np.zeros(len(best_models)),
    "avg_path_length":np.zeros(len(best_models)),
    "avg_nbr_features":np.zeros(len(best_models)),
    "f1_black_box":np.zeros(len(best_models)),              
    "bb_predict_time":np.zeros(len(best_models)),
    "accuracy_black_box":np.zeros(len(best_models)),        

"trepan_costruction_time":np.zeros(4),
"trepan_predict_time":np.zeros(4),
"trepan_accuracy":np.zeros(4),
"trepan_fidelity":np.zeros(4),
"trepan_f1":np.zeros(4),
"trepan_depth":np.zeros(4),
"trepan_nodes_internals":np.zeros(4),
"trepan_leaves":np.zeros(4),
"trepan_avg_path_length":np.zeros(4),
"trepan_avg_nbr_features":np.zeros(4),

"trepanNone_costruction_time":np.zeros(4),
"trepanNone_predict_time":np.zeros(4),
"trepanNone_accuracy":np.zeros(4),
"trepanNone_fidelity":np.zeros(4),
"trepanNone_f1":np.zeros(4),
"trepanNone_depth":np.zeros(4),
"trepanNone_nodes_internals":np.zeros(4),
"trepanNone_leaves":np.zeros(4),
"trepanNone_avg_path_length":np.zeros(4),
"trepanNone_avg_nbr_features":np.zeros(4),

}
experiments["train_time_bb"] = np.array(train_times)
best_models = best_models[:-2]

tmp = best_models[2]
best_models[2]=best_models[1]
best_models[1]=tmp
for i,model in enumerate(best_models):
    
    start_time = time.time()
    y_pred_test = np.round(model.predict(X_test))
    experiments["bb_predict_time"][i] = time.time()-start_time
    
    y_pred_train = np.round(model.predict(X_train))

    '''exp,merge_time = getExplainer(model,originalData=X_ap,oblique_is_explainer=False)
    experiments["explainer_costruction_time"][i]=merge_time
    
    start_time = time.time()
    y_exp_test = np.round(exp.predict(X_test))
    experiments["explainer_predict_time"][i] = time.time()-start_time
    
    
    experiments["accuracy_black_box"][i]=accuracy_score(y_test,y_pred_test)
    experiments["accuracy"][i] = accuracy_score(y_test,y_exp_test)
    experiments["fidelity"][i] = accuracy_score(y_pred_test,y_exp_test)
    
    experiments["f1_black_box"][i] = f1_score(y_test,y_pred_test,average='macro')
    experiments["f1"][i] = f1_score(y_test,y_exp_test,average='macro')

    print(type(model),type(exp))
    if type(model)==ensemble.BaggingClassifier:
        print("no node, dept, n leaves")
        print(check_size(exp))
    res = complexity(exp)

    
    dept = res[0]
    no_nodes = res[1]
    leaves = res[2]
    avg_path_len = res[3]
    avg_no_feats = res[4]
    
    print("dept:",dept,"no_nodes",no_nodes,"acc_Expl",experiments["accuracy"][i],"Fid",experiments["fidelity"][i])
    
    experiments["depth"][i]=dept
    experiments["nodes_internals"][i]=no_nodes
    experiments["leaves"][i]=leaves
    experiments["avg_path_length"][i]=avg_path_len
    experiments["avg_nbr_features"][i]=avg_no_feats


    '''
    #TREPAN  
    exp,trepan_time = trepan(model,originalData=X_ap,md=4)
    experiments["trepan_costruction_time"][i]=trepan_time
    
    start_time = time.time()
    y_exp_test = np.round(exp.predict(X_test))
    experiments["trepan_predict_time"][i] = time.time()-start_time
    experiments["trepan_accuracy"][i] = accuracy_score(y_test,y_exp_test)
    experiments["trepan_fidelity"][i] = accuracy_score(y_pred_test,y_exp_test)
    experiments["trepan_f1"][i] = f1_score(y_test,y_exp_test,average='macro')

    res = complexity(exp) 
    dept = res[0]
    no_nodes = res[1]
    leaves = res[2]
    avg_path_len = res[3]
    avg_no_feats = res[4]

    experiments["trepan_depth"][i]=dept
    experiments["trepan_nodes_internals"][i]=no_nodes
    experiments["trepan_leaves"][i]=leaves
    experiments["trepan_avg_path_length"][i]=avg_path_len
    experiments["trepan_avg_nbr_features"][i]=avg_no_feats
    
    #TREPAN  None
    exp,trepan_time = trepan(model,originalData=X_ap,md=None)
    experiments["trepanNone_costruction_time"][i]=trepan_time
    
    start_time = time.time()
    y_exp_test = np.round(exp.predict(X_test))
    experiments["trepanNone_predict_time"][i] = time.time()-start_time
    experiments["trepanNone_accuracy"][i] = accuracy_score(y_test,y_exp_test)
    experiments["trepanNone_fidelity"][i] = accuracy_score(y_pred_test,y_exp_test)
    experiments["trepanNone_f1"][i] = f1_score(y_test,y_exp_test,average='macro')

    res = complexity(exp) 
    dept = res[0]
    no_nodes = res[1]
    leaves = res[2]
    avg_path_len = res[3]
    avg_no_feats = res[4]

    experiments["trepanNone_depth"][i]=dept
    experiments["trepanNone_nodes_internals"][i]=no_nodes
    experiments["trepanNone_leaves"][i]=leaves
    experiments["trepanNone_avg_path_length"][i]=avg_path_len
    experiments["trepanNone_avg_nbr_features"][i]=avg_no_feats

    


for k in experiments.keys():
    print(k,experiments[k],experiments[k].shape)


model = best_models[-1]
i=4

start_time = time.time()
y_pred_test = np.round(model.predict(X_test))
experiments["bb_predict_time"][i] = time.time()-start_time

y_pred_train = np.round(model.predict(X_train))

exp1,merge_time1 = getExplainer(model,originalData=X_ap,oblique_is_explainer=False,agnostic=False)
exp,merge_time = getExplainer(exp1,originalData=X_ap,oblique_is_explainer=False)
experiments["explainer_costruction_time"][i]= merge_time + merge_time1

start_time = time.time()
y_exp_test = np.round(exp.predict(X_test))
experiments["explainer_predict_time"][i],time.time()-start_time


experiments["accuracy_black_box"][i]=accuracy_score(y_test,y_pred_test)
experiments["fidelity"][i] = accuracy_score(y_pred_test,y_exp_test)
experiments["accuracy"][i] = accuracy_score(y_test,y_exp_test)
    
experiments["f1_black_box"][i] = f1_score(y_test,y_pred_test,average='macro')
experiments["f1"][i] = f1_score(y_test,y_exp_test,average='macro')
print(type(model),type(exp))


print("no node, dept, n leaves")
print(check_size(exp))
res = complexity(exp)


dept = res[0]
no_nodes = res[1]
leaves = res[2]
avg_path_len = res[3]
avg_no_feats = res[4]

print("dept:",dept,"no_nodes",no_nodes,"acc_Expl",experiments["accuracy"][i],"Fid",experiments["fidelity"][i])

experiments["depth"][i]=dept
experiments["nodes_internals"][i]=no_nodes
experiments["leaves"][i]=leaves
experiments["avg_path_length"][i]=avg_path_len
experiments["avg_nbr_features"][i]=avg_no_feats



model = best_models[-1]
i=5

start_time = time.time()
y_pred_test = np.round(model.predict(X_test))
experiments["bb_predict_time"][i] = time.time()-start_time

y_pred_train = np.round(model.predict(X_train))

exp,merge_time = getExplainer(exp1,originalData=X_ap,oblique_is_explainer=False,case_six=True)
experiments["explainer_costruction_time"][i]= merge_time + merge_time1

start_time = time.time()
y_exp_test = np.round(exp.predict(X_test))
experiments["explainer_predict_time"][i] = time.time()-start_time


experiments["accuracy_black_box"][i] = accuracy_score(y_test,y_pred_test)
experiments["fidelity"][i] = accuracy_score(y_pred_test,y_exp_test)
experiments["accuracy"][i] = accuracy_score(y_test,y_exp_test)
    
experiments["f1_black_box"][i] = f1_score(y_test,y_pred_test,average='macro')
experiments["f1"][i] = f1_score(y_test,y_exp_test,average='macro')
print(type(model),type(exp))

print("no node, dept, n leaves")
print(check_size(exp))
res = complexity(exp)


dept = res[0]
no_nodes = res[1]
leaves = res[2]
avg_path_len = res[3]
avg_no_feats = res[4]

print("dept:",dept,"no_nodes",no_nodes,"acc_Expl",experiments["accuracy"][i],"Fid",experiments["fidelity"][i])

experiments["depth"][i]=dept
experiments["nodes_internals"][i]=no_nodes
experiments["leaves"][i]=leaves
experiments["avg_path_length"][i]=avg_path_len
experiments["avg_nbr_features"][i]=avg_no_feats


XLABELS = ["DT",
           "RF"+str(len(best_models[1].estimators_)),
           "OT",
           "ORF"+str(len(best_models[-1].estimators_)),
           "ORF",
           "ORF"
          ]
EXPLAINER_LABELS = ["ID","M","ph","ph+M","ph+M","SAME"]

XLABELS,EXPLAINER_LABELS

best_models.append("")
best_models.append("")


plt.figure(figsize=(8.1*3, 7*2))
words = ["accuracy_black_box","fidelity","accuracy"]
bar_colors = ['k','cyan','gold']
patterns = [' ',".","*"]
bar_width = .5



plt.subplot(231)
plt.title("Performancies",fontsize=22)
plt.grid(axis='y')
for i,w in enumerate(words):
    print(i*2*len(words))
    plt.bar((-bar_width*i if i>0 else 0)+i+(len(words)-1)*np.arange(len(best_models)),
            experiments[w],label=w,
            width = bar_width,
           hatch=patterns[i],
           color = bar_colors[i])
#plt.xticks(np.arange(6))
print([bar_width+i*(len(words)-1) for i in range(len(best_models))])

plt.xticks([bar_width+i*(len(words)-1) for i in range(len(best_models))],
           XLABELS,fontsize=16)
plt.ylim(min(experiments['accuracy'])-.15,1)
plt.yticks(fontsize=16)
plt.legend(fontsize=20,loc='lower left')


plt.subplot(232)
plt.title("Explainer Size",fontsize=22)
plt.grid(axis='y')
for i,t in enumerate(["leaves","nodes_internals"]):
    plt.bar(2*np.arange(len(best_models))+0.8*i,experiments[t],label=t)
plt.legend(fontsize=20)    
plt.yscale("log")
plt.xticks([2*i for i in range(len(best_models))],
           XLABELS,fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=20,loc='upper left')


measurement = ["depth","avg_path_length"]
plt.subplot(233)
plt.title("Complexity measures",fontsize=22)

plt.grid(axis='y')
t = "depth"
plt.bar(2*np.arange(len(best_models)),experiments[t],label=t)
plt.xticks([i for i in range(len(best_models))],
          XLABELS,fontsize=16)
t = "avg_path_length"
plt.bar(2*np.arange(len(best_models))+.8,experiments[t],label=t)
plt.xticks([2*i for i in range(len(best_models))],
           XLABELS,fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=20,loc='upper left')


plt.subplot(235)
plt.title("average number of features per rule",fontsize=22)
plt.grid(axis='y')
t  = "avg_nbr_features"
plt.bar(np.arange(len(best_models)),experiments[t])
plt.xticks([i for i in range(len(best_models))],
           XLABELS,fontsize=16)
plt.yticks(fontsize=16)    
#plt.legend(fontsize=20,loc='upper left')


plt.subplot(234)
plt.scatter([0],[0],label=datasetsNames[datasetChoosen])
plt.xticks([],[])
plt.yticks([],[])
plt.legend(fontsize=30)

plt.subplot(236)

plt.title("Performance Times",fontsize=22)
plt.bar(3*np.arange(len(best_models))+.8,experiments["explainer_costruction_time"],label="merge")
plt.bar(3*np.arange(len(best_models)),experiments["bb_predict_time"],label="bb predict")
plt.bar(3*np.arange(len(best_models))-.80,experiments["explainer_predict_time"],label="explainer predict")
plt.xticks([3*i for i in range(len(best_models))],
           EXPLAINER_LABELS,fontsize=16)
plt.yscale('log')
plt.yticks(fontsize=16)
plt.legend(fontsize=20)
plt.savefig("./Trepan"+datasetsNames[datasetChoosen]+" Results.png")


print(experiments["f1_black_box"])
print(experiments["f1"])

import pickle
f_n = "./Trepan_"+datasetsNames[datasetChoosen]+" Results.dict"
with open(f_n,'wb+') as file_Exp:
    pickle.dump(str(experiments),file_Exp)
for k in experiments.keys():
    if (k[:6]=='trepan'):
        print(k,experiments[k])
