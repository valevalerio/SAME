import numpy as np
from sklearn.metrics import accuracy_score,f1_score
from SuperTree import *

# pbranch means partial sub branch
#interval is a list of couples
#attributeXf is the attribute included 
def computeBranch(nodeNr,IntervalIf,attributeXf,verbose=True):
        if verbose:
            print("a call with condition on",attributeXf,IntervalIf)
            print("TREE IS")
            nodeNr.print_tree()
        if len(IntervalIf)==0:
            return [nodeNr]
        if nodeNr is None:
            return [None for i in range(len(IntervalIf))]
        if nodeNr.is_leaf:
            pbranch = []
            for i,v in enumerate(IntervalIf):
                pbranch.append(Node(labels=nodeNr.labels,is_leaf=True))
            if verbose:
                print('----in Leaf: class_',np.argmax(nodeNr.labels),"Returning",pbranch)
            return pbranch
            
        else:
            msg = '(not involved)'
            if verbose:
                print('in node with Test',nodeNr.feat,nodeNr.thresh,(msg if not (nodeNr.feat==attributeXf) else 'involved'))
            pbranch = []
            if not (nodeNr.feat==attributeXf):
                pbranch = []
                #if verbose:
                    #print("i've",len(nodeNr.children),'childs')
                
                left_children = computeBranch(nodeNr._left_child,IntervalIf,attributeXf,verbose)
                right_children = computeBranch(nodeNr._right_child,IntervalIf,attributeXf,verbose)
                if verbose:
                    print('I was not involbed got:',len(left_children),"left childrens and ",len(right_children),'rigth')
                for left_c,right_c in zip(left_children,right_children):
                    pbranch.append(Node(feat_num=nodeNr.feat,
                                        thresh=nodeNr.thresh,
                                        left_child = left_c,
                                        right_child = right_c))
                        
                if verbose:
                    print("----attribute was not involved, returning",len(pbranch),"roots, just as I am")
                    for p in pbranch:
                        p.print_tree()
                return pbranch
                   
            
            else:
                new_childs = []
                If1 = IntervalIf[IntervalIf[:,0]<nodeNr.thresh]
                If2 = IntervalIf[IntervalIf[:,0]>=nodeNr.thresh]
                if verbose:
                    print("dividing interval in",If1,If2)
                pbranch = []
                
                left_children = []
                right_children = []
                if If1.shape[0]==1:
                    if verbose:
                        print("*"*5,"rec call on left part")
                        print("just added the left child")
                    left_children = [nodeNr._left_child]#computeBranch(nodeNr._left_child,If1,attributeXf,verbose)
                if If1.shape[0]>1:
                    left_children = computeBranch(nodeNr._left_child,If1,attributeXf,verbose)
                    
                if If2.shape[0]==1:
                    if verbose:
                        print("*"*5,"rec call on right part")
                        print("just added the right child")
                    right_children = [nodeNr._right_child]
                if If2.shape[0]>1:
                    right_children = computeBranch(nodeNr._right_child,If2,attributeXf,verbose)              
                if verbose:
                    print("PRE UNIFICATION")
                    print(len(left_children),"left childs")
                    print(len(right_children),"Right childs")
                    for left_c in left_children+right_children:
                        left_c.print_tree()
                if verbose:
                    print("----attribute was involved results:")
                    print(len(left_children),"left childs")
                    print(len(right_children),"Right childs")
                

                for left_c in left_children:
                    pbranch.append(left_c)
                
                for right_c in right_children:
                    pbranch.append(right_c)
                assert len(pbranch)==len(IntervalIf), "not the same thing"
                
                if nodeNr.thresh in np.unique(IntervalIf):
                    
                        if verbose:
                            print("I'm inside the interval")
                            print(IntervalIf,"pruning on",IntervalIf[len(left_children)][0])
                        if len(left_children)==0:
                            adding_node = right_children[0]
                            left_children.append(adding_node)
                        else:
                            adding_node = Node(feat_num=nodeNr.feat,
                                                thresh=nodeNr.thresh,
                                                left_child = left_children[-1],
                                                right_child = right_children[-1])
                        
                        left_children[-1]=prune_tree(adding_node,IntervalIf[len(left_children)][0],attributeXf)

                        if verbose:
                            print("last decision IS MEEEEEEE, at index",len(left_children))
                else:
                    if verbose:
                        print("i'm outside the interval"*6)
                        print("intervals",len(If1),len(If2))
                        print(IntervalIf)
                    major_all_minors = (IntervalIf[:,0]<nodeNr.thresh)
                    minor_all_majors = (IntervalIf[:,1]>nodeNr.thresh)
                    if verbose:
                        print(major_all_minors,minor_all_majors)
                        if (major_all_minors.all()):
                            print('im the greather')
                        if (minor_all_majors.all()):
                            print('im the smaller')
                    my_newly_child_r = (right_children[-1] if len(If2)>0 else nodeNr._right_child)
                    my_newly_child_l = (left_children[-1] if len(left_children)>0 else nodeNr._left_child)
                    
                    adding_node = Node(feat_num=nodeNr.feat,
                                                thresh=nodeNr.thresh,
                                                left_child = my_newly_child_l,
                                                right_child = my_newly_child_r)
                    #se tutti i limiti superiori sono maggiori di me e io non sono nell'intervallo la mia posizione
                    #è all'estrema sinitra
                    if minor_all_majors.all():
                        if verbose:
                            print("added at extreme left")
                        adding_node = prune_tree(adding_node,IntervalIf[0][1],attributeXf)
                        if len(left_children)==0:
                            pbranch[0]=adding_node
                        else:
                            pbranch[0]=adding_node
                    #se tutti i limiti inferiori sono minori di me e io non sono nell'intervallo la mia posizione
                    #è all'estrema destra
                    if major_all_minors.all():
                        if verbose:
                            print("added at extreme right")
                        adding_node = prune_tree(adding_node,IntervalIf[-1][1],attributeXf)
                        if len(right_children)==0:
                            pbranch[len(left_children)-1]=adding_node
                        else:
                            pbranch[-1] = adding_node
                    if (not minor_all_majors.all()) and (not major_all_minors.all()):
                        my_pos = np.argmax( major_all_minors & minor_all_majors  )
                        if verbose:
                            print("added at my specific position",my_pos)
                        adding_node = prune_tree(adding_node,IntervalIf[my_pos][1],attributeXf)
                        pbranch[my_pos]=adding_node
                
                                
                
                if verbose:
                    print("result")
                    for p in pbranch:
                            p.print_tree()                   
            return pbranch

def mergeDecisionTrees(roots,num_classes,level=0,verbose=False,r=''):
    k = len(roots)
    if verbose>0:
        print("MERGE "+str(k)+" trees, level",level,"previous splits")
        print(r)
   
    if np.array([r.is_leaf for r in roots]).all():
        out_propose = [np.argmax(r.labels) for r in roots]
        val,cou = np.unique(out_propose,return_counts=True)
        labels = np.zeros(num_classes)
        
        
        for j,vv in enumerate(val):
            labels[vv]=cou[j]
        if verbose>0:
            print('all_leafs: votes',val,
                            "voters",cou,"(return superLeaf with labels ",labels,")")
            
        return SuperNode(is_leaf=True,labels=labels)
    else:
        val,cou = np.unique([r.feat for r in roots if r.feat!=None],return_counts=True)
        if np.sum(cou)<(k/2):
            majority = [np.argmax(r.labels) for r in roots if r.is_leaf]
            val_out,cou_out = 0,0
            if majority:
            
                val_out,cou_out = np.unique(majority,return_counts=True)
                if verbose==1:
                    print("Can i terminate already?",val_out,cou_out,"majority at",(k/2)+1)
                if np.max(cou_out)>=(k/2)+1:
                    labels = np.zeros(num_classes)
                    
                    for j,vv in enumerate(val_out):
                        labels[vv]=cou_out[j]
                    if verbose==1:
                        print('there are almost all_leafs,values are:',
                        "votes",val_out,"voters",cou_out,"(return superLeaf with labels",majority,")")
                        print("ENDING EARLIER!"*10)
                    return SuperNode(is_leaf=True,labels=labels)

        Xf = val[np.argmax(cou)]
        
        If = list(np.unique([r.thresh for r in roots if r.feat==Xf]))
        
        if verbose:
            print("most used features",val,cou,'superNode is X_',Xf,If)

        If = np.array(list(zip([-np.inf]+If,If+[np.inf])))
        branch = []   #sub_trees under conditions
        for i in range(k):
            # sould be an array of roots of the condition sub trees (in which each subtree respect the rule recursivey)
            if (roots[i].is_leaf):
                branch.append(computeBranch(roots[i],If,Xf,verbose=False))
            else:
                branch.append(computeBranch(roots[i],If,Xf,verbose=False))
            assert len(branch[-1])==len(If),str(computeBranch(roots[i],If,Xf,verbose=True))
        if verbose==2:
            print(len(branch),'condition trees')
            for K in range(k):
                for j,v in enumerate(If):
                    print("under interval X",Xf,"<=",v)
                    print("subTRee",K,"has",len(branch[K]),"cdTree printing subtree",j)
                    branch[K][j].print_tree()
        assert len(branch[0])==(len(If)),"Not equal sub trees intervals tree0 has "+str(len(branch[0]))+" condition trees but threre are conditions"+str(len(If))
        
        
        
        children = []
        r+= ', X_'+str(Xf)
        for j,v in enumerate(If):
            
            children.append(mergeDecisionTrees([branch[K][j] for K in range(k)],num_classes,level+1,verbose,
            r+'('+str(j+1)+'\\'+str(len(If))+')'))
        #print("ended computation of sub tree level",level)
        
        #mergeDecisionTrees()
        If = If[:,1]
        
        new_root = SuperNode(feat_num=Xf,intervals=np.array(If),children=children)
        return new_root



import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
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
param_dict =  {
    'n_estimators': [100],#,16, 20, 24,28,32],
    'base_estimator__min_samples_split': [2, 5,10,15],
    'base_estimator__min_samples_leaf': [1, 0.001, 0.01, 0.05, 0.1, 0.2],
    'base_estimator__max_depth':  [5],
    'bootstrap':[False],
    'base_estimator__class_weight': [None, 'balanced'],
    "bootstrap_features":[True],
    "random_state":[0],
    'base_estimator__random_state': [0],
}


def bestRFRefit (cv_results):
    res = pd.DataFrame(cv_results)
    return res.sort_values(["rank_test_score","param_n_estimators","param_base_estimator__max_depth"],ascending=[True,True,True]).index[0]


if __name__ == "__main__":
    from sklearn.datasets import load_iris,load_wine
    from sklearn.ensemble import BaggingClassifier
    from sklearn.model_selection import train_test_split
    from load_exp_data import *
    X, y = load_AOM(num_samples=500)
    print("prototipization of dataset")
    X, y = load_german(return_X_y=True,prototipy=True,targetEnc=True)
    #X = (X - np.mean(X,axis=0)) / np.std(X,axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
    
    print("Random Search")
    RF = BaggingClassifier(base_estimator=DecisionTreeClassifier())
    #param_dict = {"n_estimators":[i for i in range(5,15)],"bootstrap_features":[True,False]}
    random_search =  RandomizedSearchCV(RF,param_distributions = param_dict,iid=True,cv=3,refit=bestRFRefit)
    result = random_search.fit(X_train,y_train)
    report(random_search.cv_results_)

    forest = random_search.best_estimator_
    y_pred_train_forest = forest.predict(X_train)
    y_pred_test = forest.predict(X_test)
    print("pruning Trees")
    for t in forest.estimators_:
        prune_duplicate_leaves(t)
    print("merging")
    roots = np.array([rec_buildTree(t,FI_used) for t,FI_used in zip(forest.estimators_,forest.estimators_features_)])
    for r in roots:
        r.print_tree()
        #input()
    superT = mergeDecisionTrees(roots, num_classes= np.unique(y).shape[0],verbose=1)
    res = complexitiSuperTree(superT)
    dept = res[0]
    no_nodes = res[1]
    leaves = res[2]
    avg_path_len = res[3]
    avg_no_feats = res[4]
    if dept<10 and no_nodes<50:
        superT.print_superTree()
    print("size",no_nodes,"dept",dept)
    
    y_pred_train_forest = forest.predict(X_train)
    y_pred_test = forest.predict(X_test)
    print("Train",np.round(accuracy_score(y_train,y_pred_train_forest),3),
            "Test",np.round(accuracy_score(y_test,y_pred_test),3))
    print("Supertree:")
    print("Train",np.round(accuracy_score(y_train,superT.predict(X=X_train)),3),
        "Test",np.round(accuracy_score(y_test,superT.predict(X=X_test)),3))
    