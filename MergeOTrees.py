import numpy as np
from sklearn.metrics import accuracy_score,f1_score
from SuperTree import *


from Oblique_Trees import HHCart,RandCart,WODT,plot_tree
from matplotlib import patches
HHCart.HouseHolderCART
RandCart.Rand_CART
WODT.WeightedObliqueDecisionTreeClassifier


def getmaxClass(radice):
    if radice.is_leaf:
        return radice.labels 
    else:
        return getmaxClass(radice._left_child)+getmaxClass(radice._right_child)
    
#this one could be substituted by the oiginal dataset
def generateIstancies(b,n_samples =  100):
    
    a = (np.random.rand(len(b)*n_samples)*2-1)*100
    print(a.reshape(n_samples,len(b)).shape)
    a = a.reshape(n_samples,len(b))
    for k in range(len(b)):
        a[:,k] = np.clip(a[:,k],b[k,0],b[k,1])
    return a

    
def checkSynthSide(w,bo,features_involved):
    test_points = generateIstancies(bo)
    test = test_points.dot(w[:-1])- w[-1]<=0
    if test.all(): #all true
        "just left size"
        return -1
    else:
        if not test.any():#all false
            "just right size"
            return 1
        else:
            "Still neded baby"
            return 0
        
def checkOriginalOrthoSide(x_f, t, samples):
    global OriginalDataset
    if len(samples)==0:
        return [],[]
    test_points = np.array(OriginalDataset[samples,x_f])#features_involved])
    test = test_points <= t
    left_idxs = np.arange(len(samples))[test]
    right_idxs = np.arange(len(samples))[np.logical_not(test)]
    
    return samples[left_idxs] if len(left_idxs)>0 else [] ,\
            samples[right_idxs] if len(right_idxs)>0 else []

def checkOriginalSide(w,features_involved,samples):
    global OriginalDataset
    
    test_points = np.array(OriginalDataset[samples,:])#features_involved])
    test = test_points.dot(w[:-1])- w[-1]<=0
    left_idxs = np.arange(len(samples))[test]
    right_idxs = np.arange(len(samples))[np.logical_not(test)]
    
    return samples[left_idxs],samples[right_idxs]

# pbranch means partial sub branch
#interval is a list of couples
#attributeXf is the attribute included 
def mergeDecisionTrees(roots,num_classes,level=0,verbose=False,OriginalPoints = None,maximum_depth=20):
    global checkSide,OriginalDataset
    
    assert not (OriginalPoints is None),"No data provided"
    
    print("extensive approach")
    OriginalDataset = OriginalPoints
    #print("for a maximum of ",maximum_depth,'levels')
    return mergeObliqueDecisionTrees(roots,num_classes,np.arange(OriginalDataset.shape[0]),level,verbose,max_depth=maximum_depth)

def computeOBranch(nodeNr,IntervalIf,attributeXf,indexes,verbose=False):
        
        if verbose==1:
            print("a call with condition on",attributeXf,IntervalIf)
            print(np.sum([len(r) for r in indexes]),"TREE IS")
            nodeNr.print_tree()
        '''if len(IntervalIf)==0:
            return [nodeNr]
        if nodeNr is None:
            return [None for i in range(len(IntervalIf))]'''
        #LEAF CASE
        if nodeNr.is_leaf:
            pbranch = []
            for i,v in enumerate(IntervalIf):
                pbranch.append(Node(labels=nodeNr.labels,is_leaf=True))
            if verbose==1:
                print('----in Leaf: class_',np.argmax(nodeNr.labels),"Returning",pbranch)
            return pbranch

        msg = '(not involved)'
        if verbose==1:
            print('in node with Test',nodeNr.feat,nodeNr.thresh,(msg if not (nodeNr.feat==attributeXf) else 'involved'))
        pbranch = []
        #OBLIQUE NODE CASE
        if not nodeNr._weights is None: #oblique tree, I'm i needed? DO I intersect? 
            if verbose:
                print("Computing branch on Oblique",attributeXf,"is the feat")
                print(len(indexes), "sample sets present")
                print(IntervalIf,"are the new conditions")

            #check if there is intersection:
            for j,_ in enumerate(indexes):
                v = IntervalIf[j] if not attributeXf is None else [None,None]
                sub_partition_samples = indexes[j]
                if len(indexes[j])>0:
                    left_idxs,right_idxs = checkOriginalSide(nodeNr._weights,nodeNr._features_involved,indexes[j])
                else:
                    left_idxs,right_idxs = [],[]
                
                #print(len(indexes[j]),"elements under condition",v[0],"< X_",attributeXf,"<=",v[1])
                #print("I split them in 2 sets of ",len(left_idxs),len(right_idxs))
                
                if  len(left_idxs)==0 and len(right_idxs)==0:
                   
                    try:
                        new_Node = Node(labels = nodeNr.labels,is_leaf=True)
                    except AttributeError:
                        print(nodeNr.thresh,nodeNr._weights,nodeNr.impurity,)
                        print("Not enought samples")
                        exit()
                    pbranch.append(new_Node)
                elif len(left_idxs)==0: #just right side
                        #print("Just Adding Right side")
                        pbranch.append(computeOBranch(nodeNr._right_child,np.array([v]),attributeXf,[right_idxs],verbose)[0])
                elif len(right_idxs)==0: #just left side
                            #print("Just Adding Left side")
                        pbranch.append(computeOBranch(nodeNr._left_child,np.array([v]),attributeXf,[left_idxs],verbose)[0])
                else:
                    #If I'm here, I DO intersect 
                    left_children = computeOBranch(nodeNr._left_child,np.array([v]),attributeXf,[left_idxs],verbose)
                    right_children = computeOBranch(nodeNr._right_child,np.array([v]),attributeXf,[right_idxs],verbose)
                    if verbose:
                        print("I'm still nedded","my_wheigts:",nodeNr._weights)
                        print(len(left_children),len(right_children))
                    newNode = Node(weights=nodeNr._weights,
                                        left_child = left_children[0],
                                        right_child = right_children[0],
                                        labels=nodeNr.labels)
                    newNode._features_involved = nodeNr._features_involved
                    pbranch.append(newNode)
            return pbranch    
                
            
            
            
        #STANDARD NODE CASE
        else:
            # NODE NOT INVOLVED
            if not (nodeNr.feat==attributeXf):
                pbranch = []
                #if verbose:
                    #print("i've",len(nodeNr.children),'childs')
                
                left_idxs = []
                right_idxs = []
                for sub_samples in indexes:
                    l_i,r_i = checkOriginalOrthoSide(nodeNr.feat,nodeNr.thresh,sub_samples)
                    left_idxs.append(l_i)
                    right_idxs.append(r_i)
                left_children = computeOBranch(nodeNr._left_child,IntervalIf,attributeXf,left_idxs,verbose)
                right_children = computeOBranch(nodeNr._right_child,IntervalIf,attributeXf,right_idxs,verbose)
                if verbose:
                    
                    print('I was not involbed got:',len(left_children),"left childrens and ",len(right_children),'rigth')
                    
                for left_c,right_c in zip(left_children,right_children):
                    newNode = Node(feat_num=nodeNr.feat,
                                        thresh=nodeNr.thresh,
                                        left_child = left_c,
                                        right_child = right_c,
                                        labels=nodeNr.labels)
                    pbranch.append(newNode)
                        
                if verbose:
                    print("----attribute was not involved, returning",len(pbranch),"roots, just as I am")
                    for p in pbranch:
                        p.print_tree()
                return pbranch
                
            #NODE INVOLVED
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
                    left_children = computeOBranch(nodeNr._left_child,If1,attributeXf,indexes[:If1.shape[0]],verbose)
                    
                if If2.shape[0]==1:
                    if verbose:
                        print("*"*5,"rec call on right part")
                        print("just added the right child")
                    right_children = [nodeNr._right_child]
                if If2.shape[0]>1:
                    right_children = computeOBranch(nodeNr._right_child,If2,attributeXf,indexes[If1.shape[0]:],verbose)              
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
                                                right_child = right_children[-1],
                                                labels=nodeNr.labels)
                        
                        
                        left_children[-1]=prune_tree(adding_node,IntervalIf[len(left_children)][0],attributeXf)

                        if verbose==1:
                            print("last decision IS MEEEEEEE, at index",len(left_children))
                else:
                    if verbose==1:
                        print("i'm outside the interval"*6)
                        print("intervals",len(If1),len(If2))
                        print(IntervalIf)
                    major_all_minors = (IntervalIf[:,0]<nodeNr.thresh)
                    minor_all_majors = (IntervalIf[:,1]>nodeNr.thresh)
                    if verbose==1:
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
                                                right_child = my_newly_child_r,
                                                labels=nodeNr.labels)
                    #se tutti i limiti superiori sono maggiori di me e io non sono nell'intervallo la mia posizione
                    #è all'estrema sinitra
                    if minor_all_majors.all():
                        if verbose==1:
                            print("added at extreme left")
                        adding_node = prune_tree(adding_node,IntervalIf[0][1],attributeXf)
                        if len(left_children)==0:
                            pbranch[0]=adding_node
                        else:
                            pbranch[0]=adding_node
                    #se tutti i limiti inferiori sono minori di me e io non sono nell'intervallo la mia posizione
                    #è all'estrema destra
                    if major_all_minors.all():
                        if verbose==1:
                            print("added at extreme right")
                        adding_node = prune_tree(adding_node,IntervalIf[-1][1],attributeXf)
                        if len(right_children)==0:
                            pbranch[len(left_children)-1]=adding_node
                        else:
                            pbranch[-1] = adding_node
                    if (not minor_all_majors.all()) and (not major_all_minors.all()):
                        my_pos = np.argmax( major_all_minors & minor_all_majors  )
                        if verbose==1:
                            print("added at my specific position",my_pos)
                        adding_node = prune_tree(adding_node,IntervalIf[my_pos][1],attributeXf)
                        pbranch[my_pos]=adding_node
                
                                
                
                if verbose==1:
                    print("result")
                    for p in pbranch:
                            p.print_tree()                   
            return pbranch


#roots are the roots to be merged, num classes is rethoric, 

def mergeObliqueDecisionTrees(roots,num_classes,samples_indexs,level=0,verbose=False,max_depth=20,rule=''):
    #print(len(samples_indexs))
    k = len(roots)
    
    
    if level<-1:#max_depth-1:
        
        out_propose = [(np.argmax(getmaxClass(r)) if not r.is_leaf else 
                        (np.argmax(r.labels) if type(r)==Node else r.label))
                                 for r in roots]
        val,cou = np.unique(out_propose,return_counts=True)
        labels = np.zeros(num_classes)
        for j,vv in enumerate(val):
            labels[vv]=cou[j]
        return SuperNode(is_leaf=True,labels=labels)
    if np.array([r.is_leaf for r in roots]).all():
        out_propose = [(np.argmax(r.labels) if type(r)==Node else r.label) for r in roots]
        val,cou = np.unique(out_propose,return_counts=True)
        labels = np.zeros(num_classes)
        if verbose:
            print('all_leafs,values are:',val,cou,"(",out_propose,")")
        
        
        for j,vv in enumerate(val):
            labels[vv]=cou[j]
        if verbose:
            print('all_leafs,return set of labels',val,cou,"(",labels,")")
        return SuperNode(is_leaf=True,labels=labels)
    
    else:
        
        

        
        out_propose = [np.argmax(r.labels) for r in roots if r.is_leaf]
        val_out,cou_out = 0,0
        if out_propose:
            val_out,cou_out = np.unique(out_propose,return_counts=True)
            if verbose:
                print("Can i terminate already?",val_out,cou_out,int(k/2)+1)
            if np.max(cou_out)>=int(k/2)+1:
                labels = np.zeros(num_classes)
                
                for j,vv in enumerate(val_out):
                    labels[vv]=cou_out[j]
                if verbose:
                    print('there are almost all_leafs,values are:',
                    val_out,cou_out,"(",out_propose,")")
                    print("ENDING EARLIER!"*10)
                return SuperNode(is_leaf=True,labels=labels)


        if verbose < -3:
            print("MERGE "+str(k)+" trees","level",level,"rule is:",rule,samples_indexs.shape[0],"samples")
            
            '''for t in roots:
                print(type(t)==Node,t._weights is None)
                t.print_tree()
                print()'''
            plt.title("level"+str(level)+"\nrule is:"+rule+"\n"+str(samples_indexs.shape[0])+"samples")
            if len(samples_indexs)>0:
                plt.scatter(OriginalDataset[:,0],OriginalDataset[:,1],marker='+',label = str(val_out)+" votes "+str(cou_out))
                print(samples_indexs)
                if (samples_indexs!=[]):
                    plt.scatter(OriginalDataset[samples_indexs,0],OriginalDataset[samples_indexs,1])
                plt.legend()
                plt.xlim(0,5)
                plt.ylim(0,5)
                plt.show()
        #If there is root that comes from an OT
        #choses it as last one in order to keep more transparent the node

        val,cou = np.unique([r.feat for r in roots if not r.feat is None and r._weights is None],return_counts=True) 
        if val.size>0: 
            Xf = val[np.argmax(cou)]
            
            If = list(np.unique([r.thresh for r in roots if r.feat==Xf]))
            if len(samples_indexs)>0:
                my_partition =  OriginalDataset[samples_indexs,Xf]
                
                sub_partition_samples = []
                next_partition = np.arange(samples_indexs.shape[0]).astype(int)
                
                
                for nxi,v in enumerate(If):
                    '''print(nxi,v)
                    print("next_part",next_partition)
                    print("checking on",samples_indexs[next_partition])
                    print(len(next_partition)," and ",len(samples_indexs[next_partition]))'''
                    respecting_cond_indexes_of_indexes = samples_indexs[next_partition[my_partition[next_partition]<=v]]
                    
                    sub_partition_samples.append(respecting_cond_indexes_of_indexes)
                    next_partition = next_partition[my_partition[next_partition]>v]

                sub_partition_samples.append(samples_indexs[next_partition])
            else:
                sub_partition_samples = [[]]
                for _ in If:
                    sub_partition_samples.append([])
            
            If = np.array(list(zip([-np.inf]+If,If+[np.inf])))

            
            if (verbose==3):
                print("most used features",val,cou,'superNode is X_',Xf,If)
            branch = []   #sub_trees under conditions
            for i in range(k):
                # is an array of roots of the condition sub trees (in which each subtree respect the rule recursivey)
                branch.append(computeOBranch(roots[i],If,Xf,sub_partition_samples,verbose))
                assert len(branch[-1])==len(If),len(branch[-1])
            if verbose==1:
                print(len(branch),'condition trees')
                for K in range(k):
                    for j,v in enumerate(If):
                        print("under interval X",Xf,"<=",v)
                        print("subTRee",K,"has",len(branch[K]),"cdTree printing subtree",j)
                        (branch[K][j]).print_tree()
            assert len(branch[0])==(len(If)),"Not equal sub trees intervals tree0 has "+str(len(branch[0]))+" condition trees but threre are conditions"+str(len(If))
            
            
            
            children = []
            r = rule
            r+=", X_"+str(Xf)
            for j,v in enumerate(If):
                
                children.append(mergeObliqueDecisionTrees(roots = [branch[K][j] for K in range(k)],
                                                    num_classes = num_classes,
                                                    samples_indexs = sub_partition_samples[j],
                                                    level = level+1,
                                                    verbose = verbose,
                                                    max_depth=max_depth,
                                                    rule=r+'('+str(np.round(v,2))+","+str(j+1)+'\\'+str(len(If))+')'))
            
            
            #mergeDecisionTrees()
            If = If[:,1]
            
            new_root = SuperNode(feat_num=Xf,intervals=np.array(If),children=children)
            return new_root
        else:
            #chose a random ObliqueRoot
            
            ObliqueRoots  = [i for i,r in enumerate(roots) if not r._weights is None]
            if (verbose>0):
                print(len(ObliqueRoots)," Oblique roots choose the first")
            chosen_one = roots[ObliqueRoots[0]]
            if verbose == 2:
                X0 = np.linspace(0,5,50)
                X1 = np.linspace(0,5,50)
                XX = make_meshgrid(X0,X1,h=.2)
                y_deb = XX.dot(chosen_one._weights[:-1]) - chosen_one._weights[-1] <= 0
                y_deb[y_deb==False] = 0
                y_deb[y_deb==True] = 1
                print("MERGE "+str(k)+" trees","level",level,"rule is:",rule,samples_indexs.shape[0],"samples")
                
                '''for t in roots:
                    print(type(t)==Node,t._weights is None)
                    t.print_tree()
                    print()'''
                plt.title("level"+str(level)+"\nrule is:"+rule+"\n"+str(samples_indexs.shape[0])+"samples")
                plt.scatter(XX[:,0],XX[:,1],c=y_deb)
                if len(samples_indexs)>0:
                    plt.scatter(OriginalDataset[:,0],OriginalDataset[:,1],marker='+',label = str(val_out)+" votes "+str(cou_out))
                    
                    if (samples_indexs!=[]):
                        plt.scatter(OriginalDataset[samples_indexs,0],OriginalDataset[samples_indexs,1])
                    plt.legend()
                    plt.xlim(0,5)
                    plt.ylim(0,5)
                    plt.show()

            
            sub_partition_samples = []
            if len(samples_indexs)>0:
                my_partition =  OriginalDataset[samples_indexs,:]
                
                sub_partition_samples = []
                next_partition = np.arange(samples_indexs.shape[0]).astype(int)
                #print(np.argmax(chosen_one._weights[:-1]))
                for nxi in range(1):
                    
                    #print("Oblique next_part",next_partition)
                    #print("Oblique checking on",samples_indexs[next_partition])
                    #print(len(next_partition)," and ",len(samples_indexs[next_partition]))
                    respecting_cond_indexes_of_indexes = my_partition[next_partition,:].dot(chosen_one._weights[:-1])<=chosen_one._weights[-1]
                    #print(respecting_cond_indexes_of_indexes)
                    sub_partition_samples.append(samples_indexs[respecting_cond_indexes_of_indexes])
                    next_partition = np.array(list(set(samples_indexs).difference(set(samples_indexs[respecting_cond_indexes_of_indexes]))))


                sub_partition_samples.append(next_partition)
                #for idx_part,e in enumerate(sub_partition_samples):
                #    print("partition"+str(idx_part)+" has ",len(e),"elements/",len(samples_indexs))            
            else:
                for _ in range(2):
                    sub_partition_samples.append(np.array([]))

            children = [0,0]
            l_pbranch = np.array([computeOBranch(r,[None],None,[sub_partition_samples[0]],verbose)[0] for r in roots if r!=chosen_one]+[chosen_one._left_child])
            r_pbranch = np.array([computeOBranch(r,[None],None,[sub_partition_samples[1]],verbose)[0] for r in roots if r!=chosen_one]+[chosen_one._right_child])
            r=rule
            r+=', OB#'+str(np.argmax(chosen_one._weights[:-1]))+" |Ws|"+str(np.round(np.linalg.norm(chosen_one._weights),3))


            children[0] = mergeObliqueDecisionTrees(l_pbranch,
                                                    num_classes,
                                                    sub_partition_samples[0],
                                                    level+1,
                                                    verbose,
                                                    max_depth=max_depth,
                                                    rule=r+'(1/2)')
            
            children[1] = mergeObliqueDecisionTrees(r_pbranch,
                                                    num_classes,
                                                    sub_partition_samples[1],
                                                    level+1,
                                                    verbose,
                                                    max_depth=max_depth,
                                                    rule=r+'(2/2)')
            
            new_root = SuperNode(weights=chosen_one._weights,children=children)
            
            '''print(chosen_one._weights)
            print(chosen_one._features_involved)'''
            new_root._features_involved = chosen_one._features_involved
            
            
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
    'n_estimators': [10,16, 20, 24,28,32],
    'base_estimator__min_samples_split': [2, 5,10,15],
    'base_estimator__min_samples_leaf': [1, 0.001, 0.01, 0.05, 0.1, 0.2],
    'base_estimator__max_depth':  [2,3,4,5, 6],
    'base_estimator__class_weight': [None, 'balanced'],
    "bootstrap_features":[False],
    "random_state":[0],
    'base_estimator__random_state': [0],
}

def bestRFRefit (cv_results):
    res = pd.DataFrame(cv_results)
    return res.sort_values(["rank_test_score","param_n_estimators","param_base_estimator__max_depth"],ascending=[True,True,True]).index[0]

import matplotlib.pyplot as plt

if __name__ == "__main__":
    from sklearn.datasets import load_iris,load_wine,make_moons,make_multilabel_classification
    from sklearn.ensemble import BaggingClassifier
    from sklearn.model_selection import train_test_split
    from simplestdatasets import load_AOM
    from load_exp_data import *
    import time
    
    #X, y = make_moons(n_samples=1000,random_state=0)
    #X, y = load_employee(return_X_y=True,prototipy=True)
    #X, y = load_fico(return_X_y=True,prototipy=False)
    #
    
    #X, y = load_employee(return_X_y=True,prototipy=True,targetEnc=True)
    
    #X, y = load_AOM(num_samples=3000,random_state=0,prototipy=False)
    #X,y = load_iris(return_X_y=True)
    #X, y = make_multilabel_classification(n_samples=1000,n_features=10,n_classes=4)
    #X = (X - np.mean(X,axis=0)) / np.std(X,axis=0)
    X,y = load_german(prototipy=True)
    #y = np.argmax(y,axis=1)
    print(X.shape,y.shape)
    y=y.reshape(-1)
    #prototypeFeatures(X,y)
    
    '''plt.scatter(X[:,0],X[:,1],c=y)
    
    plt.show()'''
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

    MAX_DEPTH = 4
    num_estimators = 3
    '''
    merging 9 estimators
    checkSide on original
    for a maximum of  10 levels
    elapsed 0.3455159664154053 9 estimators
    size 345 dept 21 n rules 346
    '''
    '''
    MAX_DEPTH = 4
    num_estimators = 4
    RAND CART vs Ortho-DT vs HouseHolderCart
    size 54599 dept 17 vs size 163 dept 6 vs size 3363 dept 14
    '''
    
    start = time.time()
    forest = BaggingClassifier(base_estimator =
                                        #HHCart.HouseHolderCART(max_depth=MAX_DEPTH),
                                        #DecisionTreeClassifier(max_depth=MAX_DEPTH),
                                        RandCart.Rand_CART(max_depth=MAX_DEPTH),
                                        n_estimators=num_estimators,
                                        #max_features= 6,
                                        bootstrap=True,
                                        bootstrap_features=False,
                                        random_state=0)
    forest.fit(X_train,y_train)
    print("elapsed",time.time()-start,"for costruction")
    
    y_pred_train_forest = forest.predict(X_train)
    y_pred_test = forest.predict(X_test)

    T1 = Node(feat_num=0,thresh=0,is_leaf=False)
    T1._left_child = Node(feat_num=1,thresh=-1)
    T1._left_child._left_child = Node(is_leaf=True,labels=[0,10])
    T1._left_child._right_child = Node(is_leaf=True,labels=[0,10])
    T1._right_child = Node(feat_num=1,thresh=3)
    T1._right_child._left_child = Node(is_leaf=True,labels=[0,10])
    T1._right_child._right_child = Node(is_leaf=True,labels=[10,0])
    
    T2 = Node(weights=np.array([1,1,2]))
    T2._left_child = Node(is_leaf=True,labels=[0,10])
    T2._right_child = Node(weights=np.array([-1,1,4]))
    T2._right_child._left_child = Node(is_leaf=True,labels=[0,10])
    T2._right_child._right_child = Node(is_leaf=True,labels=[10,0])


    
    #plot_tree.PlotTree(forest.estimators_[0],classes=["c1","c2","c3"],features=["X"+str(i) for i in range(4)])
    n_class = len(set(y))

    '''roots = [T1,T2]
    [r.print_tree() for r in roots]
    '''
    if type(forest.estimators_[0])!=DecisionTreeClassifier:
        print("not orthogonal forest")
        roots = np.array([buildTreefromOblique(t._root,n_class,FI_used) for t,FI_used in zip(forest.estimators_,forest.estimators_features_)])
    else:

        print("orthogonal forest")
        for r in forest.estimators_:
            prune_duplicate_leaves(r)
        roots = np.array([rec_buildTree(t,FI_used) for t,FI_used in zip(forest.estimators_,forest.estimators_features_)])
    '''    for j,t in enumerate(roots):
        print("feature used",forest.estimators_features_[j])
        t.print_tree()'''

    y_pred_train_forest = forest.predict(X_train)
    y_pred_test = forest.predict(X_test)
    
    print("Train",np.round(accuracy_score(y_train,y_pred_train_forest),3),
            "Test",np.round(accuracy_score(y_test,y_pred_test),3))

    for r in roots:
        r.print_tree() 
    if type(forest.estimators_[0])==DecisionTreeClassifier:
        import MergeTrees
        superT2 = MergeTrees.mergeDecisionTrees(roots, num_classes= np.unique(y).shape[0])
        size,dept,n_rules = check_size(superT2)
        '''if dept<10 and size<50:
            superT2.print_superTree()'''
        #print("size",size,"dept",dept,"n rules",n_rules)
    
    X0 = np.linspace(min(X_train[:,0]),max(X_train[:,0]),50)
    X1 = np.linspace(min(X_train[:,1]),max(X_train[:,1]),50)
    XX = make_meshgrid(X0,X1,h=.2)
    for eeest in [num_estimators]:#range(3,num_estimators,3):#[10,20,30,40]:#range(2,num_estimators):   
            print("merging the first"+str(eeest)+" estimators")
            partition_elements_index = np.arange(X.shape[0])
            startTime = time.time()
            superT = mergeDecisionTrees(roots[:eeest],
                                        num_classes = np.unique(y).shape[0],
                                        OriginalPoints = X_train,
                                        maximum_depth=15,
                                        verbose=False)
            print("elapsed",time.time()-startTime,eeest,"estimators")
            size,dept,n_rules = check_size(superT)


            #if dept<10 and size<50:
            #    superT.print_superTree()
            print("size",size,"dept",dept,"n rules",n_rules)
            print("Supertree:")
            print("Train",np.round(accuracy_score(y_train,superT.predict(X=X_train)),3))
            print("Test",np.round(accuracy_score(y_test,superT.predict(X=X_test)),3))
            print("Fidelity Train",np.round(accuracy_score(forest.predict(X_train),superT.predict(X=X_train)),3))
            print("Fidelity Test",np.round(accuracy_score(forest.predict(X_test),superT.predict(X=X_test)),3))
            
            '''plt.subplot(1,3,1)
            plt.title("original Forest")
            #plt.scatter(XX[:,0],XX[:,1],c=forest.predict(XX),marker='s',s=100)
            plt.scatter(X_train[:,0],X_train[:,1],c=forest.predict(X_train))

            plt.subplot(1,3,2)
            plt.title("Merged")
            #plt.scatter(XX[:,0],XX[:,1],c=superT.predict(XX),marker='s',s=100)
            plt.scatter(X_train[:,0],X_train[:,1],c=superT.predict(X_train))
            
            plt.subplot(1,3,3)
            plt.title("Differencies")
            #plt.scatter(XX[:,0],XX[:,1],c=forest.predict(XX)==superT.predict(XX),marker='s',s=100)
            plt.scatter(X_train[:,0],X_train[:,1],c=forest.predict(X_train)==superT.predict(X_train),marker='s',s=100)
            
            plt.show()'''
    
    
    
    
    if type(forest.estimators_[0])==DecisionTreeClassifier:        
        print("Supertree2:")
        print("Train",np.round(accuracy_score(y_train,superT2.predict(X=X_train)),3),
            "Test",np.round(accuracy_score(y_test,superT2.predict(X=X_test)),3))
        