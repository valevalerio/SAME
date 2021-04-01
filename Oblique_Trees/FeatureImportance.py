from HHCart import *
import numpy as np


feature_importances = np.zeros(10)
def featureImportance(radix,total_samples):
  global feature_importances
  
  l_counts = r_counts = r_impurity = l_impurity = 0
    #print(radix.depth," ".join(np.round(radix._weights,2).astype(str)),radix.split_rules)
  if (not radix._left_child.is_leaf):
    l_counts,l_impurity = featureImportance(radix._left_child,total_samples)
  if (not radix._right_child.is_leaf):
    r_counts,r_impurity = featureImportance(radix._right_child,total_samples)

  N_t = np.sum(radix.counts)
  N_t_r = np.sum(r_counts)
  N_t_l = np.sum(l_counts)
  w = radix._weights[:-1] 
  contribution = ((w*w)/np.linalg.norm(w))\
                     *(N_t/total_samples )*(radix.impurity - (N_t_r/N_t) * r_impurity - (N_t_l/N_t) *l_impurity)
  feature_importances += contribution
  return radix.counts,radix.impurity

def TreeDepth(radix):
  if (radix.is_leaf):
    return 1
  else:
    return 1 + max(TreeDepth(radix._left_child),TreeDepth(radix._right_child))
def calc_total_samples(radix):
    return  np.sum(radix.counts) +\
    ((calc_total_samples(radix._left_child) +calc_total_samples(radix._right_child) ) if (not radix.is_leaf) else 0) 


def getTop(radix,total_samples,k):
    global metafeatures,node_id
    my_id = node_id
    node_id += 1
    best_topK = []
    if (radix.is_leaf):#no info on top features
        return radix.counts,0
    else:
    
        l_counts,l_impurity = getTop(radix._left_child,total_samples,k)
        r_counts,r_impurity = getTop(radix._right_child,total_samples,k)

        N_t = np.sum(radix.counts)
        N_t_r = np.sum(r_counts)
        N_t_l = np.sum(l_counts)
        w = radix._weights[:-1]
        contribution = ((w*w)/np.linalg.norm(w))\
                         *(N_t/total_samples )*(radix.impurity - (N_t_r/N_t) * r_impurity - (N_t_l/N_t) *l_impurity)
        
        topK = np.argpartition(contribution,-k)[-k:]
        importance_root = contribution[topK][:k]
        #print("so far ",best_topK,best_importance)
        
        if (importance_root>0).all() : # all the top k contribution all >0 (is an oblique node)
          #print("i propose",topK,"with importance",importance_root.sum(),(importance_root>0).all(),importance_root>0)
          importance_root=importance_root.sum()
          topK = tuple(set(topK))
          try:
            currmax = importance_root
            
            if importance_root>metafeatures[topK][0]:
              metafeatures[topK] = (importance_root,my_id)
            else:
              "already saved the metafeature, even with higher importance"
          except KeyError:
            metafeatures[topK] = (importance_root,my_id)

        return radix.counts,radix.impurity

metafeatures = {}
node_id = -1
def TopFeatures(obliqueT,k):
    global metafeatures,node_id
    node_id = -1
    metafeatures = {}
    total_samples = calc_total_samples(obliqueT._root)
    getTop(obliqueT._root,total_samples,k)
    metaDict = np.array([(k,metafeatures[k][0],metafeatures[k][1]) for k in metafeatures.keys()])
    
    for (e,k,n) in sorted(metaDict, key=lambda i: i[1], reverse=True):
      print(e,"are the features with",k,"importance","from node",n)
    return metafeatures


def FI(obliqueT,normalize = True):
    global feature_importances
    
    total_samples = calc_total_samples(obliqueT._root)
    
    feature_importances = np.zeros(obliqueT._root._weights.shape[0]-1)
    if not obliqueT._root.is_leaf:
      featureImportance(obliqueT._root,total_samples)
    else:
      print("Warning Tree is a Leaf!")
    if normalize:
        normalizer = feature_importances.sum()
        
        if normalizer > 0.0:
            feature_importances/=normalizer
    return np.array(list(feature_importances))



if __name__ == "__main__":
    from plot_tree import PlotTree
    from segmentor import *
    from sklearn.datasets import load_breast_cancer,load_iris
    import matplotlib.pyplot as plt
    X, y = load_breast_cancer(return_X_y=True)
    X = (X - np.mean(X,axis=0)) / np.std(X,axis=0) #Standarization of data
    
    hhcart = HouseHolderCART(MSE(), MeanSegmentor(),max_depth=None)
    hhcart.fit(X,y)
    
    TopFeatures(hhcart,2)
    #plt.legend()
    #plt.show()
    #from plot_tree import PlotTree
    g= PlotTree(obliqueT=hhcart,
            visual=True,
            filename="OTTest",
            classes ={i:"Y"+str(i) for i in list(set(y))},
            features = ["feature_"+str(i) for i in range(X.shape[1])])
    plt.clf()
    plt.bar([i for i in range(X.shape[1])],FI(hhcart))
    plt.show()

    