from .HHCart import *
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph


node_id = -1
normalizeVec = lambda x : (np.min(x)+x)/np.sum((np.min(x)+x))
feature_names = []
def DFSOblique(radix,UseRule,classes_names,graph):
  global node_id,feature_names 
  my_id = node_id
  node_id+=1
  if radix.is_leaf:
    #print(radix.depth,radix.label,radix.counts,"leaf")
    #print(radix.label)
    lab = classes_names[radix.label]
    graph.node(name="node_"+str(my_id),label = lab+"\n"+str(radix.counts),shape='box')
    return  np.hstack([radix.counts,my_id])
  else:
    
    #print(radix.depth," ".join(np.round(radix._weights,2).astype(str)),radix.split_rules)
    l_counts = DFSOblique(radix._left_child,UseRule,classes_names,graph)
    r_counts = DFSOblique(radix._right_child,UseRule,classes_names,graph)
    l_id = l_counts[-1]
    l_counts = l_counts[:-1]
    r_id = r_counts[-1]
    r_counts = r_counts[:-1]
    rule = ''


    '''ww = radix._weights
    x_axis = np.linspace(-3, 3)
    a = ww[1]
    b = ww[2]
    bias = ww[-1]
    y_axis = x_axis*(-b/a)+(bias/a)
    plt.plot(x_axis,y_axis,label="node_"+str(my_id))
    '''



    
    for i,f in enumerate(feature_names):
      rule+= (("+" if i!=0 and np.sign(radix._weights[i])>0 else "")+str(np.round(radix._weights[i],4))+" "+f+" ") if np.abs(radix._weights[i])>1e-1 else ""
    rule+="< "+str(np.round(radix._weights[-1],4))
    
    
    
    if (UseRule):
      graph.node(name="node_"+str(my_id),shape='box',label=str(my_id)+"\n"+rule) #rule
    else:
      plt.clf()
      plt.title(radix.impurity,fontsize=22)
      plt.barh([ i for i in range(len(radix._weights))],radix._weights,color = [('g' if np.sign(w)<0 else 'b') for w in radix._weights[:-1]]+['k'])
      plt.yticks([i for i in range(len(feature_names))],feature_names)
      plt.savefig(str(my_id)+".png")
      rule+="\n"+str(radix.counts)
      graph.node(name="node_"+str(my_id),shape='box',image =str(my_id)+".png",label=str(my_id))#Feature importance image #label=rule) #rule
    graph.edge("node_"+str(my_id),"node_"+str(l_id),label="left")
    graph.edge("node_"+str(my_id),"node_"+str(r_id),label="right")
    #print(np.hstack([radix.counts,my_id]))
    return np.hstack([radix.counts,my_id])

def PlotTree(obliqueT,filename="OT1Text",visual=True,features=None,classes = None):
    global node_id,feature_names
    node_id=-1
    if features is None:
        features = ["X"+str(i) for i in range(100)]
    
    feature_names = features
    
    
    g = Digraph('G', filename = filename,format = 'png')
    DFSOblique(radix=obliqueT._root,UseRule=not visual,classes_names=classes,graph=g)
    g.view()

    
    import cv2
    img = cv2.imread(filename+'.png')
    plt.figure(figsize = (20, 20))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    import os
    
    os.system("rm [0-9]*.png")
    try:
      os.remove("-1.png")
    except FileNotFoundError:
      45+40
    return g




if __name__ == "__main__":
   
    from sklearn.datasets import load_wine
    X, y = load_wine(return_X_y=True)
    X = (X - np.mean(X,axis=0)) / np.std(X,axis=0) #Standarization of data
    
    hhcart = HouseHolderCART(MSE(), MeanSegmentor())
    hhcart.fit(X,y)
    g= PlotTree(obliqueT=hhcart,
            visual=True,
            filename="OTTest",
            classes =["Y"+str(i) for i in range(len(set(y)))],
            features = ["feature_"+str(i) for i in range(X.shape[1])])

    plt.legend()
    #plt.colorbar()

    g.view()