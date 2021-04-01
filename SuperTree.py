import numpy as np
from sklearn.tree import DecisionTreeClassifier

class Node():
    def __init__(self,feat_num=None,weights=None,thresh=None,labels=None,is_leaf=False,impurity=1,**kwargs):
        self.feat = feat_num
        self.thresh = thresh
        self.is_leaf = is_leaf
        self._weights = weights
        self._left_child = kwargs.get('left_child', None)
        self._right_child = kwargs.get('right_child', None)
        self.children = kwargs.get('children', None)
        self.impurity = impurity
        self.labels = labels
        if not (weights is None):
            self._features_involved = np.arange(weights.shape[0]-1)
        
        else:
            if not self.children:
                self.children=[]
                if self._left_child:
                    self.children.append(self._left_child)
                if self._right_child:
                    self.children.append(self._right_child)
    
    def print_tree(Tree,level=0,feature_names=None):
        if Tree.is_leaf:
            print("|\t"*level+"|---",np.argmax(Tree.labels),Tree.labels)
        else:
            if Tree._weights is None:
                if not Tree._left_child is None:
                    print("|\t"*level+"|---","X",Tree.feat,"<=",Tree.thresh)
                    Node.print_tree(Tree._left_child,level+1)
                if not Tree._right_child is None:
                    print("|\t"*level+"|---","X",Tree.feat,">",Tree.thresh)
                    Node.print_tree(Tree._right_child,level+1)
            else:
                
                if feature_names is None:
                    feature_names = ["X_"+str(i) for i in Tree._features_involved]
                rule = ''
                for i,f in enumerate(feature_names):
                    rule+= (("+" if i!=0 and np.sign(Tree._weights[i])>0 else "")+\
                                            str(np.round(Tree._weights[i],4))+" "+f+" ") \
                                            if np.abs(Tree._weights[i])>1e-2 else ""
                left_rule = rule + "<="+str(np.round(Tree._weights[-1],4))
                right_rule = rule + "> "+str(np.round(Tree._weights[-1],4))
                if not Tree._left_child is None:
                    print("|\t"*level+"/--",left_rule)
                    Node.print_tree(Tree._left_child,level+1)
                if not Tree._right_child is None:
                    print("|\t"*level+"/--",right_rule)
                    Node.print_tree(Tree._right_child,level+1)

def buildTreefromOblique(Ocart,n_classes,feature_used):
    if Ocart.is_leaf:
        
        return Node(is_leaf=True,labels=Ocart.counts)
    else:
        equal_1 = np.where(Ocart._weights[:-1]==1)[0] 
        #it's an orthogonal Node
        if len(equal_1)==1:
            nr = Node(feat_num=feature_used[equal_1[0]],thresh=Ocart._weights[-1],labels=Ocart.counts,is_leaf=False,impurity=Ocart.impurity)
            nr._left_child = buildTreefromOblique(Ocart._left_child,n_classes,feature_used)
            nr._right_child = buildTreefromOblique(Ocart._right_child,n_classes,feature_used)
        #it's an oblique Node
        else:
            nr = Node(weights=Ocart._weights,labels=Ocart.counts,is_leaf=False,impurity=Ocart.impurity)
            nr._features_involved = np.array(feature_used)
            
            nr._left_child = buildTreefromOblique(Ocart._left_child,n_classes,feature_used)
            nr._right_child = buildTreefromOblique(Ocart._right_child,n_classes,feature_used)
        return nr




def rec_buildTree(DT:DecisionTreeClassifier,feature_used):
    nodes = DT.tree_.__getstate__()['nodes'] 
    values = DT.tree_.__getstate__()['values']
    root = nodes[0]
    def createNode(node_idx):
        #a leaf node
        line = nodes[node_idx]
        if line[0]==-1:
            return Node(feat_num=None,thresh=None,labels = values[node_idx][0],is_leaf=True)
        else:
            LC = createNode(line[0])
            RC = createNode(line[1])
            
            node = Node(feat_num=feature_used[line[2]],thresh=line[3],labels=values[node_idx],is_leaf=False,left_child=LC,right_child=RC)
            return node
    return createNode(0)

def pruneRedundant(BigTree):
    if BigTree.is_leaf:
        return BigTree
    l_c = pruneRedundant(BigTree._left_child)
    r_c = pruneRedundant(BigTree._right_child)

    if l_c.is_leaf and r_c.is_leaf:
        if np.argmax(l_c.labels)==np.argmax(r_c.labels):
            l_c.labels+=r_c.labels
            return l_c
    BigTree._left_child = l_c
    BigTree._right_child = r_c
    return BigTree
    
def prune_tree(BigTree: Node,value,Xf):
    if BigTree is None:
        return None
    if BigTree.is_leaf:
        return BigTree
    
    l_c = prune_tree(BigTree._left_child,value,Xf)
    r_c = prune_tree(BigTree._right_child,value,Xf)
    if l_c.is_leaf and r_c.is_leaf:
        if np.argmax(l_c.labels)==np.argmax(r_c.labels):            
            l_c.labels+=r_c.labels
            return l_c
    if l_c is None:
        return r_c
    if r_c is None:
        return l_c
    
    if BigTree.feat != Xf:#è un nodo split
        BigTree._left_child = l_c
        BigTree._right_child = r_c
        return BigTree
    #lo split è sulla condizione da verificare Xf<=value
    
    if BigTree.thresh < value:
        BigTree._left_child = l_c
        BigTree._right_child = r_c
        return BigTree 
    if BigTree.thresh >= value:
        return l_c
    
class SuperNode():
    def __init__(self,feat_num=None,intervals=None,weights = None,labels=None,children=None,is_leaf=False):
        self.feat = feat_num
        self.is_leaf = is_leaf
        self._weights = weights
        self.children = children
        self.labels = labels
        self.intervals = intervals
        self._features_involved = None # if bootstrap features is false
    def predict(self,X):
        def predict_datum(node,x):        
            if node.is_leaf:
                return np.argmax(node.labels)
            else:
                if not node.feat is None:
                    Xf = node.feat
                    next_node = np.argmin(node.intervals<=x[Xf])
                else:
                    bias = node._weights[-1]
                    next_node = 0 if (x[node._features_involved].dot(np.array(node._weights[:-1]).T) - node._weights[-1] <= 0) else 1
                return predict_datum(node.children[next_node],x)
        return np.array([predict_datum(self,el) for el in X])
    def print_superTree(SuperTree,level=0,feature_names = None):
        
        if SuperTree.is_leaf:
            print("|\t"*level+"|---","class:",np.argmax(SuperTree.labels))
        else:
            if SuperTree._weights is None:
                print("\t"*level,"level",level,",",len(SuperTree.children),"childs",SuperTree.feat)
                feature_names = ["X_"+str(k) for k in range(SuperTree.feat+1)]
                for c_nr,i in enumerate(SuperTree.intervals):
                    print("|\t"*level+"|---",feature_names[SuperTree.feat],"<=",i)
                    #print("\t"*(level+1)+"ch ",c_nr,end='')
                    SuperNode.print_superTree(SuperTree.children[c_nr],level+1)
            else:
                if feature_names is None:
                    feature_names = ["X_"+str(i) for i in SuperTree._features_involved]
                rule = ''
                for i,f in enumerate(feature_names):
                    rule+= (("+" if i!=0 and np.sign(SuperTree._weights[i])>0 else "")+\
                                            str(np.round(SuperTree._weights[i],4))+" "+f+" ") \
                                            if np.abs(SuperTree._weights[i])>1e-2 else ""
                left_rule = rule + "<="+str(np.round(SuperTree._weights[-1],4))
                right_rule = rule + "> "+str(np.round(SuperTree._weights[-1],4))
                
                #print left child first
                print("|\t"*level+"/---",left_rule)
                SuperNode.print_superTree(SuperTree.children[0],level+1)
                #than right
                print("|\t"*level+"/---",right_rule)
                SuperNode.print_superTree(SuperTree.children[1],level+1)
        

def check_size(node:SuperNode):
    if node.is_leaf:
        return np.array([0,1,1]) #for the size, the dept and #rules
    else:
        res = np.array([check_size(c) for c in node.children])
        return np.array([1+np.sum(res[:,0]),1+np.max(res[:,1]),np.sum(res[:,2])])


def complexityDecisionTree(DT):
    nodes = DT.tree_.__getstate__()['nodes'] 
    values = DT.tree_.__getstate__()['values']
    root = nodes[0]
    path_lenghts = []
    def complexDT(node_idx,level):
        #a leaf node
        line = nodes[node_idx]
        if line[0]==-1:
            path_lenghts.append(level+1)
            return 0
        else:
            LC = complexDT(line[0],level+1)
            RC = complexDT(line[1],level+1)
            return 1+ LC + RC#leaves
                    
    res = complexDT(0,0)
    return np.max(path_lenghts),res,len(path_lenghts),np.mean(path_lenghts),1
        #depth,	nodes internals,	leaves,	avg path length,	avg nbr features

def complexitiSuperTree(node:Node):
    path_lenghts = []
    averages = []
    def complexSuper(radix,level=0,n_feat_used=0):
        if radix.is_leaf:
            path_lenghts.append(level)
            return 0
        else:
            
            if radix._weights is None:
                my_feats = 1
            else:
                my_feats = len(set(radix._features_involved))
            res = np.array([complexSuper(c,level+1,n_feat_used+my_feats) for c in radix.children])
            return 1+np.sum(res)
    res = complexSuper(node)
    return np.max(path_lenghts),res,len(path_lenghts),np.mean(path_lenghts),np.mean(averages)

def complexityOblique(OT):
    path_lenghts = []
    averages = []
    def complexOT(radix,level=0,n_feat_used=0):
        #a leaf node
        if radix.is_leaf:
            path_lenghts.append(level+1)
            averages.append(n_feat_used/level)
            return 0
        else:
            equal_1 = np.where(radix._weights[:-1]==1)[0] 
            #it's an orthogonal Node
            if len(equal_1)==1:
                n_feat_used+=1
            else:
                n_feat_used+=radix._weights.shape[0]-1
            LC = complexOT(radix._left_child,level+1,n_feat_used)
            RC = complexOT(radix._right_child,level+1,n_feat_used)
            return 1+LC+RC
    res = complexOT(OT)
    return np.max(path_lenghts),res,len(path_lenghts),np.mean(path_lenghts),np.mean(averages)


from sklearn.tree._tree import TREE_LEAF
from sklearn.tree import DecisionTreeClassifier


def is_leaf(inner_tree, index):
    # Check whether node is leaf node
    return (inner_tree.children_left[index] == TREE_LEAF and
            inner_tree.children_right[index] == TREE_LEAF)

def prune_index(inner_tree, decisions, index=0):
    # Start pruning from the bottom - if we start from the top, we might miss
    # nodes that become leaves during pruning.
    # Do not use this directly - use prune_duplicate_leaves instead.
    if not is_leaf(inner_tree, inner_tree.children_left[index]):
        prune_index(inner_tree, decisions, inner_tree.children_left[index])
    if not is_leaf(inner_tree, inner_tree.children_right[index]):
        prune_index(inner_tree, decisions, inner_tree.children_right[index])
    # Prune children if both children are leaves now and make the same decision:
    if (is_leaf(inner_tree, inner_tree.children_left[index]) and
            is_leaf(inner_tree, inner_tree.children_right[index]) and
            (decisions[index] == decisions[inner_tree.children_left[index]]) and
            (decisions[index] == decisions[inner_tree.children_right[index]])):
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
        # print("Pruned {}".format(index))

def prune_duplicate_leaves(dt):
    # Remove leaves if both
    decisions = dt.tree_.value.argmax(axis=2).flatten().tolist()  # Decision for each node
    prune_index(dt.tree_, decisions)
