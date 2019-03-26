# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Homework 3 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
import os
import graphviz

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import sys
sys.setrecursionlimit(12000)

def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """

    # INSERT YOUR CODE HERE
    return {c: (x==c).nonzero()[0] for c in np.unique(x)}
    raise Exception('Function not yet implemented!')

#x1 = [1,2,3,2]
#print(partition(x1))

def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    # INSERT YOUR CODE HERE
    h = 0
    
    # return_counts=True returns the number of times each unique item appears
    val, counts = np.unique(y, return_counts=True)
    probs = counts.astype('float')/len(y)
    
    for p in probs:
        if p != 0.0:
            h -= p * np.log2(p)
            
    return h

    raise Exception('Function not yet implemented!')


def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """

    # INSERT YOUR CODE HERE
    mi = entropy(y)

     # return_counts=True returns the number of times each unique item appears
    val, counts = np.unique(x, return_counts=True)
    probs = counts.astype('float')/len(x)

    # We calculate a weighted average of the entropy
    for p, v in zip(probs, val):
        mi -= p * entropy(y[x == v])

    return mi

    raise Exception('Function not yet implemented!')


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    # If there is only one unique label then return the label
    if len(set(y)) == 1:
        return y
    
    # If there is only one label then return the label
    if len(y) == 0:
        return y
    
    # Create empty tree
    tree = {}

    # Select the feature that gives highest mutual information (best feature)
    m_infos = np.array([mutual_information(feature, y) for feature in x.T])
    best_feature = np.argmax(m_infos)

    # Split using the selected feature
    splits = partition(x[:, best_feature])
    
    # Grow tree based on split of the best feature
    for key, val in splits.items():
        node = x.take(val, axis=0)
        label = y.take(val, axis=0)
        tree["%s = %d" % (best_feature, key)] = id3(node, label)

    return tree

    raise Exception('Function not yet implemented!')


def predict_example(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    for key in tree.keys():
        value = x[key]
        tree = tree[key][value]
        predicted_label = 0;
        if type(tree) is dict:
            predicted_label = predict_example(x, tree)
        else:
            predicted_label = tree
            break;
            
    return predicted_label
    raise Exception('Function not yet implemented!')


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    # INSERT YOUR CODE HERE
    misclassifications = 0
    n = len(y_true)
    
    for i in y_true:
        if(y_true[i] != y_pred[i]):
            misclassifications += 1
    
    return misclassifications/n

    raise Exception('Function not yet implemented!')


def pretty_print(tree, depth=0):
    """
    Pretty prints the decision tree to the console. Use print(tree) to print the raw nested dictionary representation
    DO NOT MODIFY THIS FUNCTION!
    """
    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print('+-- [SPLIT: x{0} = {1} {2}]'.format(split_criterion[0], split_criterion[1], split_criterion[2]))

        # Print the children
        if type(sub_trees) is dict:
            pretty_print(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def render_dot_file(dot_string, save_file, image_format='png'):
    """
    Uses GraphViz to render a dot file. The dot file can be generated using
        * sklearn.tree.export_graphviz()' for decision trees produced by scikit-learn
        * to_graphviz() (function is in this file) for decision trees produced by  your code.
    DO NOT MODIFY THIS FUNCTION!
    """
    if type(dot_string).__name__ != 'str':
        raise TypeError('visualize() requires a string representation of a decision tree.\nUse tree.export_graphviz()'
                        'for decision trees produced by scikit-learn and to_graphviz() for decision trees produced by'
                        'your code.\n')

    # Set path to your GraphViz executable here
    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    graph = graphviz.Source(dot_string)
    graph.format = image_format
    graph.render(save_file, view=True)


def to_graphviz(tree, dot_string='', uid=-1, depth=0):
    """
    Converts a tree to DOT format for use with visualize/GraphViz
    DO NOT MODIFY THIS FUNCTION!
    """

    uid += 1       # Running index of node ids across recursion
    node_id = uid  # Node id of this node

    if depth == 0:
        dot_string += 'digraph TREE {\n'

    for split_criterion in tree:
        sub_trees = tree[split_criterion]
        attribute_index = split_criterion[0]
        attribute_value = split_criterion[1]
        split_decision = split_criterion[2]

        if not split_decision:
            # Alphabetically, False comes first
            dot_string += '    node{0} [label="x{1} = {2}?"];\n'.format(node_id, attribute_index, attribute_value)

        if type(sub_trees) is dict:
            if not split_decision:
                dot_string, right_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, right_child)
            else:
                dot_string, left_child, uid = to_graphviz(sub_trees, dot_string=dot_string, uid=uid, depth=depth + 1)
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, left_child)

        else:
            uid += 1
            dot_string += '    node{0} [label="y = {1}"];\n'.format(uid, sub_trees)
            if not split_decision:
                dot_string += '    node{0} -> node{1} [label="False"];\n'.format(node_id, uid)
            else:
                dot_string += '    node{0} -> node{1} [label="True"];\n'.format(node_id, uid)

    if depth == 0:
        dot_string += '}\n'
        return dot_string
    else:
        return dot_string, node_id, uid


if __name__ == '__main__':
    # Load the training and testing data
    Mtrn1 = np.genfromtxt('./monks-1.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn1 = Mtrn1[:, 0]
    Xtrn1 = Mtrn1[:, 1:]

    Mtst1 = np.genfromtxt('./monks-1.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst1 = Mtst1[:, 0]
    Xtst1 = Mtst1[:, 1:]
    
    Mtrn2 = np.genfromtxt('./monks-2.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn2 = Mtrn2[:, 0]
    Xtrn2 = Mtrn2[:, 1:]

    Mtst2 = np.genfromtxt('./monks-2.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst2 = Mtst2[:, 0]
    Xtst2 = Mtst2[:, 1:]
    
    Mtrn3 = np.genfromtxt('./monks-3.train', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytrn3 = Mtrn3[:, 0]
    Xtrn3 = Mtrn3[:, 1:]

    Mtst3 = np.genfromtxt('./monks-3.test', missing_values=0, skip_header=0, delimiter=',', dtype=int)
    ytst3 = Mtst3[:, 0]
    Xtst3 = Mtst3[:, 1:]

#=========For depth = 1 to 10, learn decision trees 
#         and compute the average training and test errors on each of 
#         the three MONK’s problems===================================

    trn_err1 = {}
    tst_err1 = {}
    
    trn_err2 = {}
    tst_err2 = {}
    
    trn_err3 = {}
    tst_err3 = {}    
    
    for d in range (1,4):
        # Learn a decision tree of depth d
        decision_tree1 = id3(Xtrn1, ytrn1, max_depth=d)
        decision_tree2 = id3(Xtrn2, ytrn2, max_depth=d)
        decision_tree3 = id3(Xtrn3, ytrn3, max_depth=d)
        
        # Predict using test set
        y_pred1 = [predict_example(x, decision_tree1) for x in Xtst1]
        y_pred2 = [predict_example(x, decision_tree2) for x in Xtst2]
        y_pred3 = [predict_example(x, decision_tree3) for x in Xtst3]
        
        # Compute and store the training errors
        trn_err1[d] = compute_error(ytrn1, y_pred1)
        trn_err2[d] = compute_error(ytrn2, y_pred2)
        trn_err3[d] = compute_error(ytrn3, y_pred3)
        #print('Test Error = {0:4.2f}%.'.format(tst_err1 * 100))
        
        # Compute and store the test errors 
        tst_err1[d] = compute_error(ytst1, y_pred1)
        tst_err2[d] = compute_error(ytst2, y_pred2)
        tst_err3[d] = compute_error(ytst3, y_pred3)
        #print('Test Error = {0:4.2f}%.'.format(tst_err1 * 100))
        
    # Plot the training and Testing errors with varying depth
    plt.figure()
    plt.plot(trn_err1.keys(), trn_err1.values(), marker='s', linewidth=3, markersize=12)
    plt.plot(tst_err1.keys(), tst_err1.values(), marker='o', linewidth=3, markersize=12)
    plt.xlabel('Tree depth', fontsize=16)
    plt.ylabel('Trn/Test error', fontsize=16)
    plt.xticks(list(trn_err1.keys()), fontsize=12)
    plt.legend(['Training Error', 'Test Error'], fontsize=16)
    
    plt.figure()
    plt.plot(trn_err2.keys(), trn_err2.values(), marker='s', linewidth=3, markersize=12)
    plt.plot(tst_err2.keys(), tst_err2.values(), marker='o', linewidth=3, markersize=12)
    plt.xlabel('Tree depth', fontsize=16)
    plt.ylabel('Trn/Test error', fontsize=16)
    plt.xticks(list(trn_err2.keys()), fontsize=12)
    plt.legend(['Training Error', 'Test Error'], fontsize=16)

    plt.figure()
    plt.plot(trn_err3.keys(), trn_err3.values(), marker='s', linewidth=3, markersize=12)
    plt.plot(tst_err3.keys(), tst_err3.values(), marker='o', linewidth=3, markersize=12)
    plt.xlabel('Tree depth', fontsize=16)
    plt.ylabel('Trn/Test error', fontsize=16)
    plt.xticks(list(trn_err3.keys()), fontsize=12)
    plt.legend(['Training Error', 'Test Error'], fontsize=16) 
    
#===========For monks-1, learned decision tree using self implemented classifier and scikit's version
#           and found the confusion matrix on the test set for depth = 1, 3, 5========================
    
    print("Self implementation on monks-1")
    for d in range(1, 6, 2):
        # Learn classifier of depth d
        self_clf = id3(Xtrn1, ytrn1, max_depth=d)
        # Pretty print decision tree to console 
        print("For depth = ",d)
        pretty_print(self_clf)
        # Visualize the tree and save it as a PNG image
        dot_str = to_graphviz(self_clf)
        render_dot_file(dot_str, './my_learned_tree_monks-1_{0}'.format(d))
        # Predict using test set
        self_y_pred = [predict_example(x, self_clf) for x in Xtst1]
        # Compute the confusion matrix
        print(confusion_matrix(ytst1, self_y_pred))
        
    print("Scikit-learn's implementation on monks-1")
    for d in range(1, 6, 2):
        # Learn classifier of depth d
        sklearn_clf = DecisionTreeClassifier(criterion='entropy', max_depth=d)
        sklearn_clf.fit(Xtrn1, ytrn1)
        # Predict using test set
        sklearn_y_pred = [predict_example(x, sklearn_clf) for x in Xtst1]
        # Compute the confusion matrix
        print(confusion_matrix(ytst1, sklearn_y_pred)) 

    X_ttt = np.loadtxt('tic-tac-toe.txt', delimiter=',', usecols=(np.arange(0,8))) 
    y_ttt = np.loadtxt('tic-tac-toe.txt', delimiter=',', usecols=(9))
    X_ttt_trn, X_ttt_tst, y_ttt_trn, y_ttt_tst = train_test_split(X_ttt, y_ttt, test_size=0.3, random_state=42)
    
    print("Self implementation on tic-tac-toe data")
    for d in range(1, 6, 2):
        # Learn classifier of depth d
        self_clf = id3(X_ttt_trn, y_ttt_trn, max_depth=d)
        # Pretty print decision tree to console 
        print("For depth = ",d)
        pretty_print(self_clf)
        # Visualize the tree and save it as a PNG image
        dot_str = to_graphviz(self_clf)
        render_dot_file(dot_str, './my_learned_tree_monks-1_{0}'.format(d))
        # Predict using test set
        self_y_pred = [predict_example(x, self_clf) for x in X_ttt_tst]
        # Compute the confusion matrix
        print(confusion_matrix(y_ttt_tst, self_y_pred))
        
    print("Scikit-learn's implementation on tic-tac-toe data")
    for d in range(1, 6, 2):
        # Learn classifier of depth d
        sklearn_clf = DecisionTreeClassifier(criterion='entropy', max_depth=d)
        sklearn_clf.fit(X_ttt_trn, y_ttt_trn)
        # Predict using test set
        sklearn_y_pred = [predict_example(x, sklearn_clf) for x in X_ttt_tst]
        # Compute the confusion matrix
        print(confusion_matrix(y_ttt_tst, sklearn_y_pred))
  