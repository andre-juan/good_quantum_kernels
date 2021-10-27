import numpy as np
np.set_printoptions(precision=3, suppress=False)

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from itertools import combinations, permutations, product

from random import sample

from qiskit import Aer
from qiskit.circuit.library import PauliFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.kernels import QuantumKernel

from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import time

from dppy.finite_dpps import FiniteDPP

from qiskit.quantum_info import Statevector

from qiskit.visualization import (plot_bloch_multivector, plot_state_city, 
                                  plot_state_hinton, plot_state_qsphere)
                                  

import warnings
warnings.filterwarnings("ignore")

from qiskit import IBMQ

seed = 42
algorithm_globals.random_seed = 42

#######################################################################################
####################################################################################### 

# ================================================================================================= #
# ================================================================================================= #
# ========================== DPPs (pre-processing) 
# ================================================================================================= #
# ================================================================================================= # 

def get_unique_list(ll):
    '''
    auxliar function to flatten and get unique elements of a list of lists
    '''
    
    return list(set([item for sublist in ll for item in sublist]))

#######################################################################################
#######################################################################################   

def row_sample_k_dpp(X, n):
    '''
    this uses a finite k-DPP to sample "n" rows from a given matrix "X",
    using the linear kernel X*X^T.
    this uses the dppy library
    '''
    # transform matrix to numpy array, if not already
    feat_mat = X.to_numpy() if not isinstance(X, np.ndarray) else X
       
    # linear kernel
    L_rows = feat_mat.dot(feat_mat.T)
    
    # finite DPP with linear kernel
    dpp_rows = FiniteDPP(kernel_type='likelihood', L=L_rows)

    # we'll sample 2 rows at a time using the DPP
    # that's especially necessary given that we're mostly interested in n=2 dimensions
    # which means X with 2 columns. This limits the rank of the matrix, which then
    # limits the number of rows sampled by the DPP.
    # in order to construct samples with more than 2 rows, we sample 2 rows several times
    # ("n_tilde" is to make sure that we'll sample an even number of rows)
    n_tilde = n + 1 if n % 2 == 1 else n
    
    for _ in range(int(n_tilde / 2)):
        dpp_rows.sample_exact_k_dpp(size=2)

    # recover sampled rows
    sampled = get_unique_list(dpp_rows.list_of_samples)
    
    # as describred above, the DPP samples 2 rows at a time, and it's definitely possible
    # that it'll sample repeated rows each time, so that we won't
    # be able to get a sample of size "n" ("n_tilde for now).
    # this is why here we keep sampling until we indeed get a sample of size "n_tilde"
    while len(sampled) < n_tilde:
        dpp_rows.sample_exact_k_dpp(size=2)
        aux = dpp_rows.list_of_samples[-1]
        sampled = list(set(sampled + aux))
    
    # since it's possible that n_tilde = n + 1, in the end we sample again "n" rows
    # to guarantee that this is the desired sample size!
    # in the worst case, this drops one of the rows selected by the DPP.
    return pd.DataFrame(X).iloc[sampled, :].sample(n)

#######################################################################################
#######################################################################################   

def row_sample_k_dpp_stratified(X, y, n=50):
    '''
    this uses a finite k-DPP to sample "n" rows from a given matrix "X",
    in a stratified manner with respect to the target, which is important to 
    keep the target distribution from the original data to be subsampled.
    for this function to work, it's important that "X" and "y" have aligned indices
    '''
    
    # get the (sample) target distribution
    prop_classes = y.value_counts(1).sort_index()
    
    # determines how many samples of each class will be drawn
    n0 = np.ceil(prop_classes.iloc[0] * n).astype(int)
    n1 = n - n0
    
    # features filtered by classes
    X_0 = X[y == prop_classes.index[0]]
    X_1 = X[y == prop_classes.index[1]]
    
    # DPP samples, size n0 and n1
    sampled_0 = row_sample_k_dpp(X_0, n0)
    sampled_1 = row_sample_k_dpp(X_1, n1)
    
    # full subsample
    return pd.concat([sampled_0, sampled_1])

#######################################################################################
#######################################################################################   

def col_sample_k_dpp(X, feat_num=2):
    '''
    this uses a finite k-DPP to sample "feat_num" columns from a given matrix "X",
    using the linear kernel X^T*X
    this function is quite similar to "row_sample_k_dpp" (see comments there)
    '''
    
    if feat_num > X.shape[1]:
        raise ValueError('Sample size must be less or equal than the number of features!')
        
    # transform matrix to numpy array, if not already
    feat_mat = X.to_numpy() if not isinstance(X, np.ndarray) else X

    L_cols = feat_mat.T.dot(feat_mat)
    dpp_cols = FiniteDPP(kernel_type='likelihood', L=L_cols)
    
    # that's almost always the case!
    if X.shape[0] > X.shape[1]:
        
        # here, I can directly sample "feat_num", given that X.shape[0] >> X.shape[1]
        dpp_cols.sample_exact_k_dpp(size=feat_num)
        
        sampled = get_unique_list(dpp_cols.list_of_samples)
        
        ans = pd.DataFrame(X).iloc[:, sampled]
    
    # quite unlikely, but it doesn't hurt to cover here...
    else:
        n_tilde = feat_num + 1 if feat_num % 2 == 1 else feat_num
        for _ in range(int(n_tilde / 2)):
            dpp_cols.sample_exact_k_dpp(size=feat_num)
           
        sampled = get_unique_list(dpp_cols.list_of_samples)

        while len(sampled) < n_tilde:
            dpp_cols.sample_exact_k_dpp(size=2)
            aux = dpp_cols.list_of_samples[-1]
            sampled = list(set(sampled + aux))
            
        ans = pd.DataFrame(X).iloc[:, sampled].sample(feat_num)
    
    return ans

#######################################################################################
#######################################################################################   

def subsample_dpp(X, y, n_rows, n_cols):
    '''
    this is a general function that takes a general matrix of features X and series of targets y,
    and returns a subsample with "n_rows" and "n_cols"
    this is just a wrapper of the previous functions
    '''
    
    aux = pd.DataFrame(StandardScaler().fit_transform(X),
                       index=X.index)
    
    if n_cols is None:
    
        ans = row_sample_k_dpp_stratified(aux, y, n=n_rows)
    
    else:
        
        ans = row_sample_k_dpp_stratified(col_sample_k_dpp(aux, feat_num=n_cols), y, n=n_rows)
    
    return ans
    
#######################################################################################
#######################################################################################

def subsample_pca(X, n):
    '''
    returns X with first n PCs
    '''

    aux = StandardScaler().fit_transform(X)
    X_pca = pd.DataFrame(PCA(n_components=n).fit_transform(aux),
                         index=X.index)
    
    return X_pca

#######################################################################################
#######################################################################################

def pre_process_data(X, y, dpp=False, n_rows=50, n_cols=2, pca=False, n_pcs=2):
    '''
    this functions wraps the 2 preprocessing otpions.
    notice that we also normalize the data by default,
    what is interesting for dpp, mandatory for pca,
    but is also interesting to have for quantum kernels,
    given the periodic nature of quantum rotations.
    '''
    
    X = pd.DataFrame(X, 
                 columns=[f"x{i+1}" 
                          for i in range(X.shape[1])]) if not isinstance(X, (pd.DataFrame, 
                                                                             pd.core.frame.DataFrame)) else X
    
    y = pd.Series(y, name="y") if not isinstance(y, (pd.DataFrame, 
                                                 pd.core.frame.DataFrame)) else y
    
    # sub-sampling ____________________________________________
    
    if dpp:
        
        X = subsample_dpp(X, y, n_rows, n_cols)
        y = y[X.index]
        
    if pca:
        
        X = subsample_pca(X, n_pcs)
        
    # scale always. if already scaled, there's no effect, so that's okay
    X = pd.DataFrame(StandardScaler().fit_transform(X),
                     index=X.index)

    # this is important to construct the feature map!
    feature_dim = X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42, stratify=y)
    
    #____________________________________________
    
    # let's now group classes together (this will be important for plotting the Gram matrix)

    df_train = pd.concat([X_train, y_train], axis=1).sort_values(y.name)

    X_train = df_train.drop(columns=y.name)
    y_train = df_train[y.name]

    ###########################################

    df_test = pd.concat([X_test, y_test], axis=1).sort_values(y.name)

    X_test = df_test.drop(columns=y.name)
    y_test = df_test[y.name]
    
    #____________________________________________
    
    
    return X_train, X_test, y_train, y_test, feature_dim

#######################################################################################
#######################################################################################

# ================================================================================================= #
# ================================================================================================= #
# ========================== VISUALIZATION 
# ================================================================================================= #
# ================================================================================================= #

def show_figure(fig):
    '''
    auxiliar function to display plot 
    even if it's not the last command of the cell
    from: https://github.com/Qiskit/qiskit-terra/issues/1682
    '''
    
    new_fig = plt.figure()
    new_mngr = new_fig.canvas.manager
    new_mngr.canvas.figure = fig
    fig.set_canvas(new_mngr.canvas)
    plt.show(fig)
    
    
#######################################################################################
#######################################################################################   
    
def plot_decision_boundary(model, X_train, y_train, X_test, y_test,
                           title="decision boundary", border=None, spacing=0.05,
                           quantum_kernel=True, reps=None, paulis=None, entanglement=None,
                           shots=None):
    
    # to construct the meshgrid below, I must look to the entire feature space.
    # in order to do so, I'll concatenate training and testing data. 
    # this is specially important given that the original feature matrix X 
    # may have been subsampled in the preprocessing step!
    X = pd.concat([X_train, X_test]).copy()
    
    if X.shape[1] != 2:
        
        error_str = "I can only plot 2-dimensional data!"
        error_str += f"\nData provided is {X.shape[1]}-dimensional!"
        error_str += "\nPlease perform dimensionality reduction, or try a different dataset."
        
        raise ValueError(error_str)
    
    ##########################################
    # mesh
    
    if not border:
        # get an additional border of 10% of most extreme value
        border = max(X.iloc[:, 0].abs().max(), X.iloc[:, 1].abs().max())*0.1
    
    x_min, x_max = X.iloc[:, 0].min() - border, X.iloc[:, 0].max() + border
    y_min, y_max = X.iloc[:, 1].min() - border, X.iloc[:, 1].max() + border
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, spacing),
                         np.arange(y_min, y_max, spacing))
    
    ##########################################
    # predictions for mesh
    
    if quantum_kernel:
        
        gram_grid = construct_quantum_gram(reps, paulis, entanglement, np.c_[xx.ravel(), yy.ravel()], X_train, shots, plot_stuff=False)
        
        Z = model.predict(gram_grid)
        Z = Z.reshape(xx.shape)
    
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
    
    ##########################################
    # plot
    
    fig = plt.figure(figsize=(10, 8))
    
    if quantum_kernel:
        plt.title("SVM with quantum kernel - " + title, fontsize=20)  
    else:
        plt.title("Classical model - " + title, fontsize=20)
    
    plt.contourf(xx, yy, Z, cmap="viridis", alpha=0.4)

    # training points
    plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1],
                c=y_train, s=20, edgecolor="black", cmap="viridis", 
                label="training data")
    
    # test points
    plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1],
                c=y_test, s=120, edgecolor="black", cmap="viridis", marker="*",
                label="test data")
    
    
    plt.legend()
    
    ax = plt.gca()
    legend = ax.get_legend()
    legend.legendHandles[0].set_color("black")
    legend.legendHandles[1].set_color("black")
    
    plt.show()
    
#######################################################################################
#######################################################################################

def plot_part(gram, use_mask=True):
    '''
    this plots a given matrix. it'll be useful to plot the quadrants of the gram matrix
    '''

    plt.figure(figsize=(4, 4))
       
    mask = np.triu(np.ones_like(gram, dtype=bool)) if use_mask else None
    
    sns.heatmap(gram, vmin=0, vmax=1, square=True, mask=mask)
        
    plt.show()
    
#######################################################################################
#######################################################################################

def plot_gram(gram):
    '''
    this plots only the lower diagonal of the training gram matrix (which is symmetric)
    '''

    plt.figure(figsize=(4, 4))
        
    mask = np.triu(np.ones_like(gram, dtype=bool))
    sns.heatmap(gram, vmin=0, vmax=1, square=True, mask=mask)
        
    plt.show()
    
    
#######################################################################################
#######################################################################################    
 
def plot_train_test_gram(g_train, g_test):
    '''
    this plots both training and test gram matrices
    '''
    
    fig, axs = plt.subplots(1, 2, figsize=(8, 5))

    mask = np.triu(np.ones_like(g_train, dtype=bool))
    sns.heatmap(g_train, vmin=0, vmax=1, square=True, mask=mask, ax=axs[0])
    axs[0].set_title("Training quantum kernel matrix")

    sns.heatmap(g_test, vmin=0, vmax=1, square=True, ax=axs[1])
    axs[1].set_title("Test quantum kernel matrix")

    plt.tight_layout()
    plt.show()

#######################################################################################
#######################################################################################

def visualize_all(feat_map, data_point, bloch=True, state_city=True,
                  state_hinton=True, qsphere=True):
    '''
    we do not plot the state_city visualization if the number of qubits 
    (i.e. underlying feature dimension) is higher than 4
    '''
    feat_dim = len(data_point)
    
    quantum_circuit = feat_map.assign_parameters(data_point)
    show_figure(quantum_circuit.decompose().draw("mpl"))

    outputstate = Statevector.from_label("0" * feat_dim).evolve(quantum_circuit)

    print(f"\nState:\t{outputstate.data}\n")

    if bloch:
        show_figure(plot_bloch_multivector(outputstate))

    if state_city and (feat_dim <= 4):
        show_figure(plot_state_city(outputstate))
    
    if state_hinton:
        show_figure(plot_state_hinton(outputstate))

    if qsphere:
        show_figure(plot_state_qsphere(outputstate))

#######################################################################################
#######################################################################################

def visualize_feature_mapping(reps, paulis, entanglement, X, idx):
    '''
    visualize the feature mapping of the "idx"-th observation of "X"
    '''
    
    feature_dim = X.shape[1]
    
    datapoint = X.iloc[idx].tolist()
    
    feature_map = PauliFeatureMap(feature_dimension=feature_dim,
                                  reps=reps,
                                  paulis=paulis,
                                  entanglement=entanglement)
    
    print(f"\nFeature mapping of {idx}-th observation:")
    print(f"\nDatapoint: {np.array(datapoint)}")
    
    visualize_all(feature_map, datapoint)

#######################################################################################
#######################################################################################

# ================================================================================================= #
# ================================================================================================= #
# ========================== GRID/RANDOM SEARCH UTILS
# ================================================================================================= #
# ================================================================================================= #

def cart_product_string(my_string, feature_dim):
    '''
    this function returns a list of all cartesian products of characters of "my_string",
    of maximum length equal to "feature_dim", and each one joined in a string.
    for instance, if my_string = "XYZ", and feature_dim = 2, the function returns the following list:
    ['X', 'Y', 'Z', 'XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ']
    obs.: here, the order matters. So, "YX" is NOT equivalent to "XY", so we must consider both cases.
    this is why we use the cartesian product.
    '''
    
    perms = []
    
    for i in range(1, feature_dim+1):
        
        for p in product(my_string, repeat=i):
            
            temp = "".join(p)
            
            if len(temp) > 0:
                
                perms.append(temp)
                
    return perms

#######################################################################################
#######################################################################################

def combs_sub_lists(my_list, at_most=2):
    '''
    this functions returns a list of lists, each one containing a "at_most" strings,
    where the pairs are obtained from the input "my_list", which is supposed to be a list of strings.
    notice that each list will be a combination of "at most" strings.
    and notice that the order of elements in each sublist is important. That is,
    ["X", "Y"] is NOT equivalent to ["Y", "X"], so both are considered.
    (this is why we use "product" here too).
    '''
            
    subs = []

    for i in range(1, at_most+1):

        temp = [list(x) for x in product(my_list, repeat=i)]

        # combinations of "at most" strings
        if len(temp) > 0:

            subs.extend(temp)
   
    return subs
    
#######################################################################################
#######################################################################################    
    
def calc_param_grid(params, feature_dim, random_search=False, n_random=100):
    '''
    this function creates the grid of parameters to be tested in the grid/random search
    important redundancies are removed in order to reduce the number of combinations to be tested
    the parameter "random_search" determines whetre or not random search is to be used.
    if True, it samples "n_random" combinations from the full parameter grid;
    if False, the full parameter grid is used.
    '''
    
    # combination of parameters
    params_list = list(product(params["reps_list"],
                               params["paulis_list"],
                               params["entanglement_list"]))

    print(f"Initial number of combinations: {len(params_list)}")
    
    if feature_dim < 3:
        
        # for 2 qubits, all entanglement strategies yield the same result, naturally
        # so, we set all entanglements to "Linear"
        aux = [(item[0], item[1], "linear") for item in params_list]

        # the step above will yield lots of repeated entries
        # for instance, the elements (1, ['X'], 'linear'), (1, ['X'], 'circular'), (1, ['X'], 'full') in "params_list"
        # will be reduced to the element (1, ['X'], 'linear') in "aux", repeated 3 times.
        # thus, i'll drop here these repeated elements, whilst maintaining the order (hence the use of "dict.fromkeys()")
        # notice that I had to turn the list ("paulis_list", index 1) into tuples,
        # in order for the list "aux" to be hashable
        aux2 = list(dict.fromkeys([(item[0], tuple(item[1]), item[2]) 
                                   for item in aux]))

        # here, go back to lists in index 1
        aux3 = [(item[0], list(item[1]), item[2]) for item in aux2]
        
        params_list = aux3.copy()
        
    else:
        
        # when only single-qubit paulis are considered ("X", "Y" or "Z"), entanglement estrategies have no action
        # (to see this, look at feature_map.draw("mpl") in these cases!)
        # therefore, the param grid can be reduced quite a lot, by considering only "linear" entanglemet in such cases
        # (eg, for "paulis_list" such as ["X", "Y"], ["Z"], etc.)
        # this first auxiliar list does that, by replacing the entanglement by "linear" in all these cases
        aux = [item if any(len(pauli_str) > 1 for pauli_str in item[1])
               else (item[0], item[1], "linear")
               for item in params_list]

        # natutrally, "aux" will have many repeated elements, since, for instance,
        # the elements (1, ['X'], 'linear'), (1, ['X'], 'circular'), (1, ['X'], 'full') in "params_list"
        # will be reduced to the element (1, ['X'], 'linear') in "aux", repeated 3 times.
        aux2 = list(dict.fromkeys([(item[0], tuple(item[1]), item[2]) 
                                   for item in aux]))

        # here, go back to lists in index 1
        aux3 = [(item[0], list(item[1]), item[2]) for item in aux2]

        params_list = aux3.copy()

    print(f"Number of combinations after dropping redundancies: {len(params_list)}")
    
    # random search
    if random_search:
        
        params_list = sample(params_list, n_random)
        
        print(f"Number of random combinations chosen: {len(params_list)}")

    # dictionary as param grid, of the form {"parameter" : [parameter_values]}, 
    # for all combinations in params_list
    # this makes the iteration through the combinations more natural and more flexible
    param_grid = {key.replace("_list", "") : [item[i] for item in params_list] for i, key in enumerate(params.keys())}

    # let's iterate by parts -- it will take a looong time, so iterating the whole thing at once
    # may be problematic, since any error in the middle would ruin the whole thing
    n_combs = len(params_list)
    n = 10
    n_pieces = int(n_combs/n)

    # list of pieces to be iterated, in the form of range objects
    range_list = [range(k*n_pieces, (k+1)*n_pieces) for k in range(n-1)] + [range((n-1)*n_pieces, n_combs)]

    del aux, aux2, aux3, params_list
        
    return param_grid, range_list

#######################################################################################
#######################################################################################

# ================================================================================================= #
# ================================================================================================= #
# ========================== QUANTUM STUFF
# ================================================================================================= #
# ================================================================================================= #

def construct_quantum_gram(reps, paulis, entanglement, X1, X2=None, shots=1024, plot_stuff=True, 
                           hardware=False, hub=None, backend=None):
    '''
    this function constructs and returns a quantum gram matrix, using the qasm simulator
    with "shots" number shots.
    it also allows to send job to actual hardware, instead of simulating 
    must specify hub and backend, and "hardware"=True
    (beware: it can take a loong time depending on the hardware acess!!)
    '''
    
    feature_dim = X1.shape[1]
    
    feature_map = PauliFeatureMap(feature_dimension=feature_dim,
                                  reps=reps,
                                  paulis=paulis,
                                  entanglement=entanglement)

    if plot_stuff:
        
        print("\nFeature map:")
        show_figure(feature_map.decompose().draw("mpl"))

    #____________________________________

    
    if hardware:
        provider = IBMQ.load_account()
        accountProvider = IBMQ.get_provider(hub=hub)
        backend = provider.get_backend(backend)
        
    else:
        backend = Aer.get_backend("qasm_simulator")

    quantum_instance = QuantumInstance(backend,
                                       shots=shots,
                                       seed_simulator=seed, seed_transpiler=seed)
    
    q_kernel = QuantumKernel(feature_map=feature_map, 
                             quantum_instance=quantum_instance)

 
    # for some quite weird reason, for relatively large number of observations,
    # the kernel matrix has entries with imaginary parts 
    # (although the iomaginary part is quiiteeee small, ~e-46, which is virtually zero)
    # whis is why i'm getting only the real part
    q_gram = q_kernel.evaluate(x_vec=X1, y_vec=X2)
    
    # checking if indeed imaginary part is virtually zero (still quite weird tho)
    if np.abs(q_gram.imag).max() > 1e-20:
        
        assert ValueError("Imaginary part of kerel not virtually zero!\n")
    
    # if pass check above, return real part
    #---------------------------------
    # TO DO: review source code. Open issue, if necessary.
    #---------------------------------
    q_gram = q_gram.real.copy()

    # another quite weird thing: there are some values in the kernel matrix which are greater than 1
    # (I haven't seen, but it's possible that there are some smaller than 0 also..?)
    # so, in order to garantee a fair comparison among different kernels, I'll normalize
    # the entris between 0 and 1
    #---------------------------------
    # TO DO: review source code. Open issue, if necessary.
    #---------------------------------
    minmax = MinMaxScaler()
    q_gram = minmax.fit_transform(q_gram)
    
    return q_gram
    
#######################################################################################
#######################################################################################

def qsvm(X_train, y_train, X_test, y_test, reps, paulis, entanglement, shots=1024, plot_stuff=True,
         plot_boundary = False, title="decision boundary", border=None, spacing=0.05, quantum_kernel=True):
    '''
    this function computes the training and test gram matrices, and train a classical SVM classifier
    with these pre-computed kernel matrices. 
    evaluation metrics are also shown by default.
    '''
    
    q_gram_train = construct_quantum_gram(reps, paulis, entanglement, X_train, None, shots, plot_stuff)
    
    # for test gram matrix, X_test must be in "x_vec" and X_train in "y_vec" in the .evaluate()
    # i.e., we must have test observations in ROWS; and training observations in COLUMNS.
    # this is why I inverted here ;)
    q_gram_test = construct_quantum_gram(reps, paulis, entanglement, X_test, X_train, shots, plot_stuff=False)
    
    #__________________________
    
    if plot_stuff:
            
        print("\nQuantum kernel matrices:\n")
        plot_train_test_gram(q_gram_train, q_gram_test)
            
    #__________________________
    
    svm = SVC(kernel="precomputed").fit(q_gram_train, y_train)
    
    print("\nSVM with quantum kernel trained!\n")
    
    y_pred = svm.predict(q_gram_test)
    cr = classification_report(y_test, y_pred, output_dict=True)
    
    if plot_stuff:
        
        print("\nTest set performance:")
          
        plot_confusion_matrix(svm, q_gram_test, y_test)
        plt.show()

        print(classification_report(y_test, y_pred))
        
    if plot_boundary:
        
        plot_decision_boundary(svm, X_train, y_train, X_test, y_test,
                               title, border, spacing, quantum_kernel, reps, paulis, entanglement,
                               shots)

    return q_gram_train, cr
    
#######################################################################################
#######################################################################################

# ================================================================================================= #
# ================================================================================================= #
# ========================== KERNEL QUALITY
# ================================================================================================= #
# ================================================================================================= #

def get_gram_blocks(train_quantum_kernel, train_target, only_plot_stuff=False):
    
    classes_counts = train_target.value_counts().sort_index()

    # indices used to mark the blocks
    idx_c0 = classes_counts.iloc[0]
    idx_c1 = classes_counts.iloc[1]

    # matrices bloks: same class (c00 and c11) and different classes (c10)
    c00 = train_quantum_kernel[:idx_c0, :idx_c0]
    c10 = train_quantum_kernel[idx_c0:idx_c0+idx_c1, :idx_c0]
    c11 = train_quantum_kernel[idx_c1:, idx_c1:]
    
    if only_plot_stuff:
        plot_part(c00)
        plot_part(c10, use_mask=False)
        plot_part(c11)
    else:
        return c00, c10, c11

#######################################################################################
#######################################################################################

def gram_eval(train_quantum_kernel, train_target):
    
    b1, b2, b3 = get_gram_blocks(train_quantum_kernel, train_target)
    
    density_b1 = b1.mean()
    density_b2 = b2.mean()
    density_b3 = b3.mean()
    
    density_diff = 0.5 * (density_b1 + density_b3) - density_b2
    print(f"\nmetric 1: {density_diff}")
    
    print(f"metric(s) 2: (00, 10, 11) = {density_b1, density_b2, density_b3}")
    
    kernel_det = np.linalg.det(train_quantum_kernel)
    print(f"metric 3: {kernel_det}")
    
    return density_diff, (density_b1, density_b2, density_b3)


#######################################################################################
#######################################################################################


def run_kernel_evaluation(X_train, y_train, X_test, y_test,
                          param_grid, range_list, j, 
                          shots=1024, 
                          plot_stuff=True, model=False):
    '''
    this functions run the grid/random search for constructing feature maps and evaluating
    the resulting quantum kernels
    '''
    
    # dict of results
    results = {"reps" : [],
               "paulis" : [],
               "entanglement" : [],
               "density_diff" : [],
               "densities" : [],
               "pauli_quantity": [],
               "pauli_diversity": []}
    
    if model:
        results["precision"] = []
        results["recall"] = []
        results["f1-score"] = []
         
    n_combs = len([x for range_ in range_list for x in range_])
    
    feature_dim = X_train.shape[1]
    
    k=0
    n_range = len(range_list[j])
    
    for i in range_list[j]:
        
        start = time.time()
        
        #____________________________________
        
        reps = param_grid["reps"][i]
        paulis = param_grid["paulis"][i]
        entanglement = param_grid["entanglement"][i]

        joint_pauli_str = "".join(paulis)
        num_paulis = len(joint_pauli_str)
        num_unique_paulis = len(set(joint_pauli_str))
        
        print("\n")
        print("="*50)
        print(f"Combination {i+1}/{n_combs}")
        print(f"(Current range: {k+1}/{n_range})")
        print("="*50)
        print(f"\t\t\t reps : {reps}")
        print(f"\t\t\t paulis : {paulis}")
        print(f"\t\t\t entanglement : {entanglement}")
        print("="*50)
        
        #____________________________________
        
        # this will take longer because of the calculation of the test kernel matrix!
        if model:
          
            q_gram_train, cr = qsvm(X_train, y_train, X_test, y_test, reps, paulis, entanglement, shots,
                                    plot_stuff, quantum_kernel=False)
   
        else:
          
            q_gram_train = construct_quantum_gram(reps, paulis, entanglement, X_train, None, shots, plot_stuff)

            if plot_stuff:

                print("\nQuantum kernel matrix:")
                plot_gram(q_gram_train)
        
        #____________________________________
        
        # evaluation
        density_diff, densities = gram_eval(q_gram_train, y_train)
        
        #____________________________________
        
        stop = time.time()
        duration = time.strftime("%H:%M:%S", time.gmtime(stop-start))
        
        print(f"\n\n----> Kernel construction duration: {duration}")
        
        k += 1
        
        #____________________________________
        
        # update dict of results
        results["reps"].append(reps)
        results["paulis"].append(paulis)
        results["entanglement"].append(entanglement)
        results["density_diff"].append(density_diff)
        results["densities"].append(densities)
        results["pauli_quantity"].append(num_paulis)
        results["pauli_diversity"].append(num_unique_paulis)
        
        if model:
            results["precision"].append(cr["weighted avg"]["precision"])
            results["recall"].append(cr["weighted avg"]["recall"])
            results["f1-score"].append(cr["weighted avg"]["f1-score"])
        
    return pd.DataFrame(results)

#######################################################################################
#######################################################################################