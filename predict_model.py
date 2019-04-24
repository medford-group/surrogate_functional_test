# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 16:12:24 2017

@author: ray
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 13:56:12 2017

@author: ray
"""

import os
import itertools
import h5py
import json
import sys
import csv
import numpy as np
from numpy import mean, sqrt, square, arange
try: import cPickle as pickle
except: import pickle
import time
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation
from keras import backend as K
import matplotlib.pyplot as plt
import pprint
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline



def convert_formation_energies(energy_dict,atomic_references,composition_dict):
    """
    Convert dictionary of energies, atomic references and compositions into a dictionary of formation energies
    :param energy_dict: Dictionary of energies for all species.
                        Keys should be species names and values
                        should be energies.
                        
    :type energy_dict: dict
    :param atomic_references: Dictionary of atomic reference compositions (e.g. {H2O:{H:2,O:2}})
    :type atomic_references: dict
    :param composition_dict: Dictionary of compositions
    :type composition_dict: dict
    .. todo:: Explain the keys and values for energy_dict, atomic_references, and composition_dict
    """
    n = len(atomic_references)
    R = np.zeros((n,n))
    e = []
    ref_offsets = {}
    atoms = sorted(atomic_references)
    print atoms
    for i,a in enumerate(atoms):
        composition = composition_dict[atomic_references[a]]
        e.append(energy_dict[atomic_references[a]])
        for j,a in enumerate(atoms):
            n_a = composition.get(a,0)
            R[i,j] = n_a
    if not np.prod([R[i,i] for i in range(0,n)]):
        raise ValueError('Reference set is not valid.')
    e1 = []
    for i in range(len(e)):
        e1.append(e[i][0])
    e1 = np.array(e1)
    try:
        R_inv = np.linalg.solve(R,np.eye(n))
    except np.linalg.linalg.LinAlgError:
        raise ValueError('Reference set is not valid.')
    x = list(np.dot(R_inv,e1))
    for i,a in enumerate(atoms):
        ref_offsets[a] = x[i]

    return ref_offsets

def get_formation_energies(energy_dict,ref_dict,composition_dict):
    formation_energies = {}
    for molecule in energy_dict:
        E = energy_dict[molecule]
        for atom in composition_dict[molecule]:
            E -= float(composition_dict[molecule][atom]) *ref_dict[atom]
        formation_energies[molecule] = E
    return formation_energies

def sae(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true))

def map_to_0_1(arr, maxx, minn):
    return np.divide(np.subtract(arr,minn),(maxx-minn))
    
def map_back_0_1(arr, maxx, minn):
    return np.add(np.multiply(arr,(maxx-minn)),minn)
    
def map_back_n1_1(arr, maxx, minn):
    temp = np.multiply(np.add(arr,1.),0.5)
    return np.add(np.multiply(temp,(maxx-minn)),minn)

def log(log_filename, text):
    with open(log_filename, "a") as myfile:
        myfile.write(text)
    return

def get_start_loss(log_filename):
    
    with open(log_filename, 'r') as f:
        for line in f:
            pass
        temp = line
    
    if temp.strip().startswith('updated'):
        return float(temp.split()[2]), temp.split()[9]
    else:
        raise ValueError

def optimization_constants(x):
    #C0I = x[0]
    #C1  = x[1]
    #CC1 = x[2]
    #CC2 = x[3]
    #IF2 = x[4]

    C1  = x[0]
    gamma = x[1]
    alpha1 = x[2]
    beta1 = x[3]
    beta2 = x[4]
    beta3 = x[5]
    beta4 = x[6]

    #return C0I, C1, CC1, CC2, IF2, gamma, alpha1, beta1, beta2, beta3, beta4
    return C1, gamma, alpha1, beta1, beta2, beta3, beta4

def G(rtrs, gamma, alpha1, beta1, beta2, beta3, beta4):
    Q0 = -2.0 * gamma * (1.0 + alpha1 * rtrs * rtrs)
    Q1 = 2.0 * gamma * rtrs * (beta1 +
                           rtrs * (beta2 +
                                   rtrs * (beta3 +
                                           rtrs * beta4)))
    G1 = Q0 * np.log(1.0 + 1.0 / Q1)
    return G1

def lda_x( n, x):
#    C0I, C1, CC1, CC2, IF2 = lda_constants()
    C1, gamma, alpha1, beta1, beta2, beta3, beta4 = optimization_constants(x)

    C0I = 0.238732414637843
    #C1 = -0.45816529328314287
    rs = (C0I / n) ** (1 / 3.)
    ex = C1 / rs
    return n*ex
    #e[:] += n * ex

def lda_c( n, x):
    #C0I, C1, CC1, CC2, IF2 = lda_constants()
    C1, gamma, alpha1, beta1, beta2, beta3, beta4 = optimization_constants(x)

    C0I = 0.238732414637843
    #C1 = -0.45816529328314287
    rs = (C0I / n) ** (1 / 3.)
    ec = G(rs ** 0.5, gamma, alpha1, beta1, beta2, beta3, beta4)
    return n*ec
    #e[:] += n * ec

def predict_LDA(n,LDA_x):

    n = np.asarray(n)

    return lda_x(n,LDA_x) + lda_c(n,LDA_x)



def predict_each_block(setup,dens,X,y):

    NN_model = setup["NN_model"]
    LDA_model = setup["LDA_model"]
    y_transform  = setup["dataset_setup"]["target_transform"]

    original_y = detransform_data(y, y_transform)

    raw_predict_y = predict_LDA(dens,LDA_model.x) + (NN_model.predict(X*1e6)/1e6)
    predict_y = detransform_data(raw_predict_y, y_transform)

    return original_y, predict_y


def detransform_data(temp_data, transform):
    if transform == "real":
        return temp_data
    elif transform == "log10":
        return np.power(10.,temp_data)
    elif transform == "neglog10":
        return np.multiply(-1.,np.power(10.,temp_data))

def transform_data(temp_data, transform):
    if transform == "real":
        return temp_data.flatten().tolist()
    elif transform == "log10":
        return np.log10(temp_data.flatten()).tolist()
    elif transform == "neglog10":
        return np.log10(np.multiply(-1., temp_data.flatten())).tolist()
    
def load_data_each_block(molecule,functional,i,j,k, dataset_setup, data_dir_full):
    data_filename = "{}_{}_{}_{}_{}_all_descriptors.hdf5".format(molecule,functional,i,j,k)

    os.chdir(data_dir_full)

    print data_dir_full
    print os.getcwd()

    data =  h5py.File(data_filename,'r')
    

    result_list = []
    y = []
    dens = []
    
    target = dataset_setup['target']
    
    if target == 'Vxc':
        target_set_name = 'V_xc'   
    if target == 'epxc':
        target_set_name = 'epsilon_xc'
    if target == 'tau':
        target_set_name = 'tau'
    if target == 'gamma':
        target_set_name = 'gamma'
    
    temp_data = np.asarray(data[target_set_name])
    y.append(transform_data(temp_data, dataset_setup['target_transform']))
    
    if int(dataset_setup['density']) == 1:
        temp_data = np.asarray(data['rho'])
        result_list.append(transform_data(temp_data, dataset_setup['density_transform']))
        dens.append(transform_data(temp_data, dataset_setup['density_transform']))

    if int(dataset_setup['gamma']) == 1:
        temp_data = np.asarray(data['gamma'])
        result_list.append(transform_data(temp_data, dataset_setup['gamma_transform']))

    if int(dataset_setup['tau']) == 1:
        temp_data = np.asarray(data['tau'])
        result_list.append(transform_data(temp_data, dataset_setup['tau_transform']))


    group_name = 'derivative'
    temp_list = dataset_setup["derivative_list"]
    if len(temp_list) > 0: 
        for derivative_count in temp_list:
            dataset_name = 'derivative_{}'.format(derivative_count)
            temp_data = np.asarray(data[group_name][dataset_name])
            result_list.append(transform_data(temp_data, dataset_setup['derivative_transform']))

    temp_list = dataset_setup["derivative_square_list"]
    if len(temp_list) > 0: 
        for derivative_count in temp_list:
            dataset_name = 'derivative_{}'.format(derivative_count)
            temp_data = np.power(np.asarray(data[group_name][dataset_name]), 2.)
            result_list.append(transform_data(temp_data, dataset_setup['derivative_square_transform']))
    
   
    group_name = 'average_density'
    temp_list = dataset_setup["average_density_r_list"]
    if len(temp_list) > 0:
        for r_list_count in temp_list:
            dataset_name = 'average_density_{}'.format(str(r_list_count).replace('.','-'))
            temp_data = np.asarray(data[group_name][dataset_name])
            result_list.append(transform_data(temp_data, dataset_setup['average_density_transform']))


    group_name = 'asym_integral'
    temp_list = dataset_setup["asym_desc_r_list"]
    if len(temp_list) > 0:
        for r_list_count in temp_list:
            dataset_name = 'asym_integral_x_{}'.format(str(r_list_count).replace('.','-'))
            temp_data = np.asarray(data[group_name][dataset_name])
            result_list.append(transform_data(temp_data, dataset_setup['asym_desc_transform']))

            dataset_name = 'asym_integral_y_{}'.format(str(r_list_count).replace('.','-'))
            temp_data = np.asarray(data[group_name][dataset_name])
            result_list.append(transform_data(temp_data, dataset_setup['asym_desc_transform']))

            dataset_name = 'asym_integral_z_{}'.format(str(r_list_count).replace('.','-'))
            temp_data = np.asarray(data[group_name][dataset_name])
            result_list.append(transform_data(temp_data, dataset_setup['asym_desc_transform']))

    
    group_name = 'MCSH'

    try:
        temp_list = dataset_setup["MCSH_0_1_r_list"]
        if len(temp_list) > 0:
            for r_list_count in temp_list:
                dataset_name = 'MCSH_0_1_{}'.format(str(r_list_count).replace('.','-'))
                temp_data = np.asarray(data[group_name][dataset_name])
                result_list.append(transform_data(temp_data, dataset_setup['MCSH_0_1_transform']))
    except:
        pass

    try:
        temp_list = dataset_setup["MCSH_1_1_r_list"]
        if len(temp_list) > 0:
            for r_list_count in temp_list:
                dataset_name = 'MCSH_1_1_{}'.format(str(r_list_count).replace('.','-'))
                temp_data = np.asarray(data[group_name][dataset_name])
                result_list.append(transform_data(temp_data, dataset_setup['MCSH_1_1_transform']))
    except:
        pass

    
    try:
        temp_list = dataset_setup["MCSH_2_1_r_list"]
        if len(temp_list) > 0:
            for r_list_count in temp_list:
                dataset_name = 'MCSH_2_1_{}'.format(str(r_list_count).replace('.','-'))
                temp_data = np.asarray(data[group_name][dataset_name])
                result_list.append(transform_data(temp_data, dataset_setup['MCSH_2_1_transform']))
    except:
        pass

    try:
        temp_list = dataset_setup["MCSH_2_2_r_list"]
        if len(temp_list) > 0:
            for r_list_count in temp_list:
                dataset_name = 'MCSH_2_2_{}'.format(str(r_list_count).replace('.','-'))
                temp_data = np.asarray(data[group_name][dataset_name])
                result_list.append(transform_data(temp_data, dataset_setup['MCSH_2_2_transform']))
    except:
        pass




    result = zip(*result_list)
    y = zip(*y)
    dens = zip(*dens)
    print "done loading: {} {} {}".format(i,j,k)

    os.chdir(setup["model_save_dir"])
    return np.asarray(dens), np.asarray(result), np.asarray(y)


def process_each_block(molecule, i,j,k, setup, data_dir_full):
    print 'started'

    h = float(setup['grid_spacing'])
    L = float(setup['box_dimension'])
    N = int(setup['number_segment_per_side'])
    functional = setup['functional']
    #log_filename = setup["predict_full_log_name"]

    start = time.time()
    dens, X,y = load_data_each_block(molecule,functional,i,j,k, setup["dataset_setup"], data_dir_full)
    original_y, predict_y = predict_each_block(setup, dens, X, y)
    dv = h*h*h
    y = original_y * dv*27.2114
    y_predict = predict_y*dv*27.2114
    y_sum = np.sum(y)
    y_predict_sum = np.sum(y_predict)
    error = y-y_predict
    absolute_error = np.abs(error)
    
#    fraction_error = np.divide(error, y)
    sum_error = y_sum - y_predict_sum
    sum_absolute_error = np.sum(absolute_error)
    
    os.chdir(setup["model_save_dir"])
    log(setup["predict_full_log_name"],"\ndone predicting: " + molecule + "\t took time: " + str(time.time()-start)+ "\t" + str(i) + "\t" + str(j) + "\t" + str(k))
    log(setup["predict_full_log_name"],"\nenergy: " + str(y_sum) + "\tpredicted energy: " + str(y_predict_sum) + "\tpredict error: " + str(sum_error) + "\tabsolute error: " + str(sum_absolute_error))

    os.chdir(setup["model_save_dir"])

    return sum_error, sum_absolute_error,y_predict_sum, y_sum 

def process_one_molecule(molecule, setup):
 

    h = float(setup['grid_spacing'])
    L = float(setup['box_dimension'])
    N = int(setup['number_segment_per_side'])

    functional = setup['functional']  

#    log_filename = setup["predict_log_name"]
    
    data_dir_name = "{}_{}_{}_{}_{}".format(molecule,functional,str(L).replace('.','-'),str(h).replace('.','-'),N)


    data_dir_full = setup["sub_database"]  + "/" + data_dir_name

#    os.chdir(data_dir_full)

    Nx = Ny = Nz = N
    i_li = range(Nx)
    j_li = range(Ny)
    k_li = range(Nz)

    
    system_sum_error = 0 
    system_sum_absolute_error = 0
    system_y_predict_sum = 0
    system_y_sum = 0 
    
    paramlist = list(itertools.product(i_li,j_li,k_li))
    for i,j,k in paramlist:
        
        sum_error, sum_absolute_error,y_predict_sum, y_sum = process_each_block(molecule, i,j,k, setup, data_dir_full)
        system_sum_error += sum_error
        system_sum_absolute_error += sum_absolute_error
        system_y_predict_sum += y_predict_sum
        system_y_sum += y_sum


    os.chdir(setup["model_save_dir"])
    log(setup["predict_log_name"],"\n\ndone predicting: " + molecule )
    log(setup["predict_log_name"],"\nenergy: " + str(system_y_sum) + "\tpredicted energy: " + str(system_y_predict_sum) + "\tpredicted error: " + str(system_sum_error)+ "\n")

    log(setup["predict_full_log_name"],"\n\ndone predicting: " + molecule )
    log(setup["predict_full_log_name"],"\nenergy: " + str(system_y_sum) + "\tpredicted energy: " + str(system_y_predict_sum) + "\tpredicted error: " + str(system_sum_error)+ "\n") 
    
    log(setup["predict_error_log_name"], "\n{}\t{}\t{}\t{}\t{}".format(molecule, system_y_sum, system_y_predict_sum, system_sum_error,system_sum_absolute_error))
    return system_sum_error,system_sum_absolute_error, system_y_predict_sum,system_y_sum


def initialize(setup,NN_model_filename):
    os.chdir(setup["model_save_dir"])
    fit_log_name = "NN_fit_log.log"
    
    LDA_model_name = "LDA_model.sav"
    NN_model_name = NN_model_filename
    #NN_model_name = "NN.h5"


    predict_log_name = "predict_log.log"
    predict_full_log_name = "predic_full_log.log"
    predict_error_log_name = "predict_error_log.log"
    predict_formation_log_name = "predict_formation_log.log"
    try:
        NN_model = load_model(NN_model_name, custom_objects={'sae': sae})
    except:
        NN_model = load_model(NN_model_name)


    LDA_model = pickle.load(open(LDA_model_name, 'rb'))

    setup["NN_model"] = NN_model
    setup["LDA_model"] = LDA_model
    setup["predict_log_name"] = predict_log_name
    setup["predict_full_log_name"] = predict_full_log_name
    setup["predict_error_log_name"] = predict_error_log_name
    setup["predict_formation_log_name"] = predict_formation_log_name



    os.chdir(setup["working_dir"])
    return


if __name__ == "__main__":

    composition_dict = {'C4H6':{'C':4, 'H':6},
                        'C2H2':{'C':2, 'H':2},
                        'C2H4':{'C':2, 'H':4},
                        'C2H6':{'C':2, 'H':6},
                        'C3H4':{'C':3, 'H':4},
                        'C3H6':{'C':3, 'H':6},
                        'C3H8':{'C':3, 'H':8},
                        'CH2':{'C':1, 'H':2},
                        'CH2OCH2':{'C':2, 'H':4, 'O':1},
                        'CH3CH2OH':{'C':2, 'H':6, 'O':1},
                        'CH3CH2NH2':{'N':1, 'C':2, 'H':7},
                        'CH3CHO':{'C':2, 'H':4, 'O':1},
                        'CH3CN':{'N':1, 'C':2, 'H':3},
                        'CH3COOH':{'C':2, 'H':4, 'O':2},
                        'CH3NO2':{'N':1, 'C':1, 'H':3, 'O':2},
                        'CH3OCH3':{'C':2, 'H':6, 'O':1},
                        'CH3OH':{'C':1, 'H':4, 'O':1},
                        'CH4':{'C':1, 'H':4},
                        'CO2':{'C':1, 'O':2},
                        'CO':{'C':1, 'O':1},
                        'H2':{'H':2},
                        'H2CCO':{'C':2, 'H':2, 'O':1},
                        'H2CO':{'C':1, 'H':2, 'O':1},
                        'H2O2':{'H':2, 'O':2},
                        'H2O':{'H':2, 'O':1},
                        'H3CNH2':{'N':1, 'C':1, 'H':5},
                        'HCN':{'N':1, 'C':1, 'H':1},
                        'C4H8':{'C':4, 'H':8},
                        'HCOOH':{'C':1, 'H':2, 'O':2},
                        'HNC':{'N':1, 'C':1, 'H':1},
                        'N2':{'N':2},
                        'N2H4':{'N':2, 'H':4},
                        'N2O':{'N':2, 'O':1},
                        'NCCN':{'N':2, 'C':2},
                        'NH3':{'N':1, 'H':3},
                        'isobutene':{'C':4, 'H':8},
                        'glycine':{'N':1, 'C':2, 'H':5, 'O':2},
                        'C2H5CN':{'N':1, 'C':3, 'H':5},
                        'butadiene':{'C':4, 'H':6},
                        '1-butyne':{'C':4, 'H':6},
                        'CCH':{'C':2, 'H':1},
                        'propanenitrile':{'N':1, 'C':3, 'H':5},
                        'NO2':{'N':1, 'O':2},
                        'NH':{'N':1, 'H':1},
                        'pentadiene':{'C':5, 'H':8},
                        'cyclobutene':{'C':4, 'H':6},
                        'NO':{'N':1, 'O':1},
                        'OCHCHO':{'C':2, 'H':2, 'O':2},
                        'cyclobutane':{'C':4, 'H':8},
                        'propyne':{'C':3, 'H':4},
                        'CH3':{'C':1, 'H':3},
                        'NH2':{'N':1, 'H':2},
                        'CH3NHCH3':{'N':1, 'C':2, 'H':7},
                        'CH':{'C':1, 'H':1},
                        'CN':{'N':1, 'C':1},
                        'z-butene':{'C':4, 'H':8},
                        '1-butene':{'C':4, 'H':8},
                        'isobutane':{'C':4, 'H':10},
                        '2-propanamine':{'N':1, 'C':3, 'H':9},
                        'cyclopentane':{'C':5, 'H':10},                
                        'butane':{'C':4, 'H':10},
                        'HCO':{'C':1, 'H':1, 'O':1},
                        'CH3CONH2':{'N':1, 'C':2, 'H':5, 'O':1},
                        'e-butene':{'C':4, 'H':8},
                        'CH3O':{'C':1, 'H':3, 'O':1},
                        'propene':{'C':3, 'H':6},
                        'OH':{'H':1, 'O':1},
                        'methylenecyclopropane':{'C':4, 'H':6},
                        'C6H6':{'C':6, 'H':6},
                        'trimethylamine':{'N':1, 'C':3, 'H':9},
                        'cyclopropane':{'C':3, 'H':6},
                        'H2CCHCN':{'N':1, 'C':3, 'H':3},
                        '1-pentene':{'C':5, 'H':10},
                        '2-butyne':{'C':4, 'H':6},
                        'O3':{'O':3}}



    print "start"
    predict_setup_filename = sys.argv[1]
    dataset_setup_database_filename = sys.argv[2]
    dataset_name = sys.argv[3]
    #functional = sys.argv[4]
    functional = "B3LYP"
    NN_model_filename = "NN.h5"

    K.set_floatx('float64')
    K.floatx()


    with open(predict_setup_filename) as f:
        setup = json.load(f)


    h = float(setup['grid_spacing'])
    L = float(setup['box_dimension'])
    N = int(setup['number_segment_per_side'])

    setup['functional'] = functional
#    functional = setup['functional']

    with open(dataset_setup_database_filename) as f:
        setup_database = json.load(f)

    dataset_setup = setup_database[dataset_name]

    setup["dataset_setup"] = dataset_setup

    working_dir = os.getcwd()  + '/models/' + dataset_name
    setup["working_dir"] = working_dir

    model_save_dir = working_dir + "/" + "NN_{}_{}_{}_{}".format(setup["fit_type"],setup["NN_setup"]["number_neuron_per_layer"], setup["NN_setup"]["number_layers"], setup["NN_setup"]["activation"])
    setup["model_save_dir"] = model_save_dir


    database_name = setup["database_directory"]
    sub_database_name = "{}_{}_{}".format(str(L).replace('.','-'),str(h).replace('.','-'),N)

    setup["sub_database"] = database_name + '/' + sub_database_name

    initialize(setup,NN_model_filename)

    setup["result_data"] = {}

    
    error_list = []
    absolute_error_list = []

    for molecule in setup["molecule_list"]:
        setup["result_data"][molecule] = {}
        setup["result_data"][molecule]["exist"] = False

    

    os.chdir(setup["model_save_dir"])

    try:
        with open(setup["predict_error_log_name"],'rb') as f:
            for line in f:
                if line.strip() != '':
                    temp = line.strip().split()
                    temp_name = temp[0]
                    temp_original_energy = float(temp[1])
                    temp_predict_energy  = float(temp[2])
                    temp_error  = float(temp[3])
                    temp_absolute_error  = float(temp[4])
                    setup["result_data"][temp_name]['predict_exc'] = temp_predict_energy
                    setup["result_data"][temp_name]['original_exc'] = temp_original_energy
                    setup["result_data"][temp_name]['sum_error'] = temp_error
                    setup["result_data"][temp_name]['sum_absolute_error'] = temp_absolute_error
                    setup["result_data"][temp_name]["exist"] = True
                    error_list.append(temp_error)
                    absolute_error_list.append(temp_absolute_error)

    except:
        pass



    print setup["result_data"]

    
    for molecule in setup["molecule_list"]:
        if setup["result_data"][molecule]["exist"] == False:
            #try:
                
            temp_error,temp_absolute_error, temp_y_predict,temp_y = process_one_molecule(molecule, setup)
            error_list.append(temp_error)
            absolute_error_list.append(temp_absolute_error)
            #setup["result_data"][molecule] = {}
            setup["result_data"][molecule]['predict_exc'] = temp_y_predict
            setup["result_data"][molecule]['original_exc'] = temp_y
            setup["result_data"][molecule]['sum_error'] = temp_error
            setup["result_data"][molecule]['sum_absolute_error'] = temp_absolute_error
            #except:
            #    log(setup["predict_log_name"],"\n\n Failed")
            #    log(setup["predict_full_log_name"],"\n\n Failed") 


    log(setup["predict_log_name"],"\n\naverage error: " + str(np.mean(error_list)) + "\tstddev error: " + str(np.std(error_list))) 
    log(setup["predict_log_name"],"\n\naverage abs error: " + str(np.mean(np.abs(error_list))) + "\tstddev abs error: " + str(np.std(np.abs(error_list))))
    log(setup["predict_log_name"],"\n\naverage abs error: " + str(sqrt(mean(square(error_list))))) 

    log(setup["predict_full_log_name"],"\n\naverage error: " + str(np.mean(error_list)) + "\tstddev error: " + str(np.std(error_list))) 
    log(setup["predict_full_log_name"],"\n\naverage abs error: " + str(np.mean(np.abs(error_list))) + "\tstddev abs error: " + str(np.std(np.abs(error_list))))
    log(setup["predict_full_log_name"],"\n\naverage abs error: " + str(sqrt(mean(square(error_list))))) 



    for molecule in setup["result_data"]:
        if 'composition' not in setup["result_data"][molecule]:
            if molecule in composition_dict:
                setup["result_data"][molecule]['composition'] = composition_dict[molecule]

    original_energy_dict = {}
    predict_energy_dict = {}
    composition_dict = {}
    for molecule in setup["result_data"]:
        if 'composition' in setup["result_data"][molecule] and 'predict_exc' in setup["result_data"][molecule] and 'original_exc' in setup["result_data"][molecule]:
            print molecule
            original_energy_dict[molecule] = setup["result_data"][molecule]['original_exc']

                                    
            predict_energy_dict[molecule] = setup["result_data"][molecule]['predict_exc']

            composition_dict[molecule] = setup["result_data"][molecule]['composition']


    #atomic_references = {'O':'CH3CH2OH','H':'C2H2','C':'C2H6'}
    atomic_references = {'N':'NH3','O':'H2O','H':'H2','C':'CH4'}
    compound_original_en_dict = {}
    compound_predict_en_dict = {}

    for key in original_energy_dict:

        compound_original_en_dict[key]     = [original_energy_dict[key]]
        compound_predict_en_dict[key]      = [predict_energy_dict[key]]



    ref_offset_original_en       = convert_formation_energies(compound_original_en_dict.copy(),atomic_references,composition_dict)
    ref_offset_predict_en        = convert_formation_energies(compound_predict_en_dict.copy(),atomic_references,composition_dict)



    formation_energies_original_en       = get_formation_energies(compound_original_en_dict.copy(),ref_offset_original_en.copy(),composition_dict)
    formation_energies_predict_en        = get_formation_energies(compound_predict_en_dict.copy(),ref_offset_predict_en.copy(),composition_dict)
    #formation_energies_compare_en   = get_formation_energies(compare_energy_dict.copy(),ref_offset_compare_en.copy(),composition_dict)


    with open(setup["predict_formation_log_name"], "wb") as f:
        #writer = csv.writer(f)
        writer = csv.writer(f, delimiter='\t')
        #for key in formation_energies_original_en.keys():
        for key in ["C2H2", "C2H4", "C2H6", "CH3OH", "CH4", "CO", "CO2", "H2", "H2O", "HCN", "HNC", "N2", "N2O", "NH3", "O3", "NCCN", "N2H4", "HCOOH", "H2O2", "H2CO", "H2CCO", "glycine", "CH3CHO", "CH3CN", "CH3NO2"]:
            temp = [key,compound_original_en_dict[key][0],compound_predict_en_dict[key][0],(compound_original_en_dict[key][0]-compound_predict_en_dict[key][0]),formation_energies_original_en[key][0],formation_energies_predict_en[key][0],(formation_energies_original_en[key][0]-formation_energies_predict_en[key][0]), setup["result_data"][key]['sum_absolute_error']]
            writer.writerow(temp)


    print '{:10}\t{}\t{}'.format('name', 'form. E. 1', 'form. E. 2')
    print '--------- 1: predicted xc energy  2: psi4 xc energy projected on fd-grid\n'
    #for key in formation_energies_original_en.keys():
    for key in ["C2H2", "C2H4", "C2H6", "CH3OH", "CH4", "CO", "CO2", "H2", "H2O", "HCN", "HNC", "N2", "N2O", "NH3", "O3", "NCCN", "N2H4", "HCOOH", "H2O2", "H2CO", "H2CCO", "glycine", "CH3CHO", "CH3CN", "CH3NO2"]:
        #print '{:10}\t{:8.5f}\t{:8.5f}'.format(key,formation_energies_original_en[key],formation_energies_predict_en[key])
        print '{}\t{}\t{}\t{}\t{}\t{}'.format(key,compound_original_en_dict[key][0],compound_predict_en_dict[key][0],formation_energies_original_en[key][0],formation_energies_predict_en[key][0], setup["result_data"][key]['sum_absolute_error'])

