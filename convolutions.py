# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:48:37 2017

@author: ray
"""

import numpy as np
from scipy.ndimage.filters import convolve
from scipy.signal import fftconvolve
from math import pi
import math
import os
try: import cPickle as pickle
except: import pickle
import itertools

#import matplotlib
#matplotlib.use('Agg') 
#import matplotlib.cm as cm
#from mpl_toolkits.mplot3d import Axes3D
#import mpl_toolkits.mplot3d.axes3d as p3
#import matplotlib.pyplot as plt

'''
Differenciations

'''


def get_first_grad_stencil(hx, hy, hz, stencil_type = 'mid', accuracy = '2'):
    '''
    n: electron density, or whatever we need to convolve
    hx, hy, hz: grid spacing at each direction
    stencil type:   mid: only mid-row has elements
                    uniform: all rows have the same elements in each dimension
                    times2:  emphasize on the middle rows
    accuracy: degree of accuracy of the finite difference method uses
    '''
                       
    fd_coefficients = {}
                       
    fd_coefficients['2'] = {'coeff':    np.asarray([-1., 0., 1.])* -1., 
                            'mid':      np.asarray([0., 1., 0.]), 
                            'uniform':  np.asarray([1., 1., 1.]) * (1./3.),
                            'times2':   np.asarray([1., 2., 1.]) * (1./4.),
                            'pad':      1,
                            'norm_fac': 2.}
                
    fd_coefficients['4'] = {'coeff':    np.asarray([1., -8., 0., 8., -1.])* -1., 
                            'mid':      np.asarray([0., 0., 1., 0., 0.]), 
                            'uniform':  np.asarray([1., 1., 1., 1., 1.]) * (1./5.),
                            'times2':   np.asarray([1., 2., 4., 2., 1.]) * (1./10.),
                            'pad':      2,
                            'norm_fac': 12.}
                            
    fd_coefficients['6'] = {'coeff':    np.asarray([-1., 9., -45., 0., 45., -9., 1.])* -1., 
                            'mid':      np.asarray([0., 0., 0., 1., 0., 0., 0.]), 
                            'uniform':  np.asarray([1., 1., 1., 1., 1., 1., 1.]) * (1./7.),
                            'times2':   np.asarray([1., 2., 4., 8., 4., 2., 1.]) * (1./22.),
                            'pad':      3,
                            'norm_fac': 60.}
                            
    fd_coefficients['8'] = {'coeff':    np.asarray([3., -32., 168., -672., 0., 672., -168., 32., 3.])* -1., 
                            'mid':      np.asarray([0., 0., 0., 0., 1., 0., 0., 0., 0.]), 
                            'uniform':  np.asarray([1., 1., 1., 1., 1., 1., 1., 1., 1.]) * (1./9.),
                            'times2':   np.asarray([1., 2., 4., 8.,16., 8., 4., 2., 1.]) * (1./46.),
                            'pad':      4,
                            'norm_fac': 840.}
                            
    if accuracy not in list(fd_coefficients.keys()):
        raise NotImplementedError
    
    if stencil_type not in list(fd_coefficients[accuracy].keys()):
        raise NotImplementedError
    
    
    coefficient = np.asarray(fd_coefficients[accuracy]['coeff'])
    extend_mat  = np.asarray(fd_coefficients[accuracy][stencil_type])
    pad =  (fd_coefficients[accuracy]['pad'],)*3
    normalization_factor = fd_coefficients[accuracy]['norm_fac']
    temp_num = len(extend_mat)
    
    # get the stencils based on finite difference coefficients
    # transpose the stencil on x direction to get the other ones
    Gx_temp = np.reshape(extend_mat,(temp_num,1,1)) * ( np.reshape(extend_mat,(1,temp_num,1)) * coefficient)
    Gy_temp = Gx_temp.copy().transpose(0,2,1)
    Gz_temp = Gx_temp.copy().transpose(2,1,0)
    
    Gx = (1./(normalization_factor*hx)) * Gx_temp.copy()
    Gy = (1./(normalization_factor*hy)) * Gy_temp.copy()
    Gz = (1./(normalization_factor*hz)) * Gz_temp.copy()
    
    G = Gx + Gy + Gz
    
    return G, Gx, Gy, Gz, pad
    

def get_second_grad_stencil(hx, hy, hz, stencil_type = 'mid', accuracy = '2'):
                       
    fd_coefficients = {}
                       
    fd_coefficients['2'] = {'coeff':    np.asarray([1.,-2., 1.]), 
                            'mid':      np.asarray([0., 1., 0.]), 
                            'uniform':  np.asarray([1., 1., 1.]) * (1./3.),
                            'times2':   np.asarray([1., 2., 1.]) * (1./4.),
                            'pad':      1,
                            'norm_fac': 1.}
                
    fd_coefficients['4'] = {'coeff':    np.asarray([-1., 16., -30., 16., -1.]), 
                            'mid':      np.asarray([0., 0., 1., 0., 0.]), 
                            'uniform':  np.asarray([1., 1., 1., 1., 1.]) * (1./5.),
                            'times2':   np.asarray([1., 2., 4., 2., 1.]) * (1./10.),
                            'pad':      2,
                            'norm_fac': 12.}
                            
    fd_coefficients['6'] = {'coeff':    np.asarray([2., -27.,  270., -490., 270., -27., 2.]), 
                            'mid':      np.asarray([0., 0., 0., 1., 0., 0., 0.]), 
                            'uniform':  np.asarray([1., 1., 1., 1., 1., 1., 1.]) * (1./7.),
                            'times2':   np.asarray([1., 2., 4., 8., 4., 2., 1.]) * (1./22.),
                            'pad':      3,
                            'norm_fac': 180.}
                            
    fd_coefficients['8'] = {'coeff':    np.asarray([-9., 128., -1008., 8064., -14350., 8064., -1008., 128., -9.]), 
                            'mid':      np.asarray([0., 0., 0., 0., 1., 0., 0., 0., 0.]), 
                            'uniform':  np.asarray([1., 1., 1., 1., 1., 1., 1., 1., 1.]) * (1./9.),
                            'times2':   np.asarray([1., 2., 4., 8.,16., 8., 4., 2., 1.]) * (1./46.),
                            'pad':      4,
                            'norm_fac': 5040.}
                            
    if accuracy not in list(fd_coefficients.keys()):
        raise NotImplementedError
    
    if stencil_type not in list(fd_coefficients[accuracy].keys()):
        raise NotImplementedError
    
    
    coefficient = np.asarray(fd_coefficients[accuracy]['coeff'])
    extend_mat  = np.asarray(fd_coefficients[accuracy][stencil_type])
    pad =  (fd_coefficients[accuracy]['pad'],)*3
    normalization_factor = fd_coefficients[accuracy]['norm_fac']
    temp_num = len(extend_mat)
    
    Gx_temp = np.reshape(extend_mat,(temp_num,1,1)) * ( np.reshape(extend_mat,(1,temp_num,1)) * coefficient)
    Gy_temp = Gx_temp.copy().transpose(0,2,1)
    Gz_temp = Gx_temp.copy().transpose(2,1,0)
    
    Gx = (1./(normalization_factor*hx*hx)) * Gx_temp.copy()
    Gy = (1./(normalization_factor*hy*hy)) * Gy_temp.copy()
    Gz = (1./(normalization_factor*hz*hz)) * Gz_temp.copy()
#    print 'start conv'
    G = Gx + Gy + Gz
    
    return G, Gx, Gy, Gz, pad
    


def get_third_grad_stencil(hx, hy, hz, stencil_type = 'mid', accuracy = '2'):
                       
    fd_coefficients = {}
               
    fd_coefficients['2'] = {'coeff':    np.asarray([-1., 2., 0., -2., 1.])* -1., 
                            'mid':      np.asarray([0., 0., 1., 0., 0.]), 
                            'uniform':  np.asarray([1., 1., 1., 1., 1.]) * (1./5.),
                            'times2':   np.asarray([1., 2., 4., 2., 1.]) * (1./10.),
                            'pad':      2,
                            'norm_fac': 2.}
                            
    fd_coefficients['4'] = {'coeff':    np.asarray([1., -8., 13., 0., -13., 8., -1.])* -1., 
                            'mid':      np.asarray([0., 0., 0., 1., 0., 0., 0.]), 
                            'uniform':  np.asarray([1., 1., 1., 1., 1., 1., 1.]) * (1./7.),
                            'times2':   np.asarray([1., 2., 4., 8., 4., 2., 1.]) * (1./22.),
                            'pad':      3,
                            'norm_fac': 8.}
                            
    fd_coefficients['6'] = {'coeff':    np.asarray([-7., 72., -338., 488., 0., -488., 338., -72., 7.])* -1., 
                            'mid':      np.asarray([0., 0., 0., 0., 1., 0., 0., 0., 0.]), 
                            'uniform':  np.asarray([1., 1., 1., 1., 1., 1., 1., 1., 1.]) * (1./9.),
                            'times2':   np.asarray([1., 2., 4., 8.,16., 8., 4., 2., 1.]) * (1./46.),
                            'pad':      4,
                            'norm_fac': 240.}#434400.
                            
    if accuracy not in list(fd_coefficients.keys()):
        raise NotImplementedError
    
    if stencil_type not in list(fd_coefficients[accuracy].keys()):
        raise NotImplementedError
    
    
    coefficient = np.asarray(fd_coefficients[accuracy]['coeff'])
    extend_mat  = np.asarray(fd_coefficients[accuracy][stencil_type])
    pad =  (fd_coefficients[accuracy]['pad'],)*3
    normalization_factor = fd_coefficients[accuracy]['norm_fac']
    temp_num = len(extend_mat)
    
    Gx_temp = np.reshape(extend_mat,(temp_num,1,1)) * ( np.reshape(extend_mat,(1,temp_num,1)) * coefficient)
    Gy_temp = Gx_temp.copy().transpose(0,2,1)
    Gz_temp = Gx_temp.copy().transpose(2,1,0)
    
    Gx = (1./(normalization_factor*hx*hx*hx)) * Gx_temp.copy()
    Gy = (1./(normalization_factor*hy*hy*hy)) * Gy_temp.copy()
    Gz = (1./(normalization_factor*hz*hz*hz)) * Gz_temp.copy()
    
    G = Gx + Gy + Gz
    
    return G, Gx, Gy, Gz, pad   




def get_fourth_grad_stencil(hx, hy, hz, stencil_type = 'mid', accuracy = '2'):
                       
    fd_coefficients = {}
               
    fd_coefficients['2'] = {'coeff':    np.asarray([1., -4., 6., -4., 1.]), 
                            'mid':      np.asarray([0., 0., 1., 0., 0.]), 
                            'uniform':  np.asarray([1., 1., 1., 1., 1.]) * (1./5.),
                            'times2':   np.asarray([1., 2., 4., 2., 1.]) * (1./10.),
                            'pad':      2,
                            'norm_fac': 1.}
                            
    fd_coefficients['4'] = {'coeff':    np.asarray([-1., 12., -78., 112., -78., 12., -1.]), 
                            'mid':      np.asarray([0., 0., 0., 1., 0., 0., 0.]), 
                            'uniform':  np.asarray([1., 1., 1., 1., 1., 1., 1.]) * (1./7.),
                            'times2':   np.asarray([1., 2., 4., 8., 4., 2., 1.]) * (1./22.),
                            'pad':      3,
                            'norm_fac': 1.}#352
                            
    fd_coefficients['6'] = {'coeff':    np.asarray([7., -76., 676., -1952., 2730., -1952., 676., -76., 7.]), 
                            'mid':      np.asarray([0., 0., 0., 0., 1., 0., 0., 0., 0.]), 
                            'uniform':  np.asarray([1., 1., 1., 1., 1., 1., 1., 1., 1.]) * (1./9.),
                            'times2':   np.asarray([1., 2., 4., 8.,16., 8., 4., 2., 1.]) * (1./46.),
                            'pad':      4,
                            'norm_fac': 1.}#434400.
                            
    if accuracy not in list(fd_coefficients.keys()):
        raise NotImplementedError
    
    if stencil_type not in list(fd_coefficients[accuracy].keys()):
        raise NotImplementedError
    
    
    coefficient = np.asarray(fd_coefficients[accuracy]['coeff'])
    extend_mat  = np.asarray(fd_coefficients[accuracy][stencil_type])
    pad =  (fd_coefficients[accuracy]['pad'],)*3
    normalization_factor = fd_coefficients[accuracy]['norm_fac']
    temp_num = len(extend_mat)
    
    Gx_temp = np.reshape(extend_mat,(temp_num,1,1)) * ( np.reshape(extend_mat,(1,temp_num,1)) * coefficient)
    Gy_temp = Gx_temp.copy().transpose(0,2,1)
    Gz_temp = Gx_temp.copy().transpose(2,1,0)
    
    Gx = (1./(normalization_factor*(hx*hx*hx*hx))) * Gx_temp.copy()
    Gy = (1./(normalization_factor*(hy*hy*hy*hy))) * Gy_temp.copy()
    Gz = (1./(normalization_factor*(hz*hz*hz*hz))) * Gz_temp.copy()
    
    G = Gx + Gy + Gz
    
    return G, Gx, Gy, Gz, pad    
    
    
    
def get_fifth_grad_stencil(hx, hy, hz, stencil_type = 'mid', accuracy = '2'):
                       
    fd_coefficients = {}
                                          
    fd_coefficients['2'] = {'coeff':    np.asarray([-1., 4., -10., 0., 10., -4., 1.]), 
                            'mid':      np.asarray([0., 0., 0., 1., 0., 0., 0.]), 
                            'uniform':  np.asarray([1., 1., 1., 1., 1., 1., 1.]) * (1./7.),
                            'times2':   np.asarray([1., 2., 4., 8., 4., 2., 1.]) * (1./22.),
                            'pad':      3,
                            'norm_fac': 30.}

                            
    if accuracy not in list(fd_coefficients.keys()):
        raise NotImplementedError
    
    if stencil_type not in list(fd_coefficients[accuracy].keys()):
        raise NotImplementedError
    
    
    coefficient = np.asarray(fd_coefficients[accuracy]['coeff'])
    extend_mat  = np.asarray(fd_coefficients[accuracy][stencil_type])
    pad = (fd_coefficients[accuracy]['pad'],)*3
    normalization_factor = fd_coefficients[accuracy]['norm_fac']
    temp_num = len(extend_mat)
    
    Gx_temp = np.reshape(extend_mat,(temp_num,1,1)) * ( np.reshape(extend_mat,(1,temp_num,1)) * coefficient)
    Gy_temp = Gx_temp.copy().transpose(0,2,1)
    Gz_temp = Gx_temp.copy().transpose(2,1,0)
    
    Gx = (1./(normalization_factor*(hx*hx*hx*hx*hx))) * Gx_temp.copy()
    Gy = (1./(normalization_factor*(hy*hy*hy*hy*hy))) * Gy_temp.copy()
    Gz = (1./(normalization_factor*(hz*hz*hz*hz*hz))) * Gz_temp.copy()

    G = Gx + Gy + Gz
    
    return G, Gx, Gy, Gz, pad    
    
    
    
    
def get_sixth_grad_stencil(hx, hy, hz, stencil_type = 'mid', accuracy = '2'):
                       
    fd_coefficients = {}
                                          
    fd_coefficients['2'] = {'coeff':    np.asarray([1., -6., 15., -20., 15., -6., 1.]), 
                            'mid':      np.asarray([0., 0., 0., 1., 0., 0., 0.]), 
                            'uniform':  np.asarray([1., 1., 1., 1., 1., 1., 1.]) * (1./7.),
                            'times2':   np.asarray([1., 2., 4., 8., 4., 2., 1.]) * (1./22.),
                            'pad':      3,
                            'norm_fac': 1.}

                            
    if accuracy not in list(fd_coefficients.keys()):
        raise NotImplementedError
    
    if stencil_type not in list(fd_coefficients[accuracy].keys()):
        raise NotImplementedError
    
    
    coefficient = np.asarray(fd_coefficients[accuracy]['coeff'])
    extend_mat  = np.asarray(fd_coefficients[accuracy][stencil_type])
    pad = (fd_coefficients[accuracy]['pad'],)*3
    normalization_factor = fd_coefficients[accuracy]['norm_fac']
    temp_num = len(extend_mat)
    
    Gx_temp = np.reshape(extend_mat,(temp_num,1,1)) * ( np.reshape(extend_mat,(1,temp_num,1)) * coefficient)
    Gy_temp = Gx_temp.copy().transpose(0,2,1)
    Gz_temp = Gx_temp.copy().transpose(2,1,0)
    
    Gx = (1./(normalization_factor*(hx*hx*hx*hx*hx*hx))) * Gx_temp.copy()
    Gy = (1./(normalization_factor*(hy*hy*hy*hy*hy*hy))) * Gy_temp.copy()
    Gz = (1./(normalization_factor*(hz*hz*hz*hz*hz*hz))) * Gz_temp.copy()

    G = Gx + Gy + Gz
    
    return G, Gx, Gy, Gz, pad    



def get_differenciation_conv(n, hx, hy, hz, gradient = 'first', stencil_type = 'mid', accuracy = '2'):
    implemented_gradient = ['first','second', 'third', 'fourth', 'fifth', 'sixth']
    if gradient not in implemented_gradient:
        raise NotImplementedError
    
    if gradient == 'first':
        G, Gx, Gy, Gz, pad = get_first_grad_stencil(hx, hy, hz, 
                                                    stencil_type = stencil_type, 
                                                    accuracy = accuracy)
    elif gradient == 'second':
        G, Gx, Gy, Gz, pad = get_second_grad_stencil(hx, hy, hz, 
                                                    stencil_type = stencil_type, 
                                                    accuracy = accuracy)                    
    elif gradient == 'third':
        G, Gx, Gy, Gz, pad = get_third_grad_stencil(hx, hy, hz, 
                                                    stencil_type = stencil_type, 
                                                    accuracy = accuracy)    
    elif gradient == 'fourth':
        G, Gx, Gy, Gz, pad = get_fourth_grad_stencil(n, hx, hy, hz, 
                                                    stencil_type = stencil_type, 
                                                    accuracy = accuracy)
                                                    
    elif gradient == 'fifth':
        G, Gx, Gy, Gz, pad = get_fifth_grad_stencil(hx, hy, hz, 
                                                    stencil_type = stencil_type, 
                                                    accuracy = accuracy)                                                    
                                                    
    elif gradient == 'sixth':
        G, Gx, Gy, Gz, pad = get_sixth_grad_stencil(hx, hy, hz, 
                                                    stencil_type = stencil_type, 
                                                    accuracy = accuracy)

    
    pad_temp = pad[0]
    
    wrapped_n = np.pad(n, pad_temp, mode='wrap')
    
    
    temp_gradient = convolve(wrapped_n,G)
    result = temp_gradient[pad_temp:-pad_temp, pad_temp:-pad_temp, pad_temp:-pad_temp]
        
    return result, pad


def get_differenciation_conv_stencil(hx, hy, hz, gradient = 'first', stencil_type = 'mid', accuracy = '2'):
    implemented_gradient = ['first','second', 'third', 'fourth', 'fifth', 'sixth']
    if gradient not in implemented_gradient:
        raise NotImplementedError
    
    if gradient == 'first':
        G, Gx, Gy, Gz, pad = get_first_grad_stencil(hx, hy, hz, 
                                                    stencil_type = stencil_type, 
                                                    accuracy = accuracy)
    elif gradient == 'second':
        G, Gx, Gy, Gz, pad = get_second_grad_stencil(hx, hy, hz, 
                                                    stencil_type = stencil_type, 
                                                    accuracy = accuracy)                    
    elif gradient == 'third':
        G, Gx, Gy, Gz, pad = get_third_grad_stencil(hx, hy, hz, 
                                                    stencil_type = stencil_type, 
                                                    accuracy = accuracy)    
    elif gradient == 'fourth':
        G, Gx, Gy, Gz, pad = get_fourth_grad_stencil(n, hx, hy, hz, 
                                                    stencil_type = stencil_type, 
                                                    accuracy = accuracy)
                                                    
    elif gradient == 'fifth':
        G, Gx, Gy, Gz, pad = get_fifth_grad_stencil(hx, hy, hz, 
                                                    stencil_type = stencil_type, 
                                                    accuracy = accuracy)                                                    
                                                    
    elif gradient == 'sixth':
        G, Gx, Gy, Gz, pad = get_sixth_grad_stencil(hx, hy, hz, 
                                                    stencil_type = stencil_type, 
                                                    accuracy = accuracy)


        
    return G, pad
                                                    
'''
Integration

Basic philosophy: Initiate the dimensions of the stencils based on the r specified
                  and the grid spacing in each direction
                  
                  Get the coordinate where the sphere center is located
                  
                  for each dV, get the coordinates of the 8 vertices and store
                  
                  for each dV, check how many vertices are within the range of the sphere
                      if none, we assume there's no overlap between the sphere and dV
                      if 8, we assume all of dV is in the sphere
                      if other, subdivided dV into 8 sub-cubes and re-run the process

'''

    
def check_in_range(point_coord, center_coord,  r2):
    # Check if a given point is within the range of the sphere defined by
    # the coordinate of the center and radius r
#    return (np.linalg.norm(np.asarray(center_coord) - np.asarray(point_coord))) <= (r)
    temp1 = (center_coord[0] - point_coord[0]) ** 2.
    temp2 = (center_coord[1] - point_coord[1]) ** 2.
    temp3 = (center_coord[2] - point_coord[2]) ** 2.
    return (temp1 + temp2 + temp3 ) <= (r2) 
    
def check_num_vertices_in_range(li_vertex_coord, center_coord,r2):
    # for a list of points, return the number of the points that's within the 
    # range of the sphere defined by the coordinate of the center and radius r

    # Here we used to check how many vertices of a cube is within the range
    num_vertex_in_range = 0
    for vertex_coord in li_vertex_coord:
        if check_in_range(vertex_coord, center_coord, r2) == True:
            num_vertex_in_range += 1
    return num_vertex_in_range



def get_list_vertices_coord_subdivided_v(x_coord,y_coord,z_coord,dx,dy,dz):
    result = []
    result.append([x_coord, y_coord, z_coord])
    result.append([x_coord+dx, y_coord, z_coord])
    result.append([x_coord, y_coord+dy, z_coord])
    result.append([x_coord, y_coord, z_coord+dz])
    result.append([x_coord+dx, y_coord+dy, z_coord])
    result.append([x_coord+dx, y_coord, z_coord+dz])
    result.append([x_coord, y_coord+dy, z_coord+dz])
    result.append([x_coord+dx, y_coord+dy, z_coord+dz])
    
    return result
    
def get_subdivided_v_vertices(original_vertices_coords):
    result = []
    x_coord, y_coord, z_coord = original_vertices_coords[0]
    dx = (original_vertices_coords[1][0] - original_vertices_coords[0][0]) / 2.
    dy = (original_vertices_coords[2][1] - original_vertices_coords[0][1]) / 2.
    dz = (original_vertices_coords[3][2] - original_vertices_coords[0][2]) / 2.
    
    temp_vertices = get_list_vertices_coord_subdivided_v(x_coord,y_coord,z_coord,dx,dy,dz)
    
    for point in temp_vertices:
        result.append(get_list_vertices_coord_subdivided_v(point[0],point[1],point[2],dx,dy,dz))
    
    return result

def determine_dv_in_sphere_ratio(li_vertex_coord, center_coord,r2, acc_level, accuracy):
    # a recursive function, used to figure out the percentage of the volume that
    # is overlaping with the sphere
    
    # volume of the section, calculated based on the level of sub-division
    V = 1./((8.)**(acc_level-1))
    num_vertex_in_sphere = check_num_vertices_in_range(li_vertex_coord, center_coord,r2)
    if num_vertex_in_sphere == 8:
        return 1.* V
    elif num_vertex_in_sphere == 0:
        return 0.
    elif acc_level > accuracy:
        return float(num_vertex_in_sphere)*V/8.
    else:
        acc_level += 1
        result = 0.
        list_of_subdivided_v_vertices = get_subdivided_v_vertices(li_vertex_coord)
        for subdivided_v_vertices in list_of_subdivided_v_vertices:
            result += determine_dv_in_sphere_ratio(subdivided_v_vertices, center_coord,r2, acc_level, accuracy)
        return result

def get_list_vertices_coord(x_index,y_index,z_index,hx,hy,hz):
    # for each dV, based in the index in the array, 
    # figure out the coordinates of the 8 vertices of the dV    
    
    result = []
    
    result.append([x_index*hx, y_index*hy, z_index*hz])
    result.append([(x_index+1)*hx, y_index*hy, z_index*hz])
    result.append([x_index*hx, (y_index+1)*hy, z_index*hz])
    result.append([x_index*hx, y_index*hy, (z_index+1)*hz])
    result.append([(x_index+1)*hx, (y_index+1)*hy, z_index*hz])
    result.append([(x_index+1)*hx, y_index*hy, (z_index+1)*hz])
    result.append([x_index*hx, (y_index+1)*hy, (z_index+1)*hz])
    result.append([(x_index+1)*hx, (y_index+1)*hy, (z_index+1)*hz])
    
    return result



def get_coordinate_array(dim_x, dim_y, dim_z, hx, hy, hz):
    temp = np.zeros((int(dim_x), int(dim_y), int(dim_z))).tolist()
    for index, x in np.ndenumerate(temp):
        temp[index[0]][index[1]][index[2]] = get_list_vertices_coord(index[0],index[1],index[2],hx,hy,hz)
    result = np.asarray(temp)
    return result

def calc_integration_stencil(hx, hy, hz, r, accuracy):
    # calculate the stencil

    # initialize the stencil with right dimensions
    dim_x = int(2.* math.ceil( r/hx ))
    dim_y = int(2.* math.ceil( r/hy ))
    dim_z = int(2.* math.ceil( r/hz ))
   
    stencil = np.zeros((dim_x, dim_y, dim_z))
    coord_arr = get_coordinate_array(dim_x, dim_y, dim_z, hx, hy, hz)
    
    # caclulate the coordinate of the sphere center
    center_x = hx * float(dim_x)/2.
    center_y = hy * float(dim_y)/2.
    center_z = hz * float(dim_z)/2.
    center_coord = [center_x, center_y, center_z]
    
    
    # for each dV, get the percentage of dV that's overlaping with the sphere
    # and assign the percentage to the stencil
    r2 = r*r
    for index, x in np.ndenumerate(stencil):
        stencil[index[0]][index[1]][index[2]] = determine_dv_in_sphere_ratio(coord_arr[index[0]][index[1]][index[2]], center_coord,r2, 1, accuracy)
    
    # normalize the stencil with the volume of dV
    stencil *= hx*hy*hz
    
    padx = int(math.ceil(float(dim_x)/2.))
    pady = int(math.ceil(float(dim_y)/2.))
    padz = int(math.ceil(float(dim_z)/2.))
    
    pad = (padx,pady,padz)
    
    return stencil, pad

def from_temp_stencil_to_stencil(dim_x, dim_y, dim_z, temp_stencil):
    stencil = np.zeros((dim_x, dim_y, dim_z))
    center_row_x = (dim_x-1)/2
    center_row_y = (dim_y-1)/2
    center_row_z = (dim_z-1)/2
    
    for index, number in np.ndenumerate(temp_stencil):
        x = index[0]# + center_row_x
        y = index[1]# + center_row_y
        z = index[2]# + center_row_z
        stencil[int(center_row_x + x)][int(center_row_y + y)][int(center_row_z + z)] = number
        stencil[int(center_row_x - x)][int(center_row_y + y)][int(center_row_z + z)] = number
        stencil[int(center_row_x + x)][int(center_row_y - y)][int(center_row_z + z)] = number
        stencil[int(center_row_x + x)][int(center_row_y + y)][int(center_row_z - z)] = number
        stencil[int(center_row_x - x)][int(center_row_y - y)][int(center_row_z + z)] = number
        stencil[int(center_row_x + x)][int(center_row_y - y)][int(center_row_z - z)] = number
        stencil[int(center_row_x - x)][int(center_row_y + y)][int(center_row_z - z)] = number
        stencil[int(center_row_x - x)][int(center_row_y - y)][int(center_row_z - z)] = number
    
    return stencil

def calc_integration_stencil2(hx, hy, hz, r, accuracy):
    # calculate the stencil

    # initialize the stencil with right dimensions
    dim_x = int(2.* math.ceil( r/hx )) + 1
    dim_y = int(2.* math.ceil( r/hy )) + 1
    dim_z = int(2.* math.ceil( r/hz )) + 1
       
    temp_stencil = np.zeros((int((dim_x + 1 )/2), int((dim_y + 1 )/2), int((dim_z + 1 )/2)))
    
    temp_coord_arr = get_coordinate_array((dim_x + 1 )/2, (dim_y + 1 )/2, (dim_z + 1 )/2, hx, hy, hz)
    
    # caclulate the coordinate of the sphere center

    
    temp_center_x = hx / 2.
    temp_center_y = hy / 2.
    temp_center_z = hz / 2.
    temp_center_coord = [temp_center_x, temp_center_y, temp_center_z]
    print temp_center_coord
    r2 = r*r
    for index, x in np.ndenumerate(temp_stencil):
       temp_stencil[index[0]][index[1]][index[2]] = determine_dv_in_sphere_ratio(temp_coord_arr[index[0]][index[1]][index[2]], temp_center_coord,r2, 1, accuracy)
    
    
    # for each dV, get the percentage of dV that's overlaping with the sphere
    # and assign the percentage to the stencil
   
    # normalize the stencil with the volume of dV
    stencil = from_temp_stencil_to_stencil(dim_x, dim_y, dim_z, temp_stencil)
    stencil *= hx*hy*hz
    
    padx = int(math.ceil(float(dim_x)/2.))
    pady = int(math.ceil(float(dim_y)/2.))
    padz = int(math.ceil(float(dim_z)/2.))
    
    pad = (padx,pady,padz)

    
    return stencil, pad

def get_auto_accuracy(hx,hy,hz, r):
    h = max([hx,hy,hz])
    temp = 5 - int(math.floor((r/h)/3.))
    if temp < 1:
        return 1
    else:
        return temp
        
def get_integration_stencil(hx, hy, hz, r, accuracy):
    standard_acc = get_auto_accuracy(hx,hy,hz, r)
    if accuracy != standard_acc:
        stencil, pad = calc_integration_stencil2(hx, hy, hz, r, accuracy)
    else:
        if max(hx, hy, hz) - min(hx, hy, hz) < 0.001:
            try:
                stencil, pad = read_integration_stencil(hx, hy, hz, r)
            
            except:
                stencil, pad = calc_integration_stencil2(hx, hy, hz, r, accuracy)
        else:
            stencil, pad = calc_integration_stencil2(hx, hy, hz, r, accuracy)

    return stencil, pad

       
        

def get_integration_conv(n, hx, hy, hz, r, accuracy = 4):
    # get the stencil and do the convolution
    
    stencil, pad = get_integration_stencil(hx, hy, hz, r, accuracy)
    return convolve(n,stencil, mode = 'wrap'), pad


def get_integration_fftconv(n, hx, hy, hz, r, accuracy = 4):
    # get the stencil and do the convolution

    stencil, pad = get_integration_stencil(hx, hy, hz, r, accuracy)
    pad_temp = int(math.ceil(r*2. / min([hx,hy,hz])))
    wrapped_n = np.pad(n, pad_temp, mode='wrap')
    temp_result = fftconvolve(wrapped_n,stencil, mode = 'same')
    return temp_result[pad_temp:-pad_temp, pad_temp:-pad_temp, pad_temp:-pad_temp], pad

def get_integral_fftconv_with_known_stencil(n, hx, hy, hz, r, stencil, pad):
    # get the stencil and do the convolution

    pad_temp = int(math.ceil(r*2. / min([hx,hy,hz])))
    wrapped_n = np.pad(n, pad_temp, mode='wrap')
    temp_result = fftconvolve(wrapped_n,stencil, mode = 'same')
    return temp_result[pad_temp:-pad_temp, pad_temp:-pad_temp, pad_temp:-pad_temp], pad


def get_fftconv_with_known_stencil_no_wrap(n, hx, hy, hz, r, stencil, pad):
    temp_result = fftconvolve(n,stencil, mode = 'same')
    return temp_result, pad



"""
Asymmetric Integration 
"""

def get_asym_integration_stencil(hx, hy, hz, r, axis):
    
    axis_choice_list = {'x':0,'y':1,'z':2}
    if axis not in axis_choice_list:
        raise NotImplementedError
    axis_index = axis_choice_list[axis]
    standard_acc = get_auto_accuracy(hx,hy,hz, r)
    temp_stencil, pad = calc_integration_stencil2(hx, hy, hz, r, standard_acc)
    
    size = temp_stencil.shape
    target_axis_dim_div_2 = (size[axis_index] - 1) / 2
    stencil = np.zeros_like(temp_stencil)
    for index, x in np.ndenumerate(temp_stencil):
        stencil[index] = x * float(index[axis_index] - target_axis_dim_div_2)
    
    return stencil, pad

def get_asym_integration_fftconv(n, hx, hy, hz, r, axis):
    # get the stencil and do the convolution
    
    axis_choice_list = {'x':0,'y':1,'z':2}
    if axis not in axis_choice_list:
        raise NotImplementedError
    stencil, pad = get_asym_integration_stencil(hx, hy, hz, r, axis)
    pad_temp = int(math.ceil(r*2. / min([hx,hy,hz])))
    wrapped_n = np.pad(n, pad_temp, mode='wrap')
    temp_result = fftconvolve(wrapped_n,stencil, mode = 'same')
    return temp_result[pad_temp:-pad_temp, pad_temp:-pad_temp, pad_temp:-pad_temp], pad

def get_asym_integral_fftconv_with_known_stencil(n, hx, hy, hz, r, stencil, pad):
    # get the stencil and do the convolution

    pad_temp = int(math.ceil(r*2. / min([hx,hy,hz])))
    wrapped_n = np.pad(n, pad_temp, mode='wrap')
    temp_result = fftconvolve(wrapped_n,stencil, mode = 'same')
    return temp_result[pad_temp:-pad_temp, pad_temp:-pad_temp, pad_temp:-pad_temp], pad

def spherical_harmonic(x, y, z, l, m):
    r = math.sqrt(x*x + y*y + z*z)

    if l == 0:
        return 1, 0

    if l == 1:
        if m == -1:
            return x/r, -y/r
        if m == 0:
            return z/r, 0
        if m == 1:
            return -x/r, -y/r

    if l == 2:
        if m == -2:
            return x*y/(r*r), 0
        if m == -1:
            return z*y/(r*r), 0
        if m == 0:
            return (-x*x-y*y+2*z*z)/(r*r), 0
        if m == 1:
            return x*z/(r*r), 0 
        if m == 2:
            return (x*x-y*y)/(r*r), 0       

def spherical_harmonic_cutoff(x, y, z, l, m, cutoff):
    r = math.sqrt(x*x + y*y + z*z)
    if r <= cutoff:
        return spherical_harmonic(x, y, z, l, m)
    else:
        return 0, 0


def get_min_max_matrix(dim_x, dim_y, dim_z, hx, hy, hz):

    center_x = int((dim_x - 1)/2) 
    center_y = int((dim_y - 1)/2)
    center_z = int((dim_z - 1)/2)

    temp = np.zeros((dim_x, dim_y, dim_z, 6))

    for i in range(dim_x):
        for j in range(dim_y):
            for k in range(dim_z):
                temp[i,j,k,0] = hx * (float(i-center_x) - 0.5)
                temp[i,j,k,1] = hx * (float(i-center_x) + 0.5)
                temp[i,j,k,2] = hx * (float(j-center_y) - 0.5)
                temp[i,j,k,3] = hx * (float(j-center_y) + 0.5)
                temp[i,j,k,4] = hx * (float(k-center_z) - 0.5)
                temp[i,j,k,5] = hx * (float(k-center_z) + 0.5)

                #print "{}\t {} \t{}: {} -- {}, \t {} -- {}, \t{} -- {}".format(i,j,k,temp[i,j,k,0],temp[i,j,k,1],temp[i,j,k,2],temp[i,j,k,3],temp[i,j,k,4],temp[i,j,k,5])

    return temp


def calc_harmonic_stencil_value(min_max, r, l, m, accuracy):
    
    

    x_min = min_max[0]
    x_max = min_max[1]
    y_min = min_max[2]
    y_max = min_max[3]
    z_min = min_max[4]
    z_max = min_max[5]

    dx = (x_max-x_min) / accuracy
    dy = (y_max-y_min) / accuracy
    dz = (z_max-z_min) / accuracy
    dv = dx*dy*dz

    x_li = np.linspace(x_min, x_max, num=accuracy)
    y_li = np.linspace(y_min, y_max, num=accuracy)
    z_li = np.linspace(z_min, z_max, num=accuracy)
    
    coord_list = list(itertools.product(x_li,y_li,z_li))

    Re = 0.0
    Im = 0.0

    for x,y,z in coord_list:
        temp_Re, temp_Im = spherical_harmonic_cutoff(x, y, z, l, m, r)
        Re += temp_Re * dv
        Im += temp_Im * dv

    return Re, Im


def plot_stencil(stencil, min_max_matrix):

    x = np.zeros_like(stencil)
    y = np.zeros_like(stencil)
    z = np.zeros_like(stencil)

    for index, _ in np.ndenumerate(stencil):
        temp = min_max_matrix[index[0]][index[1]][index[2]].tolist()
        x[index] = (temp[0] + temp[1]) / 2.
        y[index] = (temp[2] + temp[3]) / 2.
        z[index] = (temp[4] + temp[5]) / 2.

    n = stencil.flatten()
    x_temp = x.flatten()
    y_temp = y.flatten()
    z_temp = z.flatten()


    #fig = plt.figure()
    #cmap = plt.get_cmap("bwr")
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(x_temp, y_temp, z_temp, c=n, cmap=cmap,linewidths=0,s=10.0)
    #plt.show()

    return



def calc_harmonic_stencil(hx, hy, hz, r, l, m, accuracy = 5):
    # calculate the stencil

    # initialize the stencil with right dimensions
    dim_x = int(2.* math.ceil( r/hx )) + 1
    dim_y = int(2.* math.ceil( r/hy )) + 1
    dim_z = int(2.* math.ceil( r/hz )) + 1

    print dim_x, dim_y, dim_z

    stencil_Re = np.zeros((dim_x, dim_y, dim_z))
    stencil_Im = np.zeros((dim_x, dim_y, dim_z))

    min_max_matrix = get_min_max_matrix(dim_x, dim_y, dim_z, hx, hy, hz)

    for index, x in np.ndenumerate(stencil_Re):
        temp_Re, temp_Im = calc_harmonic_stencil_value(min_max_matrix[index], r, l, m, accuracy)
        stencil_Re[index] = temp_Re
        stencil_Im[index] = temp_Im
    
    # caclulate the coordinate of the sphere center

    #print stencil_Im
    #print stencil_Re

    #plot_stencil(stencil_Im, min_max_matrix)
    #plot_stencil(stencil_Re, min_max_matrix)
    
    
    padx = int(math.ceil(float(dim_x)/2.))
    pady = int(math.ceil(float(dim_y)/2.))
    padz = int(math.ceil(float(dim_z)/2.))
    
    pad = (padx,pady,padz)

    
    return stencil_Re, stencil_Im, pad

def get_harmonic_fftconv(n, hx, hy, hz, r, l, m, accuracy = 5):
    # get the stencil and do the convolution
    
    stencil_Re, stencil_Im, pad = calc_harmonic_stencil(hx, hy, hz, r, l, m, accuracy = accuracy)
    pad_temp = int(math.ceil(r*2. / min([hx,hy,hz])))
    wrapped_n = np.pad(n, pad_temp, mode='wrap')
    temp_result_Re = fftconvolve(wrapped_n,stencil_Re, mode = 'same')
    temp_result_Im = fftconvolve(wrapped_n,stencil_Im, mode = 'same')
    return temp_result_Re[pad_temp:-pad_temp, pad_temp:-pad_temp, pad_temp:-pad_temp], temp_result_Im[pad_temp:-pad_temp, pad_temp:-pad_temp, pad_temp:-pad_temp], pad







def MC_surface_spherical_harmonic(x, y, z, l, m):
    r = math.sqrt(x*x + y*y + z*z)
    x_hat = x/r
    y_hat = y/r
    z_hat = z/r

    if l == 0:
        return 1.0

    if l == 1:
        if m == 1:
            return x_hat
        if m == 2:
            return y_hat
        if m == 3:
            return z_hat

    if l == 2:
        if m == 1:
            return 3.0 * x_hat * x_hat - 1.0
        if m == 2:
            return 3.0 * x_hat * y_hat
        if m == 3:
            return 3.0 * x_hat * z_hat
        if m == 4:
            return 3.0 * y_hat * y_hat - 1.0
        if m == 5:
            return 3.0 * y_hat * z_hat
        if m == 6:
            return 3.0 * z_hat * z_hat - 1.0

    if l == 3:
        if m == 1:
            return 15.0 * x_hat * x_hat * x_hat - 9.0 * x_hat
        if m == 2:
            return 15.0 * x_hat * x_hat * y_hat - 3.0 * y_hat
        if m == 3:
            return 15.0 * x_hat * x_hat * z_hat - 3.0 * z_hat
        if m == 4:
            return 15.0 * x_hat * y_hat * y_hat - 3.0 * x_hat
        if m == 5:
            return 15.0 * x_hat * y_hat * z_hat
        if m == 6:
            return 15.0 * x_hat * z_hat * z_hat - 3.0 * x_hat
        if m == 7:
            return 15.0 * y_hat * y_hat * y_hat - 9.0 * y_hat
        if m == 8:
            return 15.0 * y_hat * y_hat * z_hat - 3.0 * z_hat
        if m == 9:
            return 15.0 * y_hat * z_hat * z_hat - 3.0 * y_hat 
        if m == 10:
            return 15.0 * z_hat * z_hat * z_hat - 9.0 * z_hat

    if l == 4:
        if m == 1:
            return 105.0 * x_hat * x_hat * x_hat * x_hat - 90.0 * x_hat * x_hat + 9.0
        if m == 2:
            return 105.0 * x_hat * x_hat * x_hat * y_hat - 45.0 * x_hat * y_hat
        if m == 3:
            return 105.0 * x_hat * x_hat * x_hat * z_hat - 45.0 * x_hat * z_hat
        if m == 4:
            return 105.0 * x_hat * x_hat * y_hat * y_hat - 15.0 * x_hat * x_hat - 15.0 * y_hat * y_hat + 3.0
        if m == 5:
            return 105.0 * x_hat * x_hat * y_hat * z_hat - 15.0 * y_hat * z_hat
        if m == 6:
            return 105.0 * x_hat * x_hat * z_hat * z_hat - 15.0 * x_hat * x_hat - 15.0 * z_hat * z_hat + 3.0
        if m == 7:
            return 105.0 * x_hat * y_hat * y_hat * y_hat - 45.0 * x_hat * y_hat
        if m == 8:
            return 105.0 * x_hat * y_hat * y_hat * z_hat - 15.0 * x_hat * z_hat
        if m == 9:
            return 105.0 * x_hat * y_hat * z_hat * z_hat - 15.0 * x_hat * y_hat
        if m == 10:
            return 105.0 * x_hat * z_hat * z_hat * z_hat - 45.0 * x_hat * z_hat
        if m == 11:
            return 105.0 * y_hat * y_hat * y_hat * y_hat - 90.0 * y_hat * y_hat + 9.0
        if m == 12:
            return 105.0 * y_hat * y_hat * y_hat * z_hat - 45.0 * y_hat * z_hat
        if m == 13:
            return 105.0 * y_hat * y_hat * z_hat * z_hat - 15.0 * y_hat * y_hat - 15.0 * z_hat * z_hat + 3.0
        if m == 14:
            return 105.0 * y_hat * z_hat * z_hat * z_hat - 45.0 * y_hat * z_hat
        if m == 15:
            return 105.0 * z_hat * z_hat * z_hat * z_hat - 90.0 * z_hat * z_hat + 9.0




def MC_surface_spherical_harmonic_cutoff(x, y, z, l, m, cutoff):
    r = math.sqrt(x*x + y*y + z*z)
    if r <= cutoff:
        return MC_surface_spherical_harmonic(x, y, z, l, m)
    else:
        return 0



def calc_MC_surface_harmonic_stencil_value(min_max, r, l, m, accuracy):
    
    

    x_min = min_max[0]
    x_max = min_max[1]
    y_min = min_max[2]
    y_max = min_max[3]
    z_min = min_max[4]
    z_max = min_max[5]

    dx = (x_max-x_min) / accuracy
    dy = (y_max-y_min) / accuracy
    dz = (z_max-z_min) / accuracy
    dv = dx*dy*dz

    x_li = np.linspace(x_min, x_max, num=accuracy)
    y_li = np.linspace(y_min, y_max, num=accuracy)
    z_li = np.linspace(z_min, z_max, num=accuracy)
    
    coord_list = list(itertools.product(x_li,y_li,z_li))

    Re = 0.0

    for x,y,z in coord_list:
        temp_Re = MC_surface_spherical_harmonic_cutoff(x, y, z, l, m, r)
        Re += temp_Re * dv

    return Re



def calc_MC_surface_harmonic_stencil(hx, hy, hz, r, l, m, accuracy = 5):
    # calculate the stencil

    # initialize the stencil with right dimensions
    dim_x = int(2.* math.ceil( r/hx )) + 1
    dim_y = int(2.* math.ceil( r/hy )) + 1
    dim_z = int(2.* math.ceil( r/hz )) + 1

    print dim_x, dim_y, dim_z

    stencil_Re = np.zeros((dim_x, dim_y, dim_z))
    stencil_Im = np.zeros((dim_x, dim_y, dim_z))

    min_max_matrix = get_min_max_matrix(dim_x, dim_y, dim_z, hx, hy, hz)

    for index, x in np.ndenumerate(stencil_Re):
        temp_Re = calc_MC_surface_harmonic_stencil_value(min_max_matrix[index], r, l, m, accuracy)
        stencil_Re[index] = temp_Re
    
    # caclulate the coordinate of the sphere center

    #print stencil_Im
    #print stencil_Re

    #plot_stencil(stencil_Im, min_max_matrix)
    #plot_stencil(stencil_Re, min_max_matrix)
    
    
    padx = int(math.ceil(float(dim_x)/2.))
    pady = int(math.ceil(float(dim_y)/2.))
    padz = int(math.ceil(float(dim_z)/2.))
    
    pad = (padx,pady,padz)

    
    return stencil_Re, pad

def get_MC_surface_harmonic_fftconv(n, hx, hy, hz, r, l, m, accuracy = 5):
    # get the stencil and do the convolution
    
    stencil_Re, pad = calc_MC_surface_harmonic_stencil(hx, hy, hz, r, l, m, accuracy = accuracy)
    pad_temp = int(math.ceil(r*2. / min([hx,hy,hz])))
    wrapped_n = np.pad(n, pad_temp, mode='wrap')
    temp_result_Re = fftconvolve(wrapped_n,stencil_Re, mode = 'same')
    return temp_result_Re[pad_temp:-pad_temp, pad_temp:-pad_temp, pad_temp:-pad_temp], pad






def MC_surface_spherical_harmonic_n(x, y, z, l, n):
    r = math.sqrt(x*x + y*y + z*z)
    x_hat = x/r
    y_hat = y/r
    z_hat = z/r

    if l == 0:
        return 1.0

    if l == 1:
        if n == "100":
            return x_hat
        if n == "010":
            return y_hat
        if n == "001":
            return z_hat

    if l == 2:
        if n == "200":
            return 3.0 * x_hat * x_hat - 1.0
        if n == "110":
            return 3.0 * x_hat * y_hat
        if n == "101":
            return 3.0 * x_hat * z_hat
        if n == "020":
            return 3.0 * y_hat * y_hat - 1.0
        if n == "011":
            return 3.0 * y_hat * z_hat
        if n == "002":
            return 3.0 * z_hat * z_hat - 1.0

    if l == 3:
        if n == "300":
            return 15.0 * x_hat * x_hat * x_hat - 9.0 * x_hat
        if n == "210":
            return 15.0 * x_hat * x_hat * y_hat - 3.0 * y_hat
        if n == "201":
            return 15.0 * x_hat * x_hat * z_hat - 3.0 * z_hat
        if n == "120":
            return 15.0 * x_hat * y_hat * y_hat - 3.0 * x_hat
        if n == "111":
            return 15.0 * x_hat * y_hat * z_hat
        if n == "102":
            return 15.0 * x_hat * z_hat * z_hat - 3.0 * x_hat
        if n == "030":
            return 15.0 * y_hat * y_hat * y_hat - 9.0 * y_hat
        if n == "021":
            return 15.0 * y_hat * y_hat * z_hat - 3.0 * z_hat
        if n == "012":
            return 15.0 * y_hat * z_hat * z_hat - 3.0 * y_hat 
        if n == "003":
            return 15.0 * z_hat * z_hat * z_hat - 9.0 * z_hat

    if l == 4:
        if n == "400":
            return 105.0 * x_hat * x_hat * x_hat * x_hat - 90.0 * x_hat * x_hat + 9.0
        if n == "310":
            return 105.0 * x_hat * x_hat * x_hat * y_hat - 45.0 * x_hat * y_hat
        if n == "301":
            return 105.0 * x_hat * x_hat * x_hat * z_hat - 45.0 * x_hat * z_hat
        if n == "220":
            return 105.0 * x_hat * x_hat * y_hat * y_hat - 15.0 * x_hat * x_hat - 15.0 * y_hat * y_hat + 3.0
        if n == "211":
            return 105.0 * x_hat * x_hat * y_hat * z_hat - 15.0 * y_hat * z_hat
        if n == "202":
            return 105.0 * x_hat * x_hat * z_hat * z_hat - 15.0 * x_hat * x_hat - 15.0 * z_hat * z_hat + 3.0
        if n == "130":
            return 105.0 * x_hat * y_hat * y_hat * y_hat - 45.0 * x_hat * y_hat
        if n == "121":
            return 105.0 * x_hat * y_hat * y_hat * z_hat - 15.0 * x_hat * z_hat
        if n == "112":
            return 105.0 * x_hat * y_hat * z_hat * z_hat - 15.0 * x_hat * y_hat
        if n == "103":
            return 105.0 * x_hat * z_hat * z_hat * z_hat - 45.0 * x_hat * z_hat
        if n == "040":
            return 105.0 * y_hat * y_hat * y_hat * y_hat - 90.0 * y_hat * y_hat + 9.0
        if n == "031":
            return 105.0 * y_hat * y_hat * y_hat * z_hat - 45.0 * y_hat * z_hat
        if n == "022":
            return 105.0 * y_hat * y_hat * z_hat * z_hat - 15.0 * y_hat * y_hat - 15.0 * z_hat * z_hat + 3.0
        if n == "013":
            return 105.0 * y_hat * z_hat * z_hat * z_hat - 45.0 * y_hat * z_hat
        if n == "004":
            return 105.0 * z_hat * z_hat * z_hat * z_hat - 90.0 * z_hat * z_hat + 9.0


def MC_surface_spherical_harmonic_cutoff_n(x, y, z, l, n, cutoff):
    r = math.sqrt(x*x + y*y + z*z)
    if r <= cutoff:
        return MC_surface_spherical_harmonic_n(x, y, z, l, n)
    else:
        return 0



def calc_MC_surface_harmonic_stencil_value_n(min_max, r, l, n, accuracy):
    
    

    x_min = min_max[0]
    x_max = min_max[1]
    y_min = min_max[2]
    y_max = min_max[3]
    z_min = min_max[4]
    z_max = min_max[5]

    dx = (x_max-x_min) / accuracy
    dy = (y_max-y_min) / accuracy
    dz = (z_max-z_min) / accuracy
    dv = dx*dy*dz

    x_li = np.linspace(x_min, x_max, num=accuracy)
    y_li = np.linspace(y_min, y_max, num=accuracy)
    z_li = np.linspace(z_min, z_max, num=accuracy)
    
    coord_list = list(itertools.product(x_li,y_li,z_li))

    Re = 0.0

    for x,y,z in coord_list:
        temp_Re = MC_surface_spherical_harmonic_cutoff_n(x, y, z, l, n, r)
        Re += temp_Re * dv

    return Re



def calc_MC_surface_harmonic_stencil_n(hx, hy, hz, r, l, n, accuracy = 5):
    # calculate the stencil

    # initialize the stencil with right dimensions
    dim_x = int(2.* math.ceil( r/hx )) + 1
    dim_y = int(2.* math.ceil( r/hy )) + 1
    dim_z = int(2.* math.ceil( r/hz )) + 1

    print dim_x, dim_y, dim_z

    stencil_Re = np.zeros((dim_x, dim_y, dim_z))
    stencil_Im = np.zeros((dim_x, dim_y, dim_z))

    min_max_matrix = get_min_max_matrix(dim_x, dim_y, dim_z, hx, hy, hz)

    for index, x in np.ndenumerate(stencil_Re):
        temp_Re = calc_MC_surface_harmonic_stencil_value_n(min_max_matrix[index], r, l, n, accuracy)
        stencil_Re[index] = temp_Re

       
    padx = int(math.ceil(float(dim_x)/2.))
    pady = int(math.ceil(float(dim_y)/2.))
    padz = int(math.ceil(float(dim_z)/2.))
    
    pad = (padx,pady,padz)

    
    return stencil_Re, pad


def sum_magnitude(li):
    result = np.zeros_like(li[0])
    for entry in li:
        result = np.add(result,np.square(entry))
    return
