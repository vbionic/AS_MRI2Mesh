import cv2 as cv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys, getopt
import pydicom
import numpy as np
from PIL import Image
import json
from pydicom.tag import Tag
from pydicom.datadict import keyword_for_tag
import os
import pathlib
from argparse import ArgumentParser
import glob
import math
import shutil
import logging
import scipy
from scipy.interpolate import RectBivariateSpline
import time
import random
import scipy.optimize as sp
import multiprocessing as mp
from scipy.spatial import cKDTree, distance


#-----------------------------------------------------------------------------------------
sys.path.append(os.getcwd())
#-----------------------------------------------------------------------------------------
from v_utils.v_json import *
from v_utils.v_polygons import *
#-----------------------------------------------------------------------------------------

cycles = 0

def process_spectrum(spectrum, min_bin_no, threshold, width):
    for i in range(min_bin_no, (len(spectrum)//2)):
        if width == 3:
            average = (spectrum[i-1] + spectrum[i+1])/2
        elif width == 5:
            average = (spectrum[i-1] + spectrum[i+1] + spectrum[i-2] + spectrum[i+2])/4
        else:
            logging.error("incorrect width of the averaging window")
            exit(1)
        if ((np.abs(spectrum[i]) > threshold*np.abs(average))|(np.abs(spectrum[i]) < np.abs(average)*0.9)):
            spectrum[i] = average
            spectrum[len(spectrum) - i] = average.conjugate()
    return spectrum
        
        
def apply_corrections(corrections, iDir, oDir):
    poly_list    = glob.glob(os.path.normpath(iDir + "/*_polygons.json"))
    poly_list.sort()
    poly_num = 0
    for poly_fn in poly_list:
        poly_fn_norm = os.path.basename(os.path.normpath(poly_fn))
#        poly_num = int(poly_fn_norm.split('_',1)[0])-1
        try:
            poly_h = open(poly_fn)
            data = json.load(poly_h)
            poly_h.close()
        except Exception as err:
            logging.error("Input data IO error: {}".format(err))
            sys.exit(1)
        if(len(data["polygons"])==0):
            try:
                if not os.path.isdir(os.path.normpath(oDir)):
                    pathlib.Path(os.path.normpath(oDir)).mkdir(mode=0o775, parents=True, exist_ok=True)
            except Exception as err:
                logging.error('creating "%s" directory failed, error "%s"'%(oDir,err))
                exit(1)
            jsonDumpSafe(os.path.normpath(oDir + "/" + poly_fn_norm), data)
            continue
        
        corr_val_x = corrections[poly_num][1]
        corr_val_y = corrections[poly_num][2]
        imsize_x   = corrections[poly_num][3]
        imsize_y   = corrections[poly_num][4]
        
#        logging.debug("Moving {} by {} {}".format(poly_fn_norm, corr_val_x, corr_val_y))
        
        for polygon in data["polygons"]:
            for i in range(0, len(polygon["outer"]["path"])):
                polygon["outer"]["path"][i][0] = polygon["outer"]["path"][i][0] + corr_val_x
                polygon["outer"]["path"][i][1] = polygon["outer"]["path"][i][1] + corr_val_y
            polygon["outer"]["box"][0] = polygon["outer"]["box"][0] + corr_val_x
            polygon["outer"]["box"][1] = polygon["outer"]["box"][1] + corr_val_y
            polygon["outer"]["box"][2] = polygon["outer"]["box"][2] + corr_val_x
            polygon["outer"]["box"][3] = polygon["outer"]["box"][3] + corr_val_y
            for inner in polygon["inners"]:
                for i in range(0, len(inner["path"])):
                    inner["path"][i][0] = inner["path"][i][0] + corr_val_x
                    inner["path"][i][1] = inner["path"][i][1] + corr_val_y
                inner["box"][0] = inner["box"][0] + corr_val_x
                inner["box"][1] = inner["box"][1] + corr_val_y
                inner["box"][2] = inner["box"][2] + corr_val_x
                inner["box"][3] = inner["box"][3] + corr_val_y
        data["box"][0] = data["box"][0] + corr_val_x
        data["box"][1] = data["box"][1] + corr_val_y
        data["box"][2] = data["box"][2] + corr_val_x
        data["box"][3] = data["box"][3] + corr_val_y
        
        #if (data["box"][0] < 0) or (data["box"][1]<0) or (data["box"][2]>imsize_x - 1) or (data["box"][3]>imsize_y - 1):
        #    logging.error("The correction shift too large - tissue in {} goes outside the image size".format(iDir))
        #    exit(1)
        try:
            if not os.path.isdir(os.path.normpath(oDir)):
                pathlib.Path(os.path.normpath(oDir)).mkdir(mode=0o775, parents=True, exist_ok=True)
        except Exception as err:
            logging.error('creating "%s" directory failed, error "%s"'%(oDir,err))
            exit(1)
        jsonDumpSafe(os.path.normpath(oDir + "/" + poly_fn_norm), data)
        jsonDumpSafe(os.path.normpath(oDir + "/" + "corrections.json"), {"corrections":corrections})
        poly_num += 1
        
def apply_corrections_img(corrections, iDir, oDir):
    img_list    = glob.glob(os.path.normpath(iDir + "/*_labels.png"))
    img_list.sort()
    img_num = 0
    for img_fn in img_list:
        img_fn_norm = os.path.basename(os.path.normpath(img_fn))
#        img_num = int(img_fn_norm.split('_',1)[0])-1 
        corr_val_x = corrections[img_num][1]
        corr_val_y = corrections[img_num][2]
        
        img = cv.imread(img_fn, cv.IMREAD_COLOR)
#        logging.debug("Moving {} by {} {}".format(img_fn, corr_val_x, corr_val_y))
        
        new_img = np.zeros((img.shape[0], img.shape[1], 3))
        
        if (corr_val_x >= 0) and (corr_val_y >= 0):
            new_img[corr_val_y:, corr_val_x:, :] = img[0:img.shape[0]-corr_val_y, 0:img.shape[1]-corr_val_x, :]
        if (corr_val_x >= 0) and (corr_val_y < 0):
            new_img[0:img.shape[0]+corr_val_y, corr_val_x:, :] = img[-corr_val_y:, 0:img.shape[1]-corr_val_x, :]
        if (corr_val_x < 0) and (corr_val_y >= 0):
            new_img[corr_val_y:, 0:img.shape[1]+corr_val_x, :] = img[0:img.shape[0]-corr_val_y, -corr_val_x:, :]
        if (corr_val_x < 0) and (corr_val_y < 0):
            new_img[0:img.shape[0]+corr_val_y, 0:img.shape[1]+corr_val_x, :] = img[-corr_val_y:, -corr_val_x:, :]
            
        try:
            if not os.path.isdir(os.path.normpath(oDir)):
                pathlib.Path(os.path.normpath(oDir)).mkdir(mode=0o775, parents=True, exist_ok=True)
        except Exception as err:
            logging.error('creating "%s" directory failed, error "%s"'%(oDir,err))
            exit(1)
        
        cv.imwrite(os.path.normpath(oDir + "/" + img_fn_norm), new_img)
        
        img_fn_2 = img_fn.rsplit('_',1)[0]+"_prob.png"
        
        
        img = cv.imread(img_fn_2, cv.IMREAD_COLOR)
        if not img is None:
            new_img = np.zeros((img.shape[0], img.shape[1], 3))
            
            if (corr_val_x >= 0) and (corr_val_y >= 0):
                new_img[corr_val_y:, corr_val_x:, :] = img[0:img.shape[0]-corr_val_y, 0:img.shape[1]-corr_val_x, :]
            if (corr_val_x >= 0) and (corr_val_y < 0):
                new_img[0:img.shape[0]+corr_val_y, corr_val_x:, :] = img[-corr_val_y:, 0:img.shape[1]-corr_val_x, :]
            if (corr_val_x < 0) and (corr_val_y >= 0):
                new_img[corr_val_y:, 0:img.shape[1]+corr_val_x, :] = img[0:img.shape[0]-corr_val_y, -corr_val_x:, :]
            if (corr_val_x < 0) and (corr_val_y < 0):
                new_img[0:img.shape[0]+corr_val_y, 0:img.shape[1]+corr_val_x, :] = img[-corr_val_y:, -corr_val_x:, :]
            
            cv.imwrite(os.path.normpath(oDir + "/" + os.path.basename(img_fn_2)), new_img)
        
        
        img_fn_3 = img_fn.rsplit('_',1)[0]+"_prob_nl.png"
        
        img = cv.imread(img_fn_3, cv.IMREAD_COLOR)
        if not img is None:
            new_img = np.zeros((img.shape[0], img.shape[1], 3))
            
            if (corr_val_x >= 0) and (corr_val_y >= 0):
                new_img[corr_val_y:, corr_val_x:, :] = img[0:img.shape[0]-corr_val_y, 0:img.shape[1]-corr_val_x, :]
            if (corr_val_x >= 0) and (corr_val_y < 0):
                new_img[0:img.shape[0]+corr_val_y, corr_val_x:, :] = img[-corr_val_y:, 0:img.shape[1]-corr_val_x, :]
            if (corr_val_x < 0) and (corr_val_y >= 0):
                new_img[corr_val_y:, 0:img.shape[1]+corr_val_x, :] = img[0:img.shape[0]-corr_val_y, -corr_val_x:, :]
            if (corr_val_x < 0) and (corr_val_y < 0):
                new_img[0:img.shape[0]+corr_val_y, 0:img.shape[1]+corr_val_x, :] = img[-corr_val_y:, -corr_val_x:, :]
            
            cv.imwrite(os.path.normpath(oDir + "/" + os.path.basename(img_fn_3)), new_img)
        img_num += 1
    jsonDumpSafe(os.path.normpath(oDir + "/" + "corrections.json"), {"corrections":corrections})
            
            
def apply_corrections_img_orig(corrections, iDir, oDir):
    for img_type in ["lsi", "csi", "nsi", "gsi"]:
        img_list    = glob.glob(os.path.normpath(iDir + "/*_{}.png".format(img_type)))
        img_list.sort()
        img_num = 0
        for img_fn in img_list:
            img_fn_norm = os.path.basename(os.path.normpath(img_fn))
#            img_num = int(img_fn_norm.split('_',1)[0])-1
            corr_val_x = corrections[img_num][1]
            corr_val_y = corrections[img_num][2]
            
            img = cv.imread(img_fn, cv.IMREAD_COLOR)
            #logging.debug("Moving {} by {} {}".format(img_fn, corr_val_x, corr_val_y))
            new_img = np.zeros((img.shape[0], img.shape[1], 3))
            
            if (corr_val_x >= 0) and (corr_val_y >= 0):
                new_img[corr_val_y:, corr_val_x:, :] = img[0:img.shape[0]-corr_val_y, 0:img.shape[1]-corr_val_x, :]
            if (corr_val_x >= 0) and (corr_val_y < 0):
                new_img[0:img.shape[0]+corr_val_y, corr_val_x:, :] = img[-corr_val_y:, 0:img.shape[1]-corr_val_x, :]
            if (corr_val_x < 0) and (corr_val_y >= 0):
                new_img[corr_val_y:, 0:img.shape[1]+corr_val_x, :] = img[0:img.shape[0]-corr_val_y, -corr_val_x:, :]
            if (corr_val_x < 0) and (corr_val_y < 0):
                new_img[0:img.shape[0]+corr_val_y, 0:img.shape[1]+corr_val_x, :] = img[-corr_val_y:, -corr_val_x:, :]
                
            try:
                if not os.path.isdir(os.path.normpath(oDir)):
                    pathlib.Path(os.path.normpath(oDir)).mkdir(mode=0o775, parents=True, exist_ok=True)
            except Exception as err:
                logging.error('creating "%s" directory failed, error "%s"'%(oDir,err))
                exit(1)
            
            cv.imwrite(os.path.normpath(oDir + "/" + img_fn_norm), new_img)
            img_num += 1
    jsonDumpSafe(os.path.normpath(oDir + "/" + "corrections.json"), {"corrections":corrections})
    
def pokaz(ax, dane, ofset = (0,0,0), c=None):
    ax.scatter([i[2][0] - ofset[0] for i in dane], [i[2][1] - ofset[1]  for i in dane], [i[2][2] - ofset[2]  for i in dane],c=c)
    
def pokaz2(ax, dane, c=None):
    ax.scatter([i[0] for i in dane], [i[1] for i in dane], [i[2] for i in dane],c=c)
    
def add_offset_to_slices(points_3D_data, offsets):
    #print(offsets)
    offsets = np.append(offsets, [0,0])
    #print(offsets)
    offsets = offsets.reshape((int(len(offsets)/2),2))
    #print(offsets)
    points_3D_data_new = []
    for p in points_3D_data:
        points_3D_data_new.append([p[0], p[1], [p[2][0]+offsets[p[0]][0], p[2][1]+offsets[p[0]][1], p[2][2]] ])
    return points_3D_data_new
        
        
def get_next_point(point3D, step, slices):
    curr_slice = point3D[0]
    curr_point = point3D[1]
    curr_slice_len = len(slices[curr_slice])
    index = (curr_point+step)%curr_slice_len
    return slices[curr_slice][index]
    
def crossKK(a,b):
    return np.array([a[1]*b[2]-a[2]*b[1], -a[0]*b[2]+a[2]*b[0], a[0]*b[1]-a[1]*b[0]])
    
def calc_normal_neighbor_triangles(slice_num, point_num, point3D, slices, slice_trees):
    a = find_closest_next_slice(slice_num, point_num, point3D, slices, slice_trees[slice_num-1], slice_trees[(slice_num+1)%len(slice_trees)])
    
    b = (get_next_point(a[0], 1, slices), get_next_point(a[0], -1, slices))
    c = (get_next_point(a[1], 1, slices), get_next_point(a[1], -1, slices))
    
    #triangle 1 normal slice +1
    #v1 = np.cross(b[1][2] - np.array(point3D[2]), a[0][2] - np.array(point3D[2]))
    v1 = crossKK(b[1][2] - np.array(point3D[2]), a[0][2] - np.array(point3D[2]))
    temp = np.linalg.norm(v1)
    if temp != 0:
        v1 /= temp
    #triangle 2 normal slice +1
    #v2 = np.cross(a[0][2] - np.array(point3D[2]), b[0][2] - np.array(point3D[2]))
    v2 = crossKK(a[0][2] - np.array(point3D[2]), b[0][2] - np.array(point3D[2]))
    temp = np.linalg.norm(v2)
    if temp != 0:
        v2 /= temp
    #triangle 3 normal slice -1
    #v3 = np.cross(c[0][2] - np.array(point3D[2]), a[1][2] - np.array(point3D[2]))
    v3 = crossKK(c[0][2] - np.array(point3D[2]), a[1][2] - np.array(point3D[2]))
    temp = np.linalg.norm(v3)
    if temp != 0:
        v3 /= temp
    #triangle 4 normal slice -1
    #v4 = np.cross(a[1][2] - np.array(point3D[2]), c[1][2] - np.array(point3D[2]))
    v4 = crossKK(a[1][2] - np.array(point3D[2]), c[1][2] - np.array(point3D[2]))
    temp = np.linalg.norm(v4)
    if temp != 0:
        v4 /= temp
    #average the triangles connecting to the slice +1
    vec1 = (v1+v2)/2
    
    #average the triangles connecting to the slice -1
    vec2 = (v3+v4)/2
    
    #normalize
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    vec1 = vec1/norm1 if norm1>0 else vec1
    vec2 = vec2/norm2 if norm2>0 else vec2
    
    return (vec1, vec2)
        
        
def calc_normal(slice_num, point_num, point3D, slices):
    a = find_closest_next_slice(slice_num, point_num, point3D, slices)
    b = find_neighbors_same_slice(slice_num, point_num, point3D, slices)
    
    v1 = np.cross(a[0][2] - np.array(point3D[2]), b[1][2] - np.array(point3D[2]))
    v2 = np.cross(b[1][2] - np.array(point3D[2]), a[1][2] - np.array(point3D[2]))
    v3 = np.cross(a[1][2] - np.array(point3D[2]), b[0][2] - np.array(point3D[2]))
    v4 = np.cross(b[0][2] - np.array(point3D[2]), a[0][2] - np.array(point3D[2]))
    
    vec = (v1+v2+v3+v4)/4
    temp = np.linalg.norm(vec)
    if temp == 0:
        temp = 1
    return vec/temp
    
# def calc_normal2(slice_num, point_num, point3D, points_3D_data, slice_centres, slices):
    # a = find_closest_next_slice(slice_num, point_num, point3D, points_3D_data, slice_centres, slices)
    # b = find_neighbors_same_slice(slice_num, point_num, point3D, points_3D_data, slice_centres, slices)
    # print("{} {}".format(slice_num, point_num))
    # print(point3D)
    # print(a[0])
    # print(a[1])
    # print(b[0])
    # print(b[1])
    
    # vec = [(point3D[0]-slice_centres[slice_num][0]), (point3D[1]-slice_centres[slice_num][1]), 0]
    # return [vec/np.linalg.norm(vec), [a[0], a[1], b[0], b[1]]]
    
def calc_tangent_plane(slice_num, point_num, point3D, slices):
    #print("Licze plaszczyzne dla punktu numer {} na slice {}, w sumie slice ma {} punktow".format(point_num, slice_num, len(slices[slice_num])))
    vec1 = calc_normal(slice_num, point_num, point3D, slices)
    temp = np.array([0,0,1])
    vec2 = np.cross(vec1,temp)
    vec3 = np.cross(vec1,vec2)
    
    vec1 = vec1/np.linalg.norm(vec1)
    vec2 = vec2/np.linalg.norm(vec2)
    vec3 = vec3/np.linalg.norm(vec3)
    return (vec1, vec2, vec3)
    
def get_dist(p1, p2):
    return np.linalg.norm((p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2]))

def get_dist_sq(p1, p2):
    return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2+ (p1[2]-p2[2])**2
    
def find_neighbors(point3D, slices, dist_max, slicespacing):
    res = []
    dm = dist_max**2
    
    max_slice_dist = int(math.ceil(dist_max / slicespacing))
    for slicenum in range(point3D[0]-max_slice_dist, point3D[0]+max_slice_dist):
        if slicenum <0:
            continue
        if slicenum >len(slices)-1:
            continue
        for pt in slices[slicenum]:
            if get_dist_sq(point3D[2], pt[2])<dm:
                res.append(pt)
    
    # for pt in points_3D_data:
        # if get_dist_sq(point3D, pt[2])<dm:
            # res.append(pt)
    return res
    
def find_neighbors2(point3D, slices, dist_max, slicespacing):
    res = []
    dm = dist_max**2
    
    max_slice_dist = int(math.ceil(dist_max / slicespacing))
    for slicenum in range(point3D[0]-max_slice_dist, point3D[0]+max_slice_dist):
        if slicenum <0:
            continue
        if slicenum >len(slices)-1:
            continue
        if slicenum == point3D[0]:
            continue #there is no sense in checking points from the same slice, since they DO NOT MOVE wrt the current point
        for pt in slices[slicenum]:
            if get_dist_sq(point3D[2], pt[2])<dm:
                res.append(pt)
    
    # for pt in points_3D_data:
        # if get_dist_sq(point3D, pt[2])<dm:
            # res.append(pt)
    return res

    
def find_closest_next_slice(slice_num, point_num, point3D, slices, slice_m1_tree, slice_p1_tree):
    min_dist_p1 = 1000000000
    closest_p1 = -1
    min_dist_m1 = 1000000000
    closest_m1 = -1
    
                
    if (slice_num == 0) or (slice_m1_tree is None):
        #first slice - no bottom neighbor
        closest_m1 = point3D
    else:
        #odl = list(list(distance.cdist([point3D[2],], [p[2] for p in slices[slice_num-1]]))[0])
        #min_odl = min(odl)
        #ind_min = odl.index(min_odl)
        #closest_m1 = slices[slice_num-1][ind_min]
        
        
        [dist, vert] = slice_m1_tree.query(point3D[2], 1, distance_upper_bound = 15)#, workers = 5)
        if math.isinf(dist):
            [dist, vert] = slice_m1_tree.query(point3D[2], 1)
        closest_m1 = slices[slice_num-1][vert]
        
        
        #for p in slices[slice_num-1]:
        #    d = get_dist_sq(point3D[2], p[2])
        #    if d<min_dist_m1:
        #        closest_m1 = p
        #        min_dist_m1 = d
    #print("{} {}".format(slice_num, len(slices)))
    if (slice_num == len(slices)-1) or (slice_p1_tree is None):
        closest_p1 = point3D
    else:
        #odl = list(list(distance.cdist([point3D[2],], [p[2] for p in slices[slice_num+1]]))[0])
        #min_odl = min(odl)
        #ind_min = odl.index(min_odl)
        #closest_p1 = slices[slice_num+1][ind_min]
        
        [dist, vert] = slice_p1_tree.query(point3D[2], 1, distance_upper_bound = 15)#, workers = 5)
        if math.isinf(dist):
            [dist, vert] = slice_p1_tree.query(point3D[2], 1)
        closest_p1 = slices[slice_num+1][vert]
        
        #for p in slices[slice_num+1]:
        #    d = get_dist_sq(point3D[2], p[2])
        #    if d<min_dist_p1:
        #        closest_p1 = p
        #        min_dist_p1 = d
        
    if closest_m1 == -1:
        closest_m1 = point3D
    if closest_p1 == -1:
        closest_p1 = point3D
        
    return (closest_p1, closest_m1)
    
    
    
def find_neighbors_same_slice(slice_num, point_num, point3D, slices):
    try:
        n1 = slices[slice_num][point_num-1]
        n2 = slices[slice_num][(point_num+1)%len(slices[slice_num])]
        return (n1, n2)
    except Exception as e:
        print("Blad {}: {} {} {} {}".format(e, slice_num, point_num, len(slices), len(slices[slice_num]) if slice_num<len(slices) else "-"))
        sys.exit(1)
    
def transform_coords(pts, coords, p0):
    new_pts = []
    #acc = np.array([0,0,0])
    #for p in pts:
    #    acc[0] +=p[2][0]
    #    acc[1] +=p[2][1]
    #    acc[2] +=p[2][2]
    #acc[0] /= len(pts)
    #acc[1] /= len(pts)
    #acc[2] /= len(pts)
    
    #acc = np.array(p0)
    
    for p in pts:
        #c0 = np.dot(np.array(p[2])-acc, coords[0])#*coords[0]
        w = (p[2][0]-p0[0], p[2][1]-p0[1], p[2][2]-p0[2])
        c0 = np.dot(w, coords[0])
        #c1 = np.dot(np.array(p[2])-acc, coords[1])#*coords[1]
        c1 = np.dot(w, coords[1])
        #c2 = np.dot(np.array(p[2])-acc, coords[2])#*coords[2]
        c2 = np.dot(w, coords[2])
        new_pts.append([p[0], p[1], [c0,c1,c2]])
    return new_pts
    
    
def approximate_cubic(points):
    macierz_liczenie_LS = []
    wektor_liczenie_LS = []
    
    for p in points:
        c = p[2]
        macierz_liczenie_LS.append([c[1]**2, c[2]**2, c[0]*c[2], c[1], c[2],1])
        wektor_liczenie_LS.append(c[0])
    
    wspolczynniki = np.linalg.lstsq(macierz_liczenie_LS, wektor_liczenie_LS, rcond=None)
    return wspolczynniki[0]
    
def calc_curvature(w):
    # a geometrical approach for automatic shape restoration of the left ventrice (2013)
    a = w[0]
    b = w[2]
    c = w[1]
    d = w[3]
    e = w[4]
    A = math.sqrt(d**2+e**2+1)
    B = a+a*e**2+c+c*d**2+b*d*e
    k1 = (B+math.sqrt(B**2-A**2*(4*a*c-b**2)))/(A**3)
    k2 = (B-math.sqrt(B**2-A**2*(4*a*c-b**2)))/(A**3)
    
    return [k1, k2]
    
    


def calc_metric_sub2(params):
    slices = params[0]
    radius = params[1]
    slicespacing = params[2]
    startslice = params[3]
    stopslice = params[4]
    slice_num = params[5]
    decim = params[6]
    total = 0
    #first, calculate normals
    for snum in range(startslice, stopslice):
        if slices[snum]:
            #decim_factor = int(math.floor(len(slices[snum])/decim))
            #if decim_factor <1:
            #    decim_factor = 1
            decim_factor = decim
            for pnum in range(0,len(slices[snum]), decim_factor):
                punkt = slices[snum][pnum]
                w_normalny = calc_normal(punkt[0], punkt[1], punkt, slices)
                punkt[3][0] = w_normalny[0]
                punkt[3][1] = w_normalny[1]
                punkt[3][2] = w_normalny[2]
    #then, calculate sum of differences between current normal and all the other normals in the neighborhood
    #decim_factor = int(math.floor(len(slices[slice_num])/decim))
    #if decim_factor <1:
    #    decim_factor = 1
    decim_factor = decim
    for snum in range(startslice, stopslice):
        if slices[snum]:
            slicetotal = 0
            for pnum in range(0,len(slices[snum]), decim_factor):
                punkt = slices[snum][pnum]
                sasiedzi = find_neighbors2(punkt, slices, radius, slicespacing)
                sum_of_differences_point = 0
                for s in sasiedzi:
                    sum_of_differences_point += np.linalg.norm(np.array(punkt[3]) - np.array(s[3]))
                total += sum_of_differences_point
                slicetotal += sum_of_differences_point
            slices[snum][0][5] = slicetotal
    return total**2


def calc_metric_sub3(params):
    slices = params[0]
    radius = params[1]
    slicespacing = params[2]
    startslice = params[3]
    stopslice = params[4]
    slice_num = params[5]
    decim = params[6]
    total = 0
    #sum of distances to the closest points on the neighboring slices
    #decim_factor = int(math.floor(len(slices[slice_num])/decim))
    #if decim_factor <1:
    #    decim_factor = 1
    decim_factor = decim
    for snum in range(startslice, stopslice):
        if slices[snum]:
            slicetotal = 0
            for pnum in range(0,len(slices[snum]), decim_factor):
                punkt = slices[snum][pnum]
                sasiedzi = find_closest_next_slice(punkt[0], punkt[1], punkt, slices)
                #find_neighbors2(punkt, slices, radius, slicespacing)
                sum_of_distances_point = 0
                for s in sasiedzi:
                    sum_of_distances_point += np.linalg.norm(np.array(punkt[2]) - np.array(s[2]))
                total += sum_of_distances_point
                slicetotal += sum_of_distances_point
            slices[snum][0][5] = slicetotal
    return total
    
def calc_metric_sub4(params):
    slices = params[0]
    radius = params[1]
    slicespacing = params[2]
    startslice = params[3]
    stopslice = params[4]
    slice_num = params[5]
    decim = params[6]
    slice_trees = params[7]
    total = 0
    #decim_factor = int(math.floor(len(slices[slice_num])/decim))
    #if decim_factor <1:
    #    decim_factor = 1
    decim_factor = decim
    for snum in range(startslice, stopslice):
        if slices[snum]:
            slicetotal = 0
            for pnum in range(0,len(slices[snum]), decim_factor):
                punkt = slices[snum][pnum]
                #calculate average normals of the neighboring triangles for the current point
                normalne_trojkatow = calc_normal_neighbor_triangles(punkt[0], punkt[1], punkt, slices, slice_trees)
                #calculate difference between the averaged normals
                temp = np.linalg.norm(normalne_trojkatow[0] - normalne_trojkatow[1])
                total += temp
                slicetotal += temp
            slices[snum][0][5] = slicetotal
    return total

def calc_metric_sub5(params):
    slices = params[0]
    radius = params[1]
    slicespacing = params[2]
    startslice = params[3]
    stopslice = params[4]
    slice_num = params[5]
    decim = params[6]
    total = 0
    #decim_factor = int(math.floor(len(slices[slice_num])/decim))
    #if decim_factor <1:
    #    decim_factor = 1
    decim_factor = decim
    for snum in range(startslice, stopslice):
        if slices[snum]:
            slicetotal = 0
            maxval_for_slice = 0
            for pnum in range(0,len(slices[snum]), decim_factor):
                punkt = slices[snum][pnum]
                #calculate average normals of the neighboring triangles for the current point
                normalne_trojkatow = calc_normal_neighbor_triangles(punkt[0], punkt[1], punkt, slices)
                #calculate difference between the averaged normals
                temp = np.linalg.norm(normalne_trojkatow[0] - normalne_trojkatow[1])
                if temp > maxval_for_slice:
                    maxval_for_slice = temp
            slicetotal = maxval_for_slice
            total += maxval_for_slice
            slices[snum][0][5] = slicetotal
    return total

def calc_metric_sub6(params):
    slices = params[0]
    radius = params[1]
    slicespacing = params[2]
    startslice = params[3]
    stopslice = params[4]
    slice_num = params[5]
    decim = params[6]
    total = 0
    #decim_factor = int(math.floor(len(slices[slice_num])/decim))
    #if decim_factor <1:
    #    decim_factor = 1
    decim_factor = decim
    
    for snum in [slice_num]:#range(startslice, stopslice):
        if slices[snum]:
            slicetotal = 0
            maxval_for_slice = 0
            for pnum in range(0,len(slices[snum]), decim_factor):
                punkt = slices[snum][pnum]
                closest = find_closest_next_slice(snum, pnum, punkt, slices)
                
                midpoint = np.array(closest[0][2]) + np.array(closest[1][2])
                midpoint /= 2
                
                distance_to_midpoint = np.linalg.norm(np.array(punkt[2])-midpoint)
                
                if distance_to_midpoint > maxval_for_slice:
                    maxval_for_slice = distance_to_midpoint
            slicetotal = maxval_for_slice
            total += maxval_for_slice
            slices[snum][0][5] = slicetotal
    return total

def calc_movement_sub6(params):
    slices = params[0]
    radius = params[1]
    slicespacing = params[2]
    startslice = params[3]
    stopslice = params[4]
    slice_num = params[5]
    decim = params[6]
    step = params[7]
    total = 0
    #decim_factor = int(math.floor(len(slices[slice_num])/decim))
    #if decim_factor <1:
    #    decim_factor = 1
    decim_factor = decim
    max_point = -1
    closest_p1 = -1
    closest_m1 = -1
    for snum in [slice_num]:#range(startslice, stopslice):
        movement_vector = np.array([0,0])
        if slices[snum]:
            maxval_for_slice = 0
            for pnum in range(0,len(slices[snum]), decim_factor):
                punkt = slices[snum][pnum]
                closest = find_closest_next_slice(snum, pnum, punkt, slices)
                
                midpoint = np.array(closest[0][2]) + np.array(closest[1][2])
                midpoint /= 2
                distance_to_midpoint = np.linalg.norm(np.array(punkt[2])-midpoint)
                
                if distance_to_midpoint >= maxval_for_slice:
                    maxval_for_slice = distance_to_midpoint
                    movement_vector = (np.array(punkt[2])-midpoint) * step
                    max_point = pnum
                    closest_p1 = closest[0][2]
                    closest_m1 = closest[1][2]
    return [movement_vector, max_point, closest_p1, closest_m1]



def move_slice(slices, slice_num, ofs):
    for i in range(len(slices[slice_num])):
        slices[slice_num][i][2][0] += ofs[0]
        slices[slice_num][i][2][1] += ofs[1]
        

def choose_best_direction2(params):
    slices = params[0]
    radius = params[1]
    slicespacing = params[2]
    startslice = params[3]
    stopslice = params[4]
    slice_num = params[5]
    step = params[6]
    decim = params[7]
    norm = params[8]
    slice_trees = params[9]
    
    max_ofs_x = 20
    min_ofs_x = -20
    max_ofs_y = 20
    min_ofs_y = -20
    
    if norm == 2:
        sum_x = sum([x[0] for x in slices[slice_num][0][4]])
        sum_y = sum([x[1] for x in slices[slice_num][0][4]])
        zero_val  = calc_metric_sub2([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        
        
        val1 = 1000000000
        if  sum_x< max_ofs_x:
            move_slice(slices, slice_num, [step,0])#1,0
            val1 = calc_metric_sub2([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        
        val2 = 1000000000
        if  sum_x< max_ofs_x and sum_y<max_ofs_y:
            move_slice(slices, slice_num, [0,step])#1,1
            val2 = calc_metric_sub2([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        
        val3 = 1000000000
        if  sum_y<max_ofs_y:
            move_slice(slices, slice_num, [-step,0])#0,1
            val3 = calc_metric_sub2([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        
        val4 = 1000000000
        if  sum_x> min_ofs_x and sum_y<max_ofs_y:
            move_slice(slices, slice_num, [-step,0])#-1,1
            val4 = calc_metric_sub2([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        
        val5 = 1000000000
        if  sum_x> min_ofs_x:
            move_slice(slices, slice_num, [0,-step])#-1,0
            val5 = calc_metric_sub2([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        
        val6 = 1000000000
        if  sum_x> min_ofs_x and sum_y>min_ofs_y:
            move_slice(slices, slice_num, [0,-step])#-1,-1
            val6 = calc_metric_sub2([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        
        val7 = 1000000000
        if  sum_y>min_ofs_y:
            move_slice(slices, slice_num, [step,0])#0,-1
            val7 = calc_metric_sub2([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        
        val8 = 1000000000
        if  sum_x< max_ofs_x and sum_y>min_ofs_y:
            move_slice(slices, slice_num, [step,0])#1,-1
            val8 = calc_metric_sub2([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        
        min_val = min((zero_val, val1, val2, val3, val4, val5, val6, val7, val8))
        #print(min_val)
    
    elif norm == 3:
        zero_val  = calc_metric_sub3([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        sum_x = sum([x[0] for x in slices[slice_num][0][4]])
        sum_y = sum([x[1] for x in slices[slice_num][0][4]])
        
        val1 = 1000000000
        if  sum_x< max_ofs_x:
            move_slice(slices, slice_num, [step,0])#1,0
            val1 = calc_metric_sub3([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        
        val2 = 1000000000
        if  sum_x< max_ofs_x and sum_y<max_ofs_y:
            move_slice(slices, slice_num, [0,step])#1,1
            val2 = calc_metric_sub3([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        
        val3 = 1000000000
        if  sum_y<max_ofs_y:
            move_slice(slices, slice_num, [-step,0])#0,1
            val3 = calc_metric_sub3([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        
        val4 = 1000000000
        if  sum_x> min_ofs_x and sum_y<max_ofs_y:
            move_slice(slices, slice_num, [-step,0])#-1,1
            val4 = calc_metric_sub3([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        
        val5 = 1000000000
        if  sum_x> min_ofs_x:
            move_slice(slices, slice_num, [0,-step])#-1,0
            val5 = calc_metric_sub3([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        
        val6 = 1000000000
        if  sum_x> min_ofs_x and sum_y>min_ofs_y:
            move_slice(slices, slice_num, [0,-step])#-1,-1
            val6 = calc_metric_sub3([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        
        val7 = 1000000000
        if  sum_y>min_ofs_y:
            move_slice(slices, slice_num, [step,0])#0,-1
            val7 = calc_metric_sub3([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        
        val8 = 1000000000
        if  sum_x< max_ofs_x and sum_y>min_ofs_y:
            move_slice(slices, slice_num, [step,0])#1,-1
            val8 = calc_metric_sub3([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        
        min_val = min((zero_val, val1, val2, val3, val4, val5, val6, val7, val8))
        #print(min_val)
    
    elif norm == 4:
        zero_val  = calc_metric_sub4([slices, radius, slicespacing, startslice, stopslice, slice_num, decim, slice_trees])
        sum_x = sum([x[0] for x in slices[slice_num][0][4]])
        sum_y = sum([x[1] for x in slices[slice_num][0][4]])
        
        val1 = 1000000000
        if  sum_x< max_ofs_x:
            move_slice(slices, slice_num, [step,0])#1,0
            val1 = calc_metric_sub4([slices, radius, slicespacing, startslice, stopslice, slice_num, decim, slice_trees])
        
        val2 = 1000000000
        if  sum_x< max_ofs_x and sum_y<max_ofs_y:
            move_slice(slices, slice_num, [0,step])#1,1
            val2 = calc_metric_sub4([slices, radius, slicespacing, startslice, stopslice, slice_num, decim, slice_trees])
        
        val3 = 1000000000
        if  sum_y<max_ofs_y:
            move_slice(slices, slice_num, [-step,0])#0,1
            val3 = calc_metric_sub4([slices, radius, slicespacing, startslice, stopslice, slice_num, decim, slice_trees])
        
        val4 = 1000000000
        if  sum_x> min_ofs_x and sum_y<max_ofs_y:
            move_slice(slices, slice_num, [-step,0])#-1,1
            val4 = calc_metric_sub4([slices, radius, slicespacing, startslice, stopslice, slice_num, decim, slice_trees])
        
        val5 = 1000000000
        if  sum_x> min_ofs_x:
            move_slice(slices, slice_num, [0,-step])#-1,0
            val5 = calc_metric_sub4([slices, radius, slicespacing, startslice, stopslice, slice_num, decim, slice_trees])
        
        val6 = 1000000000
        if  sum_x> min_ofs_x and sum_y>min_ofs_y:
            move_slice(slices, slice_num, [0,-step])#-1,-1
            val6 = calc_metric_sub4([slices, radius, slicespacing, startslice, stopslice, slice_num, decim, slice_trees])
        
        val7 = 1000000000
        if  sum_y>min_ofs_y:
            move_slice(slices, slice_num, [step,0])#0,-1
            val7 = calc_metric_sub4([slices, radius, slicespacing, startslice, stopslice, slice_num, decim, slice_trees])
        
        val8 = 1000000000
        if  sum_x< max_ofs_x and sum_y>min_ofs_y:
            move_slice(slices, slice_num, [step,0])#1,-1
            val8 = calc_metric_sub4([slices, radius, slicespacing, startslice, stopslice, slice_num, decim, slice_trees])
        
        min_val = min((zero_val, val1, val2, val3, val4, val5, val6, val7, val8))
    
    
    elif (norm == 5):
        zero_val  = calc_metric_sub5([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        sum_x = sum([x[0] for x in slices[slice_num][0][4]])
        sum_y = sum([x[1] for x in slices[slice_num][0][4]])
        
        val1 = 1000000000
        if  sum_x< max_ofs_x:
            move_slice(slices, slice_num, [step,0])#1,0
            val1 = calc_metric_sub5([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        
        val2 = 1000000000
        if  sum_x< max_ofs_x and sum_y<max_ofs_y:
            move_slice(slices, slice_num, [0,step])#1,1
            val2 = calc_metric_sub5([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        
        val3 = 1000000000
        if  sum_y<max_ofs_y:
            move_slice(slices, slice_num, [-step,0])#0,1
            val3 = calc_metric_sub5([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        
        val4 = 1000000000
        if  sum_x> min_ofs_x and sum_y<max_ofs_y:
            move_slice(slices, slice_num, [-step,0])#-1,1
            val4 = calc_metric_sub5([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        
        val5 = 1000000000
        if  sum_x> min_ofs_x:
            move_slice(slices, slice_num, [0,-step])#-1,0
            val5 = calc_metric_sub5([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        
        val6 = 1000000000
        if  sum_x> min_ofs_x and sum_y>min_ofs_y:
            move_slice(slices, slice_num, [0,-step])#-1,-1
            val6 = calc_metric_sub5([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        
        val7 = 1000000000
        if  sum_y>min_ofs_y:
            move_slice(slices, slice_num, [step,0])#0,-1
            val7 = calc_metric_sub5([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        
        val8 = 1000000000
        if  sum_x< max_ofs_x and sum_y>min_ofs_y:
            move_slice(slices, slice_num, [step,0])#1,-1
            val8 = calc_metric_sub5([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        
        min_val = min((zero_val, val1, val2, val3, val4, val5, val6, val7, val8))
    
    elif (norm == 6):
        zero_val  = calc_metric_sub6([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        sum_x = sum([x[0] for x in slices[slice_num][0][4]])
        sum_y = sum([x[1] for x in slices[slice_num][0][4]])
        
        val1 = 1000000000
        if  sum_x< max_ofs_x:
            move_slice(slices, slice_num, [step,0])#1,0
            val1 = calc_metric_sub6([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        
        val2 = 1000000000
        if  sum_x< max_ofs_x and sum_y<max_ofs_y:
            move_slice(slices, slice_num, [0,step])#1,1
            val2 = calc_metric_sub6([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        
        val3 = 1000000000
        if  sum_y<max_ofs_y:
            move_slice(slices, slice_num, [-step,0])#0,1
            val3 = calc_metric_sub6([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        
        val4 = 1000000000
        if  sum_x> min_ofs_x and sum_y<max_ofs_y:
            move_slice(slices, slice_num, [-step,0])#-1,1
            val4 = calc_metric_sub6([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        
        val5 = 1000000000
        if  sum_x> min_ofs_x:
            move_slice(slices, slice_num, [0,-step])#-1,0
            val5 = calc_metric_sub6([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        
        val6 = 1000000000
        if  sum_x> min_ofs_x and sum_y>min_ofs_y:
            move_slice(slices, slice_num, [0,-step])#-1,-1
            val6 = calc_metric_sub6([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        
        val7 = 1000000000
        if  sum_y>min_ofs_y:
            move_slice(slices, slice_num, [step,0])#0,-1
            val7 = calc_metric_sub6([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        
        val8 = 1000000000
        if  sum_x< max_ofs_x and sum_y>min_ofs_y:
            move_slice(slices, slice_num, [step,0])#1,-1
            val8 = calc_metric_sub6([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        
        min_val = min((zero_val, val1, val2, val3, val4, val5, val6, val7, val8))

    
    
    if min_val == zero_val:
        move_slice(slices, slice_num, [-step,step])#0,0
        #slices[slice_num][0][4].append([0, 0])
        slices[slice_num][0][6] = False
        #recalculate slice metrics
        if norm == 2:
            calc_metric_sub2([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        elif norm == 3:
            calc_metric_sub3([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        elif norm == 4:
            calc_metric_sub4([slices, radius, slicespacing, startslice, stopslice, slice_num, decim, slice_trees])
        elif norm == 5:
            calc_metric_sub5([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        elif norm == 6:
            calc_metric_sub6([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        return False
    elif min_val == val1:
        move_slice(slices, slice_num, [0,step])#1,0
        slices[slice_num][0][4].append([step, 0])
    elif min_val == val2:
        move_slice(slices, slice_num, [0,2*step])#1,1
        slices[slice_num][0][4].append([step, step])
    elif min_val == val3:
        move_slice(slices, slice_num, [-step,2*step])#0,1
        slices[slice_num][0][4].append([0, step])
    elif min_val == val4:
        move_slice(slices, slice_num, [-2*step,2*step])#-1,1
        slices[slice_num][0][4].append([-step, step])
    elif min_val == val5:
        move_slice(slices, slice_num, [-2*step,step])#-1,0
        slices[slice_num][0][4].append([-step, 0])
    elif min_val == val6:
        move_slice(slices, slice_num, [-2*step,0])#-1,-1
        slices[slice_num][0][4].append([-step, -step])
    elif min_val == val7:
        move_slice(slices, slice_num, [-step,0])#0,-1
        slices[slice_num][0][4].append([0, -step])
    elif min_val == val8:
        slices[slice_num][0][4].append([step, -step]) #1,-1
        pass
    
    #recalculate slice metrics
    if norm == 2:
        calc_metric_sub2([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
    elif norm == 3:
        calc_metric_sub3([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
    elif norm == 4:
        calc_metric_sub4([slices, radius, slicespacing, startslice, stopslice, slice_num, decim, slice_trees])
    elif norm == 5:
        calc_metric_sub5([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
    elif norm == 6:
        calc_metric_sub6([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        
    #mark as worth chcecking
    if startslice <0:
        startslice = 0
    if stopslice > len(slices):
        stopslice = len(slices)
    for i in range(startslice, stopslice):
        if slices[i]:
            slices[i][0][6] = True
        
    return True
    
def choose_best_direction6(params):
    
    global cycles
    slices = params[0]
    radius = params[1]
    slicespacing = params[2]
    startslice = params[3]
    stopslice = params[4]
    slice_num = params[5]
    step = params[6]
    decim = params[7]
    norm = params[8]
    oDir = params[9]
    
    if not slices[slice_num]:
        return False
    
    dump_metrics("{}/koniec_cykl{}_debug.txt".format(oDir,cycles), slices)
    
    max_ofs_x = 20
    min_ofs_x = -20
    max_ofs_y = 20
    min_ofs_y = -20
        
    zero_val  = calc_metric_sub6([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
    sum_x = sum([x[0] for x in slices[slice_num][0][4]])
    sum_y = sum([x[1] for x in slices[slice_num][0][4]])
    
    [movement, maxpoint, cl_p1, cl_m1] = calc_movement_sub6([slices, radius, slicespacing, startslice, stopslice, slice_num, decim, step])
    
    punkty_ply = []
    
    for p in slices[slice_num]:
        if p[1] == maxpoint:
            punkty_ply.append([p[2][0], p[2][1], p[2][2], 255,0,0,0,0,0])
        else:
            punkty_ply.append([p[2][0], p[2][1], p[2][2], 255,255,255,0,0,0])
    punkty_ply.append([cl_p1[0], cl_p1[1], cl_p1[2], 0,255,100,100,0,0])
    punkty_ply.append([cl_m1[0], cl_m1[1], cl_m1[2], 0,255,100,100,0,0])
    new_pos = [slices[slice_num][maxpoint][2][0] - movement[0], slices[slice_num][maxpoint][2][1] - movement[1], slices[slice_num][maxpoint][2][2]]
    punkty_ply.append([new_pos[0], new_pos[1], new_pos[2], 0,255,100,100,0,0])
    
    write_ply("{}/koniec_cykl{}_debug.ply".format(oDir,cycles), punkty_ply)
    
    
    val1 = 1000000000

    move_slice(slices, slice_num, [-movement[0], -movement[1]])
    val1 = calc_metric_sub6([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
    
    min_val = min((zero_val, val1))

    
    
    if min_val == zero_val:
        move_slice(slices, slice_num, [movement[0], movement[1]])#0,0
        slices[slice_num][0][4].append([0, 0])
        #recalculate slice metrics
        calc_metric_sub6([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
        return False
    elif min_val == val1:
        slices[slice_num][0][4].append([-movement[0], -movement[1]])

    
    #recalculate slice metrics
    calc_metric_sub6([slices, radius, slicespacing, startslice, stopslice, slice_num, decim])
    calc_metric_sub6([slices, radius, slicespacing, startslice+1, stopslice+1, slice_num+1, decim])
    calc_metric_sub6([slices, radius, slicespacing, startslice-1, stopslice-1, slice_num-1, decim])
        
    return True
    
    
def try_adjust2(params):
    slices = params[0]
    radius = params[1]
    slicespacing = params[2]
    startslice = params[3]
    stopslice = params[4]
    slice_num = params[5]
    step = params[6]
    decim = params[7]
    norm = params[8]
    oDir = params[9]
    slice_trees = params[10]
    
    moved = False
    
    #zero_val  = calc_metric_sub2([slices, radius, slicespacing, startslice, stopslice, decim])#, "start{:04d}.ply".format(slice_num)])
    
    #below choose 1 or 2
    
    #1
    #move this slice until the improvement is possible
    
    # improved = True
    # nr_tury = 1
    # while improved:
        # #print("Tura {}".format(nr_tury))
        # nr_tury += 1
        # improved = choose_best_direction2([slices, radius, slicespacing, startslice, stopslice, slice_num, step, decim])
    
    #2
    #move this slice only one step in the best direction
    if (norm != 6) and slices[slice_num]:
        if slices[slice_num][0][6]:
            moved = choose_best_direction2([slices, radius, slicespacing, startslice, stopslice, slice_num, step, decim, norm, slice_trees])
        #moved = choose_best_direction2([slices, radius, slicespacing, startslice, stopslice, slice_num, step, decim, norm, slice_trees])
    if norm == 6:
        moved = choose_best_direction6([slices, radius, slicespacing, startslice, stopslice, slice_num, step, decim, norm, oDir])
    
    return moved
        
        
    #final_val = calc_metric_sub_ply([slices, radius, slicespacing, 0, len(slices), "koniec{:04d}.ply".format(slice_num)])
    
    
def calc_metric_sub_ply2(params):
    slices = params[0]
    radius = params[1]
    slicespacing = params[2]
    startslice = params[3]
    stopslice = params[4]
    filename = params[5]
    norm = params[6]
    slice_trees = params[7]
    
    total = 0
    no_of_points = 0
    no_of_slices = 0
    punkty_ply = []
    min_total_punkt = 99999999
    min_total_slice = 99999999
    max_total_punkt = -99999999
    max_total_slice = -99999999
    slices_sum_totals = 0
    
    if norm == 2:
        #first, calculate normals
        for snum in range(startslice, stopslice):
            decim_factor = 1#int(math.floor(len(slices[snum])/decim))
            if decim_factor <1:
                decim_factor = 1
            for pnum in range(0,len(slices[snum]), decim_factor):
                punkt = slices[snum][pnum]
                w_normalny = calc_normal(punkt[0], punkt[1], punkt, slices)
                punkt[3][0] = w_normalny[0]
                punkt[3][1] = w_normalny[1]
                punkt[3][2] = w_normalny[2]
        #then, calculate sum of differences between current normal and all the other normals in the neighborhood
        for snum in range(startslice, stopslice):
            if slices[snum]:
                #only for non-empty slices
                decim_factor = 1#int(math.floor(len(slices[snum])/decim))
                if decim_factor <1:
                    decim_factor = 1
                slicetotal = 0
                for pnum in range(0,len(slices[snum]), decim_factor):
                    punkt = slices[snum][pnum]
                    sasiedzi = find_neighbors2(punkt, slices, radius, slicespacing)
                    sum_of_differences_point = 0
                    for s in sasiedzi:
                        sum_of_differences_point += np.linalg.norm(np.array(punkt[3]) - np.array(s[3]))
                    total += sum_of_differences_point
                    slicetotal += sum_of_differences_point
                    
                    no_of_points += 1
                        
                    if sum_of_differences_point < min_total_punkt:
                        min_total_punkt = sum_of_differences_point
                    if sum_of_differences_point > max_total_punkt:
                        max_total_punkt = sum_of_differences_point
                    #punkty_ply.append([punkt[2][0], punkt[2][1], punkt[2][2], total_punkt, 255 if k1_temp>k2_temp else total_punkt, 255 if k2_temp>k1_temp else total_punkt])#total_punkt, total_punkt])
                    punkty_ply.append([punkt[2][0], punkt[2][1], punkt[2][2], sum_of_differences_point, sum_of_differences_point, sum_of_differences_point, punkt[3][0], punkt[3][1], punkt[3][2]])
                    #with np.printoptions(precision=3, suppress=True):
                    #    print("{} : {:0.4f}, {:0.4f} wspolcz: {}".format(punkt, k1, k2, wspolczynniki))
                #store the total metric for a slice in the 0th point
                no_of_slices += 1
                slices[snum][0][5] = slicetotal
                if slicetotal < min_total_slice:
                    min_total_slice = slicetotal
                if slicetotal > max_total_slice:
                    max_total_slice = slicetotal
                slices_sum_totals += slicetotal
                        
                        
    if norm == 3:
        decim_factor = 1#int(math.floor(len(slices[snum])/decim))
        if decim_factor <1:
            decim_factor = 1
        for snum in range(startslice, stopslice):
            if slices[snum]:
                slicetotal = 0
                for pnum in range(0,len(slices[snum]), decim_factor):
                    punkt = slices[snum][pnum]
                    sasiedzi = find_closest_next_slice(punkt[0], punkt[1], punkt, slices)
                    #find_neighbors2(punkt, slices, radius, slicespacing)
                    sum_of_distances_point = 0
                    for s in sasiedzi:
                        sum_of_distances_point += np.linalg.norm(np.array(punkt[2]) - np.array(s[2]))
                    total += sum_of_distances_point
                    slicetotal += sum_of_distances_point
                    no_of_points += 1
                    if sum_of_distances_point < min_total_punkt:
                        min_total_punkt = sum_of_distances_point
                    if sum_of_distances_point > max_total_punkt:
                        max_total_punkt = sum_of_distances_point
                    punkty_ply.append([punkt[2][0], punkt[2][1], punkt[2][2], sum_of_distances_point, sum_of_distances_point, sum_of_distances_point, punkt[3][0], punkt[3][1], punkt[3][2]])
                #store the total metric for a slice in the 0th point
                no_of_slices += 1
                slices[snum][0][5] = slicetotal
                if slicetotal < min_total_slice:
                    min_total_slice = slicetotal
                if slicetotal > max_total_slice:
                    max_total_slice = slicetotal
                slices_sum_totals += slicetotal
                
    if norm == 4:
        decim_factor = 1#int(math.floor(len(slices[snum])/decim))
        if decim_factor <1:
            decim_factor = 1
        for snum in range(startslice, stopslice):
            if slices[snum]:
                slicetotal = 0
                for pnum in range(0,len(slices[snum]), decim_factor):
                    punkt = slices[snum][pnum]
                    #calculate average normals of the neighboring triangles for the current point
                    normalne_trojkatow = calc_normal_neighbor_triangles(punkt[0], punkt[1], punkt, slices, slice_trees)
                    #calculate difference between the averaged normals
                    sum_of_differences_point = np.linalg.norm(normalne_trojkatow[0] - normalne_trojkatow[1])
                    total += sum_of_differences_point
                    slicetotal += sum_of_differences_point
                
                    no_of_points += 1
                        
                    if sum_of_differences_point < min_total_punkt:
                        min_total_punkt = sum_of_differences_point
                    if sum_of_differences_point > max_total_punkt:
                        max_total_punkt = sum_of_differences_point
                    punkty_ply.append([punkt[2][0], punkt[2][1], punkt[2][2], sum_of_differences_point, sum_of_differences_point, sum_of_differences_point, punkt[3][0], punkt[3][1], punkt[3][2]])
                #store the total metric for a slice in the 0th point
                no_of_slices += 1
                slices[snum][0][5] = slicetotal
                if slicetotal < min_total_slice:
                    min_total_slice = slicetotal
                if slicetotal > max_total_slice:
                    max_total_slice = slicetotal
                slices_sum_totals += slicetotal
                
    if norm == 5:
        decim_factor = 1#int(math.floor(len(slices[snum])/decim))
        if decim_factor <1:
            decim_factor = 1
        for snum in range(startslice, stopslice):
            if slices[snum]:
                slicetotal = 0
                maxval_for_slice = 0
                for pnum in range(0,len(slices[snum]), decim_factor):
                    punkt = slices[snum][pnum]
                    #calculate average normals of the neighboring triangles for the current point
                    normalne_trojkatow = calc_normal_neighbor_triangles(punkt[0], punkt[1], punkt, slices)
                    #calculate difference between the averaged normals
                    sum_of_differences_point = np.linalg.norm(normalne_trojkatow[0] - normalne_trojkatow[1])
                    no_of_points += 1
                    
                    if sum_of_differences_point > maxval_for_slice:
                        maxval_for_slice = sum_of_differences_point
                        
                    if sum_of_differences_point < min_total_punkt:
                        min_total_punkt = sum_of_differences_point
                    if sum_of_differences_point > max_total_punkt:
                        max_total_punkt = sum_of_differences_point
                    punkty_ply.append([punkt[2][0], punkt[2][1], punkt[2][2], sum_of_differences_point, sum_of_differences_point, sum_of_differences_point, punkt[3][0], punkt[3][1], punkt[3][2]])
                total += maxval_for_slice * no_of_points
                slicetotal = maxval_for_slice * no_of_points
                #store the total metric for a slice in the 0th point
                no_of_slices += 1
                slices[snum][0][5] = slicetotal
                if slicetotal < min_total_slice:
                    min_total_slice = slicetotal
                if slicetotal > max_total_slice:
                    max_total_slice = slicetotal
                slices_sum_totals += slicetotal
                
    if norm == 6:
        decim_factor = 1
        
        for snum in range(startslice, stopslice):
            if slices[snum]:
                slicetotal = 0
                maxval_for_slice = 0
                punkty_ply_temp = []
                for pnum in range(0,len(slices[snum]), decim_factor):
                    punkt = slices[snum][pnum]
                    closest = find_closest_next_slice(snum, pnum, punkt, slices)
                    
                    midpoint = np.array(closest[0][2]) + np.array(closest[1][2])
                    midpoint /= 2
                    
                    distance_to_midpoint = np.linalg.norm(np.array(punkt[2])-midpoint)
                    
                    #calculate average normals of the neighboring triangles for the current point
                    #normalne_trojkatow = calc_normal_neighbor_triangles(punkt[0], punkt[1], punkt, slices)
                    #calculate difference between the averaged normals
                    #sum_of_differences_point = np.linalg.norm(normalne_trojkatow[0] - normalne_trojkatow[1])
                    
                    
                    if distance_to_midpoint >= maxval_for_slice:
                        maxval_for_slice = distance_to_midpoint
                        
                    if distance_to_midpoint < min_total_punkt:
                        min_total_punkt = distance_to_midpoint
                    if distance_to_midpoint > max_total_punkt:
                        max_total_punkt = distance_to_midpoint
                    
                    
                    punkty_ply_temp.append([punkt[2][0], punkt[2][1], punkt[2][2], distance_to_midpoint, distance_to_midpoint, distance_to_midpoint, punkt[3][0], punkt[3][1], punkt[3][2]])
                    
                for p in punkty_ply_temp:
                    if p[3] == maxval_for_slice:
                        punkty_ply.append([p[0],p[1],p[2],500,0,0,p[6],p[7],p[8]])
                    else:
                        punkty_ply.append([p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8]])
                total += maxval_for_slice
                slicetotal = maxval_for_slice
                #store the total metric for a slice in the 0th point
                no_of_slices += 1
                slices[snum][0][5] = slicetotal
                if slicetotal < min_total_slice:
                    min_total_slice = slicetotal
                if slicetotal > max_total_slice:
                    max_total_slice = slicetotal
                slices_sum_totals += slicetotal
        no_of_points = no_of_slices
        
        
    avg_total_punkt = total / no_of_points
    avg_total_slice = slices_sum_totals / no_of_slices
    
    val_for_0 = avg_total_punkt - min_total_punkt
    val_for_255 = 3*avg_total_punkt
    
    if 3*avg_total_punkt > max_total_punkt:
        val_for_255 = max_total_punkt
    
    #logging.info("Metric POINT: {}/{}/{} SLICE:{}/{}/{}".format(min_total_punkt, avg_total_punkt, max_total_punkt, min_total_slice, avg_total_slice,max_total_slice))
    for i in range(len(punkty_ply)):
        # punkty_ply[i][3] = int((punkty_ply[i][3]-min_total_punkt)/(max_total_punkt-min_total_punkt)*255)
        # punkty_ply[i][4] = int((punkty_ply[i][4]-min_total_punkt)/(max_total_punkt-min_total_punkt)*255)
        # punkty_ply[i][5] = int((punkty_ply[i][5]-min_total_punkt)/(max_total_punkt-min_total_punkt)*255)
        punkty_ply[i][3] = int((punkty_ply[i][3]-val_for_0)/(val_for_255)*255)
        punkty_ply[i][4] = int((punkty_ply[i][4]-val_for_0)/(val_for_255)*255)
        punkty_ply[i][5] = int((punkty_ply[i][5]-val_for_0)/(val_for_255)*255)
        
        if punkty_ply[i][3] >255:
            punkty_ply[i][3] = 255
        if punkty_ply[i][4] >255:
            punkty_ply[i][4] = 255
        if punkty_ply[i][5] >255:
            punkty_ply[i][5] = 255
        if punkty_ply[i][3] <0:
            punkty_ply[i][3] = 0
        if punkty_ply[i][4] <0:
            punkty_ply[i][4] = 0
        if punkty_ply[i][5] <0:
            punkty_ply[i][5] = 0
    
    if filename is not None: 
        #print("zapisuje")
        write_ply(filename, punkty_ply)
    return total

def dump_metrics(filename, slices):
    suma = 0
    fh = open(filename,"w+")
    fh.write("slice#\t\tpt.num\t\tmetric\n")
    for i in range(len(slices)):
        fh.write("{}\t\t{}\t\t".format(i, len(slices[i])))
        if slices[i]:
            fh.write("{:.4f}".format(slices[i][0][5]))
            suma += slices[i][0][5]
        fh.write("\n")
    fh.write("metric sum:{}".format(suma))
    
    fh.close()
    
def write_ply(filename, data):
    
    fh = open(filename,"w+")
    if len(data[0]) == 6:
        #x,y,z, r,g,b
        #print("{}/{}".format(os.getcwd(), filename))
        fh.write("ply\n")
        fh.write("format ascii 1.0\n")
        fh.write("element vertex {}\n".format(len(data)))
        fh.write("property float x\n")
        fh.write("property float y\n")
        fh.write("property float z\n")
        fh.write("property uchar red\n")
        fh.write("property uchar green\n")
        fh.write("property uchar blue\n")
        fh.write("end_header\n")
        
        for i in range(len(data)):
            fh.write("{} {} {} {} {} {}\n".format(data[i][0],data[i][1],data[i][2],data[i][3],data[i][4],data[i][5]))
    if len(data[0]) == 9:
        #x,y,z, normal vector, r,g,b 
        #print("{}/{}".format(os.getcwd(), filename))
        fh.write("ply\n")
        fh.write("format ascii 1.0\n")
        fh.write("element vertex {}\n".format(len(data)))
        fh.write("property float x\n")
        fh.write("property float y\n")
        fh.write("property float z\n")
        fh.write("property float nx\n")
        fh.write("property float ny\n")
        fh.write("property float nz\n")
        fh.write("property uchar red\n")
        fh.write("property uchar green\n")
        fh.write("property uchar blue\n")
        fh.write("end_header\n")
        
        for i in range(len(data)):
            fh.write("{} {} {} {} {} {} {} {} {}\n".format(data[i][0],data[i][1],data[i][2], data[i][6],data[i][7],data[i][8], data[i][3],data[i][4],data[i][5]))
    
    fh.close()
    
    
def make_polygon_dense(poly):
    newpoly = poly
    added = 0
    for i in range(len(poly)-1):
        if poly[i][0]==poly[i+1][0]:
            new =[]
            if poly[i][1]<poly[i+1][1]:
            #increasing y
                for j in range(1,poly[i+1][1]-poly[i][1]):
                    new.append([poly[i][0], poly[i][1]+j])
            else:
            #decreasing y
                for j in range(1,poly[i][1]-poly[i+1][1]):
                    new.append([poly[i][0], poly[i][1]-j])
            newpoly = [*newpoly[0:i+1+added], *new, *newpoly[i+1+added:]]
            added = added + len(new)
        if poly[i][1]==poly[i+1][1]:
            new =[]
            if poly[i][0]<poly[i+1][0]:
            #increasing x
                for j in range(1,poly[i+1][0]-poly[i][0]):
                    new.append([poly[i][0]+j, poly[i][1]])
            else:
            #decreasing x
                for j in range(1,poly[i][0]-poly[i+1][0]):
                    new.append([poly[i][0]-j, poly[i][1]])
            newpoly = [*newpoly[0:i+1+added], *new, *newpoly[i+1+added:]]
            added = added + len(new)
    #last point
    if poly[-1][0]==poly[0][0]:
        new =[]
        if poly[-1][1]<poly[0][1]:
        #increasing y
            for j in range(1,poly[0][1]-poly[-1][1]):
                new.append([poly[-1][0], poly[-1][1]+j])
        else:
        #decreasing y
            for j in range(1,poly[-1][1]-poly[0][1]):
                new.append([poly[-1][0], poly[-1][1]-j])
        newpoly = [*newpoly, *new]
    if poly[-1][1]==poly[0][1]:
        new =[]
        if poly[-1][0]<poly[0][0]:
        #increasing x
            for j in range(1,poly[0][0]-poly[-1][0]):
                new.append([poly[-1][0]+j, poly[-1][1]])
        else:
        #decreasing x
            for j in range(1,poly[-1][0]-poly[0][0]):
                new.append([poly[-1][0]-j, poly[-1][1]])
        newpoly = [*newpoly, *new]
    return newpoly
    
    
def sortowanie_slice(wpis):
    return wpis[0]
    
    
def process_dir(iDir, ishDir, oDir, imgDir, log2, ss, psx, psy, verbose, quit_on_small_margin, norm, move_step, PerformCorrections, decim):
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    poly_list    = glob.glob(ishDir + "/*_skin_polygons.json")
    poly_list.sort()
    
    global cycles

    pokaz_wykresy = False
    focus_range_in_ss = 1
    #cycle_num = 100
    #move_step = 0.2
    
    number = 0
    corrections = []
    images_exist = False
    #if os.path.isdir(iDir):
    #    images_exist = True
    
    
    label_list    = glob.glob(iDir + "/*_roi_labels.png")
    if label_list:
        images_exist = True

        label_list.sort()
        
        
        for label_file in label_list:
            #logging.info("opening file: {}".format(label_file))
            image_raw = cv.imread(label_file, cv.IMREAD_COLOR)
            image_raw = image_raw[:,:,2]
            image_raw[image_raw != 255] = 0
            
            minx = 100000000
            miny = 100000000
            maxx = -1
            maxy = -1
            num_found = 0
            
            
            for j in range(0,image_raw.shape[0]):
                for i in range(0,image_raw.shape[1]):
                    if image_raw[j,i] == 255:
                        num_found += 1
                        if j>maxy:
                            maxy = j
                        if j<miny:
                            miny = j
                        if i>maxx:
                            maxx = i
                        if i<minx:
                            minx = i
            if num_found>0:
                maxx *= psx
                minx *= psx
                maxy *= psy
                miny *= psy
                if quit_on_small_margin:
                    if (minx<3) or (miny<3) or (maxx>image_raw.shape[1]*psx-3) or (maxy>image_raw.shape[0]*psy-3):
                        logging.error("too small margin detected in ROI")
                        #exit(1)
            corrections.append([number,0,0, image_raw.shape[1], image_raw.shape[0]])
            number += 1
        
    

    
    number = 0
    points_3D_data = []
#    points3Draw = []
#    points3Draw.append([])
#    points3Draw.append([])
#    points3Draw.append([])
    slice_centres = []
    slices = []
    slices_orig = []
    slice_num = 0
    
    for poly_file in poly_list:
        #logging.info("opening file: {}".format(poly_file))
        if not images_exist:
            corrections.append([number,0,0,0,0])
            number += 1
        try:
            poly_h = open(poly_file)
            slice_data = json.load(poly_h)
            poly_h.close()
        except Exception as err:
            logging.error("Input data IO error: {}".format(err))
            sys.exit(1)
        
        #check if the array is not empty
        if not slice_data['polygons']:
            slice_centres.append([0,0])
            slices.append([])
            slices_orig.append([])
            slice_num += 1
            continue
        if not slice_data['polygons'][0]['outer']['path']:
            slice_centres.append([0,0])
            slices.append([])
            slices_orig.append([])
            slice_num += 1
            continue
        biggest_path = make_polygon_dense(slice_data['polygons'][0]['outer']['path'])
#        slice_num = int(os.path.basename(poly_file).split('_',1)[0])-1
        point_num = 0
        #print("Dodaje slice numer {} ".format(slice_num), end="")
        
        sum_x = 0
        sum_y = 0
        slices.append([])
        slices_orig.append([])
        for point in biggest_path:
            points_3D_data.append([slice_num, point_num, [point[0]*psx, point[1]*psy, slice_num*ss]])
            #print([slice_num, point_num,[point[0]*psx, point[1]*psy, slice_num*ss]])
            
            #slice number, number of point within the slice, xyz coordinates, normal vector (initialized as 0), offset from original (empty list), metric of the whole slice (for point 0), whether slice is worth checking
            slices[-1].append([slice_num, point_num,[point[0]*psx, point[1]*psy, slice_num*ss], [0,0,0], [], 0, True])
            slices_orig[-1].append([slice_num, point_num,[point[0]*psx, point[1]*psy, slice_num*ss], [0,0,0], [], 0])
#            points3Draw[0].append(point[0]*psx)
#            points3Draw[1].append(point[1]*psy)
            sum_x += point[0]*psx
            sum_y += point[1]*psy
#            points3Draw[2].append(slice_num*ss)
            point_num += 1
        slice_centres.append([sum_x/len(biggest_path), sum_y/len(biggest_path)])
        #print("ma {} punktow".format(point_num))
        #print("{} {} {}".format(len(slices), len(slices[-1]), len(points_3D_data)))
        slice_num += 1
        
    czas_start = time.time()
    
    ofs = []
    
    for i in range(2*(len(slices)-1)):
        ofs.append(0)
        
    #calc_metric_sub_ply([slices, 2*ss, ss, 0, len(slices), "plik1.ply"])
    #
    #for i in range(len(slices[1])):
    #    slices[1][i][2][0] -= 0.5
    #    slices[1][i][2][1] -= 0.5
    #calc_metric_sub_ply([slices, 2*ss, ss, 0, len(slices), "plik2.ply"])
    #for i in range(len(slices[1])):
    #    slices[1][i][2][0] -= 0.5
    #    slices[1][i][2][1] -= 0.5
    #calc_metric_sub_ply([slices, 2*ss, ss, 0, len(slices), "plik3.ply"])
    #for i in range(len(slices[1])):
    #    slices[1][i][2][0] -= 0.5
    #    slices[1][i][2][1] -= 0.5
    #calc_metric_sub_ply([slices, 2*ss, ss, 0, len(slices), "plik4.ply"])
    #for i in range(len(slices[1])):
    #    slices[1][i][2][0] -= 0.5
    #    slices[1][i][2][1] -= 0.5
    #calc_metric_sub_ply([slices, 2*ss, ss, 0, len(slices), "plik5.ply"])
    
    #slice_trees = [cKDTree([p[2] for p in s ]) for s in slices]
    
    slice_trees = [cKDTree([p[2] for p in s ], leafsize = 225) if s else None for s in slices]
    
    logging.info("starting metric: {}".format(calc_metric_sub_ply2([slices, focus_range_in_ss*ss, ss, 0, len(slices), "{}/poczatek.ply".format(oDir), norm, slice_trees])))
    slice_metric_histogram = []
    cycle_num = len(slices) * 5
    stage = 1
    previous_signatures = []
    for cycles in range(cycle_num):
        logging.info("Starting cycle {}".format(cycles))
        num_moved_slices = 0
        max_slice_metric = 0
        max_slice_metric_idx = -1
        slice_metrics = []
        slice_metrics_index = list(range(len(slices)))
        slice_metrics = [(slices[i][0][5], i) for i in slice_metrics_index if (slices[i] and slices[i][0][6])]
        
        line = ["1" if ((slices[i]) and (slices[i][0][6]))  else "0" for i in range(len(slices))]
        #for i in range(len(slices)):
        #    if slices[i]:
        #        if slices[i][0][6]:
        #            print("1",end="")
        #        else:
        #            print("0",end="")
        logging.info("".join(line))
        
#        for i in range(0,len(slices)-1):
#            if slices[i]:
#                slice_metrics.append((slices[i][0][5], i))
                #if slices[i][0][5] > max_slice_metric:
                #    max_slice_metric = slices[i][0][5]
                #    max_slice_metric_idx = i
        #slice_metric_histogram.append([m[0] for m in slice_metrics])
        slice_metrics.sort(reverse = True, key=sortowanie_slice)
        slice_metrics_orig_len = len(slice_metrics)
        
        #if (cycles == 0) or (cycles == cycle_num-1):
        #    for s in slice_metrics:
        #        logging.debug(s)
        while (num_moved_slices==0) and slice_metrics:
            max_slice_metric_idx = slice_metrics[0][1]
            slice_metrics.pop(0)
            
            minsliceno = max_slice_metric_idx-(focus_range_in_ss)
            if minsliceno <0:
                minsliceno = 0
            maxsliceno = max_slice_metric_idx+(focus_range_in_ss)
            if maxsliceno > len(slices):
                maxsliceno = len(slices)
                
            # slices = params[0]
            # radius = params[1]
            # slicespacing = params[2]
            # startslice = params[3]
            # stopslice = params[4]
            # slice_num = params[5]
            # step = params[6]
            # decim = params[7]
            
            
            slice_trees = [cKDTree([p[2] for p in s ], leafsize = 225) if s else None for s in slices]
            logging.info("Trying slice {}".format(max_slice_metric_idx))
            if try_adjust2([slices, focus_range_in_ss*ss, ss, minsliceno, maxsliceno, max_slice_metric_idx, move_step, decim, norm, oDir, slice_trees]):
                num_moved_slices+= 1
                logging.info("Moved the slice no {} in order by {}".format(slice_metrics_orig_len - len(slice_metrics), slices[max_slice_metric_idx][0][4][-1]))
#        logging.info("intermediate metric: {}, {:03.1f}% moved".format(calc_metric_sub_ply2([slices, focus_range_in_ss*ss, ss, 0, len(slices), "{}/koniec_cykl{}.ply".format(oDir, cycles), norm]), num_moved_slices/len(slices)*100))
        
        #calculate signature (should prevent getting stuck in a loop)
        moves = [a[0][4] for a in slices if a]
        sum_moves = [0.0] * 2*len(slices)
        for i,m in enumerate(moves):
            if m:
                a = sum([x[0]*1.0 for x in m])
                sum_moves[i*2] = a
                a = sum([x[1]*1.0 for x in m])
                sum_moves[i*2+1] = a
        signature = "".join(["{:2.2f} ".format(sm) for sm in sum_moves])
        #logging.info("Signature: {}".format(signature))
        
        
        if (num_moved_slices == 0):
            #nothing changes, so we are in a local minimum
            if stage < 3:
                stage += 1
                move_step /= 2
                logging.info("changing step to {}".format(move_step))
                for i in range(len(slices)):
                    if slices[i]:
                        slices[i][0][6] = True
                previous_signatures = []
            else:
                logging.info("No further movements improve metric - finishing")
                break
        
        if signature in previous_signatures:
            logging.info("Signature repeats - ending")
            break
        else:
            previous_signatures.append(signature)
                
        for i,s in enumerate(slices):
            if not s:
                continue
            try:
                corrections[i][1] = -int(round((slices_orig[i][0][2][0]-s[0][2][0])/psx))
                corrections[i][2] = -int(round((slices_orig[i][0][2][1]-s[0][2][1])/psy))
            except Exception as e:
                print(e)
                print("i: {}\ncorrections: {}\nslices_orig: {}\ns: {}".format(i, corrections, slices_orig, s))
                sys.exit(1)
            
    logging.info("Corrections:")
    for i,c in enumerate(corrections):
        logging.info("slice {}:{}".format(i, c))
    
    hist_vertices = []
    
    #for i,s in enumerate(slice_metric_histogram):
    #    max_hist = np.amax(s)
    #    for j,ss in enumerate(s):
    #        if ss==max_hist:
    #            hist_vertices.append([i,j,ss/max_hist*100,255,0,0])
    #        else:
    #            hist_vertices.append([i,j,ss/max_hist*100,255,255,255])
    
    #write_ply("{}/histogram.ply".format(oDir), hist_vertices)
    
    
    logging.info("finished metric: {}".format(calc_metric_sub_ply2([slices, focus_range_in_ss*ss, ss, 0, len(slices), "{}/koniec.ply".format(oDir), norm, slice_trees])))
    # logging.info("CORRECTIONS:")
    # for sl in slices:
        # if sl:
            # logging.info("  Slice {}:".format(sl[0][0]))
            # suma = [0,0]
            # for of in sl[0][4]:
                # logging.info("    {}".format(of))
                # suma[0] += of[0]
                # suma[1] += of[1]
            # logging.info("sum:{}".format(suma))
    
    
#    #wynik = sp.minimize(minimization_function, ofs, args=(slices, 2*ss, ss, 5), method = 'L-BFGS-B', options={'iprint':111, 'eps':1e-10, 'ftol':1e-7})
#    wynik = sp.minimize(minimization_function, ofs, args=(slices, 2*ss, ss, 5), method = 'SLSQP', options={'iprint':1, 'eps':1e-10, 'ftol':1e-7})
#        
#    #print(wynik.x)
#    print(wynik.success)
#    x = wynik.x
#    
#    
#    #x = np.array([  2.58371168,  -4.4577884,    4.11200632,  -5.58071039,   8.98490033, -17.66885867,  -5.77946386,   7.44620513,  -5.73285448,   7.56783103, -5.5241306,   16.28842053,   2.49826757,  -3.04423172, -4.24831487, 9.57445745,  -4.31326028, 5.76091189])
#    
#    #x = ofs
#    #print(calc_metric(points_3D_data, slices, 30))
#    czas_stop = time.time()
#    print("czas wykonania: {} minut".format((czas_stop-czas_start)/60))
#    offsets = np.append(x, [0,0])
#    offsets = offsets.reshape((len(slices),2))
#    
#    offsets_pix = []
#    for o in offsets:
#        offsets_pix.append([int(o[0]/psx+0.5), int(o[1]/psy+0.5)])
#    print("Ofsety do dodania do kolejnych sliceow w pikselach:\n{}".format(offsets_pix))
    
    
    if pokaz_wykresy == True:
        
        # plaszczyzna_x = []
        # plaszczyzna_y = []
        # plaszczyzna_z = []
        
        
        # for xi in range(-100,100):
            # plaszczyzna_x.append([])
            # plaszczyzna_y.append([])
            # plaszczyzna_z.append([])
            # for yi in range(-100,100):
                # x = xi/10.0
                # y = yi/10.0
                # plaszczyzna_x[-1].append(x)
                # plaszczyzna_y[-1].append(y)
                # plaszczyzna_z[-1].append(wspolczynniki[0]*x**2 + wspolczynniki[1]*y**2 + wspolczynniki[2]*x*y + wspolczynniki[3]*x + wspolczynniki[4]*y + wspolczynniki[5])
        
        
        fig = plt.figure()
        #ax = fig.add_subplot(121, projection='3d')
        #ax2 = fig.add_subplot(122, projection='3d')
        
        ax = fig.add_subplot(111, projection='3d')
        
        #ax2 = fig.add_subplot(111, projection='3d')
        
        #ax.scatter(points3Draw[0], points3Draw[1], points3Draw[2])
        #ax.scatter([i[0] for i in slice_centres], [i[1] for i in slice_centres], [i*ss for i in range(len(slice_centres))],c='r')
        
        pokaz(ax, points_3D_data, c='r', ofset = [0,0,0])
        
        points_3D_data2 = add_offset_to_slices(points_3D_data, x)
        
        pokaz(ax, points_3D_data2, c='g', ofset = [0,0,0])
        
        #pokaz(ax, sasiedzi2,c='g', ofset = punkt2[2])
        
        #pokaz(ax2, nowe_punkty,c='r')
        #pokaz(ax2, nowe_punkty2,c='g')
        
        #ax.quiver(punkt[2][0], punkt[2][1], punkt[2][2], nowy_uklad[0][0], nowy_uklad[0][1], nowy_uklad[0][2], colors='b')
        #ax.quiver(punkt[2][0], punkt[2][1], punkt[2][2], nowy_uklad[1][0], nowy_uklad[1][1], nowy_uklad[1][2], colors='b')
        #ax.quiver(punkt[2][0], punkt[2][1], punkt[2][2], nowy_uklad[2][0], nowy_uklad[2][1], nowy_uklad[2][2], colors='b')
        
        
        #ax.quiver(0, 0, 0, nowy_uklad[0][0], nowy_uklad[0][1], nowy_uklad[0][2], colors='r')
        #ax.quiver(0, 0, 0, nowy_uklad[1][0], nowy_uklad[1][1], nowy_uklad[1][2], colors='g')
        #ax.quiver(0, 0, 0, nowy_uklad[2][0], nowy_uklad[2][1], nowy_uklad[2][2], colors='b')
        
        #ax2.quiver(0, 0, 0, 1, 0, 0, colors='r')
        #ax2.quiver(0, 0, 0, 0, 1, 0, colors='g')
        #ax2.quiver(0, 0, 0, 0, 0, 1, colors='b')
        
        #ax2.plot_surface(np.array(plaszczyzna_z), np.array(plaszczyzna_x), np.array(plaszczyzna_y))
        
        ax.set_xlim(-10,10)
        ax.set_ylim(-10,10)
        ax.set_zlim(-10,10)
        
        #ax2.set_xlim(-10,10)
        #ax2.set_ylim(-10,10)
        #ax2.set_zlim(-10,10)
        
        # pokaz(ax2, pp1, c='r')
        # pokaz(ax2, np1, c='g')
        # pokaz2(ax2, sas1[1], c='b')
        # ax2.set_xlim(-20,20)
        # ax2.set_ylim(-20,20)
        # ax2.set_zlim(-20,20)
        
        #ax.scatter([i[2][0] for i in nowe_punkty], [i[2][1] for i in nowe_punkty], [i[2][2] for i in nowe_punkty],c='r')
        #ax.scatter([i[2][0] for i in nowe_punkty2], [i[2][1] for i in nowe_punkty2], [i[2][2] for i in nowe_punkty2],c='g')
        #losowy = random.randint(0,len(points_3D_data))
        #for i in nowe_punkty:
            #normalny = calc_normal(int(points_3D_data[losowy][2][2]/ss), points_3D_data[losowy][1], points_3D_data[losowy][2], points_3D_data, slice_centres)
            #print("{}: punkt {}, {}".format(i, i[2], len(slice_centres)))
            
        #    normalny = calc_normal(i[0], i[1], i[2], points_3D_data, slice_centres)
        #    ax.quiver(i[2][0], i[2][1], i[2][2], normalny[0], normalny[1], normalny[2], colors='g')
        
        plt.show()
        
    
    if PerformCorrections == True:
        #perform corrections
        logging.info("===============applying corrections===================")
        apply_corrections(corrections, iDir + "/../bones", oDir + "/bones")
        apply_corrections(corrections, iDir + "/../fat", oDir + "/fat")
        apply_corrections(corrections, iDir + "/../muscles", oDir + "/muscles")
        apply_corrections(corrections, iDir + "/../vessels", oDir + "/vessels")
        
        apply_corrections_img(corrections, iDir + "/../bones", oDir + "/bones")
        apply_corrections_img(corrections, iDir + "/../fat", oDir + "/fat")
        apply_corrections_img(corrections, iDir + "/../muscles", oDir + "/muscles")
        apply_corrections_img(corrections, iDir + "/../roi", oDir + "/roi")
        apply_corrections_img(corrections, iDir + "/../vessels", oDir + "/vessels")
        
        
        apply_corrections_img_orig(corrections, imgDir, oDir + "/images")
        
        #difficult one - shift the skin with corrected thickness
        
        apply_corrections(corrections, oDir + "/skin_closed", oDir + "/skin")
        apply_corrections_img(corrections, oDir + "/skin_closed", oDir + "/skin")
    logging.info("Done")
    
    
    
    
if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument("-iDir"  ,    "--input_dir"      ,     dest="idir"                ,    help="input directory"                          ,    metavar="PATH", required=True )
    parser.add_argument("-ishDir"  ,    "--input_shape_dir"      ,     dest="ishdir"                ,    help="input directory with corrected shape"                          ,    metavar="PATH", required=True )
    parser.add_argument("-imgDir",    "--image_dir"      ,     dest="imgdir"              ,    help="image input directory"                    ,    metavar="PATH", required=True )
    parser.add_argument("-oDir"  ,    "--output_dir"     ,     dest="odir"                ,    help="output directory"                         ,    metavar="PATH", required=True )
    parser.add_argument("-ss"    ,    "--slice_spacing"  ,     dest="ss"                  ,    help="slice spacing"                            ,    metavar="PATH", required=True )
    parser.add_argument("-psx"   ,    "--pixel_spacing_x",     dest="psx"                 ,    help="pixel spacing x"                          ,    metavar="PATH", required=True )
    parser.add_argument("-psy"   ,    "--pixel_spacing_y",     dest="psy"                 ,    help="pixel spacing y"                          ,    metavar="PATH", required=True )
    parser.add_argument("-v"     ,    "--verbose"        ,     dest="verbose"             ,    help="verbose level"                            ,                    required=False)
    parser.add_argument("-q"     ,    "--quit"           ,     dest="quit_on_small_margin",    help="quit_on_small_margin - 'on' to turn on"   ,                    required=False)
    parser.add_argument("-N"     ,    "--norm"           ,     dest="norm",    help="norm to be used"   ,                    required=False)
    parser.add_argument("-s"     ,    "--move_step"      ,     dest="move_step",    help="movement made in each step"   ,                    required=True)
    parser.add_argument("-c"     ,    "--perform_cor"    ,     dest="PerformCorrections",    help="perform correctins on the set"   ,                    required=False)
    parser.add_argument("-d"     ,    "--decim"    ,     default = "1", dest="decim",    help="decimation step"   ,                    required=False)
    
    args = parser.parse_args()
    
    verbose = 'off'                 if args.verbose is None else args.verbose
    quit_on_small_margin = True                 if args.quit_on_small_margin is None else (args.quit_on_small_margin == 'on')
    iDir 	= args.idir
    ishDir 	= args.ishdir
    oDir  	= args.odir
    ss      = args.ss
    psx     = args.psx
    psy     = args.psy
    imgDir  = args.imgdir
    decim   = int(args.decim) 
    norm    = 2 if args.norm is None else int(args.norm)
    move_step = float(args.move_step)
    PerformCorrections = True if args.PerformCorrections is None else args.PerformCorrections == 'True'
    
    logging.basicConfig(level=logging.DEBUG,format="%(asctime)s [%(levelname)s] %(message)s",handlers=[logging.FileHandler(oDir+"/correctPosition2.log",mode='w'),logging.StreamHandler(sys.stdout)])
    
    
    if not os.path.isdir(iDir):
       logging.error('Error : Input directory (%s) not found !',iDir)
       exit(1)
    if not os.path.isdir(ishDir):
        logging.error('Error : Input directory (%s) not found !',ishDir)
        exit(1)
    
    logging.info("\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    logging.info("START:     as_correctPosition2.py")
    logging.info("in:                   "    +   iDir    )
    logging.info("shape in:             "    +   ishDir    )
    logging.info("image in:             "    +   imgDir    )
    logging.info("out:                  "   +   oDir)
    logging.info("slice spacing:        "   +   ss)
    logging.info("pixel spacing x:      "   +   psx)
    logging.info("pixel spacing y:      "   +   psy)
    logging.info("move step:            "   +   str(move_step))
    logging.info("decimation:           "   +   str(decim))
    logging.info("perform corrections:  "   +   str(PerformCorrections))
    logging.info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n")
    
    
    log2 = open(oDir+"/correctPosition2_results.log","a+")
    
    if verbose == 'off':
        verbose = False
    else:
        verbose = True
    
    process_dir(iDir, ishDir, oDir, imgDir, log2, float(ss), float(psx), float(psy), verbose, quit_on_small_margin, norm, move_step, PerformCorrections, decim)
    
    log2.close()