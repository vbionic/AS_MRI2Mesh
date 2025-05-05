import cv2 as cv
from matplotlib import pyplot as plt
import sys, getopt
import pydicom
import numpy as np
from PIL import Image
import json
from pydicom.tag import Tag
from pydicom.datadict import keyword_for_tag
import os
from argparse import ArgumentParser
import glob
import math
import shutil
import logging
import scipy
from scipy.interpolate import RectBivariateSpline
import time

#-----------------------------------------------------------------------------------------
sys.path.append(os.getcwd())
#-----------------------------------------------------------------------------------------
from v_utils.v_json import *
from v_utils.v_polygons import *
#-----------------------------------------------------------------------------------------

def bufor_kolowy(buf, poz_zero, indeks):
    pozycja = poz_zero + indeks
    while(pozycja < 0):
        pozycja += len(buf)
    else:
        while(pozycja>=len(buf)):
            pozycja -= len(buf)
    return buf[pozycja]


def get_angle(center,point):
    diff_x = point[0] - center[0]
    diff_y = point[1] - center[1]
    #      
    #   (315: -/-)    | 0   (45: +/-)
    #                 |      
    #           270 ----- 90 
    #                 |      
    #   (225: -/+)    | 180  (135: +/+)
    #      
    if(diff_x>0):
        if(diff_y>0):
            #+/+
            angle = math.atan(diff_y / diff_x) / math.pi * 180 + 90
        else:
            #+/-
            angle = math.atan(diff_y / diff_x) / math.pi * 180 + 90
    else:
        if(diff_x == 0):
            if(diff_y>0):
                #0/+
                angle = 180
            else:
                #0/-
                angle = 0
        else:
            if(diff_y>0):
                #-/+
                angle = math.atan(diff_y / diff_x) / math.pi * 180 + 270
            else:
                #-/-
                angle = math.atan(diff_y / diff_x) / math.pi * 180 + 270
    return angle
    
def find_edges(lista_katow_odwiedzonych):
    edge_list_cw = []
    edge_list_ccw = []
    for i in range(1,len(lista_katow_odwiedzonych)):
        if ((lista_katow_odwiedzonych[i-1]!=0)&(lista_katow_odwiedzonych[i]==0)):
            #edge found clockwise
            edge_list_cw.append(i-1)
    for i in range(1,len(lista_katow_odwiedzonych)):
        if ((lista_katow_odwiedzonych[i]!=0)&(lista_katow_odwiedzonych[i-1]==0)):
            #edge found counterclockwise
            edge_list_ccw.append(i)
    if ((lista_katow_odwiedzonych[0]!=0)&(lista_katow_odwiedzonych[len(lista_katow_odwiedzonych)-1]==0)):
        edge_list_ccw.append(0)
    if ((lista_katow_odwiedzonych[0]==0)&(lista_katow_odwiedzonych[len(lista_katow_odwiedzonych)-1]!=0)):
        edge_list_cw.append(len(lista_katow_odwiedzonych)-1)
    return [edge_list_cw, edge_list_ccw]
    
def limit(val,minval,maxval):
    if val<minval:
        return minval
    if val>maxval:
        return maxval
    return val
    
    
def angle_greater(a1,a2):
    if(a1==-1000):
        return False
    while(a1>360):
        a1 -= 360
    while(a2>360):
        a2 -= 360
    if((a2>345)&(a1<15)):
        return True
    if((a1>345)&(a2<15)):
        return False
    if(a1>a2):
        return True
    return False

def angle_smaller(a1,a2):
    if(a1==-1000):
        return False
    while(a1>360):
        a1 -= 360
    while(a2>360):
        a2 -= 360
    while(a1<0):
        a1 += 360
    while(a2<0):
        a2 += 360
    if((a1>345)&(a2<15)):
        return True
    if((a2>345)&(a1<15)):
        return False
    if(a2>a1):
        return True
    return False


def go_cw(cy,cx,angle_map):
    cur_angle = angle_map[cy,cx]
    next_pixel_cw = []
    if(angle_greater(angle_map[      cy  ,                        limit(cx+1, 0, angle_map.shape[1]-1)],cur_angle)):
        next_pixel_cw.append([cy  ,                        limit(cx+1, 0, angle_map.shape[1]-1)])
    if(angle_greater(angle_map[limit(cy+1, 0, angle_map.shape[0]-1),limit(cx+1, 0, angle_map.shape[1]-1)],cur_angle)):
        next_pixel_cw.append([limit(cy+1, 0, angle_map.shape[0]-1),limit(cx+1, 0, angle_map.shape[1]-1)])
    if(angle_greater(angle_map[limit(cy-1, 0, angle_map.shape[0]-1),limit(cx+1, 0, angle_map.shape[1]-1)],cur_angle)):
        next_pixel_cw.append([limit(cy-1, 0, angle_map.shape[0]-1),limit(cx+1, 0, angle_map.shape[1]-1)])
    if(angle_greater(angle_map[limit(cy+1, 0, angle_map.shape[0]-1),limit(cx  , 0, angle_map.shape[1]-1)],cur_angle)):
        next_pixel_cw.append([limit(cy+1, 0, angle_map.shape[0]-1),limit(cx  , 0, angle_map.shape[1]-1)])
    if(angle_greater(angle_map[limit(cy-1, 0, angle_map.shape[0]-1),limit(cx  , 0, angle_map.shape[1]-1)],cur_angle)):
        next_pixel_cw.append([limit(cy-1, 0, angle_map.shape[0]-1),limit(cx  , 0, angle_map.shape[1]-1)])
    if(angle_greater(angle_map[limit(cy  , 0, angle_map.shape[0]-1),limit(cx-1, 0, angle_map.shape[1]-1)],cur_angle)):
        next_pixel_cw.append([limit(cy  , 0, angle_map.shape[0]-1),limit(cx-1, 0, angle_map.shape[1]-1)])
    if(angle_greater(angle_map[limit(cy+1, 0, angle_map.shape[0]-1),limit(cx-1, 0, angle_map.shape[1]-1)],cur_angle)):
        next_pixel_cw.append([limit(cy+1, 0, angle_map.shape[0]-1),limit(cx-1, 0, angle_map.shape[1]-1)])
    if(angle_greater(angle_map[limit(cy-1, 0, angle_map.shape[0]-1),limit(cx-1, 0, angle_map.shape[1]-1)],cur_angle)):
        next_pixel_cw.append([limit(cy-1, 0, angle_map.shape[0]-1),limit(cx-1, 0, angle_map.shape[1]-1)])
    return next_pixel_cw
    
def go_ccw(cy,cx,angle_map):
    cur_angle = angle_map[cy,cx]
    next_pixel_cw = []
    if(angle_smaller(angle_map[      cy  ,                        limit(cx+1, 0, angle_map.shape[1]-1)],cur_angle)):
        next_pixel_cw.append([cy  ,                        limit(cx+1, 0, angle_map.shape[1]-1)])
    if(angle_smaller(angle_map[limit(cy+1, 0, angle_map.shape[0]-1),limit(cx+1, 0, angle_map.shape[1]-1)],cur_angle)):
        next_pixel_cw.append([limit(cy+1, 0, angle_map.shape[0]-1),limit(cx+1, 0, angle_map.shape[1]-1)])
    if(angle_smaller(angle_map[limit(cy-1, 0, angle_map.shape[0]-1),limit(cx+1, 0, angle_map.shape[1]-1)],cur_angle)):
        next_pixel_cw.append([limit(cy-1, 0, angle_map.shape[0]-1),limit(cx+1, 0, angle_map.shape[1]-1)])
    if(angle_smaller(angle_map[limit(cy+1, 0, angle_map.shape[0]-1),limit(cx  , 0, angle_map.shape[1]-1)],cur_angle)):
        next_pixel_cw.append([limit(cy+1, 0, angle_map.shape[0]-1),limit(cx  , 0, angle_map.shape[1]-1)])
    if(angle_smaller(angle_map[limit(cy-1, 0, angle_map.shape[0]-1),limit(cx  , 0, angle_map.shape[1]-1)],cur_angle)):
        next_pixel_cw.append([limit(cy-1, 0, angle_map.shape[0]-1),limit(cx  , 0, angle_map.shape[1]-1)])
    if(angle_smaller(angle_map[limit(cy  , 0, angle_map.shape[0]-1),limit(cx-1, 0, angle_map.shape[1]-1)],cur_angle)):
        next_pixel_cw.append([limit(cy  , 0, angle_map.shape[0]-1),limit(cx-1, 0, angle_map.shape[1]-1)])
    if(angle_smaller(angle_map[limit(cy+1, 0, angle_map.shape[0]-1),limit(cx-1, 0, angle_map.shape[1]-1)],cur_angle)):
        next_pixel_cw.append([limit(cy+1, 0, angle_map.shape[0]-1),limit(cx-1, 0, angle_map.shape[1]-1)])
    if(angle_smaller(angle_map[limit(cy-1, 0, angle_map.shape[0]-1),limit(cx-1, 0, angle_map.shape[1]-1)],cur_angle)):
        next_pixel_cw.append([limit(cy-1, 0, angle_map.shape[0]-1),limit(cx-1, 0, angle_map.shape[1]-1)])
    return next_pixel_cw
    
def get_maps(lab_image, center):
    angle_map = np.zeros(lab_image.shape)
    radius_map = np.zeros(lab_image.shape)
    
    for i in range(0, lab_image.shape[0]):
        for j in range(0, lab_image.shape[1]):
            if(lab_image[i,j] == 255):
                angle_map [i,j] = get_angle (center,(j,i))
                radius_map[i,j] = get_radius(center,(j,i))
            else:
                angle_map [i,j] = -1000
                radius_map[i,j] = -1000
    return [angle_map, radius_map]
    
def get_maps_full(lab_image, center):
    angle_map = np.zeros(lab_image.shape)
    radius_map = np.zeros(lab_image.shape)
    
    for i in range(0, lab_image.shape[0]):
        for j in range(0, lab_image.shape[1]):
            angle_map [i,j] = get_angle (center,(j,i))
            radius_map[i,j] = get_radius(center,(j,i))
    return [angle_map, radius_map]
    
def find_edges_advanced(lab_image, center):
    edge_list_cw = []
    edge_list_ccw = []
    [angle_map, radius_map] = get_maps(lab_image, center)
    
    
    
    for a in range(10,361,10):
        max_reached_angle = a
#        print("testuje kat {}".format(a))
        temp = angle_map - a
        temp[(temp > -360) & (temp < 0)] += 360
        #candidates = np.where(np.abs(temp) == min(np.abs(temp.flatten())))
        candidates = np.where(np.abs(temp) < 2.0 )
        
        if not (candidates[0].size>0):
#            print("\tBrak kandydatow")
            continue
        
        visited = np.zeros(lab_image.shape)
        to_be_visited = []
        for i in range(0,len(candidates[0])):
            cx = candidates[1][i]
            cy = candidates[0][i]
            to_be_visited.append([cy,cx])
#            print("\tkandydat startowy: {}".format([cy,cx]))
        
        while(len(to_be_visited)>0):
            if(visited[to_be_visited[0][0],to_be_visited[0][1]] == 0):
            #not visited, let's go
#                print("\t\ttestuje {}, kat {}".format([to_be_visited[0][0],to_be_visited[0][1]], angle_map[to_be_visited[0][0],to_be_visited[0][1]]))
                if (angle_greater(angle_map[to_be_visited[0][0],to_be_visited[0][1]],max_reached_angle)):
                    max_reached_angle = angle_map[to_be_visited[0][0],to_be_visited[0][1]]
                if (angle_greater(max_reached_angle,a + 20)):
#                    print("\t\t\tDotarlismy! {} {} {}".format(max_reached_angle,a + 20, angle_map[to_be_visited[0][0],to_be_visited[0][1]]))
                    to_be_visited = []
                else:
                    visited[to_be_visited[0][0],to_be_visited[0][1]] = 1
                    next_steps = go_cw(to_be_visited[0][0],to_be_visited[0][1],angle_map)
                    if(len(next_steps)>0):
                        for i in range(0,len(next_steps)):
                            if(visited[next_steps[i][0],next_steps[i][1]]==0):
                                to_be_visited.append(next_steps[i])
                    del to_be_visited[0]
            else:
            #we've been there, just delete
                del to_be_visited[0]
        
        if (not angle_greater(max_reached_angle,a + 20)):
            if(max_reached_angle != a):
                edge_list_cw.append(max_reached_angle)
#                print("lista koncow: {}".format(edge_list_cw))
            
    for a in range(10,361,10):
        
        min_reached_angle = a
#        print("testuje kat {}".format(a))
        temp = angle_map - a
        temp[(temp > -360) & (temp < 0)] += 360
        #candidates = np.where(np.abs(temp) == min(np.abs(temp.flatten())))
        candidates = np.where(np.abs(temp) < 2.0 )
        
        #print(candidates)
        
        if not (candidates[0].size>0):
            continue
        
        visited = np.zeros(lab_image.shape)
        
        to_be_visited = []
        for i in range(0,len(candidates[0])):
            cx = candidates[1][i]
            cy = candidates[0][i]
            to_be_visited.append([cy,cx])
#            print("\tkandydat startowy: {}".format([cy,cx]))
        
        while(len(to_be_visited)>0):
            if(visited[to_be_visited[0][0],to_be_visited[0][1]] == 0):
            #not visited, let's go
#                print("\t\ttestuje {}, kat {}".format([to_be_visited[0][0],to_be_visited[0][1]], angle_map[to_be_visited[0][0],to_be_visited[0][1]]))
                if (angle_smaller(angle_map[to_be_visited[0][0],to_be_visited[0][1]],min_reached_angle)):
                    min_reached_angle = angle_map[to_be_visited[0][0],to_be_visited[0][1]]
                if (angle_smaller(min_reached_angle,a - 20)):
#                    print("\t\t\tDotarlismy! {} {} {}".format(min_reached_angle,a - 20, angle_map[to_be_visited[0][0],to_be_visited[0][1]]))
                    to_be_visited = []
                else:
                    visited[to_be_visited[0][0],to_be_visited[0][1]] = 1
                    next_steps = go_ccw(to_be_visited[0][0],to_be_visited[0][1],angle_map)
                    if(len(next_steps)>0):
                        for i in range(0,len(next_steps)):
                            if(visited[next_steps[i][0],next_steps[i][1]]==0):
                                to_be_visited.append(next_steps[i])
                    del to_be_visited[0]
            else:
            #we've been there, just delete
                del to_be_visited[0]
        #if(a==60):
        #    test_img = cv.cvtColor(lab_image, cv.COLOR_GRAY2BGR)
        #    test_img[:,:,0] = 255*visited.astype(int)
        if (not angle_smaller(min_reached_angle,a - 20)):
            if(min_reached_angle != a):
                edge_list_ccw.append(min_reached_angle)
#                print("lista koncow: {}".format(edge_list_ccw))
            
    return [edge_list_cw, edge_list_ccw]
        
        
def get_radius(center,point):
    dx = point[0]-center[0]
    dy = point[1]-center[1]
    return math.sqrt(dx*dx+dy*dy)
        
def find_radius_to_edge_int(angle, angles, data, center):
    radii = []
    for i in range(0,len(data["polygons"][0]["outer"]["path"])):
        if(int(angles[i]) == angle):
            dx = data["polygons"][0]["outer"]["path"][i][0]-center[0]
            dy = data["polygons"][0]["outer"]["path"][i][1]-center[1]
            radii.append([math.sqrt(dx*dx+dy*dy), data["polygons"][0]["outer"]["path"][i][0], data["polygons"][0]["outer"]["path"][i][1]])
    return radii

def calculate_edge_stats_near_edge(angle, lab_image, center, angle_range):
    radii = []
    max_radius = 0
    min_radius = 1000000000
    avg_radius = 0
    count = 0
    [angle_map, radius_map] = get_maps(lab_image, center)
    
    for i in range(0,angle_map.shape[0]):
        for j in range(0,angle_map.shape[1]):
            if(angle_map[i,j]!=-1000):
                dif = abs(angle - angle_map[i,j])
                if (dif > (360-angle_range)):
                    dif = 360 - dif
                    
                if(dif < angle_range):
                    rad = radius_map[i,j]
                    radii.append([rad, j, i])
                    count += 1
                    avg_radius += rad
                    if rad>max_radius:
                        max_radius = rad
                    if rad<min_radius:
                        min_radius = rad
    if count > 0:
        avg_radius /= count
    return [radii, max_radius, min_radius, avg_radius, count]
    
    
def get_angle_cw(angleStart, angleEnd):
    if (angleEnd < angleStart):
        angleEnd += 360
    temp = angleEnd - angleStart
    while (temp>=360):
        temp -= 360
    while (temp<0):
        temp += 360
    return temp
    
def get_angle_ccw(angleStart, angleEnd):
    if (angleEnd > angleStart):
        angleStart += 360
    temp = angleStart - angleEnd
    while (temp>=360):
        temp -= 360
    while (temp<0):
        temp += 360
    return temp

def get_angle_min(angle1, angle2):
    return min([get_angle_cw(angle1, angle2), get_angle_ccw(angle1, angle2)])

def fill_gap(lab_image, angle_start, angle_end, center, filename):
    #print("\n\n---+++---")
    [radii, max_radius, min_radius, avg_radius, count] = calculate_edge_stats_near_edge(angle_start, lab_image, center, 3)
    #logging.info("Stats start: max:{} min:{} avg:{} count:{}".format(max_radius, min_radius, avg_radius, count))
    width_start = max_radius - min_radius
    rad_start = min_radius
    
    [radii, max_radius, min_radius, avg_radius, count] = calculate_edge_stats_near_edge(angle_end, lab_image, center, 3)
    #logging.info("Stats end:   max:{} min:{} avg:{} count:{}".format(max_radius, min_radius, avg_radius, count))
    width_end = max_radius - min_radius
    rad_end = min_radius
    
    
    if (angle_start > angle_end):
        angle_end += 360
        
    arc = angle_end - angle_start
    #print("arc: {}".format(arc))
    [angle_map, radius_map] = get_maps_full(lab_image, center)
    #cv.imwrite(filename+"AngleMap.png",(angle_map*255/360).astype(int))
    #cv.imwrite(filename+"RadiusMap.png",(radius_map*255/radius_map.max()).astype(int))
    
    lab_image_filled = np.zeros(lab_image.shape)
    
    for i in range(0,angle_map.shape[0]):
        for j in range(0,angle_map.shape[1]):
            lab_image_filled[i,j] = lab_image[i,j] 
            #if((angle_map[i,j] > angle_start-2.0) & (angle_map[i,j] < angle_end+2.0)):
            if((get_angle_cw(angle_start-2.0, angle_map[i,j]) < arc+4) & (get_angle_ccw(angle_end+2.0, angle_map[i,j]) < arc+4)):
                #logging.info("Curr angle: {}".format(angle_map[i,j]))
                angle_dif = angle_map[i,j] - angle_start
                while(angle_dif < 0):
                    angle_dif += 360
                #logging.info("Curr angle dif: {}".format(angle_dif))
                rad_cur = rad_start + (rad_end - rad_start) / arc * angle_dif
                #logging.info("Curr radius: {}\n".format(rad_cur))
                width_cur = width_start + (width_end - width_start) / arc * angle_dif
                if (width_cur<4):
                    width_cur = 4
                if((radius_map[i,j]>=rad_cur) & (radius_map[i,j]<=(rad_cur+width_cur))):
                    lab_image_filled[i,j] = 255
    return lab_image_filled
    
    
    
    
    
    
    
def find_matching_edge_cw(cw_gap, edge_ccw):
    current_angle = cw_gap
    arc_len = 0
    result = None
    
    differences = edge_ccw - current_angle
    differences.sort()
    for curr_dif in differences:
        while (curr_dif<0):
            curr_dif += 360
        arc_len = curr_dif
        result = (current_angle + curr_dif)
        while (result<0):
            result += 360
        while (result>360):
            result -= 360
            
            
    return [result, arc_len]
    
def find_matching_edge_cw_reverse(cw_gap, edge_ccw):
    current_angle = cw_gap
    arc_len = 0
    result = None
    
    differences = -(edge_ccw - current_angle)
    differences.sort()
    for curr_dif in differences:
        while (curr_dif<0):
            curr_dif += 360
        arc_len = curr_dif
        result = current_angle - curr_dif
        while (result<0):
            result += 360
        while (result>360):
            result -= 360
            
    return [result, arc_len]
    
    
def is_closed_circle(lab_image, data, roi_image, label_min, label_max):
    sum_x = 0
    sum_y = 0
    sum_tot = 0
    
    for i in range(0,roi_image.shape[0]):
        for j in range(0,roi_image.shape[1]):
            if((roi_image[i,j] >= label_min)&(roi_image[i,j] <= label_max)):
                sum_x += roi_image[i][j] * j
                sum_y += roi_image[i][j] * i
                sum_tot += roi_image[i][j]
                
    if (sum_tot == 0):
        for i in range(0,lab_image.shape[0]):
            for j in range(0,lab_image.shape[1]):
                if((lab_image[i,j] >= label_min)&(lab_image[i,j] <= label_max)):
                    sum_x += lab_image[i][j] * j
                    sum_y += lab_image[i][j] * i
                    sum_tot += lab_image[i][j]
                
    center_x = sum_x/sum_tot
    center_y = sum_y/sum_tot
    
    angles = []
    for vertex in data["polygons"][0]["outer"]["path"]:
        angle = get_angle([center_x, center_y], vertex)
        #print("c:{} v:{} delta:{} a:{}".format([center_x, center_y], vertex, [vertex[0]-center_x, vertex[1]-center_y], angle))
        angles.append(angle)
    
    poz_zero = angles.index(max(angles))
    zamkniety = True
    monotoniczny = True
    monotonicznosci = []
    zamknietosc = []
    dotychczasowy_min = 361
    
    lista_katow_odwiedzonych = np.zeros(360)
    
    for i in range(0,len(angles)):
        temp = True
        a = bufor_kolowy(angles, poz_zero, i)
        #print("aktualny kat: {}".format(a))
        if(i==0):
            poprzedni_kat = a
        zmiana_kata = [int(a), int(poprzedni_kat)]
        zmiana_kata.sort()
        #print("od {} do {}".format(zmiana_kata[0],zmiana_kata[1]+1))
        if ((zmiana_kata[0]<20)&(zmiana_kata[1]>340)):
            for j in range(zmiana_kata[1],360):
                #print("\todwiedzony kat: {}".format(j))
                lista_katow_odwiedzonych[j] += 1
            for j in range(0,zmiana_kata[0]+1):
                #print("\todwiedzony kat: {}".format(j))
                lista_katow_odwiedzonych[j] += 1
        else:
            for j in range(zmiana_kata[0],zmiana_kata[1]+1):
                #print("\todwiedzony kat: {}".format(j))
                lista_katow_odwiedzonych[j] += 1
        poprzedni_kat = a
        if (a < dotychczasowy_min):
            dotychczasowy_min = a
        if (a > dotychczasowy_min):
            monotoniczny = False
            temp = False
        if (a > dotychczasowy_min + 20):
            zamkniety = False
            zamknietosc.append(False)
        else:
            zamknietosc.append(True)
        monotonicznosci.append(temp)
        
    #print(lista_katow_odwiedzonych)
        
    return [zamkniety, lista_katow_odwiedzonych, (center_x, center_y), angles]

def process_dir(iDir, oDir, log2, verbose):
    polygon_list    = glob.glob(iDir+"/*_polygons.json")
    polygon_list.sort()
    for polygon_file in polygon_list:
        try:
            file_data = open(polygon_file)
            data = json.load(file_data)
            file_data.close()
        except Exception as err:
            logging.error("Input data IO error: {}".format(err))
            sys.exit(1)
        #try:
        
        if(len(data["polygons"])==0):
            continue
        
        #except Exception as err:
        #    logging.error("Polygon file error")
        #    sys.exit(1)
        
        
        name = polygon_file.rsplit('_',1)[0]+"_labels.png"
        logging.info("opening file: {}".format(name))
        lab_image = cv.imread(name, cv.IMREAD_COLOR)
        lab_image = lab_image[:,:,2]

        name = polygon_file.rsplit('_',1)[0]+"_prob.png"
        prob_image = cv.imread(name, cv.IMREAD_COLOR)
        prob_image = prob_image[:,:,2]
        
        try:
            [dir_name,name] = os.path.split(polygon_file)
            roi_name = dir_name + "/../roi/" + name.split('_',1)[0]+"_roi_labels.png"
            roi_image = cv.imread(roi_name, cv.IMREAD_COLOR)
            roi_image = roi_image[:,:,2]
        except Exception as err:
            logging.error("ROI file error")
            sys.exit(1)

        
        [zamkniety, lista_katow_odwiedzonych, center, angles] = is_closed_circle(lab_image, data, roi_image, 100, 255)
        
        edge_cw = []
        edge_ccw = []
        #[edge_cw, edge_ccw] = find_edges(lista_katow_odwiedzonych)
        
        radii_cw = []
        radii_ccw = []
        
        
        if ((not zamkniety) & ((len(edge_cw) == 0) | (len(edge_ccw) == 0))):
            logging.info("Trying to close...")
            #some strange cases when two parts of the open contour overlap
            [edge_cw, edge_ccw] = find_edges_advanced(lab_image, center)
            #cv.imwrite(oDir+"/"+os.path.basename(name).rsplit('_',2)[0]+"___test___200_.png",test)
        
        #remove duplicates
        edge_cw = list(dict.fromkeys(edge_cw))
        edge_ccw = list(dict.fromkeys(edge_ccw))
        
        for edge_angle in edge_cw:
            radii_cw  = find_radius_to_edge_int(int(edge_angle), angles, data, center)
        for edge_angle in edge_ccw:
            radii_ccw = find_radius_to_edge_int(int(edge_angle), angles, data, center)
        
        #print(radii_cw)
        #print(radii_ccw)
        # if(~zamkniety | ~monotoniczny):
            # print("Kontur zamkniety: {}, monotoniczny: {}".format(zamkniety, monotoniczny))
            # print("file: {}".format(name))
            # print(angles)
            # print(monotonicznosci)
            # for i in range(0,len(angles)):
                # print("{} {}".format(angles[i], monotonicznosci[i]))
        logging.info([edge_cw, edge_ccw])
        if((not zamkniety) & ((len(edge_cw) != 0) & (len(edge_ccw) != 0))):
            #trying to close the skin shape
            for cw_gap in edge_cw:
                [ccw_gap, arc_len] = find_matching_edge_cw(cw_gap, edge_ccw)
                logging.info("Gap from {} to {}, arc: {}".format(cw_gap,ccw_gap,arc_len))
                if ((arc_len > 180) | (arc_len == 0)):
                    #arc too long, probably an overlapped skin areas, or no candidates found
                    [ccw_gap, arc_len] = find_matching_edge_cw_reverse(cw_gap, edge_ccw)
                    angle_start = ccw_gap
                    angle_end   = cw_gap
                else:
                    angle_start = cw_gap
                    angle_end   = ccw_gap
                logging.info("Gap from {} to {}, arc: {}".format(angle_start,angle_end,arc_len))
                lab_image_filled = fill_gap(lab_image, angle_start, angle_end, center, oDir+"/"+os.path.basename(name).rsplit('_',2)[0])
                lab_image_filled[lab_image_filled[:,:] < label_min] = 0
                lab_image_filled[lab_image_filled[:,:] > label_max] = 0
                tissue_polygons_out = v_polygons()
                tissue_polygons_out._mask_ndarray_to_polygons(lab_image_filled, background_val = 0, limit_polygons_num = 0)
                data = tissue_polygons_out.as_dict()
                [zamkniety, lista_katow_odwiedzonych, center, angles] = is_closed_circle(lab_image_filled, data, roi_image, 100, 255)
        else:
            lab_image_filled = np.zeros(lab_image.shape)
            for i in range(0,lab_image.shape[0]):
                for j in range(0,lab_image.shape[1]):
                    lab_image_filled[i,j] = int(lab_image[i,j]) 
                
                
            
            #for i in range(0,len(angles)):
            #    print("{} {}".format(bufor_kolowy(angles, poz_zero, i), zamknietosc[i]))
            #print(lista_katow_odwiedzonych)
            
            
            
            
        
        #print("{} {} {}".format(sum_x, sum_y, sum_tot))
        #cv.drawContours(prob_image, [np.array([(0,0),(10,0),(10,10),(0,10)])], 0, (0,255,25))
        xxx = np.array([(a[0],a[1]) for a in data["polygons"][0]["outer"]["path"]])
        lab_image_filled = cv.cvtColor(lab_image_filled.astype(np.uint8), cv.COLOR_GRAY2BGR)
        lab_image_filled[:,:,0] = 0
        lab_image_filled[:,:,1] = 0
        cv.imwrite(oDir+"/"+os.path.basename(name).rsplit('_',2)[0]+"_skin_labels.png", lab_image_filled)
        jsonDumpSafe(oDir+"/"+os.path.basename(name).rsplit('_',2)[0]+"_skin_polygons.json", data)
                    
                    
        cv.drawContours(lab_image_filled, [xxx], 0, (0,255,25))
        
        if(not zamkniety):
            for radius in radii_cw:
                cv.line(lab_image_filled, (int(center[0]), int(center[1])),(int(radius[1]),int(radius[2])),(255,0,0),1)
            for radius in radii_ccw:
                #print((int(center[0]), int(center[1])))
                #print((radius[1],radius[2]))
                cv.line(lab_image_filled, (int(center[0]), int(center[1])),(int(radius[1]),int(radius[2])),(0,255,0),1)
            cv.imwrite(oDir+"/"+os.path.basename(name).rsplit('_',2)[0]+"_filled.png",lab_image_filled)
        #print(cv.isContourConvex(xxx))
        
        
        #if(sum_tot>0):
        #    log2.write("{}\t{}\t{}\n".format(os.path.basename(name).rsplit('_',2)[0], sum_x/sum_tot, sum_y/sum_tot))



parser = ArgumentParser()

parser.add_argument("-iDir",      "--input_dir",       dest="idir",    help="input directory",      metavar="PATH", required=True)
parser.add_argument("-oDir",      "--output_dir",     dest="odir",    help="output directory",        metavar="PATH", required=True)

parser.add_argument("-v",       "--verbose",        dest="verbose",     help="verbose level",                                           required=False)

args = parser.parse_args()

verbose = 'off'                 if args.verbose is None else args.verbose
iDir 	= args.idir
oDir  	= args.odir


logging.basicConfig(level=logging.DEBUG,format="%(asctime)s [%(levelname)s] %(message)s",handlers=[logging.FileHandler(oDir+"/correctShapes.log",mode='w'),logging.StreamHandler(sys.stdout)])


if not os.path.isdir(iDir):
    logging.error('Error : Input directory (%s) not found !',iDir)
    exit(1)


logging.info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
logging.info("START:     as_correctShapes.py")
logging.info("in:  "    +   iDir    )
logging.info("out: "   +   oDir)

log2 = open(oDir+"/correctShapes_results.log","a+")

process_dir(iDir, oDir, log2, verbose)

log2.close()