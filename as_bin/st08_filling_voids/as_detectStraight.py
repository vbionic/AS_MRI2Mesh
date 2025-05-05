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
import pathlib
from argparse import ArgumentParser
import glob
import math
import shutil
import logging
import scipy
from scipy.interpolate import RectBivariateSpline
import time
import warnings



#-----------------------------------------------------------------------------------------
sys.path.append(os.getcwd())
#-----------------------------------------------------------------------------------------
from v_utils.v_json import *
from v_utils.v_polygons import *
#-----------------------------------------------------------------------------------------

def read_files(iDir, pDir, ROI_file, tissue, verbose):
    tissue_file = os.path.normpath(iDir + "/"+tissue+"/" + os.path.basename(ROI_file).rsplit('_',2)[0] + '_'+tissue+'_labels.png')
    if verbose:
        logging.info("opening {} file: {}".format(tissue, tissue_file))
    tissue_raw = cv.imread(tissue_file, cv.IMREAD_COLOR)
    tissue_raw = tissue_raw[:,:,2]
    tissue_raw[tissue_raw != 0] = 255
    
    tissue_prob_file = os.path.normpath(pDir + "/"+tissue+"/" + os.path.basename(ROI_file).rsplit('_',2)[0] + '_'+tissue+'_prob_nl.png')
    if verbose:
        logging.info("opening {} prob file: {}".format(tissue, tissue_prob_file))
    tissue_prob = cv.imread(tissue_prob_file, cv.IMREAD_COLOR)
    tissue_prob = tissue_prob[:,:,2]
    return [tissue_raw, tissue_prob]
    
def wirte_files(oDir, ROI_file, tissue, labels):
    out_path = os.path.normpath(oDir + "/"+tissue)
    out_file = os.path.normpath(oDir + "/"+tissue+"/" + os.path.basename(ROI_file).rsplit('_',2)[0] + '_'+tissue+'_labels.png')
    try:
        if not os.path.isdir(out_path):
            pathlib.Path(out_path).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        logging.error('creating "%s" directory failed, error "%s"'%(out_path,err))
        exit(1)
    cv.imwrite(out_file, labels)
    
def calculate_line(x,y, verbose):
    x = [x[0], x[-1]]
    y = [y[0], y[-1]]
    if (max(x)-min(x)) >= (max(y)-min(y)):
        #the x values should have bigger span than y, otherwise - exchange them
        coeffs = np.polyfit(x,y,1)
        version = 0
    else:
        coeffs = np.polyfit(y,x,1)
        version = 1
    return [coeffs, version]

def get_sum_of_distances(x,y, coeffs, version, verbose):
    sum_distance = 0
    max_distance = 0
    for index in range(len(x)):
        if version==0:
            a = coeffs[0]
            b = -1
            c = coeffs[1]
            x1 = x[index]
            y1 = y[index]
            d = abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))
            if y1 < a * x1 + c:
                d = -d
        else:
            a = coeffs[0]
            b = -1
            c = coeffs[1]
            x1 = y[index]
            y1 = x[index]
            d = abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))
            if y1 < a * x1 + c:
                d = -d
        if abs(d) > max_distance:
            max_distance = abs(d)
        if verbose: logging.info("D: {}".format(d)) 
        sum_distance = sum_distance + d
    return [max_distance, sum_distance]
    
def get_max_distance(x,y, coeffs, version, verbose):
    max_distance = 0
    for index in range(len(x)):
        if version==0:
            a = coeffs[0]
            b = -1
            c = coeffs[1]
            x1 = x[index]
            y1 = y[index]
            d = abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b)) 
        else:
            a = coeffs[0]
            b = -1
            c = coeffs[1]
            x1 = y[index]
            y1 = x[index]
            d = abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b)) 
        if d > max_distance:
            max_distance = d
    return max_distance
            
def process_skin_shape(poly_in, verbose):
    number = 0
    node_range = []
    #print(poly_in)
    for poly in poly_in["polygons"]:
        #print(poly)
        #print(len(poly["outer"]["path"]))
        node_range.append([])
        current_path = make_polygon_dense(poly["outer"]["path"])
        for node_num in range(len(current_path)):
            #loop over all vertices of the polygon
            results = []
            if verbose: logging.info("Processing node {}".format(node_num))
            for curr_max_node_range in range(1,max_node_range):
                #test all sets of nodes up to curr_max_node_range away from base node
                considered_nodes = [current_path[i%len(current_path)] for i in range(node_num, node_num+curr_max_node_range+1)]
                if verbose: logging.info("  Processing node list: {}".format(considered_nodes))
                x = [a[0] for a in considered_nodes]
                y = [a[1] for a in considered_nodes]
                [coeffs, version] = calculate_line(x, y, verbose)
                if verbose: logging.info("      line data: {}, version {}".format(coeffs, "x-y" if version==0 else "y-x"))
                distance_between_nodes = math.sqrt((considered_nodes[0][0]-considered_nodes[-1][0])**2+(considered_nodes[0][1]-considered_nodes[-1][1])**2)
                #distance_from_line = get_max_distance(x, y, coeffs, version, verbose)
                [distance_from_line, sum_of_dist] = get_sum_of_distances(x, y, coeffs, version, verbose)
                if verbose: logging.info("      Max distance: {}, sum: {}".format(distance_from_line, sum_of_dist))
                results.append([distance_between_nodes, distance_from_line, sum_of_dist, version])
                # if distance_from_line > max_distance_from_line:
                    # node_range[-1].append(results[-2][0])
                    # break
                # if node_num == len(poly["outer"]["path"])-1:
                if verbose: logging.info("      Wektor wynikow: {}".format(results))
            for i in range(len(results)):
                #find the biggest range between the nodes that still satisfy the max range from line requirement
                if (results[i][1] > max_distance_from_line) or (abs(results[i][2]) > max_sum_of_distances_from_line):
                    #this is too far, add the previous element
                    node_range[-1].append([results[i-1][0], i, considered_nodes[0][0], considered_nodes[0][1], considered_nodes[i][0], considered_nodes[i][1], results[i-1][3], node_num])
                    break
                if i==len(results)-1:
                    #the last element - add to the list
                    node_range[-1].append([results[i][0], i+1, considered_nodes[0][0], considered_nodes[0][1], considered_nodes[i+1][0], considered_nodes[i+1][1], results[i][3], node_num])
            if verbose: logging.info("Node range {}".format(node_range))
                
    # tissue_polygons_out = v_polygons()
    # tissue_polygons_out = v_polygons.from_dict(processed_poly)
    # raw_out = tissue_polygons_out.as_numpy_mask(fill = True,w = raw_in.shape[1], h = raw_in.shape[0])
    return node_range
    
    
    
def process_dense_vertice_list(dense_vertice_list):
    #find all fixed-flexible neighbours and mark fixed as pivot
    pivotsincr = []
    pivotsdecr = []
    for i in range(len(dense_vertice_list)):
        if dense_vertice_list[i][2] == 0 and dense_vertice_list[(i+1)%len(dense_vertice_list)][2] == 1: # i fixed and i+1 flexible
            dense_vertice_list[i][2] = 2 #pivot
            pivotsincr.append(i)
    for i in range(len(dense_vertice_list)):
        if dense_vertice_list[i][2] == 0 and dense_vertice_list[(i-1)%len(dense_vertice_list)][2] == 1: # i fixed and i-1 flexible
            dense_vertice_list[i][2] = 2 #pivot
            pivotsdecr.append(i)
    #now insert pivot in between of the just created "edge pivots"
    for p in pivotsdecr:
        print("for pivot point {} looking for match".format(p))
        num = 1
        while dense_vertice_list[(p+num)%len(dense_vertice_list)][2] == 0:  #while fixed nodes are encountered
            #print("{} still fixed".format((p+num)%len(dense_vertice_list)))
            num += 1
        dense_vertice_list[int(p+num*1/4)%len(dense_vertice_list)][2] = 2
        dense_vertice_list[int(p+num*2/4)%len(dense_vertice_list)][2] = 2
        dense_vertice_list[int(p+num*3/4)%len(dense_vertice_list)][2] = 2
        print("found matching at {} and added pivots at {}, {}, {}".format((p+num)%len(dense_vertice_list), int(p+num*1/4)%len(dense_vertice_list), int(p+num*2/4)%len(dense_vertice_list), int(p+num*3/4)%len(dense_vertice_list)))

    
def process_dir(iDir, oDir, pDir, log2, verbose):


    mark_all_as_flexible = False
    
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    file_list    = glob.glob(os.path.normpath(iDir + "/skin/*_skin_polygons.json"))
    file_list.sort()
    
    sequence_parameters_fn = os.path.normpath(pDir + "/description.json")
    sequence_parameters_h = open(sequence_parameters_fn)
    sequence_parameters = json.load(sequence_parameters_h)
    sequence_parameters_h.close()
    
    index = []
    coordinate_mm = []
    number = 1
    results = []
    corrections = []
    
    out_path = os.path.normpath(oDir + "/skinlines")
    try:
        if not os.path.isdir(out_path):
            pathlib.Path(out_path).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        logging.error('creating "%s" directory failed, error "%s"'%(out_path,err))
        exit(1)
    
    xyz_filename = os.path.normpath(out_path + '/straight_point_cloud.xyz')
    xyz_all_filename = os.path.normpath(out_path + '/point_cloud.xyz')
    xyz_file = open(xyz_filename,"w+")
    xyz_all_file = open(xyz_all_filename,"w+")
    #print(file_list)
    
    slice_number = 0
    #for polygon_file in [file_list[30]]:
    for polygon_file in file_list:
        #logging.info("opening ROI file: {}".format(ROI_file))
        
        skin_poly_h = open(polygon_file)
        skin_poly = json.load(skin_poly_h)
        skin_poly_h.close()
        
        node_ranges = process_skin_shape(skin_poly, verbose)
        #for each node_ranges[i], where i is the number of separate polygons in the input data,a list of the following for each detected line:
        #             0         1           2       3       4     5     6        7
        #node_ranges: distance, node_count, xstart, ystart, xend, yend, version, node_start
        
        #node_ranges = process_skin_shape({"polygons":[{"outer":{"path":[[5,5],[6,6],[7,7],[8,7],[9,8],[11,8],[12,6],[11,3]]}}]}, verbose)
        #print(len(node_ranges[0]))
        #print(node_ranges[0])
        
        skin_labels_file_in = os.path.normpath(iDir + "/skin/" + os.path.basename(polygon_file).rsplit('_',1)[0] + '_labels.png')
        #print(skin_labels_file_in)
        skin_labels_in = cv.imread(skin_labels_file_in, cv.IMREAD_COLOR)
        #skin_labels_in = skin_labels_in[:,:,2]
        #skin_labels_in[skin_labels_in != 0] = 255
        
        
        coordinates = []
        print("++++++++++++++++++++\nPlik: {}".format(os.path.basename(polygon_file).rsplit('_',1)[0] ))
        for i in range(len(node_ranges)):
            #for every polygon
            coordinates.append([])
            
            vertices_filename = os.path.normpath(out_path + "/" + os.path.basename(polygon_file).rsplit('_',1)[0] + '_poly{:01}_vertices.txt'.format(i))
            #vertices_file = open(vertices_filename, "w+")
            vertice_list = []
            vert_avg = [0,0]
            for vertex_out in skin_poly["polygons"][i]["outer"]["path"]:
                vertice_list.append([vertex_out[0], vertex_out[1],1])
                vert_avg[0] += vertex_out[0]
                vert_avg[1] += vertex_out[1]
            vert_avg[0] /= len(skin_poly["polygons"][i]["outer"]["path"])
            vert_avg[1] /= len(skin_poly["polygons"][i]["outer"]["path"])
            
            vertice_list.insert(0,[int(vert_avg[0]), int(vert_avg[1]),-1])
            #vertices_file.close()
            jsonDumpSafe(vertices_filename,{"vertices":vertice_list})
            
            
            current_path = make_polygon_dense(skin_poly["polygons"][i]["outer"]["path"])
            for vertex_out in current_path:
                xyz_all_file.write("{} {} {}\n".format(vertex_out[0]*sequence_parameters["pixel_spacing_x"],vertex_out[1]*sequence_parameters["pixel_spacing_y"],slice_number*sequence_parameters["distance_between_slices"]))
            
            vertices_filename = os.path.normpath(out_path + "/" + os.path.basename(polygon_file).rsplit('_',1)[0] + '_poly{:01}_vertices_dense.txt'.format(i))
            #vertices_file = open(vertices_filename, "w+")
            dense_vertice_list = []
            dense_vert_avg = [0,0]
            for vertex_out in current_path:
                dense_vertice_list.append([vertex_out[0], vertex_out[1],0])
                dense_vert_avg[0] += vertex_out[0]
                dense_vert_avg[1] += vertex_out[1]
            dense_vert_avg[0] /= len(current_path)
            dense_vert_avg[1] /= len(current_path)
            
#            dense_vertice_list.insert(0,[int(vert_avg[0]), int(vert_avg[1]),-1])
#            jsonDumpSafe(vertices_filename,{"vertices":dense_vertice_list})
            
            #draw points where the polygon edges are
#            for i2 in range(len(skin_poly["polygons"])):
#                current_path = make_polygon_dense(skin_poly["polygons"][i2]["outer"]["path"])
#                for j in range(len(current_path)):
#                    point = current_path[j]
#                    cv.circle(skin_labels_in, (point[0], point[1]), radius=0, color=(0, 0, 255), thickness=-1)
            #draw lines
            
            print("stan listy przed zmiana: {}".format(dense_vertice_list))
            for j in range(len(node_ranges[i])):
                #for each line found in contour number i in current slice
                
                #first, mark the straight line vertices as flexible
                if mark_all_as_flexible:
                    index = node_ranges[i][j][7]
                    print("linia: {} do {}".format(node_ranges[i][j][7],node_ranges[i][j][7] + node_ranges[i][j][1]))
                    index_dir = 1
                    while index != (node_ranges[i][j][7] + node_ranges[i][j][1])%len(current_path):
                        dense_vertice_list[index][2] = 1 #flexible_out
                        index = (index + index_dir) % len(current_path)
                        #print("  punkt {}".format(index))
                
                all_straight_lines = node_ranges[i]
                all_straight_lines.sort(key = lambda x: x[0])
                #choose only the longest non-overlapping lines
                already_a_line = np.zeros(len(node_ranges[i]))
                
                if all_straight_lines[j][0] > min_len_for_marking:
                    logging.info("Line detected: {} poly {}, node {}, len {}, to node {}".format(polygon_file, i, all_straight_lines[j][7], all_straight_lines[j][0], all_straight_lines[j][7]+all_straight_lines[j][1]))
                    
                    if(sum(already_a_line[all_straight_lines[j][7]:all_straight_lines[j][7]+all_straight_lines[j][1]])) == 0:
                        #there is no line here yet
                        if not mark_all_as_flexible:
                            index = all_straight_lines[j][7]
                            print("linia: {} do {}".format(all_straight_lines[j][7],all_straight_lines[j][7] + all_straight_lines[j][1]))
                            index_dir = 1
                            while index != (node_ranges[i][j][7] + node_ranges[i][j][1])%len(current_path):
                                dense_vertice_list[index][2] = 1 #flexible_out
                                index = (index + index_dir) % len(current_path)
                        cv.line(skin_labels_in,(all_straight_lines[j][2], all_straight_lines[j][3]),(all_straight_lines[j][4], all_straight_lines[j][5]),(all_straight_lines[j][6]*200,255,0),1)
                        coordinates[i].append([(all_straight_lines[j][2], all_straight_lines[j][3]), (all_straight_lines[j][4], all_straight_lines[j][5])])
                        already_a_line[all_straight_lines[j][7]:all_straight_lines[j][7]+all_straight_lines[j][1]] = 1
                        #write the x y z coordinates of the nodes to separate file
                        for vertex_out in current_path[all_straight_lines[j][7]:all_straight_lines[j][7]+all_straight_lines[j][1]]:
                            xyz_file.write("{} {} {}\n".format(vertex_out[0]*sequence_parameters["pixel_spacing_x"],vertex_out[1]*sequence_parameters["pixel_spacing_y"],slice_number*sequence_parameters["distance_between_slices"]))
            
            process_dense_vertice_list(dense_vertice_list)
            
            
            print("stan listy po zmianach: {}".format(dense_vertice_list))
            dense_vertice_list.insert(0,[int(vert_avg[0]), int(vert_avg[1]),-1])
            jsonDumpSafe(vertices_filename,{"vertices":dense_vertice_list})
            
                    
        print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n\n\n")
        skin_labels_file_out = os.path.normpath(oDir + "/skinlines/" + os.path.basename(polygon_file).rsplit('_',1)[0] + 'lines_labels.png')
        skin_coords_file_out = os.path.normpath(oDir + "/skinlines/" + os.path.basename(polygon_file).rsplit('_',1)[0] + 'lines_polygons.json')
        cv.imwrite(skin_labels_file_out, skin_labels_in)
        
        #coordinates file - for each polygon, a list of start and end points of the lines are provided for each slice
        coordinates_save = {'polygons':[]}
        #loop over all polygons in the input data and the lines within each polygon
        #but the output data will have a separate polygon for each line!
        for i in range(len(coordinates)):
            #for each polygon in input data
            for j in range(len(coordinates[i])):
                #for each line in the given polygon from the input data
                if (len(coordinates[i])==0):
                    continue
                point_list = [coordinates[i][j][0], (coordinates[i][j][0][0]+1, coordinates[i][j][0][1]), coordinates[i][j][1], (coordinates[i][j][1][0]+1, coordinates[i][j][1][1])]
                
                
                minx = min([a[0] for a in point_list])
                maxx = max([a[0] for a in point_list])
                miny = min([a[1] for a in point_list])
                maxy = max([a[1] for a in point_list])
                
                if miny == maxy:
                    #horizontal line - needs to be extended upwards
                    point_list[1] = (point_list[1][0], point_list[1][1]+1)
                    point_list[3] = (point_list[3][0], point_list[3][1]+1)
                    maxy += 1
                
                coordinates_save['polygons'].append({'outer':{"path":point_list, "box": [minx, miny, maxx, maxy]}})
        jsonDumpSafe(skin_coords_file_out, coordinates_save)
        slice_number += 1
    xyz_file.close()

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

max_node_range = 80
max_distance_from_line = 4
min_len_for_marking = 30
max_sum_of_distances_from_line = 8

parser = ArgumentParser()

parser.add_argument("-iDir",      "--input_dir"      ,     dest="idir"   ,    help="input directory" ,    metavar="PATH", required=True)
parser.add_argument("-oDir",      "--output_dir"     ,     dest="odir"   ,    help="output directory",    metavar="PATH", required=True)
parser.add_argument("-pDir",      "--param_dir"     ,     dest="pdir"   ,    help="session parameters directory",    metavar="PATH", required=True)

parser.add_argument("-v"   ,      "--verbose"        ,     dest="verbose",    help="verbose level"   ,                    required=False)

args = parser.parse_args()

verbose = 'off'                 if args.verbose is None else args.verbose
iDir 	= args.idir
oDir  	= args.odir
pDir    = args.pdir

logging.basicConfig(level=logging.DEBUG,format="%(asctime)s [%(levelname)s] %(message)s",handlers=[logging.FileHandler(oDir+"/detectStraight.log",mode='w'),logging.StreamHandler(sys.stdout)])


if not os.path.isdir(iDir):
    logging.error('Error : Input directory (%s) not found !',iDir)
    exit(1)

if not os.path.isdir(pDir):
    logging.error('Error : Set parameter directory (%s) not found !',pDir)
    exit(1)


logging.info("\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
logging.info("START:     as_detectStraight.py")
logging.info("in:       "    +   iDir    )
logging.info("out:      "    +   oDir    )
logging.info("param:      "    +   pDir    )
logging.info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n")


log2 = open(oDir+"/detectStraight_results.log","a+")

if verbose == 'off':
    verbose = False
else:
    verbose = True
    
# try:
    # conffh = open(conffn)
    # configuration = json.load(conffh)
    # conffh.close()
# except Exception as err:
    # logging.error("Input data IO error: {}".format(err))
    # sys.exit(1)
    
warnings.simplefilter('ignore', np.RankWarning)
process_dir(iDir, oDir, pDir, log2, verbose)

log2.close()