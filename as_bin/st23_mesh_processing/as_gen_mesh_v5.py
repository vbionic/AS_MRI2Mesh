import sys, getopt, shutil
import pathlib
from datetime import datetime
import time
import pydicom
import numpy as np
import json
import os
import math
import glob
import tracemalloc
import multiprocessing
#import mapbox_earcut as earcut
from itertools import permutations, combinations
from more_itertools import nth_combination
import random

import trimesh
from trimesh.transformations import rotation_matrix, translation_matrix, scale_matrix, compose_matrix
from trimesh.primitives import Cylinder, Sphere, Box
import shapely
from shapely.geometry.polygon import Polygon, LinearRing

#-----------------------------------------------------------------------------------------
from scipy import ndimage, misc
from skimage import data, img_as_float
from skimage import exposure
#-----------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits import mplot3d
#-----------------------------------------------------------------------------------------
from argparse import ArgumentParser
import logging
#-----------------------------------------------------------------------------------------
sys.path.append(os.getcwd())
#-----------------------------------------------------------------------------------------
from v_utils.v_contour import *
from v_utils.v_polygons import *
from v_utils.v_json import jsonUpdate, jsonDumpSafe
from v_utils.v_arg import arg2boolAct


#-----------------------------------------------------------------------------------------
# globals to be filled in main()
slice_dist       = 1.0
pixel_spacing_x  = 1.0
pixel_spacing_y  = 1.0
#-----------------------------------------------------------------------------------------

def show_meshes(mesh_list, colours_list = None):
    from trimesh.viewer import windowed
    import pyglet
    scene = trimesh.Scene()

    _point_to_show = []

    for mid, mesh in enumerate(mesh_list):
        if (colours_list is None):
            if(mid == 0):
                color = (0.5, 0.5, 0.5, 1.0)
            
            else:
                color = (*np.random.rand(3,), 0.65)
        elif (len(colours_list) == 3 or len(colours_list) == 4) and (not hasattr(colours_list[0], 'len')):
            color = colours_list
        else:
            color = colours_list[mid]

        if not(mesh.is_empty):
            if type(mesh) is trimesh.PointCloud:
                mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))
            else:
                mesh.visual.face_colors = np.tile(color, (len(mesh.faces), 1))
            scene.add_geometry(mesh)
        _point_to_show.extend(mesh.vertices)
        
    scene.camera_transform = scene.camera.look_at(points = _point_to_show,
                             rotation = trimesh.transformations.euler_matrix(np.pi / 2, 0, np.pi / 4))

    window = windowed.SceneViewer(scene, start_loop=False)
    window.toggle_axis()
    pyglet.app.run()
    
    
def fixes_chain(mesh, do_fix_invertion = False, do_fix_normals = False):

    mesh.remove_unreferenced_vertices()
    if(do_fix_invertion):
        trimesh.repair.fix_inversion(mesh)
    if do_fix_normals:
        trimesh.repair.fix_normals(mesh)
    mesh.merge_vertices()
    try:
        mesh.process(validate=True) # merging duplicate vertices
    except:
        logging.warning(f"  Processing failed")
    if(len(mesh.edges_face)>0):
        if((len(mesh.faces) < (mesh.edges_face.max()+1))):
            logging.warning(f" After processing I detected that mesh is corupted: I detected that number of faces {len(mesh.faces)} and max id of edge face +1 {mesh.edges_face.max()+1} do not match!")
        logging.info(f" Recreate the mesh from vertices and faces...")
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)

    return mesh

def create_meshes_from_outlines(mesh):
    outline_meshes = []
    mo = mesh.outline()
    outlines_perims_to_plug = []
    for olid, ol in enumerate(mo.entities):
        outline_varts_ids = ol.points
        ovs  = [mesh.vertices[p] for p in outline_varts_ids]
        ovs_np = np.array(ovs)
        has_depth = len(np.unique(ovs_np[:,2])) > 1

        created = False
        if(len(ovs) <= 2):
            continue
        elif(len(ovs) == 3):
            triangle_mesh = trimesh.Trimesh(vertices = ovs, faces = [(0,1,2)])
            outline_meshes.append(triangle_mesh)
            created = True

        elif not has_depth:
            try:
                p2p = Polygon(ovs)
                p2p_vertices, p2p_faces = trimesh.creation.triangulate_polygon(p2p)
                p2p_vertices3D = [[*v, find_z_of_closest_xy(v, ovs)] for v in p2p_vertices]
                plane_mesh = trimesh.Trimesh(vertices = p2p_vertices3D, faces = p2p_faces)
                outline_meshes.append(plane_mesh)
                created = True
            except:
                created = False

        if not created:
            tries_num = 0
            while not created and tries_num < 10:
                if not has_depth:
                    #convex_hull returns error if all Zs are the same
                    ovs = [[ovs[i][0], ovs[i][1], ovs[i][2] + (np.random.rand()-0.5)*0.01] for i in range(len(ovs))]
                outline_p_cloud = trimesh.PointCloud(ovs)
                if outline_p_cloud.shape[0] > 3:
                    try:
                        outline_mesh = outline_p_cloud.convex_hull
                        outline_meshes.append(outline_mesh)
                        created = True
                    except:
                        created = False
                        has_depth = False
                        tries_num += 1
                else:
                    continue
    return outline_meshes

#-----------------------------------------------------------------------------------------

def load_json(pol_path):

    try:
        pol_file 	= open(pol_path,"r");     
        pol_dict	= json.load(pol_file)
        logging.info(' loading %s file'%(os.path.basename(pol_path)))
        return(pol_dict)
    except:
        logging.error('cannot open %s file !!!'%(pol_path))
        exit(1)

#-----------------------------------------------------------------------------------------

def group_load_polygons(inputdir, name, iter_list, limit_sr):

    tmp = []
    allowed_idxs = list(range(*limit_sr))
    for idx, xpath in enumerate(iter_list):

        if idx in allowed_idxs:
            xname           	= os.path.basename(xpath)
            fname, fext     	= os.path.splitext(xname)
            fname, ftis, fsuf   = fname.split('_')

            poly_path           = inputdir + "/" + name + "/" + fname + "_" + name + "_polygons.json"

            poly_dict           = load_json(poly_path)
        
            v_polys             = v_polygons.from_dict(poly_dict)
            tmp.append(v_polys)

    return(tmp)  

#-----------------------------------------------------------------------------------------

def group_conv_polygons(poly_list, max_verts_dist):
    global pixel_spacing_x
    global pixel_spacing_y

    t_poly 	        = []
    t_cent  	    = []

    for in_polys in poly_list:

        v_polys = in_polys.convert_to_float_polys(save_int_contours = True)
        v_polys.scale(pixel_spacing_x, pixel_spacing_y)
        v_polys.fill_mass_centers2()
        removed  = v_polys.remove_colinear_verts()
        inserted = v_polys.interpolate_verts    (max_verts_dist = max_verts_dist, force_int = False)
        logging.info(f" Removed  {removed} colinear pts and inserted {inserted} pts during interpolation for max dist {max_verts_dist} points.")

        t_poly.append(v_polys)

    return(t_poly)

#-----------------------------------------------------------------------------------------
def calc_box_overlap (b1, b2):
    b1xs = b1[0],b1[2]
    b2xs = b2[0],b2[2]
    rxs = max(b1xs[0], b2xs[0]), min(b1xs[1], b2xs[1]) 
    if(rxs[1] <= rxs[0]):
        return None
    
    b1ys = b1[1],b1[3]
    b2ys = b2[1],b2[3]
    rys = max(b1ys[0], b2ys[0]), min(b1ys[1], b2ys[1]) 
    if(rys[1] <= rys[0]):
        return None
    
    overlap_box = [rxs[0], rys[0], rxs[1], rys[1]]

    return overlap_box

boxes_filled_and_square_n = 0
boxes_filled_n = 0
boxes_square_n = 0
boxes_trouble_n = 0
boxex_total_n = 0
def contours_dif_cost_overlap(cont_1, cont_2):

    global boxes_filled_and_square_n
    global boxes_filled_n
    global boxes_square_n
    global boxes_trouble_n
    global boxex_total_n 
    
    global pixel_spacing_x
    global pixel_spacing_y

    boxex_total_n += 1
    
    box1 = cont_1['box']
    box2 = cont_2['box']
    box_overlap = calc_box_overlap(box1, box2)
    if(box_overlap is None):
        ovr = 0
    else:
        box1, area1, centr1 = cont_1['box'], cont_1['area'], cont_1['centroid']
        box2, area2, centr2 = cont_2['box'], cont_2['area'], cont_2['centroid']
        count1_dx, count1_dy =  (box1[2] - box1[0], box1[3] - box1[1])
        count2_dx, count2_dy =  (box2[2] - box2[0], box2[3] - box2[1])
        box1_area = count1_dx * count1_dy
        box2_area = count2_dx * count2_dy
        cont1_squareness = min(count1_dx, count1_dy) / max(count1_dx, count1_dy)
        cont2_squareness = min(count2_dx, count2_dy) / max(count2_dx, count2_dy)
        square2round_ratio = (0.25*np.pi) / 1 
        cont1_box_fill = area1 / (box1_area * square2round_ratio)
        cont2_box_fill = area2 / (box2_area * square2round_ratio)
        boxes_square  = (cont1_squareness > 0.75 and cont2_squareness > 0.75)
        boxes_filled = ((cont1_box_fill / square2round_ratio)  > 0.8 and (cont2_box_fill / square2round_ratio)  > 0.8)

        if boxes_square and boxes_filled:
            d = np.linalg.norm(centr1 - centr2) 
            r1 = np.power((area1) / np.pi, 0.5)
            r2 = np.power((area2) / np.pi, 0.5)
            rl = max(r1, r2)
            rs = min(r1, r2)
            if(d > rl+rs):
                ovr = 0.0
            elif (d > rl):
                ovr = 0.5*(rs - (d - rl))/rs
            elif (d+rs > rl):
                ovr = 1.0 - 0.5*(d + rs - rl)/rs
            else:
                ovr = 1.0
        
            boxes_filled_and_square_n += 1
        #elif boxes_filled:
        #    area_min = min(box1_area, box2_area)
        #        
        #    countOvr_dx, countOvr_dy =  (box_overlap[2] - box_overlap[0], box_overlap[3] - box_overlap[1])
        #    boxOvr_area = countOvr_dx * countOvr_dy
        #    ovr = boxOvr_area/area_min
        #
        #    boxes_filled_n += 1
        ##elif boxes_square:
        #else:
        else:

            boxes_trouble_n += 1
            
            c1_area = cont_1['area']
            c2_area = cont_2['area']
            c1i = v_contour(path_points=cont_1['int_path'], box = cont_1['int_box' ])
            c2i = v_contour(path_points=cont_2['int_path'], box = cont_2['int_box' ])
            box_overlap_int = [math.floor(box_overlap[0]/pixel_spacing_x ), math.floor(box_overlap[1]/pixel_spacing_y),
                               math.ceil (box_overlap[2]/pixel_spacing_x ), math.ceil (box_overlap[3]/pixel_spacing_y) ]
            ddg = False
            if(ddg):
                c1i_org_img = c1i.as_image(fill=True, val = 255)
                c2i_org_img = c2i.as_image(fill=True, val = 255)
                c1i_org_img.save("c1i_org_img.png")
                c2i_org_img.save("c2i_org_img.png")
            c1i.crop_box(box_overlap_int)
            c2i.crop_box(box_overlap_int)
            if(ddg):
                c1i_crp_img = c1i.as_image(fill=True, val = 255)
                c2i_crp_img = c2i.as_image(fill=True, val = 255)
                c1i_crp_img.save("c1i_crp_img.png")
                c2i_crp_img.save("c2i_crp_img.png")
            c1i_np = c1i.as_numpy_mask(w=box_overlap_int[2]- box_overlap_int[0], h=box_overlap_int[3]- box_overlap_int[1])
            c2i_np = c2i.as_numpy_mask(w=box_overlap_int[2]- box_overlap_int[0], h=box_overlap_int[3]- box_overlap_int[1])
            ovr_int_sum = np.sum((c1i_np != 0) & (c2i_np != 0))
            ovr_sum = ovr_int_sum * pixel_spacing_x * pixel_spacing_y

            ovr = ovr_sum / min(c1_area, c2_area)

    cost = 1-ovr
    return cost

def contours_dif_cost_ang(cont_1, cont_2):
    area1, centr1 = cont_1['area'], cont_1['centroid']
    area2, centr2 = cont_2['area'], cont_2['centroid']
    pos_dist = np.linalg.norm(centr1 - centr2) 
    r1 = np.power((area1) / np.pi, 0.5)
    r2 = np.power((area2) / np.pi, 0.5)
    cost = pos_dist - max(r1, r2)
    return cost

def arg_of_min_contours_dif_cost(cont_c, cont_list, method):
    if(method == 'overlap'):
        costs = [contours_dif_cost_overlap(cont_c, cont_list[i]) for i in range(len(cont_list))]
    elif(method == 'angle'):
        costs = [contours_dif_cost_ang    (cont_c, cont_list[i]) for i in range(len(cont_list))]
    #arg_min = np.argmin(costs)
    return costs#arg_min, costs[arg_min]

def contours_lists_dif_cost(cont_list1, cont_list2, method):

    costs_l = [arg_of_min_contours_dif_cost(cont_list1[i], cont_list2, method) for i in range(len(cont_list1))]

    return np.array(costs_l)

def calc_len_of_path(pth):

    org_pth = np.array(pth[:])
    shf_pth = np.array([*pth[1:], pth[0]])
    len = np.sum(np.linalg.norm(org_pth - shf_pth, axis=1))

    return len

def split_to_counours(countour_to_split, ref_contours):
    refs_cs = [ref['centroid']               for ref in ref_contours]
    refs_as = [ref['area'    ]               for ref in ref_contours]
    refs_ls = [calc_len_of_path(ref['path']) for ref in ref_contours]
    ref_mc = np.average(refs_cs, axis=0, weights=refs_as)
            
    c2s_pth = countour_to_split['path'][:-1] # without the overlapping point

    c2s_pth_len  = len(c2s_pth)
    c2s_pth_ext = [*c2s_pth, *c2s_pth]
            

    min_cost_e1 = 1000000.0
    def_c2s_pth1_start   = -1
    def_c2s_pth1_len     = -1
    
    # znajdz przyblizony stosunek w jakim należy podzielic obwod zrodlowego poligonu na dwa mniejsze - proporcjonalnie do dlugosci obwodow docelowych poligonow
    ## stosunek dlugosci obwodow powinien byc proporcjonalny do pierwiastka stosunku powierzchni
    #refs_perim_ratio = np.power(refs_as[0] / refs_as[1], 0.5)
    refs_perim_ratio_def = refs_ls[0] / refs_ls[1]

    # 1) znajdz wstępny podział przez sprawdzenie kilku punktow startowych i tylko jednego stosunku dlugosci obwodow
    #  (zakladam rownomierne probkowanie obwodu - dlugosc bylaby proporcjonalna do liczby punktow)
    #  dodatkowo, na tym etapie nie wymagaj zeby po podziale wychodzily poprawne poligony - to dopiero w drugim etapie
    # 
    c2s_pth1_len = int(round(c2s_pth_len * (refs_perim_ratio_def / (refs_perim_ratio_def + 1.0))))
    c2s_pth2_len = c2s_pth_len - c2s_pth1_len + 1
    if(c2s_pth1_len < 3):
        c2s_pth1_len = 3
        c2s_pth2_len = c2s_pth_len - c2s_pth1_len + 1
    if(c2s_pth2_len < 3):
        c2s_pth2_len = 3
        c2s_pth1_len = c2s_pth_len - c2s_pth2_len + 1
    start_point_step = max(1, c2s_pth_len//10)
    for c2s_pth1_start in range(0, c2s_pth_len, start_point_step):

        extr_line_ends = np.array([c2s_pth_ext[c2s_pth1_start + c2s_pth1_len], c2s_pth_ext[c2s_pth1_start]])
        extr_line_contour_pth = [*extr_line_ends, extr_line_ends[0]]
        extr_line_contour = v_contour(extr_line_contour_pth)
        inserted = extr_line_contour.interpolate_path(2.5, force_int = False)
        add_points_num = inserted // 2
        add_points = extr_line_contour.store['path'][1:add_points_num+1]
        
        if((c2s_pth1_len < 3) or (c2s_pth2_len < 3)):
            continue

        c2s_pth1 = [*c2s_pth_ext[c2s_pth1_start                : c2s_pth1_start + c2s_pth1_len                + 1], *add_points      , c2s_pth_ext[c2s_pth1_start              ]]
        c2s_pth2 = [*c2s_pth_ext[c2s_pth1_start + c2s_pth1_len : c2s_pth1_start + c2s_pth1_len + c2s_pth2_len + 0], *add_points[::-1], c2s_pth_ext[c2s_pth1_start + c2s_pth1_len]]

        c2s_poly1 = v_polygons() 
        c2s_poly1.add_polygon_from_paths(c2s_pth1)
        c2s_poly1.fill_mass_centers2()
        c2s_poly2 = v_polygons() 
        c2s_poly2.add_polygon_from_paths(c2s_pth2)
        c2s_poly2.fill_mass_centers2()
                
        #ref_mc = np.average(refs_cs, axis=0, weights=refs_as)
        #div_mc = np.average([c2s_poly1["com"], c2s_poly2["com"]], axis=0, weights=[c2s_poly1['m'], c2s_poly2['m']])
        #com_dist11 = np.linalg.norm((c2s_poly1["com"]-div_mc) - (refs_cs[0]-ref_mc))
        #com_dist22 = np.linalg.norm((c2s_poly2["com"]-div_mc) - (refs_cs[1]-ref_mc))
        #com_dist12 = np.linalg.norm((c2s_poly1["com"]-div_mc) - (refs_cs[1]-ref_mc))
        #com_dist21 = np.linalg.norm((c2s_poly2["com"]-div_mc) - (refs_cs[0]-ref_mc))   
        cmd12div = c2s_poly1["com"] - c2s_poly2["com"]
        cmd12ref = refs_cs[0]       - refs_cs[1]      
        ang12div = np.arctan2(*cmd12div)
        ang12ref = np.arctan2(*cmd12ref)
        phase = (ang12div - ang12ref)
        ang12_cost = abs((phase + np.pi) % (2 * np.pi) - np.pi) / np.pi  
         
        #com_dist_cost  = np.sqrt(com_dist11**2 + com_dist22**2)
        
        div_line_cost = np.linalg.norm(extr_line_ends[1] - extr_line_ends[0]) / (2*np.sqrt(countour_to_split['area'] / np.pi))
        
        ref_ms_ratio = refs_as[0] / refs_as[1]
        div_ms_ratio = c2s_poly1['m'] / c2s_poly2['m']
        ms_ratio_sqrt = np.sqrt(np.max([ref_ms_ratio, div_ms_ratio]) / np.min([ref_ms_ratio, div_ms_ratio]))
        
        #area_and_line_cost  = np.sqrt(div_line_cost**2 + ms_ratio_sqrt**2)
        area_and_line_cost  = (div_line_cost + 2*ms_ratio_sqrt)/3

        # koszt jako iloczyn kosztu katowego i kosztu wynikajacego z roznicy powierzchni podzialow oraz dlugosci linii podzialu
        #cost = ang12_cost + 0.3* (area_and_line_cost - 1) + ang12_cost * area_and_line_cost
        cost =              0.3* (area_and_line_cost - 1) + ang12_cost * area_and_line_cost
        
        dbc = False  and countour_to_split['area']==4.375
        if dbc:
            fig = plt.figure()
            plt.gca().set_aspect('equal', adjustable='box')
            if(len(fig.axes) == 0):
                ax = fig.add_subplot()
            else:
                ax = fig.axes[0]
        if dbc and True:
            c2s_pth_np = np.array(c2s_pth)
            ref_contours_np = [np.array(ref['path']) for ref in ref_contours]
            ax.plot(c2s_pth_np        [:,0], c2s_pth_np        [:,1], c=[0.9, 0.0, 0.0], marker = "o", alpha=0.7)
            ax.plot(ref_contours_np[0][:,0], ref_contours_np[0][:,1], c=[0.0, 0.9, 0.0], marker = ".", alpha=0.3)
            ax.plot(ref_contours_np[1][:,0], ref_contours_np[1][:,1], c=[0.0, 0.0, 0.9], marker = ".", alpha=0.3)
            ax.plot(c2s_pth_np        [0,0], c2s_pth_np        [0,1], c=[0.9, 0.0, 0.0], marker = "o", alpha=0.7, markersize=14)
            #plt.show()
        if dbc:# and False:
            c2s_pth1_np = np.array(c2s_poly1['polygons'][0]['outer']['path'])
            c2s_pth2_np = np.array(c2s_poly2['polygons'][0]['outer']['path'])
            ax.plot(c2s_pth1_np        [:,0], c2s_pth1_np        [:,1], c=[0.0, 0.9, 0.1], marker = "x", alpha=0.8)
            ax.plot(c2s_pth2_np        [:,0], c2s_pth2_np        [:,1], c=[0.0, 0.1, 0.9], marker = "x", alpha=0.8)
            ax.plot(extr_line_ends     [:,0], extr_line_ends     [:,1], c=[1.0, 0.1, 0.1], marker = "o", alpha=0.8)
            print(f"cost {cost:.2f} => ang12_cost {ang12_cost:.2f}, area_and_line_cost{area_and_line_cost:.2f} (div_line_cost {div_line_cost:.2f}, ms_ratio_sqrt {ms_ratio_sqrt:.2f}), c2s_pth1_start {c2s_pth1_start:.2f}, refs_perim_ratio{refs_perim_ratio_def:.2f}")
            plt.show()
        if(ang12_cost >= 0.5) or (ms_ratio_sqrt > 2.5 and min(c2s_pth1_len, c2s_pth2_len) > 10):
            continue
        if(cost < min_cost_e1):
            min_cost_e1 = cost
            def_c2s_pth1_start = c2s_pth1_start
            def_c2s_pth1_len   = c2s_pth1_len
            
            dbc = False  and countour_to_split['area']==4.375
            if dbc:
                fig = plt.figure()
                plt.gca().set_aspect('equal', adjustable='box')
                if(len(fig.axes) == 0):
                    ax = fig.add_subplot()
                else:
                    ax = fig.axes[0]
            if dbc and True:
                c2s_pth_np = np.array(c2s_pth)
                ref_contours_np = [np.array(ref['path']) for ref in ref_contours]
                ax.plot(c2s_pth_np        [:,0], c2s_pth_np        [:,1], c=[0.9, 0.0, 0.0], marker = "o", alpha=0.7)
                ax.plot(ref_contours_np[0][:,0], ref_contours_np[0][:,1], c=[0.0, 0.9, 0.0], marker = ".", alpha=0.3)
                ax.plot(ref_contours_np[1][:,0], ref_contours_np[1][:,1], c=[0.0, 0.0, 0.9], marker = ".", alpha=0.3)
                ax.plot(c2s_pth_np        [0,0], c2s_pth_np        [0,1], c=[0.9, 0.0, 0.0], marker = "o", alpha=0.7, markersize=14)
                #plt.show()
            if dbc:# and False:
                c2s_pth1_np = np.array(c2s_poly1['polygons'][0]['outer']['path'])
                c2s_pth2_np = np.array(c2s_poly2['polygons'][0]['outer']['path'])
                ax.plot(c2s_pth1_np        [:,0], c2s_pth1_np        [:,1], c=[0.0, 0.9, 0.1], marker = "x", alpha=0.8)
                ax.plot(c2s_pth2_np        [:,0], c2s_pth2_np        [:,1], c=[0.0, 0.1, 0.9], marker = "x", alpha=0.8)
                ax.plot(extr_line_ends     [:,0], extr_line_ends     [:,1], c=[1.0, 0.1, 0.1], marker = "o", alpha=0.8)
                print(f"cost {cost:.2f} => ang12_cost {ang12_cost:.2f}, area_and_line_cost{area_and_line_cost:.2f} (div_line_cost {div_line_cost:.2f}, ms_ratio_sqrt {ms_ratio_sqrt:.2f}), c2s_pth1_start {c2s_pth1_start:.2f}, refs_perim_ratio{refs_perim_ratio_def:.2f}")
                plt.show()


      
    # 2) znajdz podział przez punktow podzialu w okolicach tych wyznaczonych w etapie 1)
    #  dodatkowo, na tym etapie wymagaj zeby po podziale wychodzily poprawne poligony, czyli
    #  nie ich obwody nie przecinaly samych siebie. 
    #       
    min_cost_e2 = 1000000.0
    found_valid = False
    best_pth1_start = def_c2s_pth1_start
    best_pth1_len   = def_c2s_pth1_len

    points_max_offset = max(1, start_point_step+1)

    while not found_valid:
        pth1_start_range = range(max(0, best_pth1_start - points_max_offset), min(best_pth1_start + points_max_offset + 1, c2s_pth_len+1))
        best_pth1_stop   = best_pth1_start + best_pth1_len - 1
        pth1_stop_range  = range(best_pth1_stop  - points_max_offset, best_pth1_stop  + points_max_offset + 1)

        for c2s_pth1_start in pth1_start_range:
           pth1_stop_range_limited = range(max(c2s_pth1_start+1, pth1_stop_range.start), pth1_stop_range.stop)
           for c2s_pth1_stop in pth1_stop_range_limited: 
                
                c2s_pth1_stop  = min(c2s_pth1_stop, c2s_pth1_start + c2s_pth_len-2)

                # zakladam rownomierne probkowanie obwodu - dlugosc bylaby proporcjonalna do liczby punktow
                c2s_pth1_len = c2s_pth1_stop - c2s_pth1_start + 1
                c2s_pth2_len = c2s_pth_len - c2s_pth1_len + 1

                if(c2s_pth1_len < 3) or (c2s_pth2_len < 3):
                    continue
                
                extr_line_ends = np.array([c2s_pth_ext[c2s_pth1_start + c2s_pth1_len], c2s_pth_ext[c2s_pth1_start]])
                extr_line_contour_pth = [*extr_line_ends, extr_line_ends[0]]
                extr_line_contour = v_contour(extr_line_contour_pth)
                inserted = extr_line_contour.interpolate_path(2.5, force_int = False)
                add_points_num = inserted // 2
                add_points = extr_line_contour.store['path'][1:add_points_num+1]

                if((c2s_pth1_len+add_points_num) <= 3 or (c2s_pth2_len+add_points_num) <= 3):
                    continue

                c2s_pth1 = [*c2s_pth_ext[c2s_pth1_start                : c2s_pth1_start + c2s_pth1_len                + 1], *add_points      , c2s_pth_ext[c2s_pth1_start               ]]
                c2s_pth2 = [*c2s_pth_ext[c2s_pth1_start + c2s_pth1_len : c2s_pth1_start + c2s_pth1_len + c2s_pth2_len + 0], *add_points[::-1], c2s_pth_ext[c2s_pth1_start + c2s_pth1_len]]

                c2s_poly1 = v_polygons() 
                c2s_poly1.add_polygon_from_paths(c2s_pth1)
                c2s_poly1.fill_mass_centers2()
                c2s_poly2 = v_polygons() 
                c2s_poly2.add_polygon_from_paths(c2s_pth2)
                c2s_poly2.fill_mass_centers2()
                
                if(countour_to_split.store['area'] != (c2s_poly1.store['m'] + c2s_poly2.store['m'])):
                    continue
                
                #ref_mc = np.average(refs_cs, axis=0, weights=refs_as)
                #div_mc = np.average([c2s_poly1["com"], c2s_poly2["com"]], axis=0, weights=[c2s_poly1['m'], c2s_poly2['m']])
                #com_dist11 = np.linalg.norm((c2s_poly1["com"]-div_mc) - (refs_cs[0]-ref_mc))
                #com_dist22 = np.linalg.norm((c2s_poly2["com"]-div_mc) - (refs_cs[1]-ref_mc))
                #com_dist12 = np.linalg.norm((c2s_poly1["com"]-div_mc) - (refs_cs[1]-ref_mc))
                #com_dist21 = np.linalg.norm((c2s_poly2["com"]-div_mc) - (refs_cs[0]-ref_mc))   
                cmd12div = c2s_poly1["com"] - c2s_poly2["com"]
                cmd12ref = refs_cs[0]       - refs_cs[1]      
                ang12div = np.arctan2(*cmd12div)
                ang12ref = np.arctan2(*cmd12ref)
                phase = (ang12div - ang12ref)
                ang12_cost = abs((phase + np.pi) % (2 * np.pi) - np.pi) / np.pi  
         
                #com_dist_cost  = np.sqrt(com_dist11**2 + com_dist22**2)
        
                div_line_cost = np.linalg.norm(extr_line_ends[1] - extr_line_ends[0]) / (2*np.sqrt(countour_to_split['area'] / np.pi))
        
                ref_ms_ratio = refs_as[0] / refs_as[1]
                div_ms_ratio = c2s_poly1['m'] / c2s_poly2['m']
                ms_ratio_sqrt = np.sqrt(np.max([ref_ms_ratio, div_ms_ratio]) / np.min([ref_ms_ratio, div_ms_ratio]))
        
                #area_and_line_cost  = np.sqrt(div_line_cost**2 + ms_ratio_sqrt**2)
                area_and_line_cost  = (div_line_cost + 2*ms_ratio_sqrt)/3

                # koszt jako iloczyn kosztu katowego i kosztu wynikajacego z roznicy powierzchni podzialow oraz dlugosci linii podzialu
                #cost = ang12_cost + 0.3* (area_and_line_cost - 1) + ang12_cost * area_and_line_cost
                cost =              0.3* (area_and_line_cost - 1) + ang12_cost * area_and_line_cost
               
                if(cost < min_cost_e2):

                    lr1 = LinearRing(coordinates = c2s_poly1['polygons'][0]['outer']['path'])
                    if (not lr1.is_valid) or (not lr1.is_simple):
                        continue
                    lr2 = LinearRing(coordinates = c2s_poly2['polygons'][0]['outer']['path'])
                    if (not lr2.is_valid) or (not lr2.is_simple):
                        continue
                    found_valid = True
                    min_cost_e2 = cost
                    best_pth1_start = c2s_pth1_start 
                    best_pth1_len   = c2s_pth1_len
                    c2s_cont1_best = c2s_poly1['polygons'][0]['outer']
                    c2s_cont2_best = c2s_poly2['polygons'][0]['outer']

                    
                    dbd = False and countour_to_split['area']==4.375
                    if dbd:
                        fig = plt.figure()
                        plt.gca().set_aspect('equal', adjustable='box')
                        if(len(fig.axes) == 0):
                            ax = fig.add_subplot()
                        else:
                            ax = fig.axes[0]
                    if dbd and True:
                        c2s_pth_np = np.array(c2s_pth)
                        ref_contours_np = [np.array(ref['path']) for ref in ref_contours]
                        ax.plot(c2s_pth_np        [:,0], c2s_pth_np        [:,1], c=[0.9, 0.0, 0.0], marker = "o", alpha=0.7)
                        ax.plot(ref_contours_np[0][:,0], ref_contours_np[0][:,1], c=[0.0, 0.9, 0.0], marker = ".", alpha=0.3)
                        ax.plot(ref_contours_np[1][:,0], ref_contours_np[1][:,1], c=[0.0, 0.0, 0.9], marker = ".", alpha=0.3)
                        ax.plot(c2s_pth_np        [0,0], c2s_pth_np        [0,1], c=[0.9, 0.0, 0.0], marker = "o", alpha=0.7, markersize=14)
                        #plt.show()
                    if dbd:# and False:
                        c2s_pth1_np = np.array(c2s_poly1['polygons'][0]['outer']['path'])
                        c2s_pth2_np = np.array(c2s_poly2['polygons'][0]['outer']['path'])
                        ax.plot(c2s_pth1_np        [:,0], c2s_pth1_np        [:,1], c=[0.0, 0.9, 0.1], marker = "x", alpha=0.8)
                        ax.plot(c2s_pth2_np        [:,0], c2s_pth2_np        [:,1], c=[0.0, 0.1, 0.9], marker = "x", alpha=0.8)
                        ax.plot(extr_line_ends     [:,0], extr_line_ends     [:,1], c=[1.0, 0.1, 0.1], marker = "o", alpha=0.8)
                        print(f"cost {cost:.2f} => ang12_cost {ang12_cost:.2f}, area_and_line_cost{area_and_line_cost:.2f} (div_line_cost {div_line_cost:.2f}, ms_ratio_sqrt {ms_ratio_sqrt:.2f}), c2s_pth1_start {c2s_pth1_start:.2f}, refs_perim_ratio{refs_perim_ratio_def:.2f}")
                        plt.show()

        
        if not found_valid: 
            if c2s_pth_len < points_max_offset:
                break
            points_max_offset = points_max_offset * 2

               
    if found_valid:
        return c2s_cont1_best, c2s_cont2_best
    else:
        return countour_to_split, None

def match_lines(slice_c, slice_n):
    #slice_c - current slice
    #slice_n - next slice
    
    max_range_betwee_lines = 30
    
    num_c = len(slice_c)  #number of lines in current slice
    num_n = len(slice_n)  #number of lines in next slice
    
    metrics = np.zeros((num_c, num_n))  #metrics for pairs of lines
    matching_lines = np.zeros(num_c)
    
    for i in range(num_c):
        #for each line in current slice calculate metrics to all lines in next slice
        for j in range(num_n):
            metrics[i,j] = get_line_metric(slice_c[i], slice_n[j])
    
    for i in range(num_c):
        minrange = min(metrics[i,:])
        if minrange < max_range_betwee_lines:
            matching_lines[i] = list(metrics[i,:]).index(minrange)
        else:
            matching_lines[i] = -1
    
    return matching_lines
            
def get_line_metric(line_c, line_n):
    #find sum of distances between ends of the lines
    start_c = line_c[0]
    start_n = line_n[0]
    end_c = line_c[1]
    end_n = line_n[1]
    dist1 = sqrt((start_c[0]-start_n[0])*(start_c[0]-start_n[0]) + (start_c[1]-start_n[1])*(start_c[1]-start_n[1])) + sqrt((end_c[0]-end_n[0])*(end_c[0]-end_n[0]) + (end_c[1]-end_n[1])*(end_c[1]-end_n[1]))
    #just to make sure - try to match in thereverse order
    dist2 = sqrt((start_c[0]-end_n[0])*(start_c[0]-end_n[0]) + (start_c[1]-end_n[1])*(start_c[1]-end_n[1])) + sqrt((end_c[0]-start_n[0])*(end_c[0]-start_n[0]) + (end_c[1]-start_n[1])*(end_c[1]-start_n[1]))
    return(min((dist1, dist2)))

def match_contours(cont_c_l, cont_n_l, method = 'overlap', 
                   z_dist=2, max_ang=-10, 
                   min_overlap = 0.2,
                   limit_matches_from_p1 = -1,
                   limit_matches_to_p2   = -1,
                   test_similar_size_polys_first = True):
    # method == 'angle'   - get cost (-1, 1) and threshold by max_ang 
    # method == 'overlap' - get cost ( 0, 1) and threshold by min_overlap
    
    if len(cont_c_l) == 0:
        return {}
    elif len(cont_n_l) == 0:
        return {id:[{"dst_id": -1, "dst_contour": None, "src_id": id, "src_contour": p, "is_complex": False}] for id, p in enumerate(cont_c_l)}

    cs = list(range(len(cont_c_l)))
    ns = list(range(len(cont_n_l)))

    # calculate cost of all possible links between contours on neighbouring layers
    m_l_cs = contours_lists_dif_cost(cont_c_l, cont_n_l, method = method)
    
    # sort linkage costs from the lowest to the highest
    p1top2_linkage_dict = {}
    for ci in cs:
        p1top2_linkage_dict[ci] = []
        
    p2top1_linkage_dict = {}
    for ni in ns:
        p2top1_linkage_dict[ni] = []

    cs_s, ns_s = np.unravel_index(np.argsort(m_l_cs, axis=None), m_l_cs.shape)
    if test_similar_size_polys_first == True:
        a_l_cs = np.array([[1-min(c["area"], n["area"])/max(c["area"], n["area"]) for n in cont_n_l] for c in cont_c_l])
        cs_s, ns_s = np.unravel_index(np.argsort(a_l_cs, axis=None), a_l_cs.shape)

    
    # get threshold cost
    if(method == 'overlap'):
        th_cost = 1-min_overlap
    elif(method == 'angle'):
        th_cost = np.tan(np.deg2rad(max_ang)) * z_dist

    # get linkages according to computed cost, starting from the least costly links
    # starts from the least costly links and respects:
    # - limits of allowed number of connections that can be established from / to a single contour
    # - threshold cost
    for ci, ni in zip(cs_s, ns_s):
        can_link_from_p1 = (limit_matches_from_p1 == -1) or (len(p1top2_linkage_dict[ci]) < limit_matches_from_p1)
        can_link_to_p2   = (limit_matches_to_p2   == -1) or (len(p2top1_linkage_dict[ni]) < limit_matches_to_p2  )
        if(m_l_cs[ci][ni] < th_cost) and can_link_from_p1 and can_link_to_p2:
            p1top2_linkage_dict[ci].append({"dst_id": ni, "dst_contour": cont_n_l[ni], "src_id": ci, "src_contour": cont_c_l[ci], "is_complex": False})
            p2top1_linkage_dict[ni].append({"dst_id": ci, "dst_contour": cont_c_l[ci], "src_id": ni, "src_contour": cont_n_l[ni], "is_complex": False})
            
    # deal with polygons without link
    for ci in cs:
        if len(p1top2_linkage_dict[ci]) == 0:
            p1top2_linkage_dict[ci].append( {"dst_id": -1, "dst_contour": None, "src_id": ci, "src_contour": cont_c_l[ci], "is_complex": False})

    # deal with polygons without link
    for ni in ns:
        if len(p2top1_linkage_dict[ni]) == 0:
            if not -1 in p1top2_linkage_dict.keys():
                p1top2_linkage_dict[-1] = []
            p1top2_linkage_dict[-1].append( {"dst_id": ni, "dst_contour": cont_n_l[ni], "src_id": -1, "src_contour": None, "is_complex": False})
            
    # deal with contour that are linked by multiple contours
    for ni in ns:
        p2top1_lnks = p2top1_linkage_dict[ni]
        if len(p2top1_lnks) == 2:
            
            src = p2top1_lnks[1]["src_contour"] 
            dsts = [x["dst_contour"] for x in p2top1_lnks]
            cont1_best, cont2_best = split_to_counours(src, dsts)

            if cont2_best is None:
                ci = p2top1_lnks[1]["dst_id"]

                # modify the corresponding entries in p1top2_linkage_dict
                src_cnt_to_copy = None
                xids_to_delete = []
                for xid, x in enumerate(p1top2_linkage_dict[ci]):
                    if (x["dst_id"] == ni):
                        src_cnt_to_copy = copy.copy(p1top2_linkage_dict[ci][xid]["src_contour"])
                        xids_to_delete.append(xid)
                if len(xids_to_delete) != 0:
                    p1top2_linkage_dict[ci] = [l for lid, l in enumerate(p1top2_linkage_dict[ci]) if not lid in xids_to_delete]
                        
                if len(p1top2_linkage_dict[ci]) == 0:
                    p1top2_linkage_dict[ci].append( {"dst_id": -1, "dst_contour": None, "src_id": ci, "src_contour": src_cnt_to_copy, "is_complex": False})

                # modify the p2top1_linkage_dict's entry
                del(p2top1_linkage_dict[ni][1])

            else:
                # extend the index info with outer contours of the best result
                p2top1_linkage_dict[ni][0]["src_contour"    ] = cont1_best
                p2top1_linkage_dict[ni][0]["src_contour_org"] = src
                p2top1_linkage_dict[ni][0]["is_complex"     ] = True
                p2top1_linkage_dict[ni][1]["src_contour"    ] = cont2_best
                p2top1_linkage_dict[ni][1]["src_contour_org"] = src
                p2top1_linkage_dict[ni][1]["is_complex"     ] = True

                ci1 = p2top1_lnks[0]["dst_id"]
                ci2 = p2top1_lnks[1]["dst_id"]
                for ci, cont_best in [(ci1, cont1_best), (ci2, cont2_best)]:
                    for x in p1top2_linkage_dict[ci]:
                        if (x["dst_id"] == ni):
                            x["dst_contour"    ] = cont_best
                            x["dst_contour_org"] = src
                            x["is_complex"     ] = True

            
    # deal with contour that are linked to multiple contours
    for ci in cs:
        p1top2_lnks = p1top2_linkage_dict[ci]
        if len(p1top2_lnks) == 2:
            src = p1top2_lnks[1]["src_contour"] 
            dsts = [x["dst_contour"] for x in p1top2_lnks]
            cont1_best, cont2_best = split_to_counours(src, dsts)
            
            if cont2_best is None:
                ni = p1top2_lnks[1]["dst_id"]

                # modify the corresponding entries in p2top1_linkage_dict
                src_cnt_to_copy = None
                xids_to_delete = []
                for xid, x in enumerate(p2top1_linkage_dict[ni]):
                    if (x["dst_id"] == ci):
                        src_cnt_to_copy = copy.copy(p2top1_linkage_dict[ni][xid]["src_contour"])
                        xids_to_delete.append(xid)
                if len(xids_to_delete) != 0:
                    p2top1_linkage_dict[ni] = [l for lid, l in enumerate(p2top1_linkage_dict[ni]) if not lid in xids_to_delete]
                    for p2top1_ld_ni in p2top1_linkage_dict[ni]:
                        if "src_contour_org" in p2top1_ld_ni.keys():
                            p2top1_ld_ni["src_contour"] = p2top1_ld_ni["src_contour_org"]
                            p2top1_ld_ni["is_complex" ] = False
                if len(p2top1_linkage_dict[ni]) == 0:
                    p2top1_linkage_dict[ni].append( {"dst_id": -1, "dst_contour": None, "src_id": ci, "src_contour": src_cnt_to_copy, "is_complex": False})

                # modify the p2top1_linkage_dict's entry
                del(p1top2_linkage_dict[ci][1])
                
                p1top2_ls_ids = list(p1top2_linkage_dict.keys())
                p1top2_ls_ids.remove(ci)
                for p1top2_ls_id in p1top2_ls_ids:
                    p1top2_ls = p1top2_linkage_dict[p1top2_ls_id]
                    for p1top2_l in p1top2_ls:
                        if p1top2_l["dst_id"] == ni:
                            if "dst_contour_org" in p1top2_l.keys():
                                p1top2_l["dst_contour"] = p1top2_l["dst_contour_org"] 
                                del p1top2_l["dst_contour_org"]
                                p1top2_l["is_complex"] = not "src_contour_org" in p1top2_l.keys()

            else:
                # extend the index info with outer contours of the best result
                p1top2_linkage_dict[ci][0]["src_contour"    ] = cont1_best
                p1top2_linkage_dict[ci][0]["src_contour_org"] = src
                p1top2_linkage_dict[ci][0]["is_complex"     ] = True
                p1top2_linkage_dict[ci][1]["src_contour"    ] = cont2_best
                p1top2_linkage_dict[ci][1]["src_contour_org"] = src
                p1top2_linkage_dict[ci][1]["is_complex"     ] = True

                ni1 = p1top2_lnks[0]["dst_id"]
                ni2 = p1top2_lnks[1]["dst_id"]
                for ni, cont_best in [(ni1, cont1_best), (ni2, cont2_best)]:
                    for x in p2top1_linkage_dict[ni]:
                        if (x["dst_id"] == ni):
                            x["dst_contour"    ] = cont_best
                            x["dst_contour_org"] = src
                            x["is_complex"     ] = True

    return p1top2_linkage_dict

def reverse_linkage(org_linkage):
    dn_linkage = {}
    if(org_linkage is None):
        return None
    for src_id, links_out in org_linkage.items():
        for up_link_dict in links_out:

            up_link_dict_outer  = up_link_dict["outer" ]
            dst_id_out = up_link_dict_outer['dst_id']
            if(dst_id_out not in dn_linkage.keys()):
                dn_linkage[dst_id_out] = []
            new_linkage_outer  = {}
            for key in up_link_dict_outer.keys():
                new_key = key.replace("src", "xxx").replace("dst", "src").replace("xxx", "dst")
                new_linkage_outer[new_key] = up_link_dict_outer[key]

            up_links_in = up_link_dict["inners"]
            new_linkage_inners = {}
            for inner_id, up_link_dict_inners in up_links_in.items():
                for up_link_dict_inner in up_link_dict_inners:
                    dst_id_in = up_link_dict_outer['dst_id']
                    if(dst_id_in not in new_linkage_inners.keys()):
                        new_linkage_inners[dst_id_in] = []
                    new_linkage_inner = {}
                    for key in up_link_dict_inner.keys():
                        new_key = key.replace("src", "xxx").replace("dst", "src").replace("xxx", "dst")
                        new_linkage_inner[new_key] = up_link_dict_inner[key]
                    new_linkage_inners[dst_id_in].append(new_linkage_inner)
            new_linkage = {'outer':new_linkage_outer, 'inners': new_linkage_inners}

            dn_linkage[dst_id_out].append(new_linkage) 

    return dn_linkage

def  assign_angles(ring, centr):
    rnorm = ring[:,0:2] - centr[0:2]
    angs = [np.arctan2(*p) for p in rnorm]

    return angs

def check_self_intersections(verts, faces):

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    triIntersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

    ray_origins      = []
    ray_directions   = []
    ray_tri_id       = []
    #ray_exp_location= []
    ray_exp_location0= []
    ray_exp_location1= []
    ray_verts        = []
    for eid, edge in enumerate(mesh.edges):
        e_vs = np.array([mesh.vertices[edge[0]], mesh.vertices[edge[1]]])
        e_zs = e_vs[:,2]
        if e_zs[0] > e_zs[1] :
            # wierzcholek o mniejszym Z jako pierwszy
            e_vs = np.array([mesh.vertices[edge[1]], mesh.vertices[edge[0]]])
            e_zs = e_vs[:,2]
        if (e_zs[0] != e_zs[1]) and np.any(e_vs[0,0:2] != e_vs[1,0:2]): # only for different Zs and different (x,y)
            #ray_tri_id.extend([mesh.edges_face[eid], mesh.edges_face[eid]])
            #e_m = np.mean(e_vs, axis = 0)
            #ray_exp_location.extend([e_m, e_m])
            #e_zm = e_m[2]
            #e_zs_d = [e_zm + (z - e_zm) * 1.01 for z in e_zs]
            #e_vs_d = np.array([[*e_vs[i][0:2], e_zs_d[i]] for i in [0,1]])
            #ray_origins.extend([e_m, e_m])
            #dir0 = e_vs_d[1] - e_vs_d[0]
            #ray_directions.extend([dir0, -dir0]) 

            ray_tri_id.append(mesh.edges_face[eid])
            e_m = np.mean(e_vs, axis = 0)
            e_zm = e_m[2]
            e_zs_d = [e_zm + (z - e_zm) * 1.0 for z in e_zs]
            e_vs_d = np.array([[*e_vs[i][0:2], e_zs_d[i]] for i in [0,1]])
            #ray_exp_location.append(e_m)
            ray_exp_location0.append(e_vs_d[0])
            ray_exp_location1.append(e_vs_d[1])
            ray_origins.append(e_vs_d[0])
            dir01 = e_vs_d[1] - e_vs_d[0]
            ray_directions.append(dir01) 
            ray_verts.append(e_vs)

    #first_tri_intersected = triIntersector.intersects_first(ray_origins, ray_directions)
    intersection_act_locations, intersection_ray_id, intersection_tri_id = triIntersector.intersects_location(ray_origins, ray_directions)
    
    intersection_exp_locations0 = np.array([ray_exp_location0[rid] for rid in intersection_ray_id])
    intersection_exp_locations1 = np.array([ray_exp_location1[rid] for rid in intersection_ray_id])
    #intersection_exp_locations = np.array([ray_origins     [rid] for rid in intersection_ray_id])
    intersection_face_id        = np.array([ray_tri_id       [rid] for rid in intersection_ray_id])
    
    dif = intersection_act_locations - intersection_exp_locations0
    is_same_tri  = np.all(np.isclose(intersection_act_locations, intersection_exp_locations0), axis = 1)
    is_same_tri |= np.all(np.isclose(intersection_act_locations, intersection_exp_locations1), axis = 1)
    faulty_ray_ids  = intersection_ray_id        [np.where(is_same_tri == False)]
    faulty_act_loc  = intersection_act_locations [np.where(is_same_tri == False)]
    faulty_exp_loc0 = intersection_exp_locations0[np.where(is_same_tri == False)]
    faulty_exp_loc1 = intersection_exp_locations1[np.where(is_same_tri == False)]
    faulty_face_id  = intersection_face_id       [np.where(is_same_tri == False)]

    no_intersection = np.all(is_same_tri)
    
    faulty_rays = []
    faulty_edges = []
    if not no_intersection:
        logging.info(f"Create mesh from the gathered vertices and faces...")
        for ray_id in range(len(faulty_ray_ids)):
            faulty_ray_id = faulty_ray_ids[ray_id]
            start_p = ray_origins[faulty_ray_id]
            stop_p  = ray_origins[faulty_ray_id] + ray_directions[faulty_ray_id]
            
            act_p = faulty_act_loc [ray_id]
            exp_p0= faulty_exp_loc0[ray_id]
            exp_p1= faulty_exp_loc1[ray_id]

            faulty_rays.append([start_p, stop_p, act_p])#, exp_p])

            already_on_list = False
            new_faulty_edge = ray_verts[faulty_ray_id]
            for fe in faulty_edges:
                if np.all(new_faulty_edge == fe) or np.all(new_faulty_edge == np.array([fe[1],fe[0]])):
                    already_on_list = True
                    break
            if not already_on_list:
                faulty_edges.append(ray_verts[faulty_ray_id])

        if False:
            ray_meshes = []
            for faulty_ray in faulty_rays:
                ray_meshes.append(trimesh.PointCloud(faulty_ray))
            show_meshes([mesh, *ray_meshes])
            _=1


    return no_intersection, faulty_edges

def connect_rings_2step(r1, r1angs, r1com, r2, r2angs, r2com, 
                  compensate_com_shift = True, 
                  high_effort = False,
                  unchanged_verts_limit = 10,
                  do_connect_convex_first = True,
                  use_connect_rings_idxs_ord = True,
                  is_outer = True,
                  do_simple = False):
    
    if np.all(r1[0] == r1[-1]) and len(r1) > 1:
        r1angs = r1angs[1:]
        r1 = r1[1:]
    if np.all(r2[0] == r2[-1]) and len(r2) > 1:
        r2angs = r2angs[1:]
        r2 = r2[1:]
        
    num_verts = len(r1) + len(r2)
    if(num_verts < 4):
        return [], [], False
    
    successfully_connected = False
    forbiden_edges = []
    num_forbiden_edges = 0
    retry_dcounter = 10
    retry_stages = ['normal', 'try_high_effort', 'try_forbid_edges', 'skip_intersection_checking']
    retry_stage = retry_stages[0]
    
    while not successfully_connected:

        if retry_stage == 'try_high_effort':
            if do_simple:
                do_simple = False
            high_effort = True
            forbiden_edges = []

        elif retry_stage == 'skip_intersection_checking':
            forbiden_edges = []
            #do_connect_convex_first = (len(r1) > 20) and (len(r2) > 20)
            unchanged_verts_limit = 10


        if do_simple:
            filled_r_idxs = []
            for idx in range(len(r1)):
                filled_r_idxs.append([0, idx])
                filled_r_idxs.append([1, idx])
            #filled_r_idxs.append([0, 0])
            #filled_r_idxs.append([1, 0])
            was_fast = True
        else:

            r1z = r1[0,2]
            r2z = r2[0,2]

            dbg = False #r1com[1] > 27 and r1com[1] < 30
            if dbg:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                fig.tight_layout()
                ax.set_xlim([ 3, 8])
                ax.set_ylim([31, 37])
            if dbg:# and False:
                ax.plot(r1   [:,0], r1   [:,1], [r1z]* len(r1   ), c=[0.9, 0.0, 0.0], marker = "o", alpha=0.2)
                ax.plot(r2   [:,0], r2   [:,1], [r2z]* len(r2   ), c=[0.0, 0.0, 0.9], marker = "o", alpha=0.2)
                ax.plot(r1   [0,0], r1   [0,1],  r1z             , c=[0.9, 0.0, 0.0], marker = "o", alpha=1, markersize=14)
                ax.plot(r2   [0,0], r2   [0,1],  r2z             , c=[0.0, 0.0, 0.9], marker = "o", alpha=1, markersize=14)

            if not do_connect_convex_first:

                filled_r_idxs, dir, was_fast = connect_rings_idxs_angdir(r1, r1angs, r1com, r2, r2angs, r2com, 
                                                         compensate_com_shift = compensate_com_shift, 
                                                         high_effort = high_effort,
                                                         unchanged_verts_limit = unchanged_verts_limit,
                                                         use_connect_rings_idxs_ord = use_connect_rings_idxs_ord,
                                                         forbiden_edges = forbiden_edges)
                if len(filled_r_idxs) > 2:
                    filled_r_idxs = filled_r_idxs[2:]

                for filed_r_idx in filled_r_idxs:
                    sel = filed_r_idx[0]
                    idx_mod = len(r1) if sel == 0 else len(r2)
                    filed_r_idx[1] %= idx_mod

                if dbg:# and False:
                    links_verts = []
                    #prev_sel = 0
                    for sel, r_idx in filled_r_idxs:
                        #if(prev_sel != sel):
                        links_verts.append(r1[r_idx] if sel==0 else r2[r_idx])
                        #    prev_sel = sel
                    links_verts = np.array(links_verts)
                    #ax.plot(links_verts[:,0], links_verts[:,1], links_verts[:,2], c=[0.0, 0.9, 0.0], marker = "", alpha=0.6)
                    for link_v_idx in range(len(links_verts)-1):
                        xvzP = links_verts[link_v_idx  ]
                        xvzK = links_verts[link_v_idx+1]
                        quiver_params = [*xvzP, *(xvzK-xvzP)]
                        ax.quiver(*quiver_params, color = [0.0, 0.9, 0.0], pivot='tail', )
                    #plt.show()

            else: #if do_connect_convex_first
                do_interpolate_convex = False
                if do_interpolate_convex:
                    interpolation_max_dist = 2
    
                lr1 = LinearRing(coordinates = r1)
                ch1 = lr1.convex_hull
                if do_interpolate_convex:
                    ch1_pth = [v2D for v2D in  zip(list(ch1.exterior.xy[0]), list(ch1.exterior.xy[1])) ]  
                    ch1_poly = v_polygons() 
                    ch1_poly.add_polygon_from_paths(ch1_pth)
                    inserted = ch1_poly.interpolate_verts(max_verts_dist = interpolation_max_dist, force_int = False)
                    r1_ch = np.array([[*v2D, r1z] for v2D in ch1_poly["polygons"][0]["outer"]["path"]])
                else:
                    r1_ch = np.array([[*v2D, r1z] for v2D in  zip(list(ch1.exterior.xy[0]), list(ch1.exterior.xy[1])) ])

                lr2 = LinearRing(coordinates = r2)
                ch2 = lr2.convex_hull
                if do_interpolate_convex:
                    ch2_pth = [v2D for v2D in  zip(list(ch2.exterior.xy[0]), list(ch2.exterior.xy[1])) ]  
                    ch2_poly = v_polygons() 
                    ch2_poly.add_polygon_from_paths(ch2_pth)
                    inserted = ch2_poly.interpolate_verts(max_verts_dist = interpolation_max_dist, force_int = False)
                    r2_ch = np.array([[*v2D, r2z] for v2D in ch2_poly["polygons"][0]["outer"]["path"]])
                else:
                    r2_ch = np.array([[*v2D, r2z] for v2D in  zip(list(ch2.exterior.xy[0]), list(ch2.exterior.xy[1])) ])
            
                if ch1.exterior.is_ccw ^ ch2.exterior.is_ccw:
                    blad += 1
                if lr1.is_ccw ^ lr2.is_ccw:
                    blad += 1

                if dbg:
    
                    ax.plot(r1_ch[:,0], r1_ch[:,1], [r1z]* len(r1_ch), c=[0.9, 0.0, 0.0], marker = "x", alpha=0.3)
                    ax.plot(r2_ch[:,0], r2_ch[:,1], [r2z]* len(r2_ch), c=[0.0, 0.0, 0.9], marker = "x", alpha=0.3)
                    ax.plot(r1_ch[0,0], r1_ch[0,1],  r1z             , c=[0.9, 0.0, 0.0], marker = "x", alpha=1, markersize=14)
                    ax.plot(r2_ch[0,0], r2_ch[0,1],  r2z             , c=[0.0, 0.0, 0.9], marker = "x", alpha=1, markersize=14)
                    #plt.show()

                r1_ch_angs = assign_angles(ring = r1_ch, centr = r1com)
                r2_ch_angs = assign_angles(ring = r2_ch, centr = r2com)
        
                r_ch_idxs, dir, was_fast = connect_rings_idxs_angdir(r1_ch, r1_ch_angs, r1com, r2_ch, r2_ch_angs, r2com, 
                                                         compensate_com_shift = compensate_com_shift, 
                                                         high_effort = True,
                                                         unchanged_verts_limit = unchanged_verts_limit,
                                                         use_connect_rings_idxs_ord = False,
                                                         use_ang_cost = True,
                                                         forbiden_edges = forbiden_edges)
                if dbg:
                    links_verts = []
                    #prev_sel = 0
                    for sel, r_ch_idx in r_ch_idxs:
                        #if(prev_sel != sel):
                        links_verts.append(r1_ch[r_ch_idx] if sel==0 else r2_ch[r_ch_idx])
                        #    prev_sel = sel
                    links_verts = np.array(links_verts)
                    ax.plot(links_verts[ :,0], links_verts[ :,1], links_verts[ :,2], c=[0.0, 0.9, 0.0], marker = "", alpha=0.6)
                    ax.plot(links_verts[ 0,0], links_verts[ 0,1], links_verts[ 0,2], c=[0.0, 0.9, 0.0], marker = "s", alpha=1, markersize=12)
                    ax.plot(links_verts[-1,0], links_verts[-1,1], links_verts[-1,2], c=[0.0, 0.9, 0.0], marker = "8", alpha=1, markersize=12)

                    ax.plot(r1com[0],          r1com[1],          r1com[2],          c=[0.1, 0.1, 0.1], marker = "x", alpha=1, markersize=8)
                    ax.plot(r2com[0],          r2com[1],          r2com[2],          c=[0.1, 0.1, 0.1], marker = "x", alpha=1, markersize=8)
                    ax.plot(links_verts[ 0,0], links_verts[ 0,1], links_verts[ 0,2], c=[0.1, 0.1, 0.1], marker = "x", alpha=1, markersize=8)
                    ax.plot(links_verts[ 1,0], links_verts[ 1,1], links_verts[ 1,2], c=[0.1, 0.1, 0.1], marker = "x", alpha=1, markersize=8)
                    #plt.show()
        
                r1_ch_idx_done  = None
                r2_ch_idx_done  = None
                r1_ch_idx_start = None
                r2_ch_idx_start = None
                vertical_ch_links = []
                ors_idx = 0
                loop_done = False
                while not loop_done:

                    osel, or_ch_idx  = r_ch_idxs[ors_idx]
            
                    state = 'find_transition'
                    dst_rs_idxs = []
                    src_rs_idxs = []

                    irs_idx = ors_idx
                    while True:
                        isel, ir_ch_idx = r_ch_idxs[irs_idx]
                
                        if (state == 'find_transition') and (osel == isel):
                            dst_rs_idxs.append(irs_idx)
                        elif (state == 'find_transition') and (osel != isel):
                            state = 'keep_till_next_transition'
                            src_rs_idxs.append(irs_idx)
                        elif(state == 'keep_till_next_transition') and (osel != isel):
                            src_rs_idxs.append(irs_idx)
                        elif(state == 'keep_till_next_transition') and (osel == isel):
                            state = 'keep_till_next_transition2'
                            dst_rs_idxs.append(irs_idx)
                        elif(state == 'keep_till_next_transition2') and (osel == isel):
                            dst_rs_idxs.append(irs_idx)
                        elif(state == 'keep_till_next_transition2') and (osel != isel):
                            break

                        irs_idx = (irs_idx+1)
                        if(irs_idx >= len(r_ch_idxs)):
                            irs_idx = irs_idx % len(r_ch_idxs)
                            loop_done = True
                            break

                    if (len(src_rs_idxs) == 0) or (len(dst_rs_idxs) == 0):
                        break

                    dists = []
                    for src_rs_idx in src_rs_idxs:
                        dist_s = []
                        s_sel, s_r_ch_idx = r_ch_idxs[src_rs_idx]
                        sv = r1_ch[s_r_ch_idx] if s_sel==0 else r2_ch[s_r_ch_idx]
                        for dst_rs_idx in dst_rs_idxs:
                            d_sel, d_r_ch_idx = r_ch_idxs[dst_rs_idx]

                            dv = r1_ch[d_r_ch_idx] if d_sel==0 else r2_ch[d_r_ch_idx]
                            if compensate_com_shift:
                                dist = np.linalg.norm(np.array(sv[0:2]) - np.array(dv[0:2]))
                            else:
                                dist = np.linalg.norm(np.array(sv) - np.array(dv))
                            dist_s.append(dist)
                        dists.append(dist_s)

                    dists = np.array(dists)

                    arg_src, arg_dst = np.unravel_index(np.argsort(dists, axis=None), dists.shape)
                    min_arg_src = arg_src[0]
                    min_arg_dst = arg_dst[0]
                    src_rs_idx_best = src_rs_idxs[min_arg_src]
                    dst_rs_idx_best = dst_rs_idxs[min_arg_dst]
                    s_sel_best, s_r_ch_idx_best = r_ch_idxs[src_rs_idx_best]
                    d_sel_best, d_r_ch_idx_best = r_ch_idxs[dst_rs_idx_best]
                    if loop_done and len(vertical_ch_links)>0:
                        if (d_sel_best, d_r_ch_idx_best)in vertical_ch_links[0] or (s_sel_best, s_r_ch_idx_best)in vertical_ch_links[0]:
                            break
                    if(s_sel == 0):
                        vertical_ch_links.append([(s_sel_best, s_r_ch_idx_best), (d_sel_best, d_r_ch_idx_best)])
                    else:
                        vertical_ch_links.append([(d_sel_best, d_r_ch_idx_best), (s_sel_best, s_r_ch_idx_best)])

                    ors_idx = max(src_rs_idx+1, dst_rs_idx_best+1) 
                
                if dbg:
                    #prev_sel = 0
                    for (s_sel, s_r_ch_idx), (d_sel, d_r_ch_idx) in vertical_ch_links:
                        s_v = ([*r1_ch[s_r_ch_idx], r1z] if s_sel==0 else [*r2_ch[s_r_ch_idx], r2z])
                        d_v = ([*r1_ch[d_r_ch_idx], r1z] if d_sel==0 else [*r2_ch[d_r_ch_idx], r2z])
                        ax.plot([s_v[0], d_v[0]], [s_v[1], d_v[1]], [s_v[2], d_v[2]], c=[0.5, 0.0, 0.0], marker = "", alpha=0.6)
                        #    prev_sel = sel
                    #plt.show()
            
                if vertical_ch_links[0] != vertical_ch_links[-1] or len(vertical_ch_links)==1:
                    vertical_ch_links.append(vertical_ch_links[0])

                vertical_links = []
                for (d0_sel, d0_idx), (d1_sel, d1_idx) in vertical_ch_links:#[1:]:
                    if not (d0_sel == 0 and d1_sel == 1):
                        blad+=1
                    r_vert = r1_ch[d0_idx]
                    matches_list = np.argwhere(np.all(r1 == r_vert, axis=1))
                    if len(matches_list) > 0:
                        r0_idx = matches_list[0][0]
                    else:
                        r0_idx = None

                    r_vert = r2_ch[d1_idx]
                    matches_list = np.argwhere(np.all(r2 == r_vert, axis=1))
                    if len(matches_list) > 0:
                        r1_idx = matches_list[0][0]
                    else:
                        r1_idx = None

                    vertical_links.append(((d0_sel, r0_idx), (d1_sel, r1_idx)))

                #check dirs
                #sanity_check = True
                #if sanity_check:
                pos_count0 = 0
                pos_count1 = 0
                vertical_link_idx = 0
                for vertical_link_idx in range(len(vertical_links)-1):
                    (s0_sel, s0_idx), (s1_sel, s1_idx) = vertical_links[vertical_link_idx  ]
                    (d0_sel, d0_idx), (d1_sel, d1_idx) = vertical_links[vertical_link_idx+1]
                    if(d0_idx - s0_idx) > 0:
                        pos_count0 += 1
                    elif (d0_idx - s0_idx) < 0:
                        pos_count0 -= 1
                    if(d1_idx - s1_idx) > 0:
                        pos_count1 += 1
                    elif(d1_idx - s1_idx) < 0:
                        pos_count1 -= 1

                #if (dir == -1) ^ ((pos_count0 <= 0 and pos_count1 <= 0) ):
                #    blad += 1

                do_reverse = ((pos_count0 <= 0 and pos_count1 <= 0) )
                if do_reverse :
                    vertical_links = vertical_links[::-1]


                if dbg:
                    #prev_sel = 0
                    for (s_sel, s_r_ch_idx), (d_sel, d_r_ch_idx) in vertical_links:
                        s_v = (r1[s_r_ch_idx] if s_sel==0 else r2[s_r_ch_idx])
                        d_v = (r1[d_r_ch_idx] if d_sel==0 else r2[d_r_ch_idx])
                        ax.plot([s_v[0], d_v[0]], [s_v[1], d_v[1]], [s_v[2], d_v[2]], c=[0.0, 0.0, 0.5], marker = "", alpha=0.6)
                        #    prev_sel = sel
                    #plt.show()
                    
                # add info about required rotation between consecutive vertical links 
                vertical_link_idx = 0
                vertical_links_wr = []
                for vertical_link_idx in range(len(vertical_links)-1):
                    (s0_sel, s0_idx), (s1_sel, s1_idx) = vertical_links[vertical_link_idx  ]
                    (d0_sel, d0_idx), (d1_sel, d1_idx) = vertical_links[vertical_link_idx+1]
                    rot_r1 = (d0_idx < s0_idx)
                    rot_r2 = (d1_idx < s1_idx)
                    vertical_links_wr.append(((s0_sel, s0_idx, rot_r1), (s1_sel, s1_idx, rot_r2)))
                (s0_sel, s0_idx), (s1_sel, s1_idx) = vertical_links[len(vertical_links)-1  ]
                vertical_links_wr.append(((s0_sel, s0_idx, False), (s1_sel, s1_idx, False)))
                has_rot_r1 = False
                has_rot_r2 = False
                for vertical_link_idx in range(len(vertical_links_wr)-1):
                    (s0_sel, s0_idx, rot_r1), (s1_sel, s1_idx, rot_r2) = vertical_links_wr[vertical_link_idx  ]
                    has_rot_r1 |=  rot_r1
                    has_rot_r2 |=  rot_r2
                if (not has_rot_r1) and (not has_rot_r2):
                    (s0_sel, s0_idx, rot_r1), (s1_sel, s1_idx, rot_r2) = vertical_links_wr[0]
                    vertical_links_wr[0] = ((s0_sel, s0_idx, True), (s1_sel, s1_idx, True))
                elif not has_rot_r1:
                    for vertical_link_idx in range(len(vertical_links_wr)-1):
                        (s0_sel, s0_idx, rot_r1), (s1_sel, s1_idx, rot_r2) = vertical_links_wr[vertical_link_idx  ]
                        if rot_r2:
                            vertical_links_wr[vertical_link_idx  ] = ((s0_sel, s0_idx, True), (s1_sel, s1_idx, True))
                            break
                elif not has_rot_r2:
                    for vertical_link_idx in range(len(vertical_links_wr)-1):
                        (s0_sel, s0_idx, rot_r1), (s1_sel, s1_idx, rot_r2) = vertical_links_wr[vertical_link_idx  ]
                        if rot_r1:
                            vertical_links_wr[vertical_link_idx  ] = ((s0_sel, s0_idx, True), (s1_sel, s1_idx, True))
                            break

                #translate from the convex_hull rings indexes to the full rings indexes
        
            
                filled_r_idxs = []#[vertical_links[0][0], vertical_links[0][1]]
                gap_line_style = "--"
                vertical_link_idx = 0
                for vertical_link_idx in range(len(vertical_links_wr)-1):
                    (s0_sel, s0_idx, rot_r1), (s1_sel, s1_idx, rot_r2) = vertical_links_wr[vertical_link_idx  ]
                    (d0_sel, d0_idx, _     ), (d1_sel, d1_idx, _     ) = vertical_links_wr[vertical_link_idx+1]

                    # found next gap to fill
                    r1_gap_verts = r1[s0_idx : d0_idx+1] if not rot_r1 else np.array([*r1[s0_idx : len(r1)], * r1[0 : d0_idx+1]])
                    r2_gap_verts = r2[s1_idx : d1_idx+1] if not rot_r2 else np.array([*r2[s1_idx : len(r2)], * r2[0 : d1_idx+1]])

                    r_gap_idxs, was_fast, conn_cost = connect_rings_idxs_ord(r1_gap_verts, r1com, r2_gap_verts, r2com, 
                                                        compensate_com_shift = compensate_com_shift, 
                                                        high_effort = high_effort,
                                                        unchanged_verts_limit = 10,
                                                        forbiden_edges = forbiden_edges)
                    gap_r_idxs = []
                    for r_gap_sel, r_gap_idx in r_gap_idxs:
                        idx_off = s0_idx       if r_gap_sel == 0 else s1_idx
                        idx_mod = len(r1)      if r_gap_sel == 0 else len(r2)
                        r_idx = (r_gap_idx + idx_off) % idx_mod
                        gap_r_idxs.append((r_gap_sel, r_idx))

                    if dbg:# and False:
                        gap_links = []
                        for sel, r_idx in gap_r_idxs[2:]:
                            gap_links.append(r1[r_idx] if sel==0 else r2[r_idx])
                        gap_links = np.array(gap_links)
                        ax.plot(gap_links[:,0], gap_links[:,1], gap_links[:,2], gap_line_style, c=[0.5, 0.5, 0.5], marker = "", alpha=1)
                        ax.plot(gap_links[0,0], gap_links[0,1], gap_links[0,2], gap_line_style, c=[0.5, 0.5, 0.5], marker = "v", alpha=1, markersize=10)
                        gap_line_style = "--" if(gap_line_style == ".-") else ".-"

                    filled_r_idxs.extend(gap_r_idxs[2:])
   
            if dbg:
                plt.show()

        #check dirs
        r1_verts_ord = [r1[idx] for sel, idx in filled_r_idxs if sel == 0]
        r2_verts_ord = [r2[idx] for sel, idx in filled_r_idxs if sel == 1]
        out_lr1 = LinearRing(coordinates = r1_verts_ord)
        out_lr2 = LinearRing(coordinates = r2_verts_ord)
    
        out_dir = 1
        #try:
        if out_lr1.is_ccw ^ out_lr2.is_ccw:
            blad += 1
        elif  out_lr1.is_ccw and out_lr2.is_ccw:
            out_dir = -1 if is_outer else  1
        else: #not ( out_lr1.is_ccw and out_lr2.is_ccw ):
                out_dir =  1 if is_outer else -1
        #except:
        #    out_dir =  1 if is_outer else -1

        best_verts, best_faces  = connect_rings_verts(out_dir, filled_r_idxs, r1, r2)

        check_degenerated_faces = True
        if check_degenerated_faces:
            deg_f_idxs = []
            for fidx, f in enumerate(best_faces):
                vs = [best_verts[vidx] for vidx in f]
                for vp in [(vs[0], vs[1]), (vs[1], vs[2]), (vs[0], vs[2])]:
                    if np.all(vp[0] == vp[1]):
                        logging.warning(f"degenerated face {vs}!")
                        deg_f_idxs.append(fidx)
            if len(deg_f_idxs) > 0:
                deg_f_idxs = np.unique(deg_f_idxs)
                deg_f_idxs = list(deg_f_idxs)
                best_faces = [f for fidx, f in enumerate(best_faces) if not (fidx in deg_f_idxs)]

        successfully_connected, faulty_edges = check_self_intersections(best_verts, best_faces)

        if (not successfully_connected) :
            if (retry_stage != 'skip_intersection_checking'):
                for fe in faulty_edges:
                    already_on_list = False
                    for forbiden_edge in forbiden_edges:
                        if np.all(fe == forbiden_edge):
                            already_on_list = True
                            break
                    if not already_on_list:
                        forbiden_edges.append(fe)
                        logging.warning(f" new face-intersecting edge {fe[0]} - {fe[1]}!")
            
            logging.warning(f" Unsuccesfull try to resolve intersection problem in {retry_stage} stage")
            try:
                logging.warning(f"  previous trajectory {prev_try_traj_r_idxs}")
                logging.warning(f"  previous trajectory {prev_try_traj_verts}")
            except:
                _=1
            logging.warning(f"  current  trajectory {filled_r_idxs}")
            logging.warning(f"  current  trajectory {best_verts}")
            prev_try_traj_r_idxs = copy.copy(filled_r_idxs)
            prev_try_traj_verts  = copy.copy(best_verts)
            if (retry_stage == 'skip_intersection_checking'):
                logging.warning(f"  breaking loop ...")
                break

            if retry_stage == 'normal':
                if(not high_effort) or do_simple:
                    retry_stage = 'try_high_effort'
                else:
                    retry_stage = 'try_forbid_edges'
            elif retry_stage == 'try_high_effort':
                retry_stage = 'try_forbid_edges'
                retry_dcounter = 5
            elif retry_stage == 'try_forbid_edges':
                retry_dcounter -= 1
                logging.warning(f"  {len(forbiden_edges)} edges intersect faces, retry connecting with those edges forbiden in {retry_stage} stage ({retry_dcounter} tries left)!")
                    
                if (retry_dcounter <= 0):
                    logging.warning(f"  Retried connecting too many times!")
                    retry_stage = 'skip_intersection_checking' 
                    
                elif (num_forbiden_edges > 0) and (num_forbiden_edges == len(forbiden_edges)):
                    logging.warning(f"  List of forbiden edges has not changed in the previous try!")
                    retry_stage = 'skip_intersection_checking'

                num_forbiden_edges = len(forbiden_edges)
                if num_forbiden_edges == (len(r1) + len(r2)):
                    logging.warning(f" all inter-layer edges ({len(forbiden_edges)}) are already forbiden! Randomly choose only 1/10 of theme!")
                    forbiden_edges = random.sample(forbiden_edges, max(1, len(forbiden_edges)//10))
                    logging.warning(f"  ({len(forbiden_edges)}) random inter-layer edges are left!")

            logging.warning(f" Retry connecting in {retry_stage} stage")

    if retry_stage != 'normal':
        if successfully_connected:
            logging.warning(f"  Succesfully resolved intersection problem in {retry_stage} stage")
        else:
            logging.warning(f"  Unsuccesfully try to resolved intersection problem in {retry_stage} stage!")


    return best_verts, best_faces, was_fast

def get_traj_cost(trajectory, verts, v2off, forbiden_edges):
    
    curr_cost = 0
    
    r1idx = 0
    r2idx = v2off

    idxs = [[0,r1idx], [1,r2idx]]
    unchanged_verts_cnt = 0
    last_was_DGD = True
    found_forbiden = False

    for lid in trajectory[2:]:
        vd10 = verts[r1idx  ] 
        vd20 = verts[r2idx  ] 


        if lid == 0:
            # D G D
            r1idx = r1idx+1
            idxs.append([0,r1idx])
            vd11 = verts[r1idx]
            cost = np.linalg.norm(vd11-vd10) + np.linalg.norm(vd11-vd20)
            
            if (not found_forbiden) and (len(forbiden_edges) != 0):
                for fe in forbiden_edges:
                    test_edge = np.array([vd11, vd20])
                    forbiden = np.all(test_edge == fe)
                    if forbiden:
                        found_forbiden = True
                        cost = 1000
                        break
        else:
            # G G D
            r2idx = r2idx+1
            idxs.append([1,r2idx])
            vd21 = verts[r2idx]
            cost = np.linalg.norm(vd21-vd10) + np.linalg.norm(vd21-vd20)
                
            if (not found_forbiden) and (len(forbiden_edges) != 0):
                for fe in forbiden_edges:
                    test_edge = np.array([vd10, vd21])
                    forbiden = np.all(test_edge == fe)
                    if forbiden:
                        found_forbiden = True
                        cost = 1000
                        break
        curr_cost += cost

    curr_cost = curr_cost/len(verts)
    return curr_cost, idxs

def shift_trajectory(in_tr, sh_groups = False, sh_len = 1):

    sh_trs = []
    for tidx in range(2, len(in_tr)-sh_len):
        tidx_n = tidx+sh_len

        if in_tr[tidx] != in_tr[tidx_n]:

            if sh_groups:
                while (tidx_n <= len(in_tr)-1) and (in_tr[tidx] != in_tr[tidx_n]):
                    tidx_n += 1
                if(tidx+sh_len != tidx_n):
                    tidx_n -= 1

            out_tr = copy.copy(in_tr)
            if (not sh_groups) or (tidx+sh_len != tidx_n):
                out_tr[tidx], out_tr[tidx_n] = in_tr[tidx_n], in_tr[tidx]
                sh_trs.append(out_tr)
    return sh_trs

def connect_rings_idxs_ord(r1, r1com, r2, r2com, 
                  compensate_com_shift = True, 
                  high_effort = False,
                  unchanged_verts_limit = 10,
                  remove_repeated_last_vert = True,
                  max_num_of_tested_trajectories = -1,
                  forbiden_edges = []):
    
    skip_already_checked_trajectories = False

    if(remove_repeated_last_vert):
        if np.all(r1[0] == r1[-1]) and len(r1) > 1:
            r1 = r1[:-1]
        if np.all(r2[0] == r2[-1]) and len(r2) > 1:
            r2 = r2[:-1]
        
    fast_break = False
    best_cost = 1000000.0

    num_verts = len(r1) + len(r2)
    if(len(r1) == 0):
        return [(1,idx) for idx in range(len(r2))], fast_break, best_cost
    if(len(r1) == 1):
        return [(0, 0), *[(1,idx) for idx in range(len(r2))]], fast_break, best_cost
    if(len(r2) == 0):
        return [(0,idx) for idx in range(len(r1))], fast_break, best_cost
    if(len(r2) == 1):
        return [(0, 0), (1, 0), *[(0,idx) for idx in range(1, len(r1))]], fast_break, best_cost
    if(num_verts < 4):
        return [], fast_break, best_cost


    # calculate threshold cost that is used to stop the connection attemtps for various start angles and directions
    #r1len = np.mean(np.linalg.norm(r1[:,0:2]-[*r1[1:,0:2], r1[0,0:2]], axis=1))
    #r2len = np.mean(np.linalg.norm(r2[:,0:2]-[*r2[1:,0:2], r2[0,0:2]], axis=1))
    #aver_in_layer_dist = (r1len + r2len) / 2
    #aver_trans_layer_dist = aver_in_layer_dist/2 + (0 if compensate_com_shift else np.linalg.norm(np.array(r2com) - np.array(r1com)))
    #exp_min_trans_l_cost = (aver_trans_layer_dist + aver_in_layer_dist)
    #th_cost = exp_min_trans_l_cost * 1.20
    #logging.info(f"   num_verts = {len(r1)},{len(r2)}, av_in_l_dist = {aver_in_layer_dist:.1f}, av_trans_l_dist = {aver_trans_layer_dist:.1f}, exp_min_trans_l_cost = {exp_min_trans_l_cost:.1f}, th_cost = {th_cost:.1f}")

    r1_len = len(r1)
    r2_len = len(r2)
    if compensate_com_shift:
        verts = np.array([*(r1-r1com), *(r2-r2com)])
        compensated_forbiden_edges = []
        for forbiden_edge in forbiden_edges:
            compensated_forbiden_edges.append([(forbiden_edge[0]-r1com), (forbiden_edge[1]-r2com)])
        forbiden_edges = np.array(compensated_forbiden_edges)
    else:
        verts = np.array([* r1       , * r2       ])
    v2off = len(r1)
    
    def place_ones_count_num_combs(size, count):
        n = size
        r = count
        num_of_combinations =  int(math.factorial(n) / math.factorial(r) / math.factorial(n-r))
        return num_of_combinations

    def place_ones(size, count, limit_len):
        is_limited = False
        n = size
        r = count
        num_of_combinations = place_ones_count_num_combs(n, r)
        if(num_of_combinations <= limit_len):
            combs = combinations(range(n), r)
        else:
            is_limited = True
            #comb_idxs = np.random.randint(0, num_of_combinations, size=(limit_len))
            rng = np.random.default_rng()
            if(num_of_combinations > sys.maxsize):
                zeros_n = n-r
                ones_n = r
                indeces = list(range(zeros_n + ones_n))
                combs = [set(random.sample(indeces, ones_n)) for i in range(limit_len)]
            else:
                comb_idxs = rng.integers(num_of_combinations, size=(limit_len))
                combs = [nth_combination(range(n), r, i) for i in comb_idxs]

        combs_l = list(combs)
        ret = np.zeros((len(combs_l), size), dtype=int)
        for cid, positions in enumerate(combs_l):
            ret[cid][list(positions)] = 1

        return ret, is_limited
    
    is_better = True
    is_limited = True
    checked_trajs = set()
    random_trajs_id = 0
    if(max_num_of_tested_trajectories == -1):
        num_of_combs = place_ones_count_num_combs(len(verts)-2, r2_len-1)
        high_limit = 1000
        low_limit = 100
        div = 40
        if ( num_of_combs > high_limit*div):
            max_num_of_tested_trajectories = high_limit
        else:
            max_num_of_tested_trajectories = max(num_of_combs//15, 200)

    trajectories, is_limited = place_ones(len(verts)-2, r2_len-1, limit_len = max_num_of_tested_trajectories)
    trajectories = list(trajectories)
    trajectories = [[0, 1, *t] for t in trajectories]
    for trajectory in trajectories:
        if is_limited and skip_already_checked_trajectories:
            traj_hash = hash(tuple(trajectory))
            checked_trajs.add(traj_hash)
        # apply dir and start angle


        curr_cost, idxs = get_traj_cost(trajectory, verts, v2off, forbiden_edges)

        if curr_cost < best_cost:
            trajectory_str = "".join([str(t) for t in trajectory])
            logging.info(f"   {'rand' if is_limited else 'full'} tr, tr = {trajectory_str}, cost = {curr_cost:10.1f}") 
            best_trajectory = copy.copy(trajectory)
            best_idxs = copy.copy(idxs)
            best_cost = curr_cost
            best_r1_start_idx = 0
            best_r2_start_idx = 0

                        
    if is_limited:
        _ = 1
        for sh_groups, sh_len in [(True, 1), (False, 3), (False, 1)]:
            refine_step_id = 0
            is_better = True
            while is_better:
                is_better = False
                refine_step_id += 1

                trajectories = shift_trajectory(best_trajectory, sh_groups = sh_groups, sh_len = sh_len)
                for trajectory in trajectories:
                    if skip_already_checked_trajectories:
                        traj_hash = hash(tuple(trajectory))
                    if (not skip_already_checked_trajectories) or (not (traj_hash in checked_trajs)):
                        if skip_already_checked_trajectories: 
                            checked_trajs.add(traj_hash)
                        curr_cost, idxs = get_traj_cost(trajectory, verts, v2off, forbiden_edges)

                        if curr_cost < best_cost:
                            is_better = True
                            trajectory_str = "".join([str(t) for t in trajectory])
                            logging.info(f"   refine {'group ' if sh_groups else 'singl'+str(sh_len)} step {refine_step_id}, tr = {trajectory_str}, cost = {curr_cost:10.1f}")  
                            best_trajectory = copy.copy(trajectory)
                            best_idxs = copy.copy(idxs)
                            best_cost = curr_cost
                            best_r1_start_idx = 0
                            best_r2_start_idx = 0

    best_idxs_no_off = [[sel, (idx + best_r2_start_idx - v2off) % r2_len] if sel==1 else [sel, (idx + best_r1_start_idx) % r1_len] for sel, idx in best_idxs]
    return best_idxs_no_off, fast_break, best_cost
    

def connect_rings_idxs_angdir(r1, r1angs, r1com, r2, r2angs, r2com, 
                  compensate_com_shift = True, 
                  high_effort = False,
                  unchanged_verts_limit = 10,
                  keep_phase = False,
                  use_connect_rings_idxs_ord = True,
                  use_ang_cost = False,
                  forbiden_edges = []):
    
    if np.all(r1[0] == r1[-1]) and len(r1) > 1:
        r1angs = r1angs[:-1]
        r1 = r1[:-1]
    if np.all(r2[0] == r2[-1]) and len(r2) > 1:
        r2angs = r2angs[:-1]
        r2 = r2[:-1]
        
    fast_break = False
    best_dir = 1

    num_verts = len(r1) + len(r2)
    if(len(r1) == 0):
        return [(1,idx) for idx in range(len(r2))], best_dir, fast_break
    if(len(r1) == 1):
        return [(0, 0), *[(1,idx) for idx in range(len(r2))]], best_dir, fast_break
    if(len(r2) == 0):
        return [(0,idx) for idx in range(len(r1))], best_dir, fast_break
    if(len(r2) == 1):
        return [(0, 0), (1, 0), *[(0,idx) for idx in range(1, len(r1))]], best_dir, fast_break
    if(num_verts < 4):
        return [], best_dir, fast_break

    best_cost = 1000000.0

    # calculate threshold cost that is used to stop the connection attemtps for various start angles and directions
    num_trans_layer_conects = num_verts
    r1len = np.mean(np.linalg.norm(r1[:,0:2]-[*r1[1:,0:2], r1[0,0:2]], axis=1))
    r2len = np.mean(np.linalg.norm(r2[:,0:2]-[*r2[1:,0:2], r2[0,0:2]], axis=1))
    aver_in_layer_dist = (r1len + r2len) / 2
    aver_trans_layer_dist = aver_in_layer_dist/2 + (0 if compensate_com_shift else np.linalg.norm(np.array(r2com) - np.array(r1com)))
    exp_min_trans_l_cost = (aver_trans_layer_dist + aver_in_layer_dist)
    th_cost = exp_min_trans_l_cost * 1.20
    logging.info(f"   num_verts = {len(r1)},{len(r2)}, av_in_l_dist = {aver_in_layer_dist:.1f}, av_trans_l_dist = {aver_trans_layer_dist:.1f}, exp_min_trans_l_cost = {exp_min_trans_l_cost:.1f}, th_cost = {th_cost:.1f}")


    

    if keep_phase:
        start_angles = [0]
    else:
        start_angles = np.arange(0.0, 2*np.pi, 2*np.pi/10)
        if(high_effort):
            start_angles = np.arange(0.0, 2*np.pi, 2*np.pi/40)
    sa_d_l = [(s,(-1)**sid) for sid, s in enumerate(start_angles)]
    tested_case_hashes = []
    for start_angle, dir in sa_d_l:
        # apply dir and start angle
        curr_cost = 0
        #r1angs_c = [(a+start_angle) if ((a+start_angle)<np.pi) else (a+start_angle-np.pi*2) for a in r1angs]
        #r2angs_c = [(a+start_angle) if ((a+start_angle)<np.pi) else (a+start_angle-np.pi*2) for a in r2angs]
        r1angs_c = [(a+start_angle + np.pi * 2) % (np.pi * 2) for a in r1angs]
        r2angs_c = [(a+start_angle + np.pi * 2) % (np.pi * 2) for a in r2angs]

        # make all angles positive - but make sure that the vertice from the first ring has the vertice with the smallest 
        #  value. It is needed because further I assume that a trajectory always starts form vertice from the first ring and is followed by the vertice from the second ring
        #min_ang = np.min([*r1angs_c, *r2angs_c])
        if(dir==-1): 
            min_ang = np.min(r2angs_c)
        else:
            min_ang = np.min(r1angs_c)
        r1angs_c = [ang - min_ang if ang >= min_ang else ang - min_ang + np.pi*2 for ang in r1angs_c]
        r2angs_c = [ang - min_ang if ang >= min_ang else ang - min_ang + np.pi*2 for ang in r2angs_c]
    
        arg_r1angs_min = np.argmin(r1angs_c)
        arg_r2angs_min = np.argmin(r2angs_c)
        test_case = (arg_r1angs_min, arg_r2angs_min, dir)
        skip_already_checked_angs_dir = True
        if skip_already_checked_angs_dir:
            test_case_hash = hash(test_case)
            if test_case_hash in tested_case_hashes:
                continue
            tested_case_hashes.append(test_case_hash)

        r1_i     = list(range(len(r1)))
        r2_i     = list(range(len(r2)))
        if keep_phase:
            r1angs_c = np.array(r1angs_c)
            r1_c     = np.array(r1)
            r2angs_c = np.array(r2angs_c)
            r2_c     = np.array(r2)
        else: # not keep_phase:
            # shift r1 so its first element has the smallest angle
            r1angs_c = np.array([*r1angs_c[arg_r1angs_min:], *r1angs_c[:arg_r1angs_min], r1angs_c[arg_r1angs_min] + 2*np.pi]  )
            r1_c     = np.array([*      r1[arg_r1angs_min:], *      r1[:arg_r1angs_min],       r1[arg_r1angs_min]]            )
            r1_i     = np.array([*    r1_i[arg_r1angs_min:], *    r1_i[:arg_r1angs_min],     r1_i[arg_r1angs_min]]            )
        

            # shift r2 so its first element is the closest to the first element in the r1_c
            r2angs_c = np.array([*r2angs_c[arg_r2angs_min:], *r2angs_c[:arg_r2angs_min], r2angs_c[arg_r2angs_min] + 2*np.pi]  )
            r2_c     = np.array([*      r2[arg_r2angs_min:], *      r2[:arg_r2angs_min],       r2[arg_r2angs_min]]            )
            r2_i     = np.array([*    r2_i[arg_r2angs_min:], *    r2_i[:arg_r2angs_min],     r2_i[arg_r2angs_min]]            )

        if(dir==-1): #np.sum(r1angs_c[:int(r1_len/2)]) > np.sum(r1angs_c[int(r1_len/2):])):
            r1angs_c = r1angs_c[::-1]
            r1_c     = r1_c    [::-1]
            r1_i     = r1_i    [::-1]
        r1_len = len(r1angs_c)

        if(dir==-1): #np.sum(r2angs_c[:int(r2_len/2)]) > np.sum(r2angs_c[int(r2_len/2):])):
            r2angs_c = r2angs_c[::-1]
            r2_c     = r2_c    [::-1]
            r2_i     = r2_i    [::-1]
        r2_len = len(r2angs_c)
    
        dbg = False
        if dbg:
            import matplotlib.pyplot as plt
            #print(f'centroid = {centr}')
            fig = plt.figure()
            ax = fig.add_subplot()
            fig.tight_layout()
            ax.plot(r1_c[:-1,0], r1_c[:-1,1], c=[0.9, 0.0, 0.0], marker = "o", alpha=0.2)
            ax.plot(r1com[   0], r1com[   1], c=[0.9, 0.0, 0.0], marker = "x", alpha=0.2)
            ax.plot(r2_c[:-1,0], r2_c[:-1,1], c=[0.0, 0.9, 0.0], marker = "o", alpha=0.2)
            ax.plot(r2com[   0], r2com[   1], c=[0.0, 0.9, 0.0], marker = "x", alpha=0.2)
            for id, (ang, point) in enumerate(zip(r1angs_c[:-1], r1_c[:-1])):
                plt.text(point[0], point[1], f'{id}:{ang*360/2/np.pi:.0f}', c=[0.9, 0.0, 0.5],  va = 'top')
            for id, (ang, point) in enumerate(zip(r2angs_c[:-1], r2_c[:-1])):
                plt.text(point[0], point[1], f'{id}:{ang*360/2/np.pi:.0f}', c=[0.0, 0.9, 0.5],  va = 'bottom')
            plt.gca().set_aspect('equal', adjustable='box')
            plt.show()

        if use_connect_rings_idxs_ord:
            idxs, conn_was_fast, curr_cost = connect_rings_idxs_ord(r1_c, r1com, r2_c, r2com, 
                                             compensate_com_shift = compensate_com_shift, 
                                             high_effort = high_effort,
                                             unchanged_verts_limit = unchanged_verts_limit,
                                             remove_repeated_last_vert=False,
                                             forbiden_edges = forbiden_edges)
            _=1

        else:
            v2off = len(r1_c)

            r1idx = 0
            r2idx = 0

            if ((dir==1) ^ (r1angs_c[0] > r2angs_c[0])):
                idxs = [[0, r1idx], [1, r2idx]]
            else:
                idxs = [[1, r2idx], [0, r1idx]]

            unchanged_verts_cnt = 0
            last_was_DGD = True

            while(r1idx < (len(r1_c)-1)) or (r2idx < (len(r2_c)-1)):
                r10ca = r1angs_c[r1idx  ]
                r20ca = r2angs_c[r2idx  ]
                r11ca = r1angs_c[r1idx+1] if (r1idx != (len(r1_c)-1)) else 10000
                r21ca = r2angs_c[r2idx+1] if (r2idx != (len(r2_c)-1)) else 10000
                vd10 = r1_c[r1idx  ] 
                vd20 = r2_c[r2idx  ] 
                vd11 = r1_c[r1idx+1] if (r1idx != (len(r1_c)-1)) else np.array([10000]*3)
                vd21 = r2_c[r2idx+1] if (r2idx != (len(r2_c)-1)) else np.array([10000]*3)

                r1_rem       = (len(r1_c)-1) - r1idx
                r2_rem       = (len(r2_c)-1) - r2idx
                r1_too_short = (r1_rem <= 2*(r2_rem // unchanged_verts_limit))
                r2_too_short = (r2_rem <= 2*(r1_rem // unchanged_verts_limit))
                if (r1_rem == 0) or (r2_rem == 0):
                    force_vert_switch = False
                else:
                    force_vert_switch = (unchanged_verts_cnt >= unchanged_verts_limit) or r1_too_short or r2_too_short
                    
                DGD_forbiden = False
                GGD_forbiden = False
                if len(forbiden_edges) != 0:
                    if not DGD_forbiden:
                        for fe in forbiden_edges:
                            DGD_forbiden = np.all(np.array([vd11, vd20]) == fe)
                            if DGD_forbiden:
                                break
                        if DGD_forbiden and (r2_rem == 0):
                            # can not prevent this connection because it is the last point, therefore add the previous connection that lead to this situation
                            DGD_forbiden = False
                            if(len(idxs) > 1):
                                new_forbiden_edge = np.array([r1_c[r1idx], r2_c[r2idx]])
                                already_present = False
                                for fe in forbiden_edges:
                                    already_present |= np.all(new_forbiden_edge == fe)
                                if not already_present:
                                    forbiden_edges = np.array([*forbiden_edges, new_forbiden_edge])
                    if not GGD_forbiden:
                        for fe in forbiden_edges:
                            GGD_forbiden = np.all(np.array([vd10, vd21]) == fe)
                            if GGD_forbiden:
                                break
                        if GGD_forbiden and (r1_rem == 0):
                            # can not prevent this connection because it is the last point, therefore add the previous connection that lead to this situation
                            GGD_forbiden = False
                            if(len(idxs) > 1):
                                new_forbiden_edge = np.array([r1_c[r1idx], r2_c[r2idx]])
                                already_present = False
                                for fe in forbiden_edges:
                                    already_present |= np.all(new_forbiden_edge == fe)
                                if not already_present:
                                    forbiden_edges = np.array([*forbiden_edges, new_forbiden_edge])

                force_DGD = r2_too_short or (force_vert_switch and  (not last_was_DGD)) or GGD_forbiden 
                force_GGD = r1_too_short or (force_vert_switch and  (    last_was_DGD)) or DGD_forbiden

                if not force_vert_switch and (unchanged_verts_cnt >= 5):
                    _=1

                if compensate_com_shift:
                    vd10 = vd10 - r1com
                    vd20 = vd20 - r2com
                    vd11 = vd11 - r1com
                    vd21 = vd21 - r2com

                dDGD = np.linalg.norm(vd11-vd10) + np.linalg.norm(vd11-vd20)
                dGGD = np.linalg.norm(vd21-vd10) + np.linalg.norm(vd21-vd20)
                
                if use_ang_cost:
                    if(dir==-1):
                        dDGD_cost_lower = (r11ca >= r21ca)
                    else:
                        dDGD_cost_lower = (r11ca <  r21ca)
                else:
                    dDGD_cost_lower = (dDGD <= dGGD)

                if force_DGD or ((not force_GGD) and dDGD_cost_lower):
                    # D G D
                    r1idx = r1idx+1
                    cost = dDGD
                    idxs.append([0, r1idx])
                    cur_was_DGD = True
                else:
                    # G G D
                    r2idx = r2idx+1
                    cost = dGGD
                    idxs.append([1, r2idx])
                    cur_was_DGD = False
                
                if(last_was_DGD == cur_was_DGD):
                    unchanged_verts_cnt += 1
                else:
                    unchanged_verts_cnt = 1
                last_was_DGD = cur_was_DGD

                curr_cost += cost

            curr_cost = curr_cost/num_trans_layer_conects

        logging.info(f"   sa = {round(start_angle*360/(2*np.pi)):5.1f}, dir = {dir:+}, cost = {curr_cost:10.1f} {'*' if curr_cost < best_cost else ' '}")    
        if curr_cost < best_cost:
            idxs_without_shift = [[rs,r1_i[idx]] if rs==0 else [rs,r2_i[idx]] for rs, idx in idxs]
            best_idxs = idxs_without_shift
            best_cost = curr_cost
            best_r1_start_idx = 0 if keep_phase else arg_r1angs_min
            best_r2_start_idx = 0 if keep_phase else arg_r2angs_min
            best_dir = dir
        
        if(best_cost < th_cost and not high_effort):
            logging.info(f"    fast break due to reaching th_cost") 
            fast_break = True
            break

    #if use_connect_rings_idxs_ord:
    #    best_idxs_no_off = [[sel, (idx + best_r2_start_idx ) % (r2_len-1)] if sel==1 else [sel, (idx + best_r1_start_idx) % (r1_len-1)] for sel, idx in best_idxs]
    #else:
    #    best_idxs_no_off = [[sel, (idx + best_r2_start_idx ) % (r2_len-1)] if sel==1 else [sel, (idx + best_r1_start_idx) % (r1_len-1)] for sel, idx in best_idxs]
    #
    #if best_dir == -1:
    #    for idxes in best_idxs_no_off:
    #        if idxes[0] == 0:
    #            idxes[1] = r1_len - 2 - idxes[1]
    #        else:
    #            idxes[1] = r2_len - 2 - idxes[1]

    return best_idxs, best_dir, fast_break

def connect_rings_verts(dir, idxs, r1, r2):
    
    faces = []
    verts = []
    
    last_r1_idx, last_r2_idx = None, None
    last_v1_idx, last_v2_idx = None, None

    # move to the first transition between layers
    while idxs[0][0]==idxs[1][0]:
        idxs = [*idxs[1:], idxs[0]]

    idxs_ext = [(vidx, lid, ridx) for vidx, (lid, ridx) in enumerate(idxs) ]
    idxs_ext = [*idxs_ext, *idxs_ext[0:2]]

    for vidx, lid, ridx in idxs_ext:

        if not (last_r1_idx is None) and not (last_r2_idx is None):

            if (lid == 0):
                # D G D
                if(dir == 1):
                    face = [last_v1_idx, last_v2_idx,        vidx]
                else:
                    face = [       vidx, last_v2_idx, last_v1_idx]
            else:
                # G G D
                if(dir == 1):
                    face = [last_v2_idx,        vidx, last_v1_idx]
                else:
                    face = [       vidx, last_v2_idx, last_v1_idx]
            faces.append(face)

        if(lid == 0):
            last_r1_idx = ridx
            last_v1_idx = vidx
            vert = r1[last_r1_idx]
        else:
            last_r2_idx = ridx
            last_v2_idx = vidx
            vert = r2[last_r2_idx]
            
        verts.append(vert)

    return verts[:-2], faces

def connect_rings(r1, r2):
    
    if np.all(r1[0] == r1[-1]) and len(r1) > 1:
        r1angs = r1angs[1:]
        r1 = r1[1:]
    if np.all(r2[0] == r2[-1]) and len(r2) > 1:
        r2angs = r2angs[1:]
        r2 = r2[1:]
        
    num_verts = len(r1) + len(r2)
    if(num_verts < 4):
        return [], [], False

    fast_break = False
    best_cost = 1000000.0

    # calculate threshold cost that is used to stop the connection attemtps for various start angles and directions
    num_trans_layer_conects = num_verts
    r1len = np.mean(np.linalg.norm(r1[:,0:2]-[*r1[1:,0:2], r1[0,0:2]], axis=1))
    r2len = np.mean(np.linalg.norm(r2[:,0:2]-[*r2[1:,0:2], r2[0,0:2]], axis=1))
    aver_in_layer_dist = (r1len + r2len) / 2
    aver_trans_layer_dist = aver_in_layer_dist/2 + (0 if compensate_com_shift else np.linalg.norm(np.array(r2com) - np.array(r1com)))
    exp_min_trans_l_cost = (aver_trans_layer_dist + aver_in_layer_dist)
    th_cost = exp_min_trans_l_cost * 1.20
    logging.info(f"   num_verts = {len(r1)},{len(r2)}, av_in_l_dist = {aver_in_layer_dist:.1f}, av_trans_l_dist = {aver_trans_layer_dist:.1f}, exp_min_trans_l_cost = {exp_min_trans_l_cost:.1f}, th_cost = {th_cost:.1f}")

    start_angles = np.arange(0.0, 2*np.pi, 2*np.pi/10)
    if(high_effort):
        start_angles = np.arange(0.0, 2*np.pi, 2*np.pi/40)
    sa_d_l = [(s,(-1)**sid) for sid, s in enumerate(start_angles)]
    for start_angle, dir in sa_d_l:
        # apply dir and start angle
        curr_cost = 0
        r1angs_c = [(a+start_angle) if (a<(np.pi*2-start_angle)) else (a+start_angle-np.pi*2) for a in r1angs]
        r2angs_c = [(a+start_angle) if (a<(np.pi*2-start_angle)) else (a+start_angle-np.pi*2) for a in r2angs]

        # make all angles positive
        min_ang = np.min([*r1angs_c, *r2angs_c])
        r1angs_c = r1angs_c - min_ang
        r2angs_c = r2angs_c - min_ang
    
        arg_r1angs_min = np.argmin(r1angs_c)
        arg_r2angs_min = np.argmin(r2angs_c)
    
        # shift r1 so its first element has the smallest angle
        r1angs_c = [*r1angs_c[arg_r1angs_min:], *r1angs_c[:arg_r1angs_min], r1angs_c[arg_r1angs_min] + 2*np.pi]
        r1_c     = [*      r1[arg_r1angs_min:], *      r1[:arg_r1angs_min],       r1[arg_r1angs_min]]
        r1_len = len(r1angs_c)
        if(dir==-1): #np.sum(r1angs_c[:int(r1_len/2)]) > np.sum(r1angs_c[int(r1_len/2):])):
            r1angs_c = r1angs_c[::-1]
            r1_c     = r1_c    [::-1]
        

        # shift r2 so its first element is the closest to the first element in the r1_c
        r2angs_c = [*r2angs_c[arg_r2angs_min:], *r2angs_c[:arg_r2angs_min], r2angs_c[arg_r2angs_min] + 2*np.pi]
        r2_c     = [*      r2[arg_r2angs_min:], *      r2[:arg_r2angs_min],       r2[arg_r2angs_min]]
        r2_len = len(r2angs_c)
        if(dir==-1): #np.sum(r2angs_c[:int(r2_len/2)]) > np.sum(r2angs_c[int(r2_len/2):])):
            r2angs_c = r2angs_c[::-1]
            r2_c     = r2_c    [::-1]
    
    
        verts = np.array([*r1_c, *r2_c])
        angs  = [*r1angs_c, *r2angs_c]
        v2off = len(r1_c)

        r1idx = 0
        r2idx = v2off

        faces = []
        unchanged_verts_cnt = 0
        last_was_DGD = True
        while(r1idx < (len(r1_c)-1)) or (r2idx < (len(angs)-1)):
            r1ca = angs[r1idx  ]
            r2ca = angs[r2idx  ]
            vd10 = verts[r1idx  ] 
            vd20 = verts[r2idx  ] 
            vd11 = verts[r1idx+1] if (r1idx != (len(r1_c)-1)) else np.array([10000]*3)
            vd21 = verts[r2idx+1] if (r2idx != (len(angs)-1)) else np.array([10000]*3)

            r1_rem       = (len(r1_c)-1) - r1idx
            r2_rem       = (len(angs)-1) - r2idx
            r1_too_short = (r1_rem <= 2*(r2_rem // unchanged_verts_limit))
            r2_too_short = (r2_rem <= 2*(r1_rem // unchanged_verts_limit))
            if (r1_rem == 0) or (r2_rem == 0):
                force_vert_switch = False
            else:
                force_vert_switch = (unchanged_verts_cnt >= unchanged_verts_limit) or r1_too_short or r2_too_short

            force_DGD = force_vert_switch and (r2_too_short or (not last_was_DGD))
            force_GGD = force_vert_switch and (r1_too_short or (    last_was_DGD))

            if not force_vert_switch and (unchanged_verts_cnt >= 5):
                _=1

            if compensate_com_shift:
                vd10 = vd10 - r1com
                vd20 = vd20 - r2com
                vd11 = vd11 - r1com
                vd21 = vd21 - r2com

            dDGD = np.linalg.norm(vd11-vd10) + np.linalg.norm(vd11-vd20)
            dGGD = np.linalg.norm(vd21-vd10) + np.linalg.norm(vd21-vd20)

            cost = min(dDGD, dGGD)

            if force_DGD or ((not force_GGD) and (dDGD <= dGGD)):
                # D G D
                if(dir == 1):
                    face = [r1idx, r2idx, r1idx+1]
                else:
                    face = [r1idx+1, r2idx, r1idx]
                r1idx = r1idx+1
                cost = dDGD
                cur_was_DGD = True
            else:
                # G G D
                if(dir == 1):
                    face = [r2idx, r2idx+1, r1idx]
                else:
                    face = [r2idx+1, r2idx, r1idx]
                r2idx = r2idx+1
                cost = dGGD
                cur_was_DGD = False
                
            if(last_was_DGD == cur_was_DGD):
                unchanged_verts_cnt += 1
            else:
                unchanged_verts_cnt = 1
            last_was_DGD = cur_was_DGD

            faces.append(face)
            curr_cost += cost

        curr_cost = curr_cost/num_trans_layer_conects

        logging.info(f"   sa = {round(start_angle*360/(2*np.pi)):5.1f}, dir = {dir:+}, cost = {curr_cost:10.1f} {'*' if curr_cost < best_cost else ' '}")    
        if curr_cost < best_cost:
            best_verts, best_faces = list(verts), faces
            best_cost = curr_cost
        
        if(best_cost < th_cost and not high_effort):
            logging.info(f"    fast break due to reaching th_cost") 
            fast_break = True
            break

    return best_verts, best_faces, fast_break

#----------------------------------------------------------------------------

color_dict = {"vessels":   (241/255.0,  12/255.0,  23/255.0, 1.00),
              "skinlines": (30/255.0,  212/255.0,  23/255.0, 0.60),
             #"skin"   :   (174/255.0, 154/255.0, 115/255.0, 0.65),
              "skin"   :   (160/255.0, 160/255.0, 160/255.0, 0.55),
              "bones"  :   (182/255.0, 190/255.0, 194/255.0, 1.00),
              "default":   (      0.5,       0.5,       0.5, 0.75)
              } 
#----------------------------------------------------
def define_mtl_str(color_dict):
    materials_def = ""
    
    for t in color_dict.keys():

        color4c = color_dict[t]

        materials_def += f"newmtl material_{t}\n"
        materials_def += f"Ka {color4c[0]} {color4c[1]} {color4c[2]}\n"
        materials_def += f"Kd {color4c[0]} {color4c[1]} {color4c[2]}\n"
        materials_def += f"Ks {color4c[0]} {color4c[1]} {color4c[2]}\n"
        materials_def += f"Ns   10.00\n"
    
        materials_def += f"illum 0\n"
        materials_def += f"d     {round(  color4c[3], 2)}\n"
        materials_def += f"Tr    {round(1-color4c[3], 2)}\n"

        materials_def += f"Tf    1.0\n"
        materials_def += f"Ni    1.50\n"

    #materials_def += f"newmtl material_skin\n"
    #materials_def += f"Ka 1.00 1.00 1.00\n"
    #materials_def += f"Kd 0.55 0.55 0.55\n"
    #materials_def += f"Ks 0.40 0.40 0.40\n"
    #materials_def += f"Ns   10.00\n"
    #
    #materials_def += f"illum 0\n"
    #materials_def += f"d     0.3\n"
    #materials_def += f"Tr    0.7\n"
    #
    #materials_def += f"Tf    1.0\n"
    #materials_def += f"Ni    1.50\n"
    #
    #materials_def += f"newmtl material_muscles\n"
    #materials_def += f"Ka 1.000000 0.200000 1.000000\n"
    #materials_def += f"Kd 1.000000 1.000000 1.000000\n"
    #materials_def += f"Ks 0.900000 0.900000 0.900000\n"
    #materials_def += f"illum 1\n"
    #materials_def += f"Tr 0.0\n"
    #materials_def += f"d  1.0\n"
    #
    #materials_def += f"newmtl material_bones\n"
    #materials_def += f"Ka 0.800000 0.800000 0.800000\n"
    #materials_def += f"Kd 0.752941 0.752941 0.752941\n"
    #materials_def += f"Ks 0.900000 0.900000 0.900000\n"
    #materials_def += f"illum 1\n"
    #materials_def += f"Tr 0.0\n"
    #materials_def += f"d  1.0\n"
    #
    #materials_def += f"newmtl material_vessels\n"
    #materials_def += f"Ka 1.000000 0.000000 0.000000\n"
    #materials_def += f"Kd 1.000000 0.000000 0.100000\n"
    #materials_def += f"Ks 0.200000 0.200000 0.200000\n"
    #materials_def += f"Ns   10.00\n"
    #
    #materials_def += f"illum 1\n"
    #materials_def += f"Tr 0.0\n"
    #materials_def += f"d  1.0\n"
    return materials_def

#----------------------------------------------------------------------------
# main
def main():
    global color_dict
    global slice_dist      
    global pixel_spacing_x 
    global pixel_spacing_y 
    
    #----------------------------------------------------------------------------
    start_time = time.time()
    start_time_total = start_time
    #----------------------------------------------------------------------------
    # initialize logging 
    script_name = os.path.basename(__file__).split(".")[0]
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    time_str = datetime.utcnow().strftime('%y_%m_%d__%H_%M_%S') #time.strftime("%y_%m_%d__%H_%M_%S.%f")
    initial_log_fn = f"_initial_{script_name}_{time_str}_pid{os.getpid()}.log"
    logging.basicConfig(level=logging.INFO, format = log_format, handlers=[logging.FileHandler(initial_log_fn, mode='w'), logging.StreamHandler(sys.stdout)])

    logging.info('*' * 50)
    logging.info(f"script {os.path.basename(__file__).split('.')[0]} start @ {time.ctime()}")
    logging.info("initial log file is {}".format(initial_log_fn))
    logging.info("*" * 50)
    
    from v_utils.v_logging_std import bind_std_2_logging
    bind_std_2_logging()
    
    logging.info(" Command line arguments:\n  {}".format(' '.join(sys.argv)))
    #----------------------------------------------------------------------------

    parser = ArgumentParser()

    parser.add_argument("-iDir",  "--in_dir",                           help="poligons input directory",   metavar="PATH",required=True)
    parser.add_argument("-sDir",  "--skin_dir",                         help="poligons input directory",   metavar="PATH",required=True)
    parser.add_argument("-oDir",  "--out_dir",                          help="mesh output directory",      metavar="PATH",required=True)
    parser.add_argument("-dDir",  "--desc_dir",                         help="description json file dir",  metavar="PATH",required=True)
    parser.add_argument("-nSfx",  "--name_sfx",      default='volume',  help="name sufix",                                required=False)
    parser.add_argument("-v",     "--verbose",                          help="verbose level",                             required=False)
    parser.add_argument("-dP",    "--do_plugs",    action=arg2boolAct, default=True ,  help="Create plugs",               required=False)
    parser.add_argument("-iH",    "--ignore_holes",action=arg2boolAct, default=False,  help="Ignore holes",               required=False)
    parser.add_argument("-lsn",   "--limit_slices_range", type = int,  help="limit range of processed slices",  nargs='*', default = [0, 1000], required=False)
    parser.add_argument("-iCs",   "--ds_polygon_clss",  type = str,  nargs='*', required=True,   metavar='STR', help="lista klas segmentow. Jednoczesnie nazwa folderow z referencyjnymi poligonami") 
    parser.add_argument("-os",    "--obj_smooth",   action=arg2boolAct, default=True ,  help="Smooth output OBJ file",    required=False)
    parser.add_argument("-fmct",   "--follow_mass_centers_trajectory",   action=arg2boolAct, default=False ,  help="If an object is not continued in a following layer then do the extrusion so the extruded object's mass centre follow the previously seen mass centre shift",    required=False)
    parser.add_argument("-dl",    "--detect_lines", action=arg2boolAct, default=False ,  help="Manage detected skin lines",               required=False)
        

    args = parser.parse_args()

    verbose 	= 'off'                 if args.verbose is None else args.verbose
    inDir  		= args.in_dir
    skinDir     = args.skin_dir
    outDir  	= args.out_dir
    descDir  	= args.desc_dir
    name_sfx    = args.name_sfx
    limit_sr    = args.limit_slices_range 
    do_plugs    = args.do_plugs
    ignore_holes= args.ignore_holes
    ts          = args.ds_polygon_clss
    obj_smooth  = args.obj_smooth
    mc_follow   = args.follow_mass_centers_trajectory
    detect_lines = args.detect_lines

    iDir 		= os.path.normpath(inDir)
    skinDir 	= os.path.normpath(skinDir)
    oDir 		= os.path.normpath(outDir)
    dDir 		= os.path.normpath(descDir)
    
    ses         = os.path.basename(oDir)
    user        = os.path.basename(os.path.dirname(oDir))

    if not os.path.isdir(iDir):
        logging.error('Error : Input directory (%s) with polygons files not found !'%iDir)
        exit(1)
    logging.info("-="*25)
    logging.info("START : as_gen_mesh.py")
    logging.info("in    : "+iDir)
    logging.info("out   : "+oDir)

    np.set_printoptions(linewidth=128)
    
    #----------------------------------------------------------------------------
    # create work dir
    logging.info("-" * 50)
    try:
        if not os.path.isdir(oDir):
            pathlib.Path(oDir).mkdir(mode=0o775, parents=True, exist_ok=True)
            logging.info("Created work dir {}".format(oDir))
    except Exception as err:
        logging.error("Creating dir ({}) IO error: {}".format(oDir, err))
        sys.exit(1)
    #----------------------------------------------------------------------------
    # redirect logging file to work directory 
    from as_bin.utils.logging_utils import redirect_log
    lDir = redirect_log(oDir, f"_{script_name}_{time_str}.log", f"_{script_name}_last.log")
    logging.info('-' * 50)
    #----------------------------------------------------------------------------
    gname           = iDir + f'/{ts[0]}/*_polygons.json'
    gname           = os.path.normpath(gname)
    gname 		    = gname.replace('[','[[]')

    cls0_poly_list  = glob.glob(gname)
    imid            = 0

    if len(cls0_poly_list)==0:
        logging.error('invalid file name or path (%s)'%gname)
        exit(1)

    cls0_poly_list.sort()

    #----------------------------------------------------
    # slice spacing
    #----------------------------------------------------
    
    logging.info("-"*50)
    logging.info("Get spacing information from sesion description file...")

    desc_path           = dDir + "/" + "description.json"
    desription          = load_json(desc_path)

    slice_dist          = desription["distance_between_slices"]
    pixel_spacing_x     = desription["pixel_spacing_x"]
    pixel_spacing_y     = desription["pixel_spacing_y"]
    logging.info(f"slice distance : {slice_dist:.3f}")
    logging.info(f"pixel_spacing_x : {pixel_spacing_x:.3f}")
    logging.info(f"pixel_spacing_y : {pixel_spacing_y:.3f}")
    
    #----------------------------------------------------

    #if True:
    #    my_rand1_path   =[    [ 2,  2],
    #                          [ 6,  2],
    #                          [ 6,  6],
    #                          [ 2,  6]]
    #    my_rand1_hole_path =[ [ 3,  3],
    #                          [ 5,  3],
    #                          [ 5,  5],
    #                          [ 3,  5]]
    #    
    #    
    #    logging.info("add two new contours to v_polygons")
    #    my_polygons = v_polygons(          ) #empty
    #    #1 - from paths (list of 2D points, or numpy ndarray of 2D points)
    #    my_polygons.add_polygon_from_paths(outer_path = my_rand1_path, holes_paths = [my_rand1_hole_path,]) 
    #    logging.info(my_polygons.to_indendent_str())
    #
    #    fn = "my_mask_image.png"
    #    logging.info("cast as pillow image and save to file: {}".format(fn))
    #    my_mask_image = my_polygons.as_image(fill=True, w=10, h=10, val = [255,0])
    #    my_mask_image.save(fn)
    #    
    #    for w in range(3,0,-1):
    #        fnd = "my_mask_image_borders_dilated_w{}.png".format(w)
    #        logging.info("Get  borders dilated by {} points and save to file: {}".format(w, fnd))
    #        dilated_border_polygons = v_polygons.from_polygons_borders(my_polygons, dilation_radius=w)
    #        my_mask_image = dilated_border_polygons.as_image(fill=True, w=10, h=10, masks_merge_type = 'over')
    #        my_mask_image.save(fnd)
    #
    #    f = Image.open(fn)
    #    my_polygons = v_polygons.from_image(f, background_val = 0, float_out = True)
    #    logging.info(my_polygons.to_indendent_str())
    #
    #    _=1

    #----------------------------------------------------
    #paramerty globalne
    #----------------------------------------------------

    scan            =  {}
    for t in ts:
        scan[t ]   =  {}
    
    if detect_lines:
        scan['skinlines']   =  {}

    #----------------------------------------------------
    # wczytanie jsonow z poligonami
    #----------------------------------------------------

    logging.info("-"*50)
    logging.info(f"Loading polygons files...")
    start_time = time.time()
    
    for t in ts:
        logging.info(f" Loading {t} polygons files...") 
        scan[t]["src"]  = group_load_polygons(skinDir, t, cls0_poly_list, limit_sr)
    
    if detect_lines:
        t = 'skinlines'
        logging.info(f" Loading {t} polygons files...") 
        scan[t]["src"]  = group_load_polygons(skinDir, t, cls0_poly_list, limit_sr)
    
        
    elapsed_time = time.time() - start_time
    logging.info("Files loaded in " + str(round(elapsed_time, 2)) + "s")

    #----------------------------------------------

    logging.info("-"*50)
    logging.info(f"Converting polygons...") 
    start_time = time.time()
    
    for t in ts:
        logging.info(f" Converting {t} polygons...") 
        scan[t]["slices"] = group_conv_polygons(scan[t]["src"], max_verts_dist =  (4.5 if ((t!= "vessels")and(t!= "skinlines")) else 2.0))
        
    elapsed_time = time.time() - start_time
    logging.info("Polygons converted in " + str(round(elapsed_time, 2)) + "s")

    #----------------------------------------------
    #dodatkowe parametry
    #----------------------------------------------
    first_tiss = ts[0]
    total_l         = len(scan[first_tiss]["slices"])
    
    #----------------------------------------------

    logging.info("-"*50)
    logging.info(f"Check polygons linkage through layers...") 
    start_time = time.time()

    for tn in ts:
        logging.info(f" {tn}...") 
        start_time_tiss = time.time()

        scan[tn]["linkage"] = []

        for lid in range(total_l):

            lz = slice_dist*lid
            tpoly_p = scan[tn]["slices"][lid-1]  if(lid !=         0) else v_polygons()
            tpoly_c = scan[tn]["slices"][lid  ]
            tpoly_n = scan[tn]["slices"][lid+1]  if(lid != total_l-1) else v_polygons()

            linkage_dict = {pid:[] for pid in range(-1, len(tpoly_c['polygons']))}

            for tp1, tp2, dir in [(tpoly_c, tpoly_n, "up")]:
                # linkage to the upper layer
        
                p1_co_l = [p['outer'] for p in tp1['polygons']]
                p2_co_l = [p['outer'] for p in tp2['polygons']]


                if (tn == 'vessels'):
                    # cost = contours centers distance - max(contours radius)
                    # max_angle = (-90, 0) -> contours should overap greatly
                    # max_angle =   0      -> contours should overlap
                    # max_angle = ( 0, 90) -> contours can be separated more and more
                    method = 'angle'
                    max_ang = 40
                    min_overlap = None
                    limit_matches_from_p1 = 2 # when going up vessel can only connect into one bigger vessel and can no split
                    test_similar_size_polys_first = False
                elif (tn == 'skinlines'):
                    # cost = contours centers distance - max(contours radius)
                    # max_angle = (-90, 0) -> contours should overap greatly
                    # max_angle =   0      -> contours should overlap
                    # max_angle = ( 0, 90) -> contours can be separated more and more
                    method = 'angle'
                    max_ang = 20
                    min_overlap = None
                    limit_matches_from_p1 = 2 # when going up vessel can only connect into one bigger vessel and can no split
                    test_similar_size_polys_first = False
                elif (tn == 'skin'):
                    method = 'angle'
                    max_ang = 0
                    min_overlap = None
                    limit_matches_from_p1 = 2 # when going up skin can only connect into one bigger polygon and can no split
                    test_similar_size_polys_first = True
                else: #bones
                    method = 'overlap'
                    min_overlap = 0.350
                    max_ang = None
                    limit_matches_from_p1 = 2 # when going up bones can connect into an elbow
                    test_similar_size_polys_first = True
                linkage_outter_dict_c = match_contours(p1_co_l, p2_co_l, 
                                                        method = method,
                                                        z_dist = slice_dist, max_ang = max_ang,
                                                        min_overlap = min_overlap,
                                                        limit_matches_from_p1 = limit_matches_from_p1, limit_matches_to_p2 = 2,
                                                        test_similar_size_polys_first = test_similar_size_polys_first)
                
                for p1id, v in linkage_outter_dict_c.items():
                    for contour_link_dict in v:
                        linkage_dict[p1id].append({"outer": contour_link_dict, "inners": {}})

                if not ignore_holes:
                    for p1id, p1 in enumerate(tp1['polygons']):
                        has_inners = len(p1['inners']) > 0
                        if has_inners:
                            p1_ci_l = p1['inners']
                            for p1link in linkage_dict[p1id]:
                                if p1link["outer"]["dst_id"]!=-1:
                                    p2 = tp2['polygons'][ p1link["outer"]["dst_id"] ]
                                    p2_ci_l = p2['inners']
                                else:
                                    p2_ci_l = []
                                if (tn == 'vessels'):
                                    method = 'angle'
                                    max_ang = 20
                                    min_overlap = None
                                    test_similar_size_polys_first = False
                                elif (tn == 'skinlines'):
                                    method = 'angle'
                                    max_ang = 10
                                    min_overlap = None
                                    test_similar_size_polys_first = False
                                elif (tn == 'skin'):
                                    method = 'angle'
                                    max_ang = 0
                                    min_overlap = None
                                    limit_matches_from_p1 = 1 # when going up skin can only connect into one bigger polygon and can no split
                                    test_similar_size_polys_first = True
                                else:
                                    method = 'overlap'
                                    min_overlap = 0.2
                                    max_ang = None
                                    test_similar_size_polys_first = True
                                linkage_up_inner_dict = match_contours(p1_ci_l, p2_ci_l, 
                                                                       method = method,
                                                                       z_dist = slice_dist, max_ang = max_ang,
                                                                       min_overlap = min_overlap,
                                                                       limit_matches_from_p1 = 1, limit_matches_to_p2 = 1,
                                                                       test_similar_size_polys_first = test_similar_size_polys_first)
                                for p1_iid, p2_iid_l in linkage_up_inner_dict.items():
                                    p1link["inners"][p1_iid] = p2_iid_l


            #up_linkage_dict_lm1 = scan[tn]["linkage"][-1] if lid > 0 else None
            #reversed_linkage = reverse_linkage(up_linkage_dict_lm1)

            scan[tn]["linkage"  ].append(linkage_dict)
    
        elapsed_time = time.time() - start_time_tiss
        logging.info(f" {tn} polygons linked in " + str(round(elapsed_time, 2)) + "s") 
    
    logging.info(f"Linkage for overlap method: {boxes_filled_and_square_n} times used simplified version (ring-like-assumed polys), and {boxes_trouble_n} times used full polygons overlapping version")
    elapsed_time = time.time() - start_time
    logging.info("Polygons linked in " + str(round(elapsed_time, 2)) + "s") 

    #----------------------------------------------

    logging.info("-"*50)
    logging.info(f"Re-check polygons linkage and remove redundant connections...") 
    start_time = time.time()

    for tn in ts:
        if (tn == 'vessels'):
            logging.info(f" {tn}...") 
            start_time_tiss = time.time()

            for lid in range(total_l-2):
            
                linkage_dict_l0 = scan[tn]["linkage"][lid  ]
                linkage_dict_l1 = scan[tn]["linkage"][lid+1]
                linkage_dict_l2 = scan[tn]["linkage"][lid+2]

                for c_src_id_l0 in linkage_dict_l0.keys():
                    c_linkage_l0 = linkage_dict_l0[c_src_id_l0]
                    if (c_src_id_l0 != -1) and len(c_linkage_l0) > 1:
                        #has point with multiple outs
                        c_dst_ids_l0 = [link["outer"]["dst_id"] for link in c_linkage_l0]
                        c_dst_ids_l1_lists = []
                        for c_src_id_l1 in c_dst_ids_l0:
                            c_linkage_l1 = linkage_dict_l1[c_src_id_l1]
                            if (c_src_id_l1 != -1) and len(c_linkage_l1) > 0:
                                # and some of its destination pointa has multiple outs
                                c_dst_ids_l1 = [link["outer"]["dst_id"] for link in c_linkage_l1]
                                c_dst_ids_l1_lists.append(c_dst_ids_l1)
                        if(len(c_dst_ids_l1_lists) > 1):
                            c_dst_ids_l1_flat = [v for l in c_dst_ids_l1_lists for v in l]
                            c_dst_ids_l1_uniq = set(c_dst_ids_l1_flat)
                            have_loop = len(c_dst_ids_l1_flat) > len(c_dst_ids_l1_uniq)
                            if have_loop:
                                logging.info(f"  Remove {tn} loop started at layers {lid} - {lid+2}")

                                dst_area_last = c_linkage_l0[-1]["outer"]["dst_contour"]["area"]
                                dst_area_firs = c_linkage_l0[ 0]["outer"]["dst_contour"]["area"]
                                if(dst_area_last > 2* dst_area_firs):
                                    # polaczenie pierwsze na liscie mialo nizszy koszt laczenia i zgodnie z tym
                                    # nalezaloby zostawic polaczenie z indeksem 0, ale roznica w rozmiarach 
                                    # docelowego poligonu sklania mnie do teo zeby zamienic ten domyslny wybor
                                    remo_id, left_id = 0, -1
                                else:
                                    remo_id, left_id = -1, 0

                                # find any other link to the same l1 contourand restore its dst_contour
                                l1_c_id = c_linkage_l0[remo_id]['outer']['dst_id']
                                for co_src_id_l0 in linkage_dict_l0.keys():
                                    if(co_src_id_l0 != -1) and (co_src_id_l0 != c_src_id_l0):
                                        co_linkage_l0 = linkage_dict_l0[co_src_id_l0]
                                        for co_link in co_linkage_l0:
                                            if co_link["outer"]["dst_id"] == l1_c_id:
                                                co_link["outer"]["dst_contour"] = co_link["outer"]["dst_contour_org"]
                                     
                                del c_linkage_l0[remo_id]
                                c_linkage_l0[left_id]["outer"]["src_contour"] = c_linkage_l0[left_id]["outer"]["src_contour_org"]
                has_loop = False


            elapsed_time = time.time() - start_time_tiss
            logging.info(f" {tn} linkage re-checked in " + str(round(elapsed_time, 2)) + "s") 
        
        
        if (tn == 'skinlines'):
            logging.info(f" {tn}...") 
            start_time_tiss = time.time()

            for lid in range(total_l-2):
            
                linkage_dict_l0 = scan[tn]["linkage"][lid  ]
                linkage_dict_l1 = scan[tn]["linkage"][lid+1]
                linkage_dict_l2 = scan[tn]["linkage"][lid+2]

                for c_src_id_l0 in linkage_dict_l0.keys():
                    c_linkage_l0 = linkage_dict_l0[c_src_id_l0]
                    if (c_src_id_l0 != -1) and len(c_linkage_l0) > 1:
                        #has point with multiple outs
                        c_dst_ids_l0 = [link["outer"]["dst_id"] for link in c_linkage_l0]
                        c_dst_ids_l1_lists = []
                        for c_src_id_l1 in c_dst_ids_l0:
                            c_linkage_l1 = linkage_dict_l1[c_src_id_l1]
                            if (c_src_id_l1 != -1) and len(c_linkage_l1) > 0:
                                # and some of its destination pointa has multiple outs
                                c_dst_ids_l1 = [link["outer"]["dst_id"] for link in c_linkage_l1]
                                c_dst_ids_l1_lists.append(c_dst_ids_l1)
                        if(len(c_dst_ids_l1_lists) > 1):
                            c_dst_ids_l1_flat = [v for l in c_dst_ids_l1_lists for v in l]
                            c_dst_ids_l1_uniq = set(c_dst_ids_l1_flat)
                            have_loop = len(c_dst_ids_l1_flat) > len(c_dst_ids_l1_uniq)
                            if have_loop:
                                logging.info(f"  Remove {tn} loop started at layers {lid} - {lid+2}")

                                dst_area_last = c_linkage_l0[-1]["outer"]["dst_contour"]["area"]
                                dst_area_firs = c_linkage_l0[ 0]["outer"]["dst_contour"]["area"]
                                if(dst_area_last > 2* dst_area_firs):
                                    # polaczenie pierwsze na liscie mialo nizszy koszt laczenia i zgodnie z tym
                                    # nalezaloby zostawic polaczenie z indeksem 0, ale roznica w rozmiarach 
                                    # docelowego poligonu sklania mnie do teo zeby zamienic ten domyslny wybor
                                    remo_id, left_id = 0, -1
                                else:
                                    remo_id, left_id = -1, 0

                                # find any other link to the same l1 contourand restore its dst_contour
                                l1_c_id = c_linkage_l0[remo_id]['outer']['dst_id']
                                for co_src_id_l0 in linkage_dict_l0.keys():
                                    if(co_src_id_l0 != -1) and (co_src_id_l0 != c_src_id_l0):
                                        co_linkage_l0 = linkage_dict_l0[co_src_id_l0]
                                        for co_link in co_linkage_l0:
                                            if co_link["outer"]["dst_id"] == l1_c_id:
                                                co_link["outer"]["dst_contour"] = co_link["outer"]["dst_contour_org"]
                                     
                                del c_linkage_l0[remo_id]
                                c_linkage_l0[left_id]["outer"]["src_contour"] = c_linkage_l0[left_id]["outer"]["src_contour_org"]
                has_loop = False


            elapsed_time = time.time() - start_time_tiss
            logging.info(f" {tn} linkage re-checked in " + str(round(elapsed_time, 2)) + "s") 
    
    elapsed_time = time.time() - start_time
    logging.info("Linakge re-checked in " + str(round(elapsed_time, 2)) + "s") 

    #----------------------------------------------

    logging.info("-"*50)
    logging.info(f"Linking polygons through layers...") 
    start_time = time.time()

    meshes = {}
    for tn in ts:
        
        logging.info("-"*50)
        logging.info(f" {tn}...") 
        start_time_tiss = time.time()

        connected_fast_n = 0
        connected_total_n = 0
        verts_ls, faces_ls = [], []
        for lid in range(total_l):
            logging.info(f"  layer{lid}")    

            lz = slice_dist*lid
        
           #tpoly_c = scan[tn]["slices"][lid  ]
           #tpoly_n = scan[tn]["slices"][lid+1]  if(lid != total_l-1) else v_polygons()

        
            layer_linkage    = scan[tn]["linkage"][lid  ]
            layer_linkage_m1 = scan[tn]["linkage"][lid-1] if lid > 0 else None
            layer_linkage_dn = reverse_linkage(layer_linkage_m1)
            for cp_outer_idx in layer_linkage.keys():
                if(cp_outer_idx == -1):
                    continue

                c_linkages_up = layer_linkage   [cp_outer_idx]
                c_linkages_dn = layer_linkage_dn[cp_outer_idx] if (not layer_linkage_dn is None) and (cp_outer_idx in layer_linkage_dn.keys()) else None
                
                cp_n_outers_verts_2_plug = []
                cp_c_outers_verts_2_plug = [] 
                cp_n_inners_verts_2_plug = []
                cp_c_inners_verts_2_plug = [] 

                outlines_perims_to_plug  = []
                
                for c_linkage_up in c_linkages_up:
                    
                    c_link_up_out = c_linkage_up['outer']
                    sh_ratio = 0.5

                    # plugs
                    up_need_plug = (c_link_up_out["dst_contour"] is None)
                    dn_need_plug = (c_linkages_dn is None) or (len(c_linkages_dn) == 0)
                    if not dn_need_plug:
                        dn_need_plug = True
                        for c_linkage_dn in c_linkages_dn:
                            dn_need_plug &= c_linkage_dn['outer']['dst_contour'] is None


                    c_outer_verts = np.array([[*v2D, lz         ] for v2D in c_link_up_out["src_contour"]['path']]) 
                    c_outer_centr = np.array([*c_link_up_out["src_contour"]['centroid'], lz])                             
                    c_outer_area  =            c_link_up_out["src_contour"]['area'    ]  
                    
                    if  dn_need_plug:
                        sh_dz = - slice_dist * sh_ratio
                        if mc_follow and (not up_need_plug):
                            up_dst_cont = c_link_up_out['dst_contour']
                            next_centriod = up_dst_cont['centroid']
                            centriod_dif = np.array(next_centriod[0:2] - c_outer_centr[0:2])
                            sh_dxy = -centriod_dif * sh_ratio
                        else:
                            sh_dxy = [0, 0]
                        c_outer_verts += np.array([*sh_dxy, sh_dz])                           
                        c_outer_centr += np.array([*sh_dxy, sh_dz]) 

                    if(up_need_plug):
                        sh_dz = slice_dist * sh_ratio
                        n_is_c_copy = True
                        if dn_need_plug:
                            sh_dz += slice_dist * sh_ratio
                        if mc_follow and (not c_linkages_dn is None) and (len(c_linkages_dn) == 1):
                            dn_dst_cont = c_linkages_dn[0]['outer']['dst_contour']
                            if not dn_dst_cont is None:
                                n_is_c_copy = False
                                prev_centriod = dn_dst_cont['centroid']
                                centriod_dif = np.array(n_outer_centr[0:2] - prev_centriod[0:2])
                                sh_dxy = centriod_dif * sh_ratio
                            else:
                                sh_dxy = [0, 0]
                        else:
                            sh_dxy = [0, 0]
                        n_outer_verts = c_outer_verts + [*sh_dxy, sh_dz]
                        n_outer_centr = c_outer_centr + [*sh_dxy, sh_dz]
                        n_outer_area  = c_outer_area                           
                        n_id          = c_link_up_out["dst_id"]  

                        #continue
                    else:
                        n_is_c_copy = False
                        n_outer_verts = np.array([[*v2D, lz+slice_dist] for v2D in c_link_up_out["dst_contour"]['path']]) 
                        n_outer_centr = np.array([*c_link_up_out["dst_contour"]['centroid'], lz+slice_dist])                            
                        n_outer_area  =            c_link_up_out["dst_contour"]['area'    ]
                        n_id          =            c_link_up_out["dst_id"]                                                              
                
                    # get angles
                    c_angs = assign_angles(ring = c_outer_verts, centr = c_outer_centr)
                    n_angs = assign_angles(ring = n_outer_verts, centr = n_outer_centr)

                    contours_rel_size = c_outer_area/n_outer_area if n_outer_area>0 else 1000
                    contours_rel_size = contours_rel_size if contours_rel_size > 1.0 else 1/contours_rel_size
                    min_verts_num = min(len(c_outer_verts), len(n_outer_verts))
                    test_all_pos = min_verts_num <= 5
                    high_effort = (not test_all_pos) and (c_link_up_out["is_complex"] or (contours_rel_size > 2))
                    do_connect_convex_first = (not test_all_pos) and ((tn == "bones") or high_effort)
                    compensate_com_shift = (tn == "vessels") or (tn == "skinlines") or do_connect_convex_first
                    # connect
                    l_verts, l_faces, fb = connect_rings_2step(r1 = c_outer_verts, r1angs = c_angs, r1com = c_outer_centr,
                                                         r2 = n_outer_verts, r2angs = n_angs, r2com = n_outer_centr,
                                                         compensate_com_shift = compensate_com_shift,
                                                         high_effort = high_effort,
                                                         unchanged_verts_limit = 4,
                                                         do_connect_convex_first = do_connect_convex_first,
                                                         use_connect_rings_idxs_ord = test_all_pos,
                                                         is_outer = True,
                                                         do_simple = n_is_c_copy)
                    if fb:
                        connected_fast_n += 1
                    connected_total_n += 1
            
                    verts_ls.append(l_verts) 
                    faces_ls.append(l_faces) 


                    if not ignore_holes:
                        for cp_inner_idx in c_linkage_up['inners'].keys():
                            if(cp_inner_idx == -1):
                                continue
                            c_linkage_up_ins = c_linkage_up['inners'][cp_inner_idx]
                            for c_linkage_up_in in c_linkage_up_ins:
                                
                                # plugs need check:

                                c_inner_contour_id = c_linkage_up_in["src_id"]
                                c_linkage_dn_in = c_linkages_dn[0]['inners'] if (not c_linkages_dn is None) else None
                                cp_inner_is_linked_up = not(c_linkage_up_in["dst_contour"] is None)
                                
                                dn_need_plug_in = (c_linkages_dn is None) or (len(c_linkages_dn) == 0)
                                if not dn_need_plug_in:
                                    dn_need_plug_in = True
                                    for c_linkage_dn in c_linkages_dn:
                                        c_linkage_dn_ins = c_linkage_dn['inners']
                                        for _, c_linkage_dn_in in c_linkage_dn_ins.items():
                                            for c_ldi in c_linkage_dn_in:
                                                if not c_ldi['src_contour'] is None:
                                                    if (c_ldi['src_contour']['centroid'] == c_linkage_up_in["src_contour"]['centroid']).all():
                                                        dn_need_plug_in &= c_ldi['dst_contour'] is None
                                                        if not c_ldi['dst_contour'] is None:
                                                            c_inner_contour_dn_link = c_ldi['dst_contour']
                                                            break

                                cp_inner_is_linked_dn = not dn_need_plug_in
                                #dn_need_plug_in = not cp_inner_is_linked_dn
                                up_need_plug_in = not cp_inner_is_linked_up

                                #vertical connections
                                c_inner_verts = np.array([[*v2D, lz           ] for v2D in c_linkage_up_in["src_contour"]['path']]) 
                                c_inner_centr = np.array([*c_linkage_up_in["src_contour"]['centroid'], lz])                                                    
                                c_inner_area  =            c_linkage_up_in["src_contour"]['area'    ]   
                                if dn_need_plug_in:
                                    sh_dz = (- slice_dist * sh_ratio) if (dn_need_plug) else (- slice_dist * 0.01)
                                    if mc_follow and (not c_linkage_up_in["dst_contour"] is None):
                                        up_dst_cont = c_linkage_up_in['dst_contour']
                                        next_centriod = up_dst_cont['centroid']
                                        centriod_dif = np.array(next_centriod[0:2] - c_inner_centr[0:2])
                                        sh_dxy = -centriod_dif * sh_ratio
                                    else:
                                        sh_dxy = [0, 0]
                                    c_inner_verts += np.array([*sh_dxy, sh_dz])                           
                                    c_inner_centr += np.array([*sh_dxy, sh_dz])               

                                if(up_need_plug_in):
                                    sh_dz = slice_dist * sh_ratio if (up_need_plug) else (slice_dist * 0.01)
                                    n_is_c_copy_in = True
                                    if dn_need_plug_in:
                                        sh_dz += slice_dist * sh_ratio
                                    if mc_follow and cp_inner_is_linked_dn:
                                        dn_dst_cont = c_inner_contour_dn_link['dst_contour']
                                        if not dn_dst_cont is None:
                                            n_is_c_copy_in = False
                                            prev_centriod = dn_dst_cont['centroid']
                                            centriod_dif = np.array(n_inner_centr[0:2] - prev_centriod[0:2])
                                            sh_dxy = centriod_dif * sh_ratio
                                        else:
                                            sh_dxy = [0, 0]
                                    else:
                                        sh_dxy = [0, 0]
                                    n_inner_verts = c_inner_verts + [*sh_dxy, sh_dz]                    
                                    n_inner_centr = c_inner_centr + [*sh_dxy, sh_dz]

                                    n_inner_area  = c_inner_area                                          
                                    n_id          = c_linkage_up_in["dst_id"]                                                         
                                else:
                                    n_is_c_copy_in = False
                                    n_inner_verts = np.array([[*v2D, lz+slice_dist] for v2D in c_linkage_up_in["dst_contour"]['path']]) 
                                    n_inner_centr = np.array([*c_linkage_up_in["dst_contour"]['centroid'], lz+slice_dist])                             
                                    n_inner_area  =            c_linkage_up_in["dst_contour"]['area'    ]
                                    n_id          = c_linkage_up_in["dst_id"]                                                              
                
                                # get angles
                                c_angs = assign_angles(ring = c_inner_verts, centr = c_inner_centr)
                                n_angs = assign_angles(ring = n_inner_verts, centr = n_inner_centr)
                                
                                contours_rel_size = c_inner_area/n_inner_area if n_inner_area>0 else 1000
                                contours_rel_size = contours_rel_size if contours_rel_size > 1.0 else 1/contours_rel_size
                                rel_sizes_differs = (contours_rel_size > 2)
                                high_effort = c_linkage_up_in["is_complex"] or rel_sizes_differs
                                do_connect_convex_first = (tn == "bones") or high_effort
                                compensate_com_shift = (tn == "vessels") or (tn == "skinlines") or do_connect_convex_first
                                # connect
                                l_verts, l_faces, fb = connect_rings_2step(r1 = c_inner_verts, r1angs = c_angs, r1com = c_inner_centr,
                                                                     r2 = n_inner_verts, r2angs = n_angs, r2com = n_inner_centr,
                                                                     compensate_com_shift = compensate_com_shift,
                                                                     high_effort = high_effort,
                                                                     unchanged_verts_limit = 10 if rel_sizes_differs else 4,
                                                                     do_connect_convex_first = do_connect_convex_first,
                                                                     use_connect_rings_idxs_ord = (tn == "vessels") or (tn == "skinlines"),
                                                                     is_outer = False,
                                                                     do_simple = n_is_c_copy_in)
                                verts_ls.append(l_verts) 
                                faces_ls.append(l_faces) 
                                if fb:
                                    connected_fast_n += 1
                                connected_total_n += 1
                                
                                # plugs:
                                if up_need_plug_in:
                                    cp_n_inners_verts_2_plug.append(n_inner_verts)
                                if dn_need_plug_in:
                                    cp_c_inners_verts_2_plug.append(c_inner_verts)

                    # plugs
                    if up_need_plug:
                        cp_n_outers_verts_2_plug.append(n_outer_verts)
                 

                    if dn_need_plug:
                        cp_c_outers_verts_2_plug.append(c_outer_verts)
                    
                ##############################################################################
                if len(cp_n_outers_verts_2_plug) > 0:
                    for cp_n_outer_verts_2_plug in cp_n_outers_verts_2_plug:
                        outlines_perims_to_plug.append((cp_n_outer_verts_2_plug, cp_n_inners_verts_2_plug, 1))
                else:
                    for cp_n_inner_verts_2_plug in cp_n_inners_verts_2_plug:
                        outlines_perims_to_plug.append((cp_n_inner_verts_2_plug, [], -1))

                if len(cp_c_outers_verts_2_plug) > 0:
                    for cp_c_outer_verts_2_plug in cp_c_outers_verts_2_plug:
                        outlines_perims_to_plug.append((cp_c_outer_verts_2_plug, cp_c_inners_verts_2_plug, -1))
                else:
                    for cp_c_inner_verts_2_plug in cp_c_inners_verts_2_plug:
                        outlines_perims_to_plug.append((cp_c_inner_verts_2_plug, [], 1))

                ##############################################################################

                if(do_plugs):# and (lid != total_l-1)):   
                    for op2p_id, (p2p_outer_verts3D, p2p_inners_verts3D, dir) in enumerate(outlines_perims_to_plug):
                        if(len(p2p_outer_verts3D) < 4):
                            continue
                        logging.info(f"  create polygon {op2p_id}...")
                        p2p_outer_verts2D  = p2p_outer_verts3D [:,0:2]
                        p2p_inners_verts2D = [p2p_inner_verts3D[:,0:2] for p2p_inner_verts3D in p2p_inners_verts3D]
                        p2p_z = p2p_outer_verts3D[0,2]
                        p2p = Polygon(shell=p2p_outer_verts2D, holes=p2p_inners_verts2D)
                        logging.info(f"  triangulate polygon {op2p_id}...")
                        # triange arguments: https://www.cs.cmu.edu/~quake/triangle.switch.html
                        # S0 - 0 Stainer points.
                        # Y  - no Stainer points at the mesh boundary. 
                        # a - imposes a maximum triangle area constraint
                        inter_layer_triangle_area =  slice_dist* pixel_spacing_x / 2
                        maximum_plug_area = int(np.ceil(inter_layer_triangle_area * 5))
                        logging.info(f"   force no Stainer points at polygon perimeter and max triangle area to {maximum_plug_area}...")
                        triangle_args = f'pYa{maximum_plug_area}' 
                        #triangle_args = f'pYq45a{maximum_plug_area}' # Y - no Stainer points at the mesh boundary - points added to existing vertices. 
                        # Forced because the stainer point it can appear at perimeter of a plug and in the final mesh 
                        # the stainer point will be only at plug but not at a wall so those will not create a solid mesh together
                        try:
                            p2p_vertices, p2p_faces = trimesh.creation.triangulate_polygon(p2p, triangle_args=triangle_args)
                        except:
                            p2p_vertices, p2p_faces = trimesh.creation.triangulate_polygon(p2p.buffer(distance=0), triangle_args=triangle_args)
                        logging.info(f"  add Zs to vertices of the triangulated polygon {op2p_id}...")
                        p2p_outer_verts3D = [[*v, p2p_z] for v in p2p_vertices]

                        if(dir == -1):
                           p2p_faces = [[f[0], f[2], f[1]] for f in p2p_faces]
                
                        logging.info(f"  append vertices and faces to the existing mesh vertices and faces {op2p_id}...")
                        verts_ls.append(p2p_outer_verts3D) 
                        faces_ls.append(p2p_faces)
                
                        if False:
                            plug_mesh = trimesh.Trimesh(vertices=p2p_outer_verts3D, faces=p2p_faces)
                            show_meshes([plug_mesh])
                
            if(False):
                logging.info(f"Create mesh from the gathered vertices and faces...")
                mesh = trimesh.Trimesh(vertices=l_verts, faces=l_faces)
                show_meshes([meshe])
            #logging.info(f" fix inversions, normals and merge duplicate vertices...")
            #fixes_chain(mesh)
        perc_fast = (connected_fast_n/connected_total_n*100) if connected_total_n!= 0 else 100
        logging.info(f" {tn} connected. {connected_total_n} polygons connected, {perc_fast:.0f}% connected fast.") 

        logging.info(f"Concatenate all {tn} vertices and faces ...")
        flat_verts = []
        flat_faces = []
        for vs, fs in zip(verts_ls, faces_ls):
            o = len(flat_verts)
            fs_Off =  [[f[0]+o, f[1]+o, f[2]+o] for f in fs]
            flat_verts.extend(vs)
            flat_faces.extend(fs_Off)
        
        #from collections import Counter
        #individual_rows_counter = Counter(map(tuple, flat_verts))

        logging.info(f"Create {tn} mesh from the gathered vertices and faces...")
        #num_fs = len(flat_faces)
        mesh = trimesh.Trimesh(vertices=flat_verts, faces=flat_faces)

        if False:
            #https://github.com/mikedh/trimesh/issues/857
            # find the four vertices in every `mesh.face_adjacency` row
            check = np.column_stack((mesh.face_adjacency_edges,
                                     mesh.face_adjacency_unshared))
            # make sure rows are sorted
            check.sort(axis=1)

            # find the indexes of unique face adjacency
            adj_unique = trimesh.grouping.unique_rows(check)[0]

            # find the unique indexes of the original faces
            faces_mask = trimesh.grouping.unique_bincount(
                mesh.face_adjacency[adj_unique].reshape(-1))

            # apply the mask to remove non-unique faces
            mesh.update_faces(faces_mask)
            
        do_resolve_overlapping_faces = True
        if do_resolve_overlapping_faces:
            logging.info(f"Check for overlaping faces/triangles...")
                            
            #from scipy.spatial.qhull import Delaunay
            #from trimesh.geometry import plane_transform

            def _remove_overlapping_triangles(m: trimesh.Trimesh) -> trimesh.Trimesh:
                cleaned_mesh = m

                num_impoved = 0
                improved = True
                while improved:
                    improved = False
                    # For all facets reconstruct the surfaces using Delaunay triangulation
                    for f_adj_idx, (face_p_id, face_n_id) in enumerate(cleaned_mesh.face_adjacency):
                        #if True:
                        #    fp_v_ids = cleaned_mesh.faces[face_p_id]
                        #    fn_v_ids = cleaned_mesh.faces[face_n_id]
                        #    fnp_e_v_ids = cleaned_mesh.face_adjacency_edges[f_adj_idx]
                        #    fp_v_id_uniq =  [ v_id for v_id in fp_v_ids if not (v_id in fnp_e_v_ids)][0]
                        #    fn_v_id_uniq =  [ v_id for v_id in fn_v_ids if not (v_id in fnp_e_v_ids)][0]
                        #    fp_v_uniq = cleaned_mesh.vertices[fp_v_id_uniq]
                        #    fn_v_uniq = cleaned_mesh.vertices[fn_v_id_uniq]
                        #    fnp_e_vs =  cleaned_mesh.vertices[fnp_e_v_ids]
                        #    found = True
                        #    for v in [fp_v_uniq, fn_v_uniq, *fnp_e_vs]:
                        #        if v[0] < 4 or v[0] > 6:
                        #            found = False
                        #        if v[1] < 45 or v[1] > 49.6:
                        #            found = False
                        #        if v[2] > 4:
                        #            found = False
                        #    if found:
                        #        _=1
                        normals_negative = np.all(np.array(cleaned_mesh.face_normals[face_p_id], dtype = np.float16) == -np.array(cleaned_mesh.face_normals[face_n_id], dtype = np.float16))
                        if normals_negative:
                            logging.warning(f" Found a new pair of overlaping faces/triangles ({face_p_id}, {face_n_id})! Divide those into nonoverlapping faces and recreate a mesh...")
                            newly_added_faces = []
                            removed_faces = []
                            fp_v_ids = cleaned_mesh.faces[face_p_id]
                            fn_v_ids = cleaned_mesh.faces[face_n_id]
                            fnp_e_v_ids = cleaned_mesh.face_adjacency_edges[f_adj_idx]
                            #find cross vertex - new vertex at crossing of the two ovelapping faces
                            fp_v_id_uniq =  [ v_id for v_id in fp_v_ids if not (v_id in fnp_e_v_ids)][0]
                            fn_v_id_uniq =  [ v_id for v_id in fn_v_ids if not (v_id in fnp_e_v_ids)][0]
                            fp_v_uniq = cleaned_mesh.vertices[fp_v_id_uniq]
                            fn_v_uniq = cleaned_mesh.vertices[fn_v_id_uniq]
                            fnp_e_vs =  cleaned_mesh.vertices[fnp_e_v_ids]
                            fp_cv_ids = [fnp_e_v_id for fnp_e_v_id in fnp_e_v_ids if not cleaned_mesh.vertices[fnp_e_v_id][2]==fp_v_uniq[2]]
                            fn_cv_ids = [fnp_e_v_id for fnp_e_v_id in fnp_e_v_ids if not cleaned_mesh.vertices[fnp_e_v_id][2]==fn_v_uniq[2]]

                            found = False
                            fs_cv_ids = zip([*fp_cv_ids, *fp_cv_ids], [*fn_cv_ids, *fn_cv_ids[::-1]]) if (len(fn_cv_ids)+len(fp_cv_ids))>2 else [[*fp_cv_ids, *fn_cv_ids]]
                            for (fp_cv_id, fn_cv_id) in fs_cv_ids:

                                fp_cv = cleaned_mesh.vertices[fp_cv_id]
                                fn_cv = cleaned_mesh.vertices[fn_cv_id]
                                fn_cv_v_dir = np.array(fn_cv-fn_v_uniq) #np.cross(np.array(fp_cv-fp_v_uniq) /np.linalg.norm(fp_cv-fp_v_uniq)**2, [1.0, 0.0, 0.0])
                                fn_cv_v_dir /= np.linalg.norm(fn_cv_v_dir)**2

                                new_v = [None, False]
                                for cross_vector in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]):
                                    fn_pln_norm = np.cross(fn_cv_v_dir, cross_vector)

                                    #fp_cv, fp_pln_norm=np.cross(np.array(fp_cv-fp_v_uniq) /np.linalg.norm(fp_cv-fp_v_uniq)**2, [0.0, 1.0, 0.0]), 
                                    #endpoints=[fn_v_uniq,fn_cv]
                                    new_v = trimesh.intersections.plane_lines(plane_origin=fn_cv, plane_normal=fn_pln_norm, endpoints=[fp_v_uniq,fp_cv], line_segments=False)
                                    if(new_v[1]):                                   
                                        break

                                if(not new_v[1]):
                                    logging.warning("Could not resolve overlapping triangles/faces issue because could not find intersection point")
                                    continue

                                new_v = new_v[0][0]
                                zs = np.array(cleaned_mesh.vertices[fp_v_ids][:,2], dtype=np.float64) 
                                z_low = np.min(zs)
                                z_hig = np.max(zs)
                                sigma = (z_hig - z_low) * 0.0001
                                z_low = z_low + sigma
                                z_hig = z_hig - sigma
                                is_same_as_np_e_v = np.any([np.allclose(cc, new_v) for cc in fnp_e_vs])
                                if (new_v[2] > z_low) and (new_v[2] < z_hig) and (not is_same_as_np_e_v):
                                    logging.info(f"Found intersection point {new_v}")
                                    found = True
                                    break

                            if(found):
                                num_impoved += 1
                                verts = [*cleaned_mesh.vertices, new_v]
                                new_v_id = len(verts)-1

                                faces = [*cleaned_mesh.faces]
                                removed_faces.append(faces[face_p_id])
                                removed_faces.append(faces[face_n_id])
                                # switch to the newly created vertex
                                faces[face_p_id] = [v_id if (v_id!=fp_cv_id) else new_v_id for v_id in faces[face_p_id]]
                                faces[face_n_id] = [v_id if (v_id!=fn_cv_id) else new_v_id for v_id in faces[face_n_id]]
                            
                                newly_added_faces.append(faces[face_p_id])
                                newly_added_faces.append(faces[face_n_id])

                                # divide sourounding triangles into two that uses a new vertex also
                                for f_adj_idx, (v0_id, v1_id) in enumerate(cleaned_mesh.face_adjacency_edges):
                                    div_n = ((v0_id == fp_v_id_uniq) and (v1_id == fp_cv_id)) or ((v1_id == fp_v_id_uniq) and (v0_id == fp_cv_id))
                                    div_p = ((v0_id == fn_v_id_uniq) and (v1_id == fn_cv_id)) or ((v1_id == fn_v_id_uniq) and (v0_id == fn_cv_id))
                                    if div_n or div_p:
                                        face_div_id0, face_div_id1 = cleaned_mesh.face_adjacency[f_adj_idx]
                                        face_div_id = face_div_id0 if (face_div_id1 in [face_p_id, face_n_id]) else face_div_id1
                                        face_div = cleaned_mesh.faces[face_div_id]
                                        face_div_v_rep_ids = [v for vid, v in enumerate(face_div) if v in (v0_id, v1_id)]
                                        new_div_faces = []
                                        for v_rep_id in face_div_v_rep_ids:
                                            new_div_face = [new_v_id if (v==v_rep_id) else v  for v in face_div]
                                            new_div_faces.append(new_div_face)
                                        removed_faces.append(faces[face_div_id])
                                        faces[face_div_id] = new_div_faces[0]
                                        faces.append(new_div_faces[1])
                                        newly_added_faces.extend(new_div_faces)
                                        
                                do_resolve_overlapping_faces_dbg = False
                                if do_resolve_overlapping_faces_dbg:
                                    rem_faces_mesh = trimesh.Trimesh(vertices=verts, faces=removed_faces)
                                    new_faces_meshs = [trimesh.Trimesh(vertices=verts, faces=[f]) for f in newly_added_faces]
                                    pvuniq = Sphere(radius = 0.1, center = fp_v_uniq)
                                    nvuniq = Sphere(radius = 0.1, center = fn_v_uniq)
                                    cvuniq = Sphere(radius = 0.1, center = new_v) 
                                    new_faces_meshs_clrs = [(0.9, 0.0, 0.0, 0.85) for msh in new_faces_meshs]
                                    show_meshes([rem_faces_mesh, *new_faces_meshs, pvuniq, nvuniq, cvuniq], \
                                                colours_list=[(0.0, 0.0, 0.0, 0.5), *new_faces_meshs_clrs, (0.0, 0.6, 0.0, 0.85), (0.0, 0.6, 0.0, 0.85), (0.0, 0.0, 0.9, 0.85)])
                                    _=1

                                check_degenerated_faces = False
                                if check_degenerated_faces:
                                    deg_f_idxs = []
                                    for fidx, f in enumerate(faces):
                                        vs = [verts[vidx] for vidx in f]
                                        for vp in [(vs[0], vs[1]), (vs[1], vs[2]), (vs[0], vs[2])]:
                                            if np.all(vp[0] == vp[1]):
                                                logging.warning(f"degenerated face {vs}!")
                                                deg_f_idxs.append(fidx)
                                    if len(deg_f_idxs) > 0:
                                        deg_f_idxs = np.unique(deg_f_idxs)
                                        deg_f_idxs = list(deg_f_idxs)
                                        cleaned_faces = [f for fidx, f in enumerate(faces) if not (fidx in deg_f_idxs)]
                                        faces = cleaned_faces

                                cleaned_mesh = trimesh.Trimesh(vertices=verts, faces=faces)


                                improved = True
                                break
                            else: 
                                do_resolve_overlapping_faces_dbg = False
                                if do_resolve_overlapping_faces_dbg:
                                    removed_faces.append(faces[face_p_id])
                                    removed_faces.append(faces[face_n_id])
                                    rem_faces_mesh = trimesh.Trimesh(vertices=verts, faces=removed_faces)
                                    pvuniq = Sphere(radius = 0.1, center = fp_v_uniq)
                                    nvuniq = Sphere(radius = 0.1, center = fn_v_uniq)
                                    show_meshes([rem_faces_mesh, pvuniq, nvuniq], \
                                                colours_list=[(0.0, 0.0, 0.0, 0.5), (0.0, 0.6, 0.0, 0.85), (0.0, 0.6, 0.0, 0.85)])
                                    _=1

                return cleaned_mesh, num_impoved
            

            mesh, overlaps_resolved_num = _remove_overlapping_triangles(mesh)
            if(overlaps_resolved_num):
                logging.warning(f"Resolved {overlaps_resolved_num} overlaps of faces/triangles!")




        #num_fs_m = len(mesh.faces)
        #if num_fs_m != num_fs:
        #    logging.warning(f"Generating mesh removed {num_fs - num_fs_m} faces!")
        #deg_fs_mask = mesh.remove_degenerate_faces()
        #num_fs_mrd = len(mesh.faces)
        #if num_fs_mrd != num_fs_m:
        #    logging.warning(f"Sanity degenerated faces remove has removed {num_fs_m - num_fs_mrd} faces!")
        meshes[tn] = mesh
        #mesh = fixes_chain(mesh)
        if(False):
            show_meshes([mesh])


        elapsed_time = time.time() - start_time_tiss
        logging.info(f" {tn} polygons connected in " + str(round(elapsed_time, 2)) + "s")

    elapsed_time = time.time() - start_time
    logging.info("Polygons connected in " + str(round(elapsed_time, 2)) + "s")
    
    #----------------------------------------------

    #logging.info("-"*50)
    #logging.info("Chain of fixes for all tissue's meshes")
    #start_time = time.time()
    #
    #for t in ts:
    #    meshes[t] = fixes_chain(meshes[t], do_fix_invertion = False, do_fix_normals = False)
    #    
    #elapsed_time = time.time() - start_time
    #logging.info("Fixed in " + str(round(elapsed_time, 2)) + "s")

    #----------------------------------------------

    if(False):
        logging.info("-"*50)
        logging.info(f"Colour tissue's meshes")

        meshes_for_show = copy.deepcopy(meshes)
        for t in ts:
            if(t in color_dict):
                meshes_color = color_dict[t]
            else:
                meshes_color = color_dict["default"]
            meshes_for_show[t].visual.face_colors = np.tile(meshes_color, (len(meshes[t].faces), 1))
        show_meshes(meshes_for_show)
    
    #----------------------------------------------
    if(len(ts) > 1):
        logging.info("-"*50)
        logging.info(f"Concatenate all tissue's vertices and faces ...")
        start_time = time.time()

        flat_verts = []
        flat_faces = []
        flat_faces_colors = []
        for t in ts:
            mesh = meshes[t]
            vs, fs = mesh.vertices, mesh.faces
            #if(t in color_dict):
            #    meshes_color = color_dict[t]
            #else:
            #    meshes_color = color_dict["default"]
            #flat_faces_colors = np.array([*flat_faces_colors, *np.tile(meshes_color, (len(fs), 1))])
            vo = len(flat_verts)
            fs_Off =  [[f[0]+vo, f[1]+vo, f[2]+vo] for f in fs]
            flat_verts.extend(vs)
            flat_faces.extend(fs_Off) 
            
        elapsed_time = time.time() - start_time
        logging.info("Concatenated in " + str(round(elapsed_time, 2)) + "s")
    else:
        t = ts[0]
        flat_verts          = meshes[t].vertices
        flat_faces          = meshes[t].faces
        #flat_faces_colors   = meshes[t].visual.face_colors
        
    #-----------------------------------------------------------------------------------------
    
    logging.info("-"*50)
    logging.info(f"Create {t} mesh from the gathered vertices and faces...")
    start_time = time.time()

    mesh_merged = trimesh.Trimesh(vertices=flat_verts, faces=flat_faces)
    #mesh_merged.visual.face_colors = flat_faces_colors

    elapsed_time = time.time() - start_time
    logging.info("Took " + str(round(elapsed_time, 2)) + "s")
    
    #-----------------------------------------------------------------------------------------
    logging.info("-"*50)
    logging.info(f"Prepare material info file...")

    tissues_names = ""
    for t in ts:
        tissues_names += f"_{t}"
    mtl_fn = f"{user}_{ses}{tissues_names}_mesh_{name_sfx}.mtl"
    mtl_pth = os.path.join(oDir, mtl_fn)

    logging.info(f" saving {mtl_pth}...")

    with open(mtl_pth, 'w') as f:
        mtl_str = define_mtl_str(color_dict)
        f.write(mtl_str)

    #-----------------------------------------------------------------------------------------
    # STLs and OBJs export 

    logging.info("-"*50)
    logging.info(f"Exporting meshes to files...")
    start_time = time.time()
    
    #-----------------------------------------------------------------------------------------
    # STLs and OBJs separate tissues
    
    if(len(ts) > 0):
    
        logging.info(f" Add normals to meshes...")
    
        for t in ts:
            logging.info(f"  {t}...")
            #fn = meshes[t].face_normals
            vn = meshes[t].vertex_normals
    
    #-----------------------------------------------------------------------------------------
    # STLs and OBJs separate tissues
    
    if(len(ts) > 1):

        logging.info(f" Exporting separate tissues to files...")

        for t in ts:
            out_fn = f"{user}_{ses}_{t}_mesh_{name_sfx}.stl"
            out_pth = os.path.join(oDir, out_fn)

            logging.info(f"  Exporting to {out_pth}. STL file (no color info)...")
            meshes[t].export(out_pth)
    
    #-----------------------------------------------------------------------------------------
    # export merged tissues

    logging.info(f" Exporting {'merged' if(len(ts) > 1) else ''} tissues to files...")

    all_tiss_names = ""
    for t in ts:
        all_tiss_names += f"_{t}"
        
    #-----------------------------------------------------------------------------------------
    # STLs merged tissues

    out_fn = f"{user}_{ses}{all_tiss_names}_mesh_{name_sfx}.stl"
    out_pth = os.path.join(oDir, out_fn)

    logging.info(f"  Exporting to {out_pth}. STL file (no color info)...")
    with open(out_pth, 'bw') as f:
        if mesh_merged.body_count > 0:
            mesh_merged.export( f, "stl")
    
    #-----------------------------------------------------------------------------------------
    # OBJs merged

    logging.info(f" Exporting merged tissues...")

    out_fn = f"{user}_{ses}{all_tiss_names}_mesh_{name_sfx}.obj"
    out_pth = os.path.join(oDir, out_fn)

    obj_export_as_scene = True
    if( obj_export_as_scene):

        scene = trimesh.Scene()
        for t in ts:
            scene.add_geometry(geometry = meshes[t], geom_name=t)

        logging.info(f"  Exporting to {out_pth}. OBJ file (has color info)...")
        with open(out_pth, 'w') as f:
            #for t in ts:
            #    obj_str = trimesh.exchange.obj.export_obj(meshes[ts[0]], include_normals=True)
            #    f.write(f"\no {t}\n")
            #    f.write(obj_str)
            #scene.export( f, "obj")
            obj_str = trimesh.exchange.obj.export_obj(scene, include_normals=True)
            obj_str = obj_str.split("\n")
            
            #get normals strings for each tissue by creating obj file string for a separate tissue and filtering only normals' lines
            tiss_obj_str_normals = []
            for t in ts:
                tis_obj_str = trimesh.exchange.obj.export_obj(meshes[ts[0]], include_normals=True)
                tis_obj_str = tis_obj_str.split("\n")
                tis_obj_str_normals = [l for l in tis_obj_str if l[0:2]=="vn"]
                tis_obj_str_normals = "\n".join(tis_obj_str_normals)
                tiss_obj_str_normals.append(tis_obj_str_normals)
                
            #insert normals infor into obj with all tissues
            for t in ts[::-1]:
                found_tissue_start = False
                for lidx, l in enumerate(obj_str):
                    if len(l)>3 and l[0] == 'o':
                        if(t == l[2:]):
                            found_tissue_start = True
                            logging.info(f"    {t}...")
                    elif found_tissue_start and len(l)>3 and l[0] == 'f':
                        l = tis_obj_str_normals + "\n" + l

                        obj_str[lidx] = l
                        break


            for l in obj_str:
                f.write(l + "\n")
            #_=1

        logging.info(f"   Adding materials info to {out_pth}...")
        with open(out_pth, 'r') as f:
            
            all_obj_lines = f.readlines()

        for lidx, l in enumerate(all_obj_lines):
            if l[0] == 'o':
                t = l[2:-1]
                logging.info(f"    {t}...")
                mtl_insert_txt  = ""
                mtl_insert_txt += f"#mtl file\n"           
                mtl_insert_txt += f"mtllib  {mtl_fn}\n"   
                mtl_insert_txt += f"#material for {t}\n"  
                if t in color_dict.keys():
                    mtl_insert_txt += f"usemtl material_{t}\n"
                else:
                    mtl_insert_txt += f"usemtl material_default\n"

                l += mtl_insert_txt

                all_obj_lines[lidx] = l
        if obj_smooth:
            all_obj_lines[0] = all_obj_lines[0] + "s 1\n"

        with open(out_pth, 'w') as f:
            
            all_obj_lines = "".join(all_obj_lines)
            f.write(all_obj_lines)
            
    elapsed_time = time.time() - start_time
    logging.info("Exported in " + str(round(elapsed_time, 2)) + "s")
    #-----------------------------------------------------------------------------------------
    
    logging.info(f"-" * 50)
    elapsed_time = time.time() - start_time_total
    logging.info("Total time: " + str(round(elapsed_time, 2)) + "s")
    logging.info(f"Done work in {oDir}")
    logging.info(f"=" * 50)

if __name__ == '__main__':
    main()