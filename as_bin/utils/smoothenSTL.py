import sys, getopt
import numpy as np
import json
import os
import pathlib
from argparse import ArgumentParser
import glob
import math
import shutil
import logging
import time
import random
import trimesh
from scipy.spatial import cKDTree
from multiprocessing import Process, Queue, Array
import copy
from datetime import datetime


#-----------------------------------------------------------------------------------------
sys.path.append(os.getcwd())
#-----------------------------------------------------------------------------------------
from v_utils.v_json import *
from v_utils.v_polygons import *
from v_utils.v_arg import arg2boolAct
#-----------------------------------------------------------------------------------------
from as_bin.utils.mesh_utils import mesh_repair
from as_bin.utils.offset_verts import offset_verts
#-----------------------------------------------------------------------------------------

def calc_normal(vertices, faces, face_normals, vertex_num):
    normals_to_consider = [0,0,0]
    number = 0
    for i,f in enumerate(faces):
        if vertex_num in f:
            normals_to_consider += face_normals[i]
            number += 1
    return normals_to_consider / number
    


def run_point(vertices, faces, vertex_normals, vertex_num, radius, sign, vert_tree):
    sasiedzi_ind = find_neighbors3(vertices, vertex_num, radius, vert_tree)
    sasiedzi = [vertices[i] for i in sasiedzi_ind]
    nowy_uklad = calc_tangent_plane(vertices, faces, vertex_normals, vertex_num, sasiedzi_ind)
    nowe_punkty = transform_coords(sasiedzi, nowy_uklad, vertices[vertex_num])
    wspolczynniki = approximate_cubic(nowe_punkty)
    vertice_new = vertices[vertex_num] + sign * np.array(wspolczynniki[5]*nowy_uklad[0])
    return vertice_new


def calc_tangent_plane(vertices, faces, vertex_normals, vertex_num, sasiedzi_ind):
    #vec1 = calc_normal(vertices, faces, face_normals, vertex_num)
    
    if False:
        vec1 = vertex_normals[vertex_num]
    else:
        sum_norm = [0, 0, 0]
        for s in sasiedzi_ind:
            sum_norm[0] += vertex_normals[s][0]
            sum_norm[1] += vertex_normals[s][1]
            sum_norm[2] += vertex_normals[s][2]
        vec1 = [sum_norm[0]/len(sasiedzi_ind), sum_norm[1]/len(sasiedzi_ind), sum_norm[2]/len(sasiedzi_ind)]
        
        
    
    
    temp = np.array([0,0,1])
    vec2 = np.cross(vec1,temp)

    if np.linalg.norm(vec2) == 0:
        temp = np.array([0,1,0])
        vec2 = np.cross(vec1,temp)

    vec3 = np.cross(vec1,vec2)
    
    vec1 = vec1/np.linalg.norm(vec1)
    vec2 = vec2/np.linalg.norm(vec2)
    vec3 = vec3/np.linalg.norm(vec3)
    return (vec1, vec2, vec3)
    
def get_dist_sq(p1, p2):
    return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2+ (p1[2]-p2[2])**2

def find_neighbors(vertices, vertex_num, dist_max, mesh):
    res = []
    dm = dist_max**2
    for v in vertices:
        if get_dist_sq(vertices[vertex_num], v) < dm:
            res.append(v)
    return res
    
def find_neighbors2(vertices, vertex_num, dist_max, mesh):
    res = []
    #to_check = [*mesh.vertex_neighbors[vertex_num],]
    to_check = [vertex_num,]
    dm = dist_max**2
    
    while to_check:
        if get_dist_sq(vertices[vertex_num], vertices[to_check[0]]) < dm:
            res.append(to_check[0])
            for v in mesh.vertex_neighbors[to_check[0]]:
                if (not v in to_check) and (not v in res):
                    to_check.append(v)
        to_check.pop(0)
        
    return [vertices[i] for i in res]
    
def find_neighbors3(vertices, vertex_num, dist_max, vert_tree):
    res = []
    closest_verts = vert_tree.query_ball_point(vertices[vertex_num], dist_max)
    #return [vertices[i] for i in [*closest_verts]]
    return [*closest_verts]
    
def transform_coords(pts, coords, p0):
#    start_time = time.time()
    ws = [(p[0]-p0[0], p[1]-p0[1], p[2]-p0[2]) for p in pts]
    new_pts = [[np.dot(w, coords[0]), np.dot(w, coords[1]), np.dot(w, coords[2])] for w in ws]
    
    #new_pts = []
    ##acc = np.array([0,0,0])
    ##for p in pts:
    ##    acc[0] +=p[2][0]
    ##    acc[1] +=p[2][1]
    ##    acc[2] +=p[2][2]
    ##acc[0] /= len(pts)
    ##acc[1] /= len(pts)
    ##acc[2] /= len(pts)
    #
    ##acc = np.array(p0)
    #for p in pts:
    #    #c0 = np.dot(np.array(p[2])-acc, coords[0])#*coords[0]
    #    w = (p[0]-p0[0], p[1]-p0[1], p[2]-p0[2])
    #    c0 = np.dot(w, coords[0])
    #    #c1 = np.dot(np.array(p[2])-acc, coords[1])#*coords[1]
    #    c1 = np.dot(w, coords[1])
    #    #c2 = np.dot(np.array(p[2])-acc, coords[2])#*coords[2]
    #    c2 = np.dot(w, coords[2])
    #    new_pts.append([c0,c1,c2])
    
#    elapsed_time = time.time() - start_time
#    logging.info(f"Total time {elapsed_time} s")    
    return new_pts
    
    
def approximate_cubic(points):
    macierz_liczenie_LS = [[c[1]**2, c[2]**2, c[1]*c[2], c[1], c[2],1] for c in points]
    wektor_liczenie_LS = [c[0] for c in points]
    
    #for p in points:
    #    c = p
    #    macierz_liczenie_LS.append([c[1]**2, c[2]**2, c[1]*c[2], c[1], c[2],1])
    #    wektor_liczenie_LS.append(c[0])
    
    wspolczynniki = np.linalg.lstsq(macierz_liczenie_LS, wektor_liczenie_LS, rcond=None)
    return wspolczynniki[0]
    
    
def process_piece(vertexy, krawedz, start_v, stop_v, trojkaty, normalne, radius, sign, result, num_done, idx_done):
    vert_tree = cKDTree(vertexy)
    krawedz_full = np.zeros(max([max(krawedz), stop_v])+1, dtype=bool)
    krawedz_full[krawedz] = True
    do_shrink_radius_at_border = True
    if do_shrink_radius_at_border:
        krawedz_vs = np.array(vertexy)[krawedz]
        kdtree = cKDTree(krawedz_vs)
        vids2boundary_dists, vids2boundary_vidx = kdtree.query(vertexy, distance_upper_bound=radius)
        radiouses = vids2boundary_dists
        radiouses[np.where(vids2boundary_dists>radius)[0]] = radius
    else:
        radiouses = np.ones(len(vertexy)) * radius
    for i in range(start_v, stop_v):
        if not krawedz_full[i]:
            result.put([i, run_point(vertexy, trojkaty, normalne, i, radiouses[i], sign, vert_tree)])
        num_done[idx_done] += 1
        #print(i)
        

def process_piece_singlethread(vertexy, krawedz, start_v, stop_v, trojkaty, normalne, radius, sign, result, num_done, idx_done):
    vert_tree = cKDTree(vertexy)
    krawedz_full = np.zeros(max([max(krawedz), stop_v])+1, dtype=bool)
    krawedz_full[krawedz] = True
    do_shrink_radius_at_border = True
    if do_shrink_radius_at_border:
        krawedz_vs = np.array(vertexy)[krawedz]
        kdtree = cKDTree(krawedz_vs)
        vids2boundary_dists, vids2boundary_vidx = kdtree.query(vertexy, distance_upper_bound=radius)
        radiouses = vids2boundary_dists
        radiouses[np.where(vids2boundary_dists>radius)[0]] = radius
    else:
        radiouses = np.ones(len(vertexy)) * radius
    for i in range(start_v, stop_v):
        if not krawedz_full[i]:
            result.append([i, run_point(vertexy, trojkaty, normalne, i, radiouses[i], sign, vert_tree)])
        num_done[idx_done] += 1
        #print(i)

    
def run(file_in, file_out, radius, sign, threads, dbg_files=False):
    mesh_in = trimesh.load(file_in)
    
    vertexy = list(mesh_in.vertices)
    trojkaty = list(mesh_in.faces)
    normalne = list(mesh_in.vertex_normals)
    
    krawedzie = [oe.points for oe in mesh_in.outline().entities]
    krawedz = []
    for krawedz_ps in krawedzie:
        krawedz.extend(krawedz_ps)
    krawedz = list(np.unique(krawedz))
    dbg_read_from_file = False
    if dbg_read_from_file:
        dbg_file = file_out.replace(".stl", "_dbg.stl")
        mesh_in = trimesh.load(dbg_file)
        vertexy = list(mesh_in.vertices)
        trojkaty = list(mesh_in.faces)
        normalne = list(mesh_in.vertex_normals)
    else:
        factor = 1
        
        points_per_thread = math.ceil(len(vertexy)/threads/factor)
        
        pid = 0
        
        result = []
        
        processes = []
        if threads == 1:
            modyfikacje_1 = []
            process_piece_singlethread(vertexy, krawedz, 0, points_per_thread, trojkaty, normalne, radius, sign, modyfikacje_1, [0], 0)
        else:
            num_verts_done = Array('i', [0 for r in range(threads)])
            for i in range(threads):
                p_name=f"{pid}_smooth_{i}"
                if points_per_thread*(i+1) > len(vertexy):
                    temp = len(vertexy)
                else:
                    temp = points_per_thread*(i+1)
                result.append(Queue())
                #print("Processing vertices from {} to {}".format(points_per_thread*i, temp-1))
                p = Process(target=process_piece, \
                        args=(vertexy, krawedz, points_per_thread*i, temp, trojkaty, normalne, radius, sign, result[i], num_verts_done, i), \
                        name=p_name)
                processes.append(p)
                pid+=1
            for p in processes:
                p.start()
                
            modyfikacje_1 = []
            done = False
            while not done:
                all_finished = True
                for i,p in enumerate(processes):
                    if p.is_alive():
                        all_finished = False
                    while not result[i].empty():
                        modyfikacje_1.append(list(result[i].get()))
                if all_finished:
                    done = True
                logging.info("{}/{}".format(sum(num_verts_done), len(vertexy)/factor))
                #logging.info("{}, {}".format([i.qsize() for i in result], len(modyfikacje_1)))
                time.sleep(3)
            print("Joining")
            for p in processes:
                p.join()
                
            for i in range(len(processes)):
                while not result[i].empty():
                    modyfikacje_1.append(list(result[i].get()))

        for m in modyfikacje_1:
            vertexy[m[0]] = m[1]

        if dbg_files:
            dbg_mesh = trimesh.Trimesh(vertexy, trojkaty)
            dbg_file_out = file_out.replace(".stl", "_dbg.stl")
            logging.info("Writing output file {}".format(dbg_file_out))
            dbg_mesh.export(dbg_file_out)

    do_use_offset_script_for_vertices_translation = False
    if do_use_offset_script_for_vertices_translation:
        vertex_normal_np_filter = vertexy - mesh_in.vertices
        vertex_magnitude_filter = np.linalg.norm(vertex_normal_np_filter, axis =1)
        vertex_normal_np_filter[vertex_magnitude_filter!= 0.0] = vertex_normal_np_filter[vertex_magnitude_filter!= 0.0] / vertex_magnitude_filter[vertex_magnitude_filter!= 0.0, np.newaxis]
        vertex_filter_tot = np.where(vertex_magnitude_filter!= 0.0)[0]
        vertex_magnitude_filter = vertex_magnitude_filter[vertex_filter_tot]
        vertex_normal_np_filter = vertex_normal_np_filter[vertex_filter_tot]
        logging.info(f" 3) Execute offset function...")
        logging.getLogger().setLevel(logging.ERROR)
        mesh_result  = offset_verts(mesh_in, magnitude  = 1.0,
                                    vertex_filter    = vertex_filter_tot, 
                                    vertex_magnitude = vertex_magnitude_filter,
                                    vertex_normal    = vertex_normal_np_filter,
                                    
                                    pitch = 0.5,
                                    do_adjust_z_at_top    = False,
                                    do_original_z_at_top  = False,
                                    do_keep_outline_verts = True,
                                    do_simple             = False,
                                    #do_update_vertices_normals_at_each_step =  False,
                                    #smooth_mesh_at_each_step =  False,

                                    dbg_files             = False,
                                    #dbg_o_dir             = tmp_out_dir,
                                    #dbg_o_fn_ptrn         = tmp_fn_ptrn,
                                    return_type_is_meshio = False
                                    )
    else:
        if dbg_files:
            tmp_dir = os.path.join(os.path.dirname(file_out), "_tmp")
            if not os.path.isdir(tmp_dir):
                os.makedirs(tmp_dir)
        new_faces, new_points, pass_id, changed = mesh_repair( trojkaty, vertexy,
                    keep_verts                      = krawedz, 
                    check_close_points              = True, 
                    check_duplicated_points         = False, # i tak nic z tym nie robie 
                    check_degenerated_faces         = True, 
                    check_overlapping_faces         = True, 
                    check_boundary_dangling_faces   = True,
                    check_boundaryonly_faces        = False, 
                    check_alternating_faces         = True, 
                    check_narrow_faces              = True, 
                    max_passes                      = 10, 
                    short_edge_th                   = 0.1, 
                    max_adjecency_angle             = 120, 
                    stop_when_no_degenerated_overlapping_alternating_faces = True, 
                    do_return_merged_points_list    = False, 
                    do_remove_unused_points         = True, 
                    log_level                       = 1 ,
                    dbg_dir = tmp_dir if dbg_files else None)
        mesh_result = trimesh.Trimesh(new_points, new_faces)
    
    
    #mesh_result = trimesh.Trimesh(vertexy, trojkaty)
    #trimesh.repair.fix_normals(mesh_result)
    
    try:
        logging.info("Writing output file {}".format(file_out))
        mesh_result.export(file_out)
    except Exception as e:
        logging.error("Error wriing file {}".format(file_out))
        logging.error("  {}".format(e))
    
    
def main():
    #----------------------------------------------------------------------------
    # initialize logging 
    script_name = os.path.basename(__file__).split(".")[0]
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    time_str = datetime.utcnow().strftime('%y_%m_%d__%H_%M_%S') #time.strftime("%y_%m_%d__%H_%M_%S.%f")
    initial_log_fn = f"_initial_{script_name}_{time_str}_pid{os.getpid()}.log"
#    initial_log_fn = f"{script_name}.log"
    logging.basicConfig(level=logging.INFO, format = log_format, handlers=[logging.FileHandler(initial_log_fn, mode='w'), logging.StreamHandler(sys.stdout)])
    
    try:
        os.chmod(initial_log_fn, 0o666)
    except:
        logging.warning(f"Could not change log file permitions {initial_log_fn}. I'm not the owner of the file?")
    
    
    logging.info(f'*' * 50)
    logging.info(f"script {script_name} start @ {time.ctime()}")
    logging.info(f"log file is {initial_log_fn}")
    logging.info(f"*" * 50)
    logging.info(f"Parse command line arguments...")
    
    start_time = time.time()
    
    #----------------------------------------------------------------------------
    parser = ArgumentParser()
    logging.info(' -' * 25)
    logging.info(" Command line arguments:\n  {}".format(' '.join(sys.argv)))
    parser = ArgumentParser()
    parser.add_argument("-if",  "--inputfile" ,  help="input file",             metavar="PATH", required=True)
    parser.add_argument("-of",  "--outputfile",  help="output file",            metavar="PATH", required=True)
    parser.add_argument("-r" ,  "--radius"    ,  help="smooth radius",                          required=True)
    parser.add_argument("-s" ,  "--sign"      ,  help="offset sign [+1, -1]",                   required=True)
    parser.add_argument("-th",  "--threads"   ,type = int , default = -2       ,help="Number of simultaneous processes",       required=False)
    parser.add_argument("-v" ,  "--verbose"   ,  help="verbose level",                          required=False)
    parser.add_argument(        "--dbg_files",   default=False, action=arg2boolAct, help="Leave all files for debug", required=False)
    args = parser.parse_args()

    verbose 	        = False                 if args.verbose is None else (args.verbose == 'on')
    
    file_in   = args.inputfile
    file_out  = args.outputfile
    radius    = float(args.radius)
    sign      = int(args.sign)

    if not os.path.exists(file_in):
        logging.error(f'Input file {file_in} not found !')
        exit(1)
        
    logging.info("-="*25                                )
    logging.info(f"START             : {script_name}"   )
    logging.info(f"input STL         : {file_in}"       )
    logging.info(f"output STL        : {file_out}"      )
    logging.info(f"radius            : {radius}"        )
    logging.info(f"sign              : {sign}"          )
    
    threads     = args.threads 
    if args.threads <= 0:
        threads = max(1, (os.cpu_count() - abs(args.threads)))
    
    run(file_in, file_out, radius, sign, threads, args.dbg_files)
    
    
    
    stop_time = time.time()
    logging.info(f"script {script_name} finish @ {time.ctime()}")
    run_time = stop_time - start_time
    
    logging.info(f"Finished in {run_time} seconds")


if __name__=='__main__':
    main()