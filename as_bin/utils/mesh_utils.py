import numpy as np
import logging
import meshio
import trimesh
import copy
import os

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
        elif (len(colours_list) == 3 or len(colours_list) == 4) and (not hasattr(colours_list[0], 'len')) and (not type(colours_list[0]) == type(tuple())):
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
    def callback(scene):
        if scene.pause:
            return

        translation = np.random.uniform(0, 1, (3,))
        axis = trimesh.creation.axis()
        axis.apply_translation(translation)
        scene.add_geometry(axis)
        scene.set_camera()

    window = windowed.SceneViewer(scene, 
        callback=callback,
        callback_period=0.1,
        start_loop=False)

    scene.pause = True
    window.toggle_axis()
    
    #@window.event
    #def on_key_press(symbol, modifiers):
    #    if symbol == pyglet.window.key.W:
    #        window.toggle_wireframe()
    #        window.update_flags()
    #    elif symbol == pyglet.window.key.A:
    #        window.toggle_axis()
    #        window.update_flags()
    #    elif symbol == pyglet.window.key.C:
    #        window.toggle_culling()
    #        window.update_flags()
    #    elif symbol == pyglet.window.key.F:
    #        window.toggle_fullscreen()
    #        window.update_flags()
    #    elif symbol == pyglet.window.key.G:
    #        window.toggle_grid()
    #        window.update_flags()
    #    elif symbol == pyglet.window.key.Q:
    #        window.close()
    #    elif symbol == pyglet.window.key.P:
    #        scene.pause = not scene.pause
    #    _=1
            
    pyglet.app.run()
def v2s(vector, precisiion = 1):
    strs = [f"{v:.1f}" for v in vector]
    str_r = ", ".join(strs)
    str_out = f"[{str_r}]"
    return str_out
    
def find_neighbouring_verts(mesh, v):
    nvis = []
    tris = mesh.cells_dict['triangle']
    tids, own_vids = np.where(tris==v)
    for tid in tids:
        tvis = tris[tid]
        nvis.append(list(tvis))
    nvis = np.unique(nvis)
    nvis = nvis[nvis!=v]
    return nvis

#def smooth_mesh_and_rewrite(pth, verts, steps = 3, depth = 0, save_backup_of_input_file = False):
def smooth_mesh_and_rewrite(msh, vids_to_smooth, steps = 3, depth = 0, 
                            save_backup_of_input_file = False, 
                            vids_blocked = None, do_increase_depth_when_no_change = True):
    
    #msh = read_meshio_with_retry(pth, max_tries = 1)
    if type(msh) is trimesh.base.Trimesh:
        num_of_msh_points = len(msh.vertices)
    else:
        num_of_msh_points = len(msh.points)
    # Make sure all the vids_to_smooth are present in the input mesh
    # (in the current application, there can be vids_to_smooth which are tissue's Steiner points and not the points from the input ROI mesh)
    vids_to_smooth = np.array(vids_to_smooth)
    if not vids_blocked is None:
        vids_to_smooth = np.setdiff1d(vids_to_smooth, vids_blocked)
    vids_to_smooth = vids_to_smooth[vids_to_smooth<num_of_msh_points]
        
    if type(msh) is trimesh.base.Trimesh:
        vs_dict = {vid:{"ns_vid": msh.vertex_neighbors[vid]} for vid in vids_to_smooth}
    else:
        vs_dict = {vid:{} for vid in vids_to_smooth}

        for vid in vids_to_smooth:
            vid_ns = find_neighbouring_verts(msh, vid)
            vs_dict[vid]["ns_vid"] = vid_ns

    if depth > 0:
        logging.info(f"  Extend vids_to_smooth-to-smooth by neighbours of those vids_to_smooth and retry...")
        verts_ext = [vs_dict[vid]["ns_vid"] for vid in vids_to_smooth]
        verts_ext.append(vids_to_smooth)
        verts_ext = [item for sublist in verts_ext for item in sublist]
        verts_ext = list(np.unique(verts_ext))
        
        #smooth_mesh_and_rewrite(pth, verts_ext, steps = steps+1, depth = depth-1)
        msh = smooth_mesh_and_rewrite(msh, verts_ext, steps = steps+1, depth = depth-1, vids_blocked=vids_blocked, do_increase_depth_when_no_change=do_increase_depth_when_no_change)
        return msh

    for sid in range(steps):

        valid_diff = False
        if type(msh) is trimesh.base.Trimesh:
            org_poss = np.array([         msh.vertices[vid]                                                    for vid in vids_to_smooth])
            nm_poss  = np.array([np.mean([msh.vertices[vid], *msh.vertices[vs_dict[vid]["ns_vid"]]], axis = 0) for vid in vids_to_smooth])
        else:
            org_poss = np.array([         msh.points  [vid]                                                    for vid in vids_to_smooth])
            nm_poss  = np.array([np.mean([msh.points  [vid], *msh.points  [vs_dict[vid]["ns_vid"]]], axis = 0) for vid in vids_to_smooth])
        new_dposs = (nm_poss - org_poss) / steps
        valid_diff = np.any(new_dposs) 
        new_poss = org_poss + new_dposs
        #for vid in vids_to_smooth:
        #    vdict = vs_dict[vid]
        #    if type(msh) is trimesh.base.Trimesh:
        #        org_pos = msh.vertices[vid]
        #        ns_pos  = msh.vertices[vdict["ns_vid"]]
        #    else:
        #        org_pos = msh.points[vid]
        #        ns_pos  = msh.points[vdict["ns_vid"]]
        #        
        #    if len(vdict["ns_vid"]) > 0:
        #        nm_pos = np.mean(ns_pos, axis = 0)
        #        new_dpos = (nm_pos - org_pos) / steps
        #        vdict["new_pos"] = org_pos + new_dpos
        #        valid_diff |= np.any(new_dpos)
        #    else:
        #        vdict["new_pos"] = org_pos

        #valid_diff = np.any(np.array([np.any(np.array(vs_dict[vid]["new_dpos"]) != 0.0) for vid in vids_to_smooth]))
        if not valid_diff:
            if do_increase_depth_when_no_change:
                logging.info(f"  No change to vertices position during smoothing. Extend vids_to_smooth-to-smooth by neighbours of those vids_to_smooth and retry...")
                #smooth_mesh_and_rewrite(pth, vids_to_smooth, steps = steps + 1, depth = depth+1)
                msh = smooth_mesh_and_rewrite(msh, vids_to_smooth, steps = steps + 1, depth = depth+1, vids_blocked=vids_blocked, do_increase_depth_when_no_change=do_increase_depth_when_no_change)
            return msh

        if type(msh) is trimesh.base.Trimesh:
            msh.vertices[vids_to_smooth] = new_poss
            #for vid in vids_to_smooth:
            #    msh.vertices[vid] = vs_dict[vid]["new_pos"]
        else:
            msh.points[vids_to_smooth] = new_poss
            #for vid in vids_to_smooth:
            #    msh.points[vid] = vs_dict[vid]["new_pos"]

    #if save_backup_of_input_file:
    #    bkp_pth = pth.replace('.', '_bkp.')
    #    if not os.path.exists(bkp_pth):
    #        logging.info(f"  rename the source file {pth} -> {bkp_pth}...")
    #        #os.remove(bkp_pth)
    #        os.rename(pth, bkp_pth)

    #logging.info(f"  save the mesh with the faulty vertices smoothed to {pth}...")
    #msh.write(pth)
    return msh

def adjust_z_at_top(points, faces, dz_th, max_normal_deviation_deg=None, force_z_val = None):
    ps_z = points[:,2]
    max_z = np.max(ps_z)
    vid_ids = np.where(np.logical_and((ps_z > (max_z - dz_th)), (ps_z < max_z)))[0]
    changed = False
    logging.info(f" Found {len(vid_ids)} vertices close to the plane at maximum Z. Change its Z coordinate so those are at the plane.")
    
    new_points = copy.copy(points)
    if len(vid_ids) > 0:
        changed = True
        logging.info(f"  Change its Z coordinate so those are at the plane...")
        
        if force_z_val is None:
            new_points[vid_ids,2] = max_z
        else:
            changed = np.any(new_points[vid_ids,2] != force_z_val)
            new_points[vid_ids,2] = force_z_val

        if not max_normal_deviation_deg is None:
            logging.info(f"  Check normals' deviation...")
            tmesh1 = trimesh.Trimesh(vertices=points,     faces=faces, process=False)
            tmesh2 = trimesh.Trimesh(vertices=new_points, faces=faces, process=False)
            vn1 = tmesh1.vertex_normals[vid_ids]
            vn2 = tmesh2.vertex_normals[vid_ids] 
            vn1_u = vn1 / np.linalg.norm(vn1, axis = 1)[:,None]
            vn2_u = vn2 / np.linalg.norm(vn2, axis = 1)[:,None]
            #vn_dif_rad = np.abs(np.arccos(np.clip(np.dot(vn1_u, vn2_u)       , -1.0, 1.0)))
            #vn_dif_rad  = np.abs(np.arccos(np.clip(np.sum(vn1_u*vn2_u, axis=1), -1.0, 1.0)))
            #vn_dif_rad  = np.abs(np.arccos(np.clip(       vn1_u[:,2]*vn2_u[:,2], -1.0, 1.0)))
            vn1_z_rad  = np.arcsin(vn1_u[:,2])
            vn2_z_rad  = np.arcsin(vn2_u[:,2])
            vn_dif_rad = vn2_z_rad - vn1_z_rad
            vn_dif_rad = (vn_dif_rad + np.pi) % (2 * np.pi) - np.pi
            max_normal_deviation_rad = np.abs(np.deg2rad(max_normal_deviation_deg))
            mask_out = np.where(vn_dif_rad > max_normal_deviation_rad)[0]
            if len(mask_out):
                logging.info(f"   Found {len(mask_out)} vertices with deviation of its normal greater than {max_normal_deviation_deg} degrees. Rstore its coordinates...")
                
                new_points[mask_out] = points[mask_out]
                changed = len(mask_out) < len(vid_ids)
            else:
                logging.info(f"   Found no vertices with deviation of its normal greater than {max_normal_deviation_deg} degrees.")
                
    return new_points, faces, changed
    
def mesh_remove_plane_at_coor(faces, points, \
    coordinates_to_remove = [None, None, None],
    #z = 0.0,
    do_remove_unused_points = True, log_level = 3):
    tmesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)
    points_change_dict = {}
    
    at_cs_p_mask = np.zeros(len(points), dtype=bool)
    for aidx in [0,1,2]:
        if (not coordinates_to_remove[aidx] is None):
            c = coordinates_to_remove[aidx]
            at_c_pids = np.where(points[:,aidx] == c)[0]
            at_c_ps_neihs_pids = tmesh.vertex_neighbors
            at_c_neihs =  np.where([np.all(points[at_c_ps_neihs_pids[pid],aidx]==c) for pid in at_c_pids])[0]
            at_c_pids_with_atzneighs = at_c_pids[at_c_neihs]
            at_cs_p_mask[at_c_pids_with_atzneighs] = True
    at_cs_pid = np.where(at_cs_p_mask)[0]

    faces2rem = np.unique(tmesh.vertex_faces[at_cs_pid])
    faces2rem = faces2rem[faces2rem>=0]
    new_faces = np.delete(faces, faces2rem, axis = 0)
    if do_remove_unused_points:
        new_tmesh = trimesh.Trimesh(vertices=points, faces=new_faces, process=True)
        changed = len(at_cs_pid) > 0 
        return new_tmesh.faces, new_tmesh.vertices, changed
    else:
        changed = len(faces2rem) > 0
        return new_faces, points, changed

def mesh_repair(faces, points, keep_verts=None, \
    check_close_points = True, check_duplicated_points = False, check_degenerated_faces = True, \
    check_overlapping_faces = True, check_boundary_dangling_faces = True, check_boundaryonly_faces = False, \
    check_alternating_faces = True, check_narrow_faces = True, \
    max_passes = 1000, short_edge_th = 0.0, max_adjecency_angle=120.0, \
    stop_when_no_degenerated_overlapping_alternating_faces = False, \
    do_return_merged_points_list = False, \
    do_remove_unused_points = True, \
    log_level = 3, dbg_dir = None):
    
    changed = False
    current_pass_changed = False
    try_next_pass = True
    prev_face_adjacency_sharp_face_vids = []
    prev_face_sharp_face_vids           = []
    depth = 0
    pass_id = 0
    org_points_num = len(points)
    if do_return_merged_points_list:
        total_merged_points = []
    if not keep_verts is None:
        keep_verts_mask = np.zeros(len(points), dtype=bool)
        keep_verts_mask[keep_verts] = True
    while try_next_pass:
        tmesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)
        #dbg_now = True
        #if dbg_now:          
        #    tmesh_outline_vids = tmesh.outline().referenced_vertices
        #    if len(tmesh_outline_vids)>0:
        #        tmesh_outline_verts = [e.points for e in tmesh.outline().entities]
        #        tmesh_outline_verts_PCs  = [trimesh.PointCloud(tmesh.vertices[vs]) for vs in tmesh_outline_verts]
        #        show_meshes([tmesh, *tmesh_outline_verts_PCs])
        #        _=1
        if not dbg_dir is None:
            tmesh.export(os.path.join(dbg_dir, f"_pass{pass_id}_in.stl"))
        already_merged_v_mask = np.zeros(len(points), dtype=bool)
        if log_level >= 1:
            logging.info(f"Pass {pass_id}")
        had_degenerated_faces = False
        had_overlapping_faces = False
        had_alternating_faces = False
        current_pass_changed = False
        if check_close_points:
            points_change_dict = {}
            short_edges = []
            unique_edges_short = np.where(tmesh.edges_unique_length < short_edge_th)
            short_edges = tmesh.edges_unique[unique_edges_short]
            unique_edges_short_len = tmesh.edges_unique_length[unique_edges_short]
            len_idxs = np.argsort(unique_edges_short_len)
            short_edges = short_edges[len_idxs]
            unique_edges_short_len = unique_edges_short_len[len_idxs]
            if len(short_edges) == 0:
                if log_level >= 1:
                    logging.warning(f" No short edges found, therefore, no close points to merge.")
            else:
                if log_level >= 1:
                    logging.warning(f" Found {len(short_edges)} short edges of lengths from {min(unique_edges_short_len):.3f}mm to {max(unique_edges_short_len):.3f}mm. Merge points for those edges.")

                if log_level >= 2:
                    for e, l in list(zip(short_edges, unique_edges_short_len)):
                        logging.info(f"  {e}: {l:.3f} mm!")
                for vidx0, vidx1 in short_edges:
                    v0, v1 = points[vidx0], points[vidx1]
                    if not keep_verts is None:
                        if already_merged_v_mask[vidx0] or already_merged_v_mask[vidx1]:
                            continue
                        v0_force_keep = keep_verts_mask[vidx0]
                        v1_force_keep = keep_verts_mask[vidx1]
                        if v0_force_keep and v1_force_keep:
                            if log_level >= 2:
                                logging.info(f"  Close points {vidx0} and {vidx1} ({v2s(v0)}, {v2s(v1)}) are in keep_verts. Keep the first one")
                            v1_force_keep = False
                            already_merged_v_mask[vidx0] = True
                            already_merged_v_mask[vidx1] = True
                            points[vidx0] = np.average([points[vidx0], points[vidx1]], axis = 0)
                            points[vidx1] = points[vidx0]
                        elif not v0_force_keep and not v1_force_keep:
                            
                            v0_force_keep = (v0[0] < v1[0]) or ((v0[0] == v1[0]) and (v0[1] < v1[1])) or ((v0[0] == v1[0]) and (v0[1] == v1[1]) and (v0[2] < v1[2]))
                            v1_force_keep = not v0_force_keep
                            already_merged_v_mask[vidx0] = True
                            already_merged_v_mask[vidx1] = True
                            points[vidx0] = np.average([points[vidx0], points[vidx1]], axis = 0)
                            points[vidx1] = points[vidx0]
                            if log_level >= 2:
                                logging.info(f"  Close points {vidx0} and {vidx1} ({v2s(v0)}, {v2s(v1)}). keep the {'1st' if v0_force_keep else '2nd'} one")
                            
                        else:
                            already_merged_v_mask[vidx1 if v0_force_keep else vidx0] = True
                            if log_level >= 2:
                                logging.info(f"  Close points {vidx0} and {vidx1} ({v2s(v0)}, {v2s(v1)}) {vidx0 if v0_force_keep else vidx1} in keep_verts so keep it")
                            
                    else:
                        v0_force_keep = (v0[0] < v1[0]) or ((v0[0] == v1[0]) and (v0[1] < v1[1])) or ((v0[0] == v1[0]) and (v0[1] == v1[1]) and (v0[2] < v1[2]))
                        v1_force_keep = not v0_force_keep
                        points[vidx0] = np.average([points[vidx0], points[vidx1]], axis = 0)
                        points[vidx1] = points[vidx0]
                        already_merged_v_mask[vidx0] = True
                        already_merged_v_mask[vidx1] = True

                    if log_level >= 2:
                        logging.info(f"  Merge close points {vidx0} and {vidx1} ({v2s(v0)}, {v2s(v1)})...")

                    if v0_force_keep or v1_force_keep:
                        if v0_force_keep:
                            changed_point_idx = vidx0
                            points_change_dict[vidx1] = vidx0
                        elif v1_force_keep:   
                            changed_point_idx = vidx1
                            points_change_dict[vidx0] = vidx1
                        
                        if log_level >= 2:
                            logging.info(f"  remaining point will be the {changed_point_idx} ({v2s(points[changed_point_idx])})...")
                if do_return_merged_points_list and len(points_change_dict) > 0:
                    total_merged_points.append(points_change_dict)

                if log_level >= 1:
                    logging.warning(f"  Replace ids of merged points in faces with the ids of the remaining ones.")
                for changed_pidx in points_change_dict.keys():
                    new_pidx = points_change_dict[changed_pidx]
                    if log_level >= 2:
                        logging.info(f"   Apply {changed_pidx} -> {new_pidx}")
                    face_vert_idx_to_change  = np.argwhere(faces==changed_pidx)
                    if log_level >= 2:
                        logging.info(f"    Change at {face_vert_idx_to_change}")
                    for fid, vid in face_vert_idx_to_change:
                        faces[fid][vid] = new_pidx
                    current_pass_changed = True
        
        if check_duplicated_points:
            #np.unique(points, axis=0, return_counts=True)
            unique_points, counts = np.unique(np.sort(points, axis = 1),axis=0, return_counts=True)
            duplicated_points_ids = np.argwhere(counts>1)
            if log_level >= 1:
                logging.warning(f" Found {len(duplicated_points_ids)} duplicated points. Do nothing with those.")
            if log_level >= 2:
                for duplicated_points_id in duplicated_points_ids:
                    duplicated_point = unique_points[duplicated_points_id]
                    count = counts[duplicated_points_id]
                    logging.warning(f" Point {duplicated_point} occurs {count} times in a mesh! ")
        if check_degenerated_faces:
            if current_pass_changed:
                tmesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)
            deg_f_idxs = np.where(np.all(tmesh.triangles_cross == 0, axis=1))[0]

            if len(deg_f_idxs) > 0:
                had_degenerated_faces = True
                if log_level >= 1:
                    logging.warning(f" Found {len(deg_f_idxs)} degenerated faces (a face with repeated vertex)! Removing those...")
                faces = np.delete(np.array(faces), deg_f_idxs, axis=0)
                current_pass_changed = True
                
        if check_overlapping_faces:
            if current_pass_changed:
                tmesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)
            faces_sorted = np.sort(faces, axis = 1)
            unique_faces, unique_face_id_2_first_face_id, counts = np.unique(faces_sorted, axis=0, return_index = True, return_counts=True)
            duplicated_faces_ids = np.where(counts>1)[0]
            face_ids_2_remove = []
            if len(duplicated_faces_ids) > 0:
                if log_level >= 1:
                    logging.warning(f" Found {len(duplicated_faces_ids)} duplicated faces to remove! Removing those...")
                for duplicated_faces_id in duplicated_faces_ids:
                    duplicated_face = unique_faces[duplicated_faces_id]
                    duplicated_face_candidate_ids, duplicated_face_candidate_count = np.unique(np.where(faces_sorted==duplicated_face)[0], return_counts=True) 
                    duplicated_face_ids = duplicated_face_candidate_ids[ duplicated_face_candidate_count==3]
                    count = counts[duplicated_faces_id]
                    if log_level >= 2:
                        logging.info(f"  Face with vertices {duplicated_face} occurs {count} times in a mesh! At {duplicated_face_ids}.")
                    #face_ids_2_remove.append(first_face_id)
                    if count%2 == 1:
                        dbg_now = False
                        if dbg_now:  
                            tmesh_double_outline_faces  = [trimesh.Trimesh(vertices=tmesh.vertices, faces=[tmesh.faces[fid]]) for fid in duplicated_face_ids]
                            show_meshes([tmesh, *tmesh_double_outline_faces])
                            _=1
                        duplicated_faces = faces[duplicated_face_ids]
                        max_arg_ids =  np.argmax(duplicated_faces, axis=1)
                        duplicated_faces_sorted = [[duplicated_faces[id][max_arg_ids[id]+0], duplicated_faces[id][(max_arg_ids[id]+1)%3], duplicated_faces[id][(max_arg_ids[id]+2)%3]] for id,_ in enumerate(duplicated_faces)]
                        d = {}
                        idx_to_remain = None
                        for i in range(len(duplicated_faces_sorted)):
                            o_duplicated_faces_sorted = [f for tid, f in enumerate(duplicated_faces_sorted) if tid!=i]
                            c_duplicated_faces_sorted = duplicated_faces_sorted[i]
                            s = np.sum([c_duplicated_faces_sorted == o for o in o_duplicated_faces_sorted])
                            d[i] = s
                            if s%2 == 1:
                                idx_to_remain = i
                        if not idx_to_remain is None:
                            duplicated_face_ids = [df for df_idx, df in enumerate(duplicated_face_ids) if df_idx!=idx_to_remain]
                    face_ids_2_remove.extend(duplicated_face_ids)
            if len(face_ids_2_remove) > 0:
                faces = np.delete(np.array(faces), face_ids_2_remove, axis=0)
                current_pass_changed = True
                had_overlapping_faces = True
        if check_boundaryonly_faces:  
            if current_pass_changed:
                tmesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)          
            tmesh_outline_vids = tmesh.outline().referenced_vertices
            if len(tmesh_outline_vids)>0:
                outline_entities = tmesh.outline().entities
                outline_entities = [oe for oe in outline_entities if len(oe.points)>2]
                if len(outline_entities)>0:
                    outline_entities_u = [np.unique(oe.points) for oe in outline_entities]
                    max_outline_len = max([len(op) for op in outline_entities_u])
                    tmesh_outline_verts = [np.pad(op, (0,max_outline_len-len(op)), 'constant', constant_values=(-1, -1)) for op in outline_entities_u]
                    outline_verts_unique, outline_verts_count = np.unique(tmesh_outline_verts, return_counts=True)
                    if outline_verts_unique[0]==-1:
                        outline_verts_unique = outline_verts_unique[1:]
                        outline_verts_count  = outline_verts_count [1:]
                    #if max(outline_verts_count)>1:
                    #    double_outline_vert_lids = np.where(outline_verts_count>1)[0]
                    #    double_outline_vert_vids = outline_verts_unique[double_outline_vert_lids]
                    if True:
                        double_outline_vert_vids = outline_verts_unique
                        double_outline_face_ids0 = [tmesh.vertex_faces[vid] for vid in double_outline_vert_vids]
                        double_outline_face_ids1 = np.unique(double_outline_face_ids0)
                        double_outline_face_ids2 = np.array([fid for fid in double_outline_face_ids1 if fid!=-1])
                        double_outline_face_ids_with_all_verts_in_outlines = [np.all([vid in outline_verts_unique for vid in tmesh.faces[fid]]) for fid in double_outline_face_ids2]
                        double_outline_face_ids = double_outline_face_ids2[double_outline_face_ids_with_all_verts_in_outlines]
                        
                        if len(double_outline_face_ids)>0:
                            if log_level >= 1:
                                logging.warning(f" Found {len(double_outline_face_ids)} double outline faces (a face that has a vertex that is a part of more than one outline)! Removing those...")
                            
                            dbg_now = False
                            if dbg_now:  
                                tmesh_double_outline_faces  = [trimesh.Trimesh(vertices=tmesh.vertices, faces=[tmesh.faces[fid]]) for fid in double_outline_face_ids]
                                show_meshes([tmesh, *tmesh_double_outline_faces])
                                _=1
                            
                            faces = np.delete(np.array(faces), double_outline_face_ids, axis=0)
                            current_pass_changed = True
                            
        if check_boundary_dangling_faces:
            #np.unique(points, axis=0, return_counts=True)
            unique_face_vids, vid_2_flatened_face_idx, counts = np.unique(np.array(faces), return_index = True, return_counts=True)
            vid_2_faces_idx = vid_2_flatened_face_idx//3
            boundary_face_idxs = vid_2_faces_idx[np.argwhere(counts==1)]
            #strang_face_idxs   = vid_2_faces_idx[np.argwhere(counts==2)]
            #if(len(strang_face_idxs) > 0):
            #    if log_level >= 1:
            #        logging.warning(f" Found {len(strang_face_idxs)} strange faces (a face with a vertex that occurs in only two faces)! Do not know what to do with this.")
            if(len(boundary_face_idxs) > 0):
                if log_level >= 1:
                    logging.warning(f" Found {len(boundary_face_idxs)} boundary faces (a face with a vertex that do not occurs in any other face)! Removing those...")
                faces = np.delete(np.array(faces), boundary_face_idxs, axis=0)
                current_pass_changed = True
            
        if (pass_id < max_passes):
            if check_alternating_faces or check_narrow_faces:
                if current_pass_changed:
                    if log_level >= 1:
                        logging.warning(f" Create new trimesh mesh...")
                    tmesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)
                if log_level >= 1:
                    logging.warning(f"  Fix normals...")
                tmesh.fix_normals() #if one triangle is flipped it can make all the procedure faulty

                if check_alternating_faces:
                    if log_level >= 1:
                        logging.warning(f" Find faces' adjacency statistics...")
                    face_adjacency_th_angle = max_adjecency_angle
                    face_adjacency_th_angle_rad = face_adjacency_th_angle * (2 * np.pi / 360.0)
                    face_adjacency_sharp_pair_id = np.unique(np.argwhere(tmesh.face_adjacency_angles > face_adjacency_th_angle_rad))
                    face_adjacency_sharp_id, face_adjacency_sharp_id_count = np.unique(tmesh.face_adjacency[face_adjacency_sharp_pair_id], return_counts=True)
                    face_adjacency_sharp_face_vids = np.unique(tmesh.faces[face_adjacency_sharp_id])
                else:
                    face_adjacency_sharp_face_vids = []

                if check_narrow_faces:
                    if log_level >= 1:
                        logging.warning(f" Find faces' sharpness statistics...")
                    face_th_angle = 10
                    face_th_angle_rad = face_th_angle * (2 * np.pi / 360.0)
                    la = np.any( (tmesh.face_angles <              face_th_angle_rad ), axis = 1)
                    ha = np.any( (tmesh.face_angles > (2 * np.pi - face_th_angle_rad)), axis = 1)
                    face_sharp_id = np.unique(np.argwhere(la | ha))
                    face_sharp_face_vids = np.unique(tmesh.faces[face_sharp_id])
                else:
                    face_sharp_face_vids = []
            
                #verts = find_neighbouring_verts(mesh, v)
                has_sharp_edge = len(face_adjacency_sharp_face_vids) > 0
                if has_sharp_edge:
                    had_alternating_faces = True
                    if log_level >= 3:
                        logging.warning(f" Found {len(face_adjacency_sharp_pair_id)} faces with adjecency angle above {face_adjacency_th_angle} deg! Smooth position of {len(face_adjacency_sharp_face_vids)} vertices.")
                has_sharp_face = len(face_sharp_face_vids) > 0
                if has_sharp_face:
                    if log_level >= 3:
                        logging.warning(f" Found {len(face_sharp_id)} faces with sharp angles below {face_th_angle} deg or above {180-face_th_angle} deg! Smooth position of {len(face_sharp_face_vids)} vertices.")
                
                if has_sharp_edge or has_sharp_face:
                    vids_to_smooth = np.unique([*face_adjacency_sharp_face_vids, *face_sharp_face_vids])
                    if log_level >= 2:
                        logging.warning(f"  Smooth position of {len(vids_to_smooth)} vertices.")
                    mesh_meshio = meshio.Mesh(points = tmesh.vertices, cells = [('triangle', tmesh.faces   )])
                    #mesh_meshio.write(tiss_dict[t]['remeshed_surf_pth'].replace('.stl', '_old.stl'))
                    if  (len(face_adjacency_sharp_face_vids)>0 and len(face_adjacency_sharp_face_vids) == len(prev_face_adjacency_sharp_face_vids)) or \
                        (len(face_sharp_face_vids          )>0 and len(face_sharp_face_vids          ) == len(prev_face_sharp_face_vids          )):
                        if depth < 8:
                            depth += 1
                            if log_level >= 1:
                                logging.warning(f"  Increase depth to {depth}.")
                    prev_face_adjacency_sharp_face_vids = copy.copy(face_adjacency_sharp_face_vids)
                    prev_face_sharp_face_vids           = copy.copy(face_sharp_face_vids          )
                    mesh_new = smooth_mesh_and_rewrite(mesh_meshio, vids_to_smooth, steps = 3, depth = depth, vids_blocked = keep_verts, do_increase_depth_when_no_change = (keep_verts is None))
                    #mesh_new.write(tiss_dict[t]['remeshed_surf_pth'].replace('.stl', '_new.stl'))
                    current_pass_changed = True
                    faces = mesh_new.cells_dict['triangle']
                    points = mesh_new.points
                
        try_next_pass = current_pass_changed
        if stop_when_no_degenerated_overlapping_alternating_faces:
            had_one_of_critical_face = had_degenerated_faces or had_overlapping_faces or had_alternating_faces
            if not had_one_of_critical_face:
                if log_level >= 1:
                    logging.info(f"  Had no degenerated, overlapping or alternating faces at this pass. Stop further repairing at pass {pass_id}.")
            try_next_pass = had_one_of_critical_face
        changed = changed or current_pass_changed
        pass_id += 1

    if do_remove_unused_points:
        tmesh   = trimesh.Trimesh(vertices=points, faces=faces, process=True)
        points  = tmesh.vertices
        faces   = tmesh.faces

    after_points_num = len(points)
    if after_points_num != org_points_num:
        logging.warning(f"  Number of points changed {org_points_num} -> {after_points_num}")
            
    if do_return_merged_points_list:
        return faces, points, pass_id, changed, total_merged_points
    else:
        return faces, points, pass_id, changed