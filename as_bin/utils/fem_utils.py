import os, sys
import numpy as nm
import copy
import logging
import random
from scipy.spatial import cKDTree
from sfepy.discrete.fem import Mesh
from sfepy.base.base import Struct, IndexedStruct
from sfepy.postprocess.utils_vtk import get_vtk_from_mesh,\
    get_vtk_by_group, get_vtk_surface, get_vtk_edges, write_vtk_to_file,\
    tetrahedralize_vtk_mesh

#-----------------------------------------------------------------------------------------
tid2tname = {
    0: "omega"  ,
    1: "bones"  ,
    2: "other"  ,
    3: "vessels",
    4: "skin"   ,
    }
tname2tid = {vd:kd for kd,vd in tid2tname.items()}
#-----------------------------------------------------------------------------------------
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / nm.linalg.norm(vector)
#-----------------------------------------------------------------------------------------

def create_region(self, name, select, kind='cell', parent=None, check_parents=True, extra_options=None, functions=None, add_to_regions=True, allow_empty=False):
    """
    Region factory constructor. Append the new region to
    self.regions list.
    """
        
    from sfepy.discrete.common.region import (Region, get_dependency_graph,
                                                sort_by_dependency, get_parents)
    from sfepy.discrete.parse_regions import create_bnf, visit_stack
    from sfepy.discrete.common.domain import region_leaf, region_op
    if check_parents:
        parents = get_parents(select)
        for p in parents:
            if p not in [region.name for region in self.regions]:
                msg = 'parent region %s of %s not found!' % (p, name)
                raise ValueError(msg)

    stack = self._region_stack
    try:
        self._bnf.parseString(select)
    except ParseException:
        logging.error(f'parsing failed: {select}')
        raise

    region = visit_stack(stack, region_op,
                            region_leaf(self, self.regions, select, functions))
    region.name = name
    region.definition = select
    region.set_kind(kind)
    region.finalize(allow_empty=allow_empty)
    region.parent = parent
    region.extra_options = extra_options
    region.update_shape()

    if add_to_regions:
        self.regions.append(region)

    return region


#---------------------------------------------------------------------------- 
def unifyVertsInVolumeAndSurfMeshes(volumeMesh, surfMeshes, tname2tid, matching_verts_max_dist = 1e-10):
    
    # tetras
    cells_4        = volumeMesh.get_conn('3_4')
    cells_groups_4 = nm.array(volumeMesh.cmesh.cell_groups, dtype= float)
    points         = volumeMesh.coors
    points_4       = copy.copy(points     )

    num_of_4_points   = len(points_4) 

    cells_3        =                     nm.array([])
    cells_3_groups =                     nm.array([])
    
    for tname in surfMeshes.keys():
        surf_data = surfMeshes[tname]
        if surf_data is None:
            continue
        surf_tid = tname2tid[tname]
    
        #lists of 3 nodes describing each element in a mesh:
        surf_cells       = surf_data.get_conn('2_3')
        surf_cell_groups = nm.array([surf_tid] * len(surf_cells), dtype=float)
        surf_points      = surf_data.coors

        surf_vids = nm.unique(surf_cells)

        kdtree = cKDTree(points_4)
        
        surf2omega_dists, surf2omega_vidx = kdtree.query(surf_points, distance_upper_bound=matching_verts_max_dist)
        surf_omega_common_vs_mask = surf2omega_dists <= matching_verts_max_dist
        surf_omega_common_vs_num = surf_omega_common_vs_mask.sum()
        assert(len(surf_points) == surf_omega_common_vs_num)

        surf_cells_in_omega = surf2omega_vidx[surf_cells]

        if(len(cells_3)==0):
            cells_3        = copy.copy(surf_cells_in_omega)
            cells_3_groups = copy.copy(surf_cell_groups   )
        else:
            cells_3        = nm.append(cells_3,        surf_cells_in_omega, axis=0)
            cells_3_groups = nm.append(cells_3_groups, surf_cell_groups   , axis=0)

        
    # sfepy volume Mesh
    cells_3_dict          = [cells_3       ]
    cells_3_groups_dict   = [cells_3_groups]
    triangles_mesh = Mesh.from_data(name="surfs_mesh", 
                                coors=points_4, 
                                ngroups=None, 
                                conns=cells_3_dict,#[connectivities], 
                                mat_ids=cells_3_groups_dict, 
                                descs=['2_3'])

    return triangles_mesh
    
#----------------------------------------------------------------------------
def get_cells_having_vertex_ids(cells, verts_ids, require_verts = 1):
    verts_ids_sorted = nm.sort(verts_ids)
    verts_ids_sorter = nm.array(range(len(verts_ids)))
    #indexes_l   = [nm.searchsorted(verts_ids, vol_conn, sorter=verts_ids_sorter) for vol_conn in cells] # where the indeces of a cell's vertices should be placed if you try to place them on the verts_ids list
    indexes_l   = nm.searchsorted(verts_ids, cells, sorter=verts_ids_sorter) # where the indeces of a cell's vertices should be placed if you try to place them on the verts_ids list
    verts_ids_ext = nm.append(verts_ids_sorted, [-1])
    cells_e = nm.array([verts_ids_ext[indexes] for indexes in indexes_l], dtype = cells.dtype)
    verts_per_cell = len(cells[0])
    if require_verts == 1:
        cells_having_voxel = nm.where(nm.any(cells==cells_e, axis=1))[0]
    elif require_verts == verts_per_cell:
        cells_having_voxel = nm.where(nm.all(cells==cells_e, axis=1))[0]
    else:
        cells_having_voxel = nm.where(nm.sum(cells==cells_e, axis=1)>=require_verts)[0]
    return cells_having_voxel
      
#----------------------------------------------------------------------------
def create_surf_from_mesh(input_mesh, tname, tname2tid, work_dir, dbg_files=False, out_fn_prefix = None):    
    ri = random.randint(0, 65200)
    vtkdata = get_vtk_from_mesh(input_mesh, None, f'postproc_{ri}')
    tids_present = nm.unique(input_mesh.cmesh.cell_groups)
    tid = tname2tid[tname]
    if tid == 0:
        matrix = get_vtk_by_group(vtkdata, min(tids_present), max(tids_present))
    elif tid in tids_present:
        matrix = get_vtk_by_group(vtkdata, tid, tid)
    else:
        matrix = None

    if not matrix is None:
        matrix_surf = get_vtk_surface(matrix)
        matrix_surf_tri = tetrahedralize_vtk_mesh(matrix_surf)
    
        if not out_fn_prefix is None:
            out_fn = f"{out_fn_prefix}.vtk"
        else:
            out_fn = f"_mat_{tname}_surface_tmp.vtk"
        out_pth = os.path.normpath(os.path.join(work_dir, out_fn))
        logging.info(f"  Save the result to {out_pth} VTK...")
        write_vtk_to_file(out_pth, matrix_surf_tri)
                

        try:
            surf = Mesh.from_file(out_pth)
        except:
            logging.info(f" Error reading {out_pth}. Delete this file.")
            surf = None

        if not dbg_files:
            os.remove(out_pth)

    else:
        logging.warning(f" {tname} not present in the scaled up mesh!")
        surf = None
    return surf
          
def create_tissue_outer_surf(input_mesh, tname, tname2tid, work_dir, dbg_files, cell_max_normal_to_XYplane_angle_deg = None):
    logging.info(f" Create tissue surf...")
    tissue_surf_org  = create_surf_from_mesh(input_mesh,  tname, tname2tid, work_dir, dbg_files)
    tissue_surf_org  = unifyVertsInVolumeAndSurfMeshes(input_mesh, {tname: tissue_surf_org }, tname2tid, matching_verts_max_dist = 1e-5)
    tissue_surf_tris  = tissue_surf_org.get_conn('2_3')

    logging.info(f" Create omega surf...")
    omega_surf_org = create_surf_from_mesh(input_mesh, "omega", tname2tid, work_dir, dbg_files)
    omega_surf_org = unifyVertsInVolumeAndSurfMeshes(input_mesh, {"omega":omega_surf_org}, tname2tid, matching_verts_max_dist = 1e-5)
    omega_surf_tris = omega_surf_org.get_conn('2_3')

    
    logging.info(f" get intersection of both surfs...")
    vert_ids_omega_surf = nm.unique( omega_surf_tris)

    tissue_tri_ids_omega = get_cells_having_vertex_ids(tissue_surf_tris, vert_ids_omega_surf, require_verts = 3)
    tissue_omega_surf_cells_3_dict          = [tissue_surf_tris[tissue_tri_ids_omega]       ]
    tissue_omega_surf_cells_3_groups_dict   = [nm.array([tname2tid['skin']] * len(tissue_tri_ids_omega), dtype=float)]
    
    logging.info(f" Actual creation of the tissue outer surf...")
    tissue_omega_surf_mesh = Mesh.from_data(name="surfs_mesh", 
                                coors   =input_mesh.coors, 
                                ngroups =None, 
                                conns   =tissue_omega_surf_cells_3_dict,
                                mat_ids =tissue_omega_surf_cells_3_groups_dict, 
                                descs   =['2_3'])
    
    if not cell_max_normal_to_XYplane_angle_deg is None:
        logging.info(f" Limit the cells to those whose angle between normal and XY plane is less than {cell_max_normal_to_XYplane_angle_deg} deg...")
        normals = get_triangles_normals(tissue_omega_surf_mesh)
        cell_max_normal_to_XYplane_angle_rad = cell_max_normal_to_XYplane_angle_deg / 360.0 * 2 * nm.pi
        cell_max_normal_to_XYplane_tan = nm.tan(cell_max_normal_to_XYplane_angle_rad)
        cell_mask = nm.array([abs(n[2]/nm.linalg.norm(n[0:1])) < cell_max_normal_to_XYplane_tan for n in normals])
        tissue_tri_ids_omega = tissue_tri_ids_omega[cell_mask]
        
        tissue_omega_surf_cells_3_dict          = [tissue_surf_tris[tissue_tri_ids_omega]       ]
        tissue_omega_surf_cells_3_groups_dict   = [nm.array([tname2tid['skin']] * len(tissue_tri_ids_omega), dtype=float)]
    
        logging.info(f" Actual creation of the tissue outer surf...")
        tissue_omega_surf_mesh = Mesh.from_data(name="surfs_mesh", 
                                    coors   =input_mesh.coors, 
                                    ngroups =None, 
                                    conns   =tissue_omega_surf_cells_3_dict,
                                    mat_ids =tissue_omega_surf_cells_3_groups_dict, 
                                    descs   =['2_3'])

    return tissue_omega_surf_mesh
#----------------------------------------------------------------------------
def get_triangles_normals(mesh_surfs, triCells_filter_ids = None):
    tri_verts_from_surfs       = mesh_surfs.get_conn('2_3')
    if triCells_filter_ids is None:
        tris_vids = tri_verts_from_surfs
    else:
        tris_vids = tri_verts_from_surfs[triCells_filter_ids]
    tris_coors = mesh_surfs.coors[tris_vids]
    tris_normals = [nm.cross(tri_coors[1]-tri_coors[0], tri_coors[2]-tri_coors[0]) for tri_coors in tris_coors]
    tris_normals = [ftn/nm.linalg.norm(ftn) for ftn in tris_normals]
    return tris_normals

def get_triangles_normal(mesh_surfs, triCells_filter_ids):
    tris_normals = get_triangles_normals(mesh_surfs, triCells_filter_ids)
    tris_normal = nm.mean(tris_normals, axis=0)
    return tris_normal

def get_triangles_aver_angles_between_normals(mesh_surfs, triCells_filter_ids=None, min_tris_at_vertex = 3, return_radians = False):
    tri_verts_from_surfs       = mesh_surfs.get_conn('2_3')
    if triCells_filter_ids is None:
        tris_vids = tri_verts_from_surfs
    else:
        tris_vids = tri_verts_from_surfs[triCells_filter_ids]

    vids, counts = nm.unique(tris_vids, return_counts=True)
    vids_filtered = vids[counts>min_tris_at_vertex]

    neighbouring_tris = [nm.where(tris_vids==v)[0] for v in vids_filtered]

    tris_normals = nm.array(get_triangles_normals(mesh_surfs, triCells_filter_ids))
    
    neighbouring_vertex_normals = nm.array([tris_normals[ntid_list] for ntid_list in neighbouring_tris],dtype=object)
    average_vertex_normal       = nm.array([unit_vector(nm.mean(nns_list, axis=0)) for nns_list in neighbouring_vertex_normals])
    
    average_vertices2neighs_normal_dots  = nm.array([nm.dot(nnl,an) for an, nnl in zip(average_vertex_normal, neighbouring_vertex_normals)],dtype=object)
    average_vertices2neighs_normal_dots  = nm.array([nm.clip(d,-1.0,1.0) for d in average_vertices2neighs_normal_dots], dtype=object)
    average_vertices2neighs_normal_angs  = nm.array([nm.arccos(an_dots) for an_dots in average_vertices2neighs_normal_dots],dtype=object)
    average_vertices2neighs_normal_ang   = nm.array([nm.mean(an_angs) for an_angs in average_vertices2neighs_normal_angs])
    #average_vertices2neighs_normal_ang_filterNan = average_vertices2neighs_normal_ang[~nm.isnan(average_vertices2neighs_normal_ang)]
    average_vertex2neighs_normal_ang = nm.mean(average_vertices2neighs_normal_ang)
    if not return_radians:
        average_vertex2neighs_normal_ang = average_vertex2neighs_normal_ang*360.0/(2*nm.pi)

    return average_vertex2neighs_normal_ang
        
def get_triangles_centers(mesh_surfs, triCells_filter_ids=None):
    tri_verts_from_surfs       = mesh_surfs.get_conn('2_3')
    if triCells_filter_ids is None:
        tris_vids = tri_verts_from_surfs
    else:
        tris_vids = tri_verts_from_surfs[triCells_filter_ids]
    tris_coors = mesh_surfs.coors[tris_vids]
    tris_centers = nm.mean(tris_coors, axis=1)
    return tris_centers

def get_triangles_center(mesh_surfs, triCells_filter_ids=None):
    tri_verts_from_surfs       = mesh_surfs.get_conn('2_3')
    if triCells_filter_ids is None:
        tris_vids = tri_verts_from_surfs
    else:
        tris_vids = tri_verts_from_surfs[triCells_filter_ids]
    tris_coors = mesh_surfs.coors[tris_vids]
    tris_crosss= [nm.cross(tri_coors[1]-tri_coors[0], tri_coors[2]-tri_coors[0]) for tri_coors in tris_coors]
    tris_area  = [nm.linalg.norm(tri_cross)/2 for tri_cross in tris_crosss]
    tris_centers = nm.mean(tris_coors, axis=1)
    tris_center  = nm.average(tris_centers, axis=0, weights=tris_area)
    return tris_center
        
def get_triangles_areas(mesh_surfs, triCells_filter_ids = None):
    tri_verts_from_surfs       = mesh_surfs.get_conn('2_3')
    if triCells_filter_ids is None:
        tris_vids = tri_verts_from_surfs
    else:
        tris_vids = tri_verts_from_surfs[triCells_filter_ids]
    tris_coors = mesh_surfs.coors[tris_vids]
    tris_crosss= [nm.cross(tri_coors[1]-tri_coors[0], tri_coors[2]-tri_coors[0]) for tri_coors in tris_coors]
    tris_areas  = [nm.linalg.norm(tri_cross)/2 for tri_cross in tris_crosss]
    return tris_areas

def get_triangles_aver_area(mesh_surfs, triCells_filter_ids = None):
    tris_areas = get_triangles_areas(mesh_surfs, triCells_filter_ids)
    tris_area_aver = nm.mean(tris_areas)
    return tris_area_aver

def get_triangles_total_area(mesh_surfs, triCells_filter_ids = None):
    tris_areas = get_triangles_areas(mesh_surfs, triCells_filter_ids)
    tris_area = nm.sum(tris_areas)
    return tris_area
    
#----------------------------------------------------------------------------
def lame_from_bulk_mu(bulk, mu):
    r"""
    Compute first Lamé parameter from bulk modulus and the second Lamé parameter (Shear modulus).

    .. math::
        \lambda = \gamma - {2 \over 3} \mu
    """
    #bulk = lam + 2.0 * mu / 3.0
    lam = bulk - 2.0 * mu / 3.0
    return lam

#----------------------------------------------------------------------------
def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):

    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    A     = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    B     = (x3*x3 * (y1 - y2) + x2*x2 * (y3 - y1) + x1*x1 * (y2 - y3)) / denom
    C     = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom

    xv = -B / (2*A)
    yv = C - B*B / (4*A)
    return xv, yv
#----------------------------------------------------------------------------
def find_volume_for_tissues(volume_mesh, tid2tname, do_fill_empty = False):
    tids = nm.unique(volume_mesh.cmesh.cell_groups)
    if not 0 in tids:
        tids = nm.array([0, *tids])
    
    if do_fill_empty:
        vol_dict = {v:0.0 for k,v in tid2tname.items()}
    else:
        vol_dict = {}

    for tid in tids:

        tname = tid2tname[tid]

        all_vol_tets                = volume_mesh.get_conn('3_4')
        if not tname == "omega":
            tet_is_curr_tiss = nm.where(volume_mesh.cmesh.cell_groups == tid)[0]
            tiss_tets = all_vol_tets[tet_is_curr_tiss]
            tiss_cell_groups = volume_mesh.cmesh.cell_groups[tet_is_curr_tiss]
        else:
            tiss_tets = all_vol_tets
            tiss_cell_groups = volume_mesh.cmesh.cell_groups

        tiss_cells_4_dict        = [tiss_tets]
        tiss_cells_4_dict_groups_dict = [nm.array(tiss_cell_groups, dtype=float)]
        tiss_off_mesh = Mesh.from_data(name="volume_mesh", 
                                    coors   =volume_mesh.coors, 
                                    ngroups =None, 
                                    conns   =tiss_cells_4_dict,
                                    mat_ids =tiss_cells_4_dict_groups_dict, 
                                    descs   =['3_4'])
        tiss_vols = tiss_off_mesh.cmesh.get_volumes(3)
        tiss_vol = nm.sum(tiss_vols)

        vol_dict[tname] = tiss_vol

    return vol_dict
#----------------------------------------------------------------------------
def find_dist_scale_optimal(surf_mesh, volume_mesh, org_dist, initial_scales, min_scale_multiplicative_difference, \
    tname2tid, work_dir, \
    used_metric = "skin_area", \
    dbg_files = False, mesh_use_meters = False, header_str = "    "):
    
    if used_metric == "skin_area":
        j = "mm^2/mm^3"
        logging.info(f"{header_str}Optimize for {used_metric} in {j}")
    elif used_metric == "skin_angle":
        j = "deg"
        logging.info(f"{header_str}Optimize for {used_metric} - mean angle between skin faces")
    else:
        logging.error(f"{header_str}Optimize for unknown {used_metric}!")
        sys.exit(1)

    process_dict = {}
    process_dict['out_raw']    = {}
    process_dict['out']        = {}
    process_dict[used_metric]  = {}
    process_dict[used_metric+"_raw"]  = {}
    process_dict['scales']     = {}
    new_scales = initial_scales
    prev_metrics_sorted = -1
    area_in_mm2_factor = 1000*1000 if mesh_use_meters else 1
    def scale_to_key(val):
        return f"{val:.3f}"
    while len(new_scales) != 0:
        logging.info(f"{header_str}Try new scales {new_scales}...")
        for scale in new_scales:
            logging.info(f"{header_str} Scale {scale:.3f}...")
            scale_key = scale_to_key(scale)
            process_dict['out_raw'][scale_key] = org_dist * scale
            process_dict['scales' ][scale_key] = scale

            #out = state.create_output_dict(extend=extend)
            output_data_Struct = Struct()
            output_data_Struct.update({'name':'output_data', 
                                        'mode':'vertex', 
                                        'data':process_dict['out_raw'][scale_key], 
                                        'var_name':'u', 
                                        'dofs': ['u.0', 'u.1', 'u.2']})
                           
            process_dict['out'][scale_key] = output_data_Struct
    
            #----------------------------------------------------------------------------
            out_fn = f"_skin_surf_offseted_{scale}.vtk"
            out_pth = os.path.normpath(os.path.join(work_dir, out_fn))
            skin_range_surf_tris                = surf_mesh.get_conn('2_3')
            skin_range_surf_cells_3_dict        = [skin_range_surf_tris]
            skin_range_surf_cells_3_groups_dict = [nm.array([tname2tid['skin']] * len(skin_range_surf_tris), dtype=float)]
            shifted_points  = copy.deepcopy(surf_mesh.coors)
            shifted_points += process_dict['out_raw'][scale_key]
            skin_range_surf_off_mesh = Mesh.from_data(name="surfs_mesh", 
                                        coors   =shifted_points, 
                                        ngroups =None, 
                                        conns   =skin_range_surf_cells_3_dict,
                                        mat_ids =skin_range_surf_cells_3_groups_dict, 
                                        descs   =['2_3'])
            
            omega_range_vol_tets                = volume_mesh.get_conn('3_4')
            omega_range_vol_cells_4_dict        = [omega_range_vol_tets]
            omega_range_vol_cells_4_dict_groups_dict = [nm.array([0] * len(omega_range_vol_tets), dtype=float)]
            omega_range_vol_off_mesh = Mesh.from_data(name="volume_mesh", 
                                        coors   =shifted_points, 
                                        ngroups =None, 
                                        conns   =omega_range_vol_cells_4_dict,
                                        mat_ids =omega_range_vol_cells_4_dict_groups_dict, 
                                        descs   =['3_4'])
            omega_vols = omega_range_vol_off_mesh.cmesh.get_volumes(3)
            omega_vol = nm.sum(omega_vols)

            if dbg_files:
                logging.info(f"{header_str}  Shift the region skin surface mesh and save it to {out_pth}...")
                skin_range_surf_off_mesh.write(out_pth, io='auto', out=None)
                
            if used_metric == 'skin_area':
                skin_range_surf_off_tris_metric_raw = get_triangles_total_area(skin_range_surf_off_mesh)#*area_in_mm2_factor
                skin_range_surf_off_tris_metric = skin_range_surf_off_tris_metric_raw/omega_vol#*area_in_mm2_factor
            elif used_metric == 'skin_angle':
                skin_range_surf_off_tris_metric_raw = get_triangles_aver_angles_between_normals(skin_range_surf_off_mesh)
                skin_range_surf_off_tris_metric = skin_range_surf_off_tris_metric_raw
            logging.info(f"{header_str}  Calculated {used_metric} of the surface mesh {skin_range_surf_off_tris_metric} {j}")
            process_dict[used_metric+"_raw"][scale_key] = skin_range_surf_off_tris_metric_raw
            process_dict[used_metric       ][scale_key] = skin_range_surf_off_tris_metric
        #----------------------------------------------------------------------------
        # find 3 scales with the smallest area
        _metrics  = nm.array(list(process_dict[used_metric].values()))
        _scales = nm.array(list(process_dict['scales'   ].values()))

        ascending_area_idxs = _metrics.argsort()
        scales_sorted = _scales[ascending_area_idxs]
        metrics_sorted  = _metrics [ascending_area_idxs]

        new_scales = []
        if len(ascending_area_idxs) >= 3:
            ns, na = \
            calc_parabola_vertex(scales_sorted[0], metrics_sorted[0], \
                                 scales_sorted[1], metrics_sorted[1], \
                                 scales_sorted[2], metrics_sorted[2] )

            logging.info(f"{header_str} Candidate for a new scale = {ns:.3f} with expected metric = {na} {j} where actual smallest metric = {metrics_sorted[0]} {j} for scale = {scales_sorted[0]:.3f}")

            curr_metrics_sorted = nm.sum(metrics_sorted[0:2])/3.0
            scale_dif_to_prev = nm.min([ abs(ns-ps) for ps in process_dict['scales'].values()])
            scale_part_to_prev = abs(scale_dif_to_prev / ns) if ns != 0 else 0.0 # procentowa zmiana w stosunku do poprzedniej najblizszej wartosci skali
            has_a_new_scale = (scale_dif_to_prev > min_scale_multiplicative_difference)
            has_a_new_data_points = (curr_metrics_sorted != prev_metrics_sorted)
            prev_metrics_sorted = curr_metrics_sorted
            if has_a_new_data_points and has_a_new_scale:
                max_prev_scale = max(process_dict['scales'].values())
                if ns > (2 * max_prev_scale):
                    new_scales = [(max_prev_scale+ns)/2.0, ns * 1.5]
                else:
                    new_scales = [ns]
                
        if len(new_scales) == 0:
            if len(ascending_area_idxs) < 3:
                logging.info(f"{header_str}End scale search because there is not enough data points to estimate a scale candidate")
            elif not has_a_new_data_points:
                logging.info(f"{header_str}End scale search because the best previous candidateds did not change since last try")
            elif not has_a_new_scale:
                logging.info(f"{header_str}End scale search because scale_dif_to_prev={scale_dif_to_prev:.3f})<{min_scale_multiplicative_difference}")
            process_dict["best_scale"]           = scales_sorted [0]
            process_dict["best_scale_key"]       = scale_to_key(scales_sorted[0])
            process_dict[f"best_{used_metric}"]  = metrics_sorted[0]
            
    return process_dict
    
#----------------------------------------------------------------------------
def find_dist_scale_volume(volume_mesh, org_dist, initial_scales, \
    max_volume_ratio_difference, \
    tname2tid, work_dir, \
    required_volume_ratios = [1.0], \
    dbg_files = False, mesh_use_meters = False, header_str = "    "):
    used_metric = "volume_ratio"
    process_dict = {}
    process_dict['out_raw']    = {}
    process_dict['out']        = {}
    process_dict[used_metric]  = {}
    process_dict['scales']     = {}
    new_scales = initial_scales
    prev_metrics_sorted = -1
    input_vols = volume_mesh.cmesh.get_volumes(3)
    input_vol = nm.sum(input_vols)
    def scale_to_key(val):
        return f"{val:.3f}"
    while len(new_scales) != 0:
        logging.info(f"{header_str}Try new scales {new_scales}...")
        for scale in new_scales:
            logging.info(f"{header_str} Scale {scale:.3f}...")
            scale_key = scale_to_key(scale)
            process_dict['out_raw'][scale_key] = org_dist * scale
            process_dict['scales' ][scale_key] = scale

            #out = state.create_output_dict(extend=extend)
            output_data_Struct = Struct()
            output_data_Struct.update({'name':'output_data', 
                                        'mode':'vertex', 
                                        'data':process_dict['out_raw'][scale_key], 
                                        'var_name':'u', 
                                        'dofs': ['u.0', 'u.1', 'u.2']})
                           
            process_dict['out'][scale_key] = output_data_Struct
    
            #----------------------------------------------------------------------------
            out_fn = f"_mesh_offseted_{scale}.vtk"
            out_pth = os.path.normpath(os.path.join(work_dir, out_fn))
            shifted_points  = copy.deepcopy(volume_mesh.coors)
            shifted_points += process_dict['out_raw'][scale_key]
            
            omega_range_vol_tets                = volume_mesh.get_conn('3_4')
            omega_range_vol_cells_4_dict        = [omega_range_vol_tets]
            omega_range_vol_cells_4_dict_groups_dict = [nm.array(volume_mesh.cmesh.cell_groups, dtype=float)]
            omega_range_vol_off_mesh = Mesh.from_data(name="volume_mesh", 
                                        coors   =shifted_points, 
                                        ngroups =None, 
                                        conns   =omega_range_vol_cells_4_dict,
                                        mat_ids =omega_range_vol_cells_4_dict_groups_dict, 
                                        descs   =['3_4'])
            omega_vols = omega_range_vol_off_mesh.cmesh.get_volumes(3)
            omega_vol = nm.sum(omega_vols)
            current_volume_ratio = omega_vol/input_vol

            if dbg_files:
                logging.info(f"{header_str}  Shift vertices of the mesh and save it to {out_pth}...")
                omega_range_vol_off_mesh.write(out_pth, io='auto', out=None)
            

            logging.info(f"{header_str}  Calculated {used_metric} of the mesh {current_volume_ratio}")
            process_dict[used_metric       ][scale_key] = current_volume_ratio
        #----------------------------------------------------------------------------
        # find 3 scales with the smallest area
        _volume_ratios = nm.array(list(process_dict[used_metric].values()))
        _scales        = nm.array(list(process_dict['scales'   ].values()))

        new_scales = []
        for required_volume_ratio in required_volume_ratios:
            process_dict[required_volume_ratio] = {}
            ascending_vol_idxs = _volume_ratios.argsort()
            scales_sorted   = _scales        [ascending_vol_idxs]
            metrics_sorted  = _volume_ratios [ascending_vol_idxs]

            new_scale_idx = nm.searchsorted(metrics_sorted, required_volume_ratio)
            if new_scale_idx == 0:
                v0,v1 =  metrics_sorted[new_scale_idx  ], metrics_sorted[new_scale_idx+1]
                s0,s1 =  scales_sorted [new_scale_idx  ], scales_sorted [new_scale_idx+1]   
            elif new_scale_idx==len(metrics_sorted):
                v0,v1 =  metrics_sorted[new_scale_idx-2], metrics_sorted[new_scale_idx-1]
                s0,s1 =  scales_sorted [new_scale_idx-2], scales_sorted [new_scale_idx-1]  
            else:
                v0,v1 =  metrics_sorted[new_scale_idx-1], metrics_sorted[new_scale_idx  ]
                s0,s1 =  scales_sorted [new_scale_idx-1], scales_sorted [new_scale_idx  ] 

            v0_is_good = (abs(v0-required_volume_ratio) <= max_volume_ratio_difference) 
            v1_is_good = (abs(v1-required_volume_ratio) <= max_volume_ratio_difference)
            if v0_is_good or v1_is_good:
                # already found
                if v0_is_good:
                    scale_key = scale_to_key(s0)
                    process_dict[required_volume_ratio]["found_scale"         ] =        s0
                    process_dict[required_volume_ratio][f"found_{used_metric}"] =        v0
                else: #if v1_is_good
                    scale_key = scale_to_key(s1)
                    process_dict[required_volume_ratio]["found_scale"         ] =        s1
                    process_dict[required_volume_ratio][f"found_{used_metric}"] =        v1
                process_dict[required_volume_ratio]["found_scale_key"     ] = scale_key
                process_dict[required_volume_ratio][f"out_raw"            ] = process_dict['out_raw'][scale_key]
                process_dict[required_volume_ratio][f"out"                ] = process_dict['out'    ][scale_key]
                continue
            else:
                dv = v1-v0
                ds = s1-s0
                dvm = (required_volume_ratio - v0) / dv
                ns = s0 + ds * dvm       
                logging.info(f"{header_str} Candidate for a new scale = {ns:.3f} with expected metric = {required_volume_ratio} where actual closest metrics = {v0} and {v1} for scales = {s0:.3f} and {s1:.3f}")

                max_prev_scale = max(process_dict['scales'].values())
                if ns > (2 * max_prev_scale):
                    new_scales.extend([(max_prev_scale+ns)/2.0, ns * 1.5])
                else:
                    new_scale_key = scale_to_key(ns)
                    if new_scale_key in process_dict['scales'   ].keys():
                        logging.info(f"{header_str}  {ns:.3f} already tested! Skip further search.")

                        _volume_ratios_dif = nm.abs(nm.array(list(process_dict[used_metric].values())) - required_volume_ratio)
                        _volume_ratios     = nm.array(list(process_dict[used_metric].values()))
                        _scales            = nm.array(list(process_dict['scales'   ].values()))
                        ascending_vol_idxs = _volume_ratios_dif.argsort()
                        scales_sorted   = _scales        [ascending_vol_idxs]
                        metrics_sorted  = _volume_ratios [ascending_vol_idxs]
                        closest_scale   = scales_sorted [0]
                        closest_metrics = metrics_sorted[0]

                        scale_key = scale_to_key(closest_scale)
                        process_dict[required_volume_ratio]["found_scale"         ] =        closest_scale
                        process_dict[required_volume_ratio][f"found_{used_metric}"] =        closest_metrics
                        process_dict[required_volume_ratio]["found_scale_key"     ] = scale_key
                        process_dict[required_volume_ratio][f"out_raw"            ] = process_dict['out_raw'][scale_key]
                        process_dict[required_volume_ratio][f"out"                ] = process_dict['out'    ][scale_key]
                        continue
                    else:
                        new_scales.append(ns)
                            
    return process_dict