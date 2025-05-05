#!/usr/bin/env python
# coding: utf-8

import sys, os
import collections.abc
from PIL import Image, ImageDraw
import numpy as np
import copy
import logging
from skimage import data, filters, measure
from scipy.ndimage.measurements import center_of_mass
#from skimage.segmentation import flood, flood_fill
from skimage.draw import line, polygon, polygon_perimeter #polygon2mask - dopiero od scikit-image=0.16.2
import skimage.morphology
from numbers import Number


#-----------------------------------------------------------------------------------------
#curr_script_path = os.path.dirname(os.path.abspath(__file__))
#flexnet_path = os.path.normpath(os.path.join(curr_script_path, ".."))
#flexnet_host_path = os.path.normpath(os.path.join(flexnet_path, ".."))
#sys.path.append(flexnet_host_path)
#-----------------------------------------------------------------------------------------
from v_utils.v_contour import v_contour
from v_utils.v_json import jsonUpdate, jsonDumpSafe
#-----------------------------------------------------------------------------------------

class v_polygons(collections.abc.MutableMapping):
    """A dictionary that applies an arbitrary key-altering
       function before accessing the keys"""

    def __init__(self):
        self.store = dict()
        self.store["polygons"]  = []
        self.store["box"]       = []
        
    @staticmethod
    def from_contours(outer_contour, hole_contours_list = []):

        if not (type(outer_contour) is v_contour):
            logging.error("outer_contour is not of type v_contour!")
            sys.exit(20)

        for hole_contour in hole_contours_list:
            if not (type(hole_contour) is v_contour):
                logging.error("hole_contour is not of type v_contour!")
                sys.exit(21)

        new_polygons = v_polygons()
        self.add_polygon_from_contours(copy.deepcopy(outer_contour), holes_contours=hole_contours_list)

        return new_polygons
    
    @staticmethod
    def from_dict(src_dict):
        if not (type(src_dict) is dict):
            logging.error("src_dict is not of type dict!")
            sys.exit(22)
        new_polygons = v_polygons()
        new_polygons.update(src_dict)
        for polygon_id, polygon in enumerate(new_polygons["polygons"]):
            if("outer" in polygon.keys()):
                polygon["outer"] = v_contour(polygon["outer"])
            else:
                polygon["outer"] = v_contour()
            if("inners" in polygon.keys()):
                for hole_id, hole in enumerate(polygon["inners"]):
                    polygon["inners"][hole_id] = v_contour(hole)
            else:
                polygon["inners"] = []
        return new_polygons
    
    @staticmethod
    def from_image(src_img, background_val = None, float_out = False):
        if not(Image.isImageType(src_img)):
            logging.error("src_img is not of type Image!")
            sys.exit(23)
        new_polygons = v_polygons()
        new_polygons._mask_image_to_polygons(src_img, background_val = background_val, float_out = float_out)
        return new_polygons
    
    @staticmethod
    def from_ndarray(src_array, float_out = False):
        if not(type(src_array) is np.ndarray):
            logging.error("src_array is not of type ndarray!")
            sys.exit(24)
        new_polygons = v_polygons()
        new_polygons._mask_ndarray_to_polygons(src_array, float_out)
        return new_polygons
    
    def convert_to_float_polys(self, save_int_contours =False):
        
        new_float_polygons = v_polygons()
        for poly_dict in self['polygons']:

            poly = v_polygons()
            poly['polygons'].append(poly_dict)
            poly.update_box()
            src_np_mask = poly.as_numpy_mask(fill = True) 
            new_float_polygon = v_polygons()
            new_float_polygon._mask_ndarray_to_polygons(src_np_mask, float_out = True)

            if(save_int_contours):
                for pid, polygon_org in enumerate(poly["polygons"]):
                    if(pid > 0):
                        logging.error("expected a single polygon only!")
                        sys.exit(-10)
                    polygon_org = poly["polygons"][pid]
                    
                    new_float_polygon["polygons"][-1]["outer"]["int_path"] = polygon_org["outer"]["path"]
                    new_float_polygon["polygons"][-1]["outer"]["int_box" ] = polygon_org["outer"]["box" ]
                    for hole_id in range(len(polygon_org["inners"])):
                        if(hole_id < len(new_float_polygon["polygons"][-1]["inners"])):
                            new_float_polygon["polygons"][-1]["inners"][hole_id]["int_path"] = polygon_org["inners"][hole_id]["path"]
                            new_float_polygon["polygons"][-1]["inners"][hole_id]["int_box" ] = polygon_org["inners"][hole_id]["box" ]

            new_float_polygons["polygons"].extend(new_float_polygon["polygons"])

        new_float_polygons.update_box()

        return new_float_polygons

    @staticmethod
    def from_polygons_borders(src_polygons, dilation_radius = 0.75):
        if not (type(src_polygons) is v_polygons):
            logging.error("src is not of type as_polygon!")
            sys.exit(25)

        src_copy = copy.deepcopy(src_polygons)
        org_box = src_copy.get_box()
        if(len(org_box) == 0):
            border_polygons = v_polygons()
        else:
            (org_x1, org_y1, org_x2, org_y2) = org_box

            dilation_radius_ceil = np.int16(np.ceil(dilation_radius))

            dx = dilation_radius_ceil - org_x1
            dy = dilation_radius_ceil - org_y1

            w = org_x2 - org_x1 + 1 + 2*dilation_radius_ceil 
            h = org_y2 - org_y1 + 1 + 2*dilation_radius_ceil  

            src_copy.move2point((dx, dy))

            #borders_1p_np = src_copy.as_numpy_mask (fill = False, w=w, h=h, val = 255, masks_merge_type = 'or')
            #borders_wp_np = skimage.morphology.dilation(borders_1p_np, v_polygons.disk_fr(float_radius=dilation_radius))

            polygon_np = src_copy.as_numpy_mask (fill = True, w=w, h=h, val = 255, masks_merge_type = 'or')
            disc = v_polygons.disk_fr(float_radius=dilation_radius)
            polygon_dil_np = skimage.morphology.dilation(polygon_np, disc)
            polygon_ero_np = skimage.morphology.erosion (polygon_np, disc)
            borders_wp_np = np.where((polygon_dil_np != 0) & (polygon_ero_np == 0), np.uint8(255), np.uint8(0))

            border_polygons = v_polygons.from_ndarray(borders_wp_np)
            
            border_polygons.move2point((-dx, -dy))
    
        return border_polygons
    
    @staticmethod
    def disk_fr(float_radius, dtype=np.uint8):
        """Generates a flat, disk-shaped structuring element.

        A pixel is within the neighborhood if the euclidean distance between
        it and the origin is no greater than radius.

        Parameters
        ----------
        radius : float
            The radius of the disk-shaped structuring element.

        Other Parameters
        ----------------
        dtype : data-type
            The data type of the structuring element.

        Returns
        -------
        selem : ndarray
            The structuring element where elements of the neighborhood
            are 1 and 0 otherwise.
        """
        L = np.arange(-np.ceil(float_radius-0.5), np.ceil(float_radius-0.5) + 1)
        X, Y = np.meshgrid(L, L)
        return np.array((X ** 2 + Y ** 2) <= float_radius ** 2, dtype=dtype)

    @staticmethod
    def group_polygons(ungrouped_polygons_list):
        
        grouped_polygons = v_polygons()
        for polygon in ungrouped_polygons_list:
            grouped_polygons["polygons"].append(polygon)
            
        grouped_polygons.update_box()
            
        return grouped_polygons
        
    def ungroup_polygons(self):
        
        ungrouped_polygons_list = []
        for polygon in self.store["polygons"]:
            ungrouped_polygons_list.append(v_polygons())
            ungrouped_polygons_list[-1]["polygons"].append(polygon)
            
        for ungrouped_polygons in ungrouped_polygons_list:
            ungrouped_polygons.update_box()
            
        return ungrouped_polygons_list

    def regroup_polygons_by_key(self, key):
    
        if(len(self) < 1):
            return self
        elif(not (key in self["polygons"][0].keys())):
            logging.error("Key \"{}\" not found in polygon dictionary keys ({})".format(key, self["polygons"][0].keys()))
            return None
            
        regrouped_polygons_list = [v_polygons()]
        key_val = self["polygons"][0][key]
        regrouped_polygons_list[-1]["polygons"].append(self["polygons"][0])
        for polygon in self.store["polygons"][1:]:
            new_key_val = polygon[key]
            if(new_key_val != key_val):
                regrouped_polygons_list.append(v_polygons())
                key_val = new_key_val
            regrouped_polygons_list[-1]["polygons"].append(polygon)
            
        for regrouped_polygons in regrouped_polygons_list:
            regrouped_polygons.update_box()
            
        return regrouped_polygons_list
                
    
    def __getitem__(self, key):
        return self.store[self.__keytransform__(key)]

    def __setitem__(self, key, value):
        self.store[self.__keytransform__(key)] = value

    def __delitem__(self, key):
        del self.store[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __keytransform__(self, key):
        return key
        
    def __str__(self):
        return str(self.store)
        
    def __repr__(self): 
        return self.__str__()
        
    def to_indendent_str(self, indent = 4):
        ident_str = " " * indent
        intendent_str = "{}num of polygons: {}\n".format(ident_str, len(self["polygons"]))
        for polygon_id, polygon in enumerate(self["polygons"]):
            intendent_str += "{}{}:\n".format(ident_str*2, polygon_id)
            for key in list(polygon):
                intendent_str += "{}{}:{}\n".format(ident_str*3, key, polygon[key])
        return intendent_str
    
    def add_polygon_from_paths(self, outer_path, holes_paths = [], polygon_name = None):

        if not (type(outer_path) is list or type(outer_path) is np.ndarray):
            logging.error(" Unexpected type of data.")
            sys.exit(26)
            
        outer_v_contour = v_contour(outer_path)
        holes_v_contours = [v_contour(x) for x in holes_paths]

        self.add_polygon_from_contours(outer_contour=outer_v_contour, holes_contours=holes_v_contours, polygon_name=polygon_name)

    def add_polygon_from_contours(self, outer_contour, holes_contours = [], polygon_name = None):

        if not (type(outer_contour) is v_contour):
            logging.error(" Unexpected type of data.")
            sys.exit(27)

        if(polygon_name == None):
            polygon_name = "{:02}".format(len(self.store["polygons"]))

        new_polygon = {
               "outer": outer_contour,
               "inners": [],
               "id"     : polygon_name
           }

        self.store["polygons"].append(new_polygon)

        for hole_contour in holes_contours:
            if not (type(hole_contour) is v_contour):
                logging.error(" Unexpected type of data.")
                sys.exit(28)

            new_polygon["inners"].append(hole_contour)
        
        self.update_box()
        
    def remove_inners(self):
        for polygon in self.store["polygons"]:
            polygon["inners"] = []


    def fill_mass_centers_for_outer_contours(self, overwrite = False):
        
        for polygon in  self.store["polygons"]:
            do_calc = overwrite or (len(polygon["outer"]["mass_center"]) == 0)
            if(do_calc):
                polygon["outer"].update_mass_center()

    def fill_mass_centers2(self):
        
        for pid, polygon in  enumerate(self['polygons']):
            
            (area_out, centroid_out, is_clockwise) = _area_centroid_cc(polygon['outer']['path'])
            polygon['outer']["centroid"] = centroid_out
            polygon['outer']["area"    ] = area_out
            poly_com = centroid_out
            poly_m   = area_out
            for inpoly in polygon['inners']:
                (area_in, centroid_in, is_clockwise) = _area_centroid_cc(inpoly['path'])
                inpoly["centroid"] = centroid_in
                inpoly["area"    ] = area_in
                poly_com = (np.array(poly_com)*poly_m - np.array(centroid_in)*area_in) / (poly_m-area_in)
                poly_m   = (poly_m-area_in)
                        
            polygon['com']   = poly_com
            polygon['m']     = poly_m

        ms      = np.array([p[  'm'] for p in self["polygons"]])
        coms    = np.array([p['com'] for p in self["polygons"]])
        self[  'm'] = sum(ms)
        if self[  'm'] != 0:
            self['com'] = sum(coms * ms[:,np.newaxis]) / sum(ms)
        else:
            self['com'] = 0

    def fill_mass_centers(self):
        my_ung_polygons = self.ungroup_polygons()
        for pid, polygon in  enumerate(my_ung_polygons):
            box = np.array(polygon["box"])
            box_pos = box[0:2]
            box_size = box[2:4] - box_pos
            cropped_poly = v_polygons()
            polygon.crop_2box(box, out=cropped_poly)
            poly_np = cropped_poly.as_numpy_mask(fill = True, val = 1)
            com = center_of_mass(poly_np)[::-1] + box_pos

            m = poly_np.sum()
            
            self["polygons"][pid]['com']   = com
            self["polygons"][pid]['m']     = m

        ms      = np.array([p[  'm'] for p in self["polygons"]])
        coms    = np.array([p['com'] for p in self["polygons"]])
        self[  'm'] = sum(ms)
        if self[  'm'] != 0:
            self['com'] = sum(coms * ms[:,np.newaxis]) / sum(ms)
        else:
            self['com'] = 0

    def update_box(self):
        
        self.store["box"        ] = []

        if(len(self["polygons"]) > 0):

            min_box_coords = len(min(self.store["polygons"], key=lambda x: len(x["outer"]["box"]))["outer"]["box"])

            if(min_box_coords == 4):
                self.store["box"        ].append(min(self.store["polygons"], key=lambda x: x["outer"]["box"][0])["outer"]["box"][0])
                self.store["box"        ].append(min(self.store["polygons"], key=lambda x: x["outer"]["box"][1])["outer"]["box"][1])
                self.store["box"        ].append(max(self.store["polygons"], key=lambda x: x["outer"]["box"][2])["outer"]["box"][2])
                self.store["box"        ].append(max(self.store["polygons"], key=lambda x: x["outer"]["box"][3])["outer"]["box"][3])
            else:
                raise ValueError('Could not calculate box for contours. Not all contours have boxes specified.')
            #logging.info(self.store["box"        ])

    def get_box(self):
        return self.store["box"]

    def get_boxes(self):
        boxes = []
        for polygon in self.store["polygons"]:
            boxes.append(polygon["outer"].get_box())
        return boxes

    def crop_2box(self, box, out=None):
        point = box[0:2]
        size = [box[2]-box[0]+1, box[3]-box[1]+1]
        if not out is None:
            self.copy(out)
            out.crop(point, size)
        else:
            self.crop(point, size)

    def copy(self, out=None):
        create_new = (out is None)
        if create_new:
            out = v_polygons()
        out.store = copy.deepcopy(self.store)
        if create_new:
            return out

    def crop(self, point, size):
        for polygon in self.store["polygons"]:
            polygon["outer"].crop(point, size)
            for hole_contour in polygon["inners"]:
                hole_contour.crop(point, size)
        self.update_box()
        
    def crop_around_own_box_center(self, size):
        box_center = [(self.store["box"][0] + self.store["box"][2])/2, (self.store["box"][1] + self.store["box"][3])/2]
        crop_start = np.round(box_center - size/2)
        self.crop(crop_start, size)
        
    def move2point(self, point):
        for polygon in self.store["polygons"]:
            polygon["outer"].move2point(point)
            for hole_contour in polygon["inners"]:
                hole_contour.move2point(point)

        if(len(self.store["box"])>0):
            self.store["box"][0] += point[0]
            self.store["box"][1] += point[1]
            self.store["box"][2] += point[0]
            self.store["box"][3] += point[1]
            
    def scale(self, sx, sy):
        for polygon in self.store["polygons"]:
            polygon["outer"].scale(sx, sy)
            for hole_contour in polygon["inners"]:
                hole_contour.scale(sx, sy)

        if(len(self.store["box"])>0):
            self.store["box"][0] *= sx
            self.store["box"][1] *= sy
            self.store["box"][2] *= sx
            self.store["box"][3] *= sy
        
    def as_dict(self):
        my_dict = copy.deepcopy(dict(self))
        for polygon in my_dict["polygons"]:
            polygon["outer"] = dict(polygon["outer"])
            for hole_id in range(len(polygon["inners"])):
                polygon["inners"][hole_id] = dict(polygon["inners"][hole_id])
        return my_dict
        
    def as_image(self, fill = False, w=None, h=None, force_labelRGB = False, val = 255, masks_merge_type = 'or', crop_points_outside_array = True):    
        """
        Parameters:
        Image_obj = v_polygons_obj.as_image(fill = False, w=None, h=None, val = None)
        fill (bool, default=False): If False just a perimeter is marked in the image, where a polygon is also marked if fill==True
        val(uint8 / [uint8, ...]): 
            if a single value is given it is a value of all parimeters and it's inner polygons
            if a list of values is given than consequtive values are assigned to consequtive parimeters & polygons
        w,h (int): size of the output Image, if not provided than the size is taken from "box" key in the v_polygons object
        Returns:
        PIL Image: uint8 Pillow Image with marked contours/ polygons

       """
        numpy_mask = self.as_numpy_mask(fill = fill, w=w, h=h, force_labelRGB = force_labelRGB, val=val, masks_merge_type=masks_merge_type, crop_points_outside_array=crop_points_outside_array)
        if(force_labelRGB):
            mask_image = Image.fromarray(numpy_mask, mode="RGB")
        else:
            mask_image = Image.fromarray(numpy_mask, mode="L")
        return mask_image

    def as_numpy_mask(self, fill = False, w=None, h=None, force_labelRGB = False, val = 255, masks_merge_type = 'or', crop_points_outside_array=True, line_type = ""): 
        """
        Parameters:
        numpy_array_obj = v_polygons_obj.as_numpy_mask(fill = False, w=None, h=None, val = None)
        fill (bool): default=False. If False just a perimeter is marked in the array, where a polygon is also marked if fill==True
        val(uint8 / [uint8, ...]): 
            if a single value is given it is a value of all parimeters and it's inner polygons
            if a list of values is given than consequtive values are assigned to consequtive parimeters & polygons
        w,h (int): size of the output Array, if not provided than the size is taken from "box" key in the v_polygons object
        Returns:
        array: uint8 numpy array with marked contours/ polygons
        """
        num_polygons = len(self.store["polygons"])
        if h == None:
            if(num_polygons != 0):
                h = self.store["box"][3]+1 #ymax
            else:
                h = 10
        if w == None:
            if(num_polygons != 0):
                w = self.store["box"][2]+1 #xmax
            else:
                w = 10
                    
        if force_labelRGB:
            
            if(num_polygons == 0):
                vals = [1]
            else:
                pol_vals_step = int((255-0)/num_polygons)
                if(pol_vals_step == 0 or num_polygons > 255):
                    vals = [x%254 + 1 for x in range(0, num_polygons)]
                else:
                    vals = [x for x in range(255, 0, -pol_vals_step)]
                    #vals_holes = [x for x in range(255, 0, -hs_vals_step)]

            num_holes = 0
            for polygon in self.store["polygons"]:
                num_holes += len(polygon["inners"])
            if(num_holes == 0):
                vals_holes = [1]
            else:
                visable_holes = False
                if visable_holes:
                    hs_vals_step = int((255-0)/num_holes)
                    if(hs_vals_step == 0 or num_holes > 255):
                        vals_holes = [x%254 + 1 for x in range(0, num_holes)]
                    else:
                        vals_holes = [x for x in range(255, 0, -hs_vals_step)]
                else:
                    vals_holes = [x%254 + 1 for x in range(0, num_holes)]
        else:
            if(type(val) is list):
                vals = val 
            else:
                vals = [val] 
            
        mask_na = np.zeros((h,w), dtype=np.uint8) #numpy h,w (y,x) not w,h (x,y) !!!
        if force_labelRGB:
            mask_holes_na = np.zeros((h,w), dtype=np.uint8) #numpy h,w (y,x) not w,h (x,y) !!!
        hole_in_img_id = 0

        for polygon_id, polygon in enumerate(self.store["polygons"]):
            
            val_curr = vals[polygon_id% len(vals)]
            mask_na_curr = np.zeros((h,w), dtype=np.uint8)
            if force_labelRGB:
                mask_holes_na_curr = np.zeros((h,w), dtype=np.uint8) #numpy h,w (y,x) not w,h (x,y) !!!
            polygon["outer"].write_2_numpy_mask(mask_na_curr, fill=fill, val = val_curr, deal_with_points_outside_array= crop_points_outside_array, line_type= line_type)
            for hole_id in range(len(polygon["inners"])):
                polygon["inners"][hole_id].write_2_numpy_mask(mask_na_curr, fill=True,  perimeter = False, val = 0       , deal_with_points_outside_array= crop_points_outside_array)
                polygon["inners"][hole_id].write_2_numpy_mask(mask_na_curr, fill=False, perimeter = True,  val = val_curr, deal_with_points_outside_array= crop_points_outside_array, line_type= line_type)

                if force_labelRGB:
                    hole_in_img_id += 1
                    val_hole = vals_holes[hole_in_img_id-1]
                    polygon["inners"][hole_id].write_2_numpy_mask(mask_holes_na_curr, fill=True,  perimeter = False, val = val_hole, deal_with_points_outside_array= crop_points_outside_array)
                    polygon["inners"][hole_id].write_2_numpy_mask(mask_holes_na_curr, fill=False, perimeter = True,  val =        0, deal_with_points_outside_array= crop_points_outside_array)

                    
            if masks_merge_type == 'or':
                np.bitwise_or(mask_na, mask_na_curr, mask_na)
            elif masks_merge_type == 'over':
                mask_na = np.where(mask_na_curr!=0, mask_na_curr, mask_na)
                
            if force_labelRGB:
                if masks_merge_type == 'or':
                    np.bitwise_or(mask_holes_na, mask_holes_na_curr, mask_holes_na)
                elif masks_merge_type == 'over':
                    mask_holes_na = np.where(mask_holes_na_curr!=0, mask_holes_na_curr, mask_holes_na)

            
        if force_labelRGB:
            r = mask_na
            g = mask_holes_na
            b = np.zeros((h,w), dtype=np.uint8)
            rgb = np.dstack([r, g, b])  # stacks 3 h x w arrays -> h x w x 3
            
            return rgb

        else:
            return mask_na
        
    def _mask_image_to_polygons(self, img, background_val = None, float_out = False):
        #get paths and sort them from the longest to shortest
        returned_paths = _mask_image_to_paths_lists(img, background_val, float_out)
        #paths_list.sort(key=len, reverse=True) #sort reversed - from longest
    
        for path, path_holes, org_val, area in returned_paths:
            self.add_polygon_from_paths(outer_path = path, holes_paths = path_holes)
            self.store["polygons"][-1]["org_val"] = org_val
                
    def _mask_ndarray_to_polygons(self, mask_ndarray, background_val = None, limit_polygons_num = 0, float_out = False):
        #get paths and sort them from the largest-area to smallest-area
        returned_paths = _mask_ndarray_to_paths_lists(mask_ndarray, background_val, float_out)
        #paths_list.sort(key=len, reverse=True) #sort reversed - from longest
    
        for pid, [path, path_holes, org_val, area] in enumerate(returned_paths):
            if((limit_polygons_num <= 0) or (pid < limit_polygons_num)):
                self.add_polygon_from_paths(outer_path = path, holes_paths = path_holes)
                self.store["polygons"][-1]["org_val"] = org_val
                self.store["polygons"][-1]["area"] = area
            else:
                logging.debug("Skip polygon {} due to limitation on maximum polygons number ({}).".format(pid, limit_polygons_num))
            
    def interpolate_verts(self, max_verts_dist = 1.0, force_int = True):
        inserted = 0
        for polygon in self["polygons"]:
            contour = polygon["outer"]
            inserted = inserted + contour.interpolate_path(max_verts_dist, force_int = force_int)
            polygon["outer"] = contour
            for hole_id in range(len(polygon["inners"])):
                contour = polygon["inners"][hole_id]
                inserted = inserted + contour.interpolate_path(max_verts_dist, force_int = force_int)
                polygon["inners"][hole_id] = contour
        return inserted
    
    def remove_colinear_verts(self):
        removed = 0
        for polygon in self["polygons"]:
            contour = polygon["outer"]
            removed = removed + contour.remove_colinear_verts()
            polygon["outer"] = contour
            for hole_id in range(len(polygon["inners"])):
                contour = polygon["inners"][hole_id]
                removed = removed + contour.remove_colinear_verts()
                polygon["inners"][hole_id] = contour
        return removed

def _remove_redundand(in_list):
    if(len(in_list)==0):
        return np.array([])

    # the first point
    out_list=np.array([in_list[0]])

    #from the second to the last but one point
    for i in range(1, len(in_list)-1):
        if not np.all(in_list[i] == out_list[-1]):
            same_x = (out_list[-1][0] == in_list[i][0]) and (in_list[i][0] == in_list[i+1][0])
            same_y = (out_list[-1][1] == in_list[i][1]) and (in_list[i][1] == in_list[i+1][1])
            is_str_line_x = same_x and (((out_list[-1][1] < in_list[i][1]) and (in_list[i][1] <= in_list[i+1][1])) or ((out_list[-1][1] > in_list[i][1]) and (in_list[i][1] >= in_list[i+1][1])))
            is_str_line_y = same_y and (((out_list[-1][0] < in_list[i][0]) and (in_list[i][0] <= in_list[i+1][0])) or ((out_list[-1][0] > in_list[i][0]) and (in_list[i][0] >= in_list[i+1][0])))

            if(not (is_str_line_x or is_str_line_y)):
                out_list = np.append(out_list, [in_list[i]], axis=0)
    # the last point
    if not np.all(in_list[-1] == out_list[-1]):
        out_list = np.append(out_list, [in_list[-1]], axis=0)

    return out_list

def _mask_image_to_paths_lists(img, background_val = None, float_out = False):

    if(img.mode == "RGB" or img.mode == "RGBA"):
        img = img.convert('L')

    numpy_mask = np.asarray(img)
    return _mask_ndarray_to_paths_lists(numpy_mask, background_val, float_out)

def _area_oriented(ndarray_of_points):
    area = 0
    for pid in range(len(ndarray_of_points)-1):
        x1, y1 = ndarray_of_points[pid]
        x2, y2 = ndarray_of_points[pid+1]
        area += (x2-x1)*(y2+y1)
    area = area / 2
    return area

def _area_centroid_cc(ndarray_of_points):
    area = 0
    centroid_x = 0
    centroid_y = 0
    for pid in range(len(ndarray_of_points)-1):
        x1, y1 = ndarray_of_points[pid]
        x2, y2 = ndarray_of_points[pid+1]
        area += (x2-x1)*(y2+y1)
        centroid_x += (x1+x2)*(x1*y2 - x2*y1)
        centroid_y += (y1+y2)*(x1*y2 - x2*y1)
    is_clockwise = area > 0
    area = abs(area / 2)
    if(area > 0):
        centroid_x = abs(centroid_x/(6*area))
        centroid_y = abs(centroid_y/(6*area))
        centroid = np.array([centroid_x, centroid_y])
    else:
        centroid = np.mean(ndarray_of_points[:-1], axis = 0)

    return (area, centroid, is_clockwise)


def _is_clockwise(ndarray_of_points):
    _area_oriented(ndarray_of_points)
    return area > 0

def _mask_ndarray_to_paths_lists(numpy_mask, background_val = None, float_out = False):
    mask_unique_vals = np.unique(numpy_mask)

    if(background_val==None):
        corners_val = [numpy_mask[0,0], numpy_mask[-1,0], numpy_mask[-1,-1], numpy_mask[0,-1]]
        corners_counts = np.bincount(corners_val)
        background_val = np.argmax(corners_counts)
    
    mask_unique_vals = mask_unique_vals[mask_unique_vals!=background_val]

    input_shape = numpy_mask.shape
    if(len(input_shape) > 2):
        input_shape = input_shape[0:2]
        
    paths_lists = []
    for unique_val in mask_unique_vals:
        #binary_mask = np.array(dtype='uint8', shape=)
        boolean_mask = numpy_mask==unique_val
        binary_mask = boolean_mask.astype(int)
        #we need to transpose matrix becuse numpy index images with y,x and we index contours with x,y 
        binary_mask = binary_mask.T
        # skimage.measure.find_contours(): "Output contours are not guaranteed to be closed: contours which intersect the array edge will be left open. All other contours will be closed."
        #  therefore padding is needed. Later, the detected contours needs to be shifted by (1,1)
        binary_mask_padded = np.pad(binary_mask, pad_width = 1, mode='constant', constant_values=0)

        # Find contours at a constant value of 0.9
        if(float_out):
            contours = measure.find_contours(binary_mask_padded, 0.5, fully_connected = 'high', positive_orientation = 'low')
        else:
            contours = measure.find_contours(binary_mask_padded, 0.9, fully_connected = 'high', positive_orientation = 'low')
        for contour_id, contour in enumerate(contours):
            is_closed = np.all(contour[0] == contour[-1])
            if(not is_closed):
                logging.error("I expected only closed polygons")

        # cast to list of integers, and remove consequtive duplicated values, and shifted by (1,1) in order to compensate for padding added before
        if float_out:
            contours_sh    = [(contour-[1.0,1.0]) for contour in contours]
            contours_simp  = [_remove_redundand(contour) for contour in contours_sh]
        else:
            contours_sh    = [((np.round(contour).astype(int))-[1,1]) for contour in contours  ]
            contours_simp  = [_remove_redundand(contour) for contour in contours_sh  ]
        
        contour_hole_dicts = []
        contour_hill_dicts = []
        for contour_id, contour in enumerate(contours_simp):
            
            contour_simp  = contours_simp[contour_id].tolist()
            contour_sh    = contours_sh  [contour_id].tolist()
            area_oriented = _area_oriented(contours_simp[contour_id])
            contour_dict  = {"simplified":contour_simp, "full":contour_sh, "area":abs(area_oriented)}
            is_hole = (area_oriented <  0)
            is_hill = not is_hole
            
            if is_hole:
                if len(contour_simp) > 3 : # 3 points means a straight line 
                    contour_hole_dicts.append(contour_dict)
            elif is_hill:
                contour_hill_dicts.append(contour_dict)

        ## if only holes exists than it means that whole the region can be a hill - not valid after padding operation
        #if(len(contour_hill_dicts) == 0):
        #    contour_hill_dicts.append({"simplified":[[0,0], [0,binary_mask.shape[1]-1], [binary_mask.shape[0]-1, binary_mask.shape[1]-1], [binary_mask.shape[0]-1, 0]]})
        if(len(contour_hill_dicts) > 1):
            def sortByArea(val): 
                return -val["area"]
            contour_hill_dicts.sort(key = sortByArea)

        for contour_hill_dict in contour_hill_dicts:

            contour_hill        = contour_hill_dict["simplified"]
            contour_hill_full   = contour_hill_dict["full"]
            holes_of_curr_hill  = []

            if(len(contour_hill_dicts) == 1):
                holes_of_curr_hill = [ x["simplified"] for x in contour_hole_dicts]
            else:
                num_holes = len(contour_hole_dicts)

                for contours_hole_id in range(num_holes-1, -1, -1):

                    sing_hole_point = contour_hole_dicts[contours_hole_id]["simplified"][0]
                    is_inside = measure.points_in_poly([sing_hole_point], contour_hill)[0]

                    if not is_inside:
                        #special case when a point of a hole can be one of the outside perimeter, and it still should count as inside
                        is_inside = sing_hole_point in contour_hill_full 
                   
                    if(is_inside):
                        contour_hole_dict = contour_hole_dicts.pop(contours_hole_id)
                        holes_of_curr_hill.append(contour_hole_dict["simplified"])

            paths_lists.append((contour_hill, holes_of_curr_hill, int(unique_val), contour_hill_dict["area"]))

        #check if all holes have been asigned to some hill 
        if(len(contour_hill_dicts) > 1):
            if(len(contour_hole_dicts) > 0):
                logging.error("Some holes do not have its polygon! Something went wrong. Dump to \"hole_error.json\". Probably hole is a spline with area=0")
                error_dict = { }
                for id, contour_hole_dict in enumerate(contour_hole_dicts):
                    error_dict["contour without its polygon id {}".format(id)] = contour_hole_dict
                error_dict["hills"] = contour_hill_dicts
                jsonDumpSafe("hole_error.json", error_dict)
                sys.exit(29)

    return paths_lists
##############################################################################
# MAIN
##############################################################################
def main():
    import json
                
    #----------------------------------------------------------------------------
    # initialize logging 
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    initial_log_fn = "_v_polygons.log"
    logging.basicConfig(level=logging.INFO, format = log_format, handlers=[logging.FileHandler(initial_log_fn, mode='w'), logging.StreamHandler(sys.stdout)])
    
    #----------------------------------------------------------------------------
    my_rand1_path   =[    [ 2,  2],
                          [ 4,  9],
                          [ 8,  5],
                          [ 8,  2]]
    my_rand1_hole_path =[ [ 4,  4],
                          [ 7,  4],
                          [ 7,  7],
                          [ 4,  7]]
    my_rand1_contour        = v_contour(my_rand1_path)
    my_rand1_hole_contour   = v_contour(my_rand1_hole_path)
    
    # Shifted copy of rand1 contours
    my_rand2_contour        = copy.deepcopy(my_rand1_contour)
    my_rand2_contour.move2point([7,6])    
    my_rand2_hole_contour   = copy.deepcopy(my_rand1_hole_contour)
    my_rand2_hole_contour.move2point([7,6])   
    
    logging.info("v_polygons initialized")
    my_polygons = v_polygons(          ) #empty
    logging.info(my_polygons)
    logging.info(my_polygons.to_indendent_str())
    
    
    logging.info("add two new contours to v_polygons")
    #1 - from paths (list of 2D points, or numpy ndarray of 2D points)
    my_polygons.add_polygon_from_paths(outer_path = my_rand1_path, holes_paths = [my_rand1_hole_path,]) 
    #2
    #my_polygons.add_polygon_from_contours(outer_contour = my_rand2_contour, holes_contours = [my_rand2_hole_contour,]) # no holes. Parameter holes_contours equals to [] on default
 
    logging.info(my_polygons.to_indendent_str())

    logging.info("add mass centers to outter contour of those polygons that do not have one already")
    my_polygons.fill_mass_centers()
    logging.info(my_polygons.to_indendent_str())

    logging.info("cast to dictionary")
    my_dict =my_polygons.as_dict()
    logging.info(my_dict)
    
    
    logging.info("cast to dictionary and save to json")
    meta_path = os.path.normpath('x_shape_polygons.json')
    jsonDumpSafe(meta_path, my_polygons.as_dict())
   

    logging.info("read from json file")
    with open (meta_path) as f:
        polygons_dict_data= json.load(f)
    my_read_polygons = v_polygons.from_dict(polygons_dict_data)
    logging.info(my_read_polygons.to_indendent_str())

    image_vals_for_polygons=[70, 70, 110, 150, 200, 255]
    
    fn = "my_mask_image.png"
    logging.info("cast as pillow image and save to file: {}".format(fn))
    my_mask_image = my_polygons.as_image(val=image_vals_for_polygons, fill=True, masks_merge_type = 'over')
    my_mask_image.save(fn)
    
    fn = "my_mask_image_borders_dilated_w0.png"
    logging.info("cast as pillow image and save to file: {}".format(fn))
    my_mask_image = my_polygons.as_image(val=image_vals_for_polygons, fill=False, w=20, h=16, masks_merge_type = 'over')
    my_mask_image.save(fn)
    
    for w in range(3,0,-1):
        fn = "my_mask_image_borders_dilated_w{}.png".format(w)
        logging.info("Get  borders dilated by {} points and save to file: {}".format(w, fn))
        dilated_border_polygons = v_polygons.from_polygons_borders(my_polygons, dilation_radius=w)
        my_mask_image = dilated_border_polygons.as_image(val=image_vals_for_polygons, fill=True, w=20, h=16, masks_merge_type = 'over')
        my_mask_image.save(fn)

        
    fn = "my_mask_image_croped2box.png"
    logging.info(" cropped to polygons box: {}".format(fn))
    my_polygons.crop_2box(my_polygons["box"])
    my_mask_image = my_polygons.as_image(val=image_vals_for_polygons, fill=True, masks_merge_type = 'over')
    my_mask_image.save(fn)
    
    fn = "my_mask_image_croped10x10.png"
    logging.info(" cropped from point(2,2) to size(10,10): {}".format(fn))
    my_polygons.crop(point=[2,2], size=[10,10])
    my_mask_image = my_polygons.as_image(val=image_vals_for_polygons, fill=True, masks_merge_type = 'over')
    my_mask_image.save(fn)
    
    out_fn = "my_mask_image_croped10x10_rec.png"
    logging.info(" Read previously saved mask image ({}), convert it to v_polygons and save as new image {}".format(fn, out_fn))
    f = Image.open(fn)
    my_polygons = v_polygons.from_image(f)
    #logging.info(my_polygons.to_indendent_str())

    my_mask_image = my_polygons.as_image(val=image_vals_for_polygons, fill=True, masks_merge_type = 'over')#, w=17, h=17)
    my_mask_image.save(out_fn)
    
    fn = "my_mask_image_moved_22_17x17.png"
    logging.info(" previously cropped contour moved to point (2,2) and image extended to size (17,17): {}".format(fn))
    my_polygons.move2point(point=[2,2])
    my_mask_image = my_polygons.as_image(val=image_vals_for_polygons, fill=True, w=17, h=17, masks_merge_type = 'over')
    my_mask_image.save(fn)
    
    out_fn = "my_mask_image_moved_22_17x17_rec.png"
    logging.info(" Read previously saved mask image ({}), convert it to v_polygons and save as new image {}".format(fn, out_fn))
    f = Image.open(fn)
    my_polygons = v_polygons.from_image(f)
    #logging.info(my_polygons.to_indendent_str())

    my_mask_image = my_polygons.as_image(val=image_vals_for_polygons, fill=True, masks_merge_type = 'over')#, w=17, h=17)
    my_mask_image.save(out_fn)

if __name__ == '__main__':
    main()