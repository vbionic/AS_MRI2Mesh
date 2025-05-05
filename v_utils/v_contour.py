#!/usr/bin/env python
# coding: utf-8
 
import sys
import collections.abc
from PIL import Image, ImageDraw
import numpy as np
import copy
import logging
from skimage import data, filters, measure
#from skimage.segmentation import flood, flood_fill
from skimage.draw import line, polygon, polygon_perimeter #polygon2mask - dopiero od scikit-image=0.16.2 
from numbers import Number

def _unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

class v_contour(collections.abc.MutableMapping):
    """A dictionary that applies an arbitrary key-altering
       function before accessing the keys"""

    def __init__(self, path_points=None, box = None, mass_center = None):
        self.store = dict()
        self.store["path"           ] = []
        self.store["box"            ] = []
        self.store["mass_center"    ] = []
        
        if((not (path_points is None)) and (type(path_points) is np.ndarray)):
            path_points = path_points.tolist()
        if((not (path_points is None)) and (type(path_points) is list or type(path_points) is np.ndarray)):
            self.store["path"       ] = path_points
        elif(type(path_points) is dict):
            self.store.update(path_points)
            
        if not(mass_center is None):
            self.store["mass_center"] = mass_center
            
        if not(box is None):
            self.store["box"        ] = box
        elif(len(self.store["path"       ]) != 0):
            self.update_box()
            
        #self.update(dict(*args, **kwargs))  # use the free update to set keys

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
        
    def update_box(self):
        self.store["box"        ] = []
        self.store["box"        ].append(min(self.store["path"       ], key=lambda x: x[0])[0])
        self.store["box"        ].append(min(self.store["path"       ], key=lambda x: x[1])[1])
        self.store["box"        ].append(max(self.store["path"       ], key=lambda x: x[0])[0])
        self.store["box"        ].append(max(self.store["path"       ], key=lambda x: x[1])[1])
        
    def get_box(self):
        return self.store["box"]
        
    def set_mass_center(self, mass_center):
        self.store["mass_center"] = mass_center
        
    def get_mass_center(self):
        return self.store["mass_center"]
    
    def update_mass_center(self):
        ax = 0
        ay = 0
        n  = 0
        
        for v in self.store["path"]:
            ax += v[0]
            ay += v[1]
            n  += 1

        ax = ax/n
        ay = ay/n
        
        self.set_mass_center([round(ax,1),round(ay,1)])
        
    def crop_box(self, box):
        #if(self.store["box"][0] >= box[0] and self.store["box"][1] >= box[1] and self.store["box"][2] <= box[2] and self.store["box"][3] <= box[3]):
        #    self.store["box"][0] = self.store["box"][0] - box[0] if (self.store["box"][0] > box[0]) else 0
        #    self.store["box"][2] = self.store["box"][2] - box[0] if (self.store["box"][2] > box[0]) else 0
        #    self.store["box"][1] = self.store["box"][1] - box[1] if (self.store["box"][1] > box[1]) else 0
        #    self.store["box"][3] = self.store["box"][3] - box[1] if (self.store["box"][3] > box[1]) else 0
        dx = box[2] - box[0] + 1
        dy = box[3] - box[1] + 1
        max_x = dx-1 
        max_y = dy-1 
        cropped_np = np.array(self.store["path"])-[box[0],box[1]]
        np.clip(cropped_np, [0,0], [max_x,max_y], out=cropped_np)
        self.store["path"] = cropped_np.tolist()
        #for i in range(len(self.store["path"])):
        #    self.store["path"][i][0] = self.store["path"][i][0] - box[0] if (self.store["path"][i][0] > box[0]) else 0
        #    self.store["path"][i][1] = self.store["path"][i][1] - box[1] if (self.store["path"][i][1] > box[1]) else 0 
        #    if (self.store["path"][i][0] > max_x):
        #        self.store["path"][i][0] = max_x
        #    if (self.store["path"][i][1] > max_y): 
        #        self.store["path"][i][1] = max_y        
        self.update_box()
        self.update_mass_center()
        
    def crop(self, point, size):
        dx, dy = size
        box = list(point)
        box.append(point[0]+dx-1)
        box.append(point[1]+dy-1)
        self.crop_box(box)
        
    def move2point(self, point):        
        for i in range(len(self.store["path"])):
            self.store["path"][i][0] += point[0]
            self.store["path"][i][1] += point[1]
            
        if(len(self.store["box"])>0):
            self.store["box"        ][0] += point[0]
            self.store["box"        ][1] += point[1]
            self.store["box"        ][2] += point[0]
            self.store["box"        ][3] += point[1]
        
        if(len(self.store["mass_center"])>0):
            self.store["mass_center"][0] += point[0]
            self.store["mass_center"][1] += point[1]
            
    def scale(self, sx, sy):        
        for i in range(len(self.store["path"])):
            self.store["path"][i][0] *= sx
            self.store["path"][i][1] *= sy
            
        if(len(self.store["box"])>0):
            self.store["box"        ][0] *= sx
            self.store["box"        ][1] *= sy
            self.store["box"        ][2] *= sx
            self.store["box"        ][3] *= sy
        
        if(len(self.store["mass_center"])>0):
            self.store["mass_center"][0] *= sx
            self.store["mass_center"][1] *= sy
            
    def interpolate_path(self, max_verts_dist = 2, force_int = True):
        
        if(len(self.store["path"]) < 3):
            return 0

        if(force_int and (round(max_verts_dist,0) != max_verts_dist)):
            logging.error(f"Assuming integer output but max_verts_dist is not an integer value ({max_verts_dist})")
            sys.exit(1)

        org_path    = np.array(self.store["path"][:-1], dtype = np.float32)
        sh_path     = [*org_path[1:], org_path[0]]
        path_dists  = np.linalg.norm(org_path - sh_path, axis=1)
        inter_need  = np.where(path_dists > max_verts_dist)
        inter_need  = inter_need[0][::-1]
        if len(inter_need) == 0:
            return 0
        else:
            inserted = 0
            extended_path = np.array([*org_path, org_path[0]])
            for pid in inter_need:
                before  = extended_path[:pid+1]
                after   = extended_path[pid+1:]
                start_p = before[-1]
                end_p   = after[0]
                vector = end_p - start_p
                dist = np.linalg.norm(vector)
                int_points_num = int(dist // max_verts_dist)
                dv = vector / (int_points_num+1)
                if force_int:
                    if dv[0] != round(dv[0]) or dv[1] != round(dv[1]):
                        if dv[0] != 0 and dv[1] != 0:
                            if abs(dv[0]) != abs(dv[1]):
                                logging.warning(f"Interpolation: dv = {dv} found while integers are forced")
                in_p_l = [start_p + dv*(i+1) for i in range(int_points_num)]
                test_p_l  = np.array([start_p, *in_p_l, end_p])
                if force_int:
                    test_p_l  = np.array([list([int(round(p[0])), int(round(p[1]))]) for p in test_p_l])
                test_p_lu = np.unique(test_p_l, axis=0)
                if(len(test_p_lu) != len(in_p_l)+2):
                    in_p_l = test_p_lu[1:-1]
                    int_points_num = len(test_p_l)
                inserted = inserted + int_points_num
                extended_path = [*before, *in_p_l, *after]
            if force_int:
                path_new = [list([int(round(p[0],0)), int(round(p[1],0))]) for p in extended_path]
            else:
                path_new = [list(p) for p in extended_path]
            self.store["path"] = path_new
            return inserted
        
    def remove_colinear_verts(self):

        if(len(self.store["path"]) < 3):
            return 0

        org_path    = np.array(self.store["path"][:-1])
        shp_path     = [              *org_path[1:  ], org_path[0]]
        shm_path     = [org_path[-1], *org_path[ :-1]             ]
            
        dp_l = shp_path - org_path
        dm_l = org_path - shm_path
            
        vp_l = np.array([_unit_vector(dp_l[i]) for i in range(len(dp_l))])
        vm_l = np.array([_unit_vector(dm_l[i]) for i in range(len(dm_l))])

        #path_dists  = np.linalg.norm(shp_path - shm_path, axis=1)
        not_colinear = np.where((vp_l != vm_l).any(axis=1))[0]
        removed = org_path.shape[0] - not_colinear.shape[0]
        trimmed_path = org_path[not_colinear]
        trimed_path_l = [list(p) for p in trimmed_path]
        trimed_path_l.append(trimed_path_l[0])
        self.store["path"] = trimed_path_l
        return removed

    def as_dict(self):
        my_dict = dict(self)
        return my_dict
        
    def as_image(self, fill = False, perimeter = True, w=None, h=None, val = None):    
        """
        Parameters:
        Image_obj = v_contour_obj.as_image(fill = False, w=None, h=None, val = None)
        perimeter (bool, default=True): If True a perimeter is marked
        fill (bool, default=True): If True a polygon is marked inside the perimeter
        val(uint8 / [uint8, uint8]): 
            if a single value is given it is a value of parimeter and it's inner polygon
            if a list of values is given than a first value is a value of a parimeter and the second is a value of the parimeter's inner polygon
        w,h (int): size of the output Image, if not provided than the size is taken from "box" key in the v_polygons object
        Returns:
        PIL Image: uint8 Pillow Image with marked contours/ polygons

       """
        numpy_mask = self.as_numpy_mask(fill = fill, perimeter = perimeter, w=w, h=h, val=val)
        mask_image = Image.fromarray(numpy_mask, mode="L")
        return mask_image

    def as_numpy_mask(self, fill = True, perimeter = True, w=None, h=None, val=1, deal_with_points_outside_array=True, line_type = ""):
        """
        Parameters:
        numpy_array_obj = v_contour_obj.as_numpy_mask(fill = False, w=None, h=None, val = None)
        perimeter (bool, default=True): If True a perimeter is marked
        fill (bool, default=True): If True a polygon is marked inside the perimeter
        val(uint8 / [uint8, uint8]): 
            if a single value is given it is a value of parimeter and it's inner polygon
            if a list of values is given than a first value is a value of a parimeter and the second is a value of the parimeter's inner polygon
        w,h (int): size of the output Array, if not provided than the size is taken from "box" key in the v_contour object
        Returns:
        array: uint8 numpy array with marked contour/ polygon
        """
        if h == None:
            h = self.store["box"][3]+1 #ymax
        if w == None:
            w = self.store["box"][2]+1 #xmax
              
        numpy_mask = np.zeros((h,w), dtype=np.uint8) #numpy h,w (y,x) not w,h (x,y) !!!

        self.write_2_numpy_mask(numpy_mask, fill = fill, perimeter = perimeter, val=val, deal_with_points_outside_array = deal_with_points_outside_array, line_type = line_type)
        
        return numpy_mask
       
    def write_2_numpy_mask(self, numpy_mask, fill = True, perimeter = True, val=1, deal_with_points_outside_array=True, line_type = ""):
        """
        Parameters:
        numpy_array_obj = v_contour_obj.as_numpy_mask(numpy_mask, fill = False, w=None, h=None, val = None)
        numpy_mask: numpy mask array to write to
        perimeter (bool, default=True): If True a perimeter is marked
        fill (bool, default=True): If True a polygon is marked inside the perimeter
        val(uint8 / [uint8, uint8]): 
            if a single value is given it is a value of parimeter and it's inner polygon
            if a list of values is given than a first value is a value of a parimeter and the second is a value of the parimeter's inner polygon
        Returns:
        """
              
        if(type(val) is list):
            val_perimeter = val[0]   
            if( len(val)>1):
                val_fill = val[1]  
            else:
                val_fill = val_perimeter 
        else:
            val_perimeter = val
            val_fill = val_perimeter 

        path_as_npa = np.array(self.store["path"], dtype=np.int16)
        rs = path_as_npa[:,1]
        #logging.info(rs)
        cs = path_as_npa[:,0]
        #logging.info(cs)
        if(fill):
            if(len(rs)>2):
                if deal_with_points_outside_array:
                    # nie shape tylko shape + poczatek wyciecia (org_point)
                    rr, cc = polygon(rs, cs, shape=numpy_mask.shape)
                else:
                    rr, cc = polygon(rs, cs)
                    
                if(line_type == "."):
                    del_range = np.arange(0, rr.size, 2)
                    rr = np.delete(rr, del_range)
                    cc = np.delete(cc, del_range)
                elif(line_type == "-"):
                    del_range0 = np.arange(0, rr.size, 5)
                    del_range1 = np.arange(1, rr.size, 5)
                    del_range = np.hstack([del_range0, del_range1])
                    rr = np.delete(rr, del_range)
                    cc = np.delete(cc, del_range)

                numpy_mask[rr, cc] = val_fill  #numpy h,w (y,x) not w,h (x,y) !!!
        if(perimeter):
            #try:
            #    if(len(rs)>2):
            #        if deal_with_points_outside_array:
            #            # nie shape tylko shape + poczatek wyciecia (org_point)
            #            rr, cc = polygon_perimeter(rs, cs, shape=numpy_mask.shape, clip=True)
            #        else:
            #            rr, cc = polygon_perimeter(rs, cs)
            #        
            #        if(line_type == "."):
            #            del_range = np.arange(0, rr.size, 2)
            #            rr = np.delete(rr, del_range)
            #            cc = np.delete(cc, del_range)
            #        elif(line_type == "-"):
            #            del_range0 = np.arange(0, rr.size, 5)
            #            del_range1 = np.arange(1, rr.size, 5)
            #            del_range2 = np.arange(2, rr.size, 5)
            #            del_range = np.hstack([del_range0, del_range1, del_range2])
            #            rr = np.delete(rr, del_range)
            #            cc = np.delete(cc, del_range)
            #
            #    elif(len(rs)>1):
            #   
            #        rr, cc = line(rs[0], cs[0], rs[1], cs[1])
            #        if deal_with_points_outside_array:
            #            for i in range(len(rr)-1, -1, -1):
            #                # nie shape tylko shape + poczatek wyciecia (org_point)
            #                out = rr[i] < 0 or rr[i] >= numpy_mask.shape[1] or cc[i] < 0 or c[i] >= numpy_mask.shape[0]
            #                if out:
            #                    rr = np.delete(rr,i)
            #                    cc = np.delete(cc,i)
            #    else:
            #        if deal_with_points_outside_array:
            #            # nie shape tylko shape + poczatek wyciecia (org_point)
            #            if(rs>=0 and cs>=0 and rs<numpy_mask.shape[0] and cs<numpy_mask.shape[1]):
            #                rr, cc = (rs, cs)
            #        else:
            #            rr, cc = (rs, cs)
            #except IndexError as err:
            #    logging.error("!Error in \"as_numpy_mask\": {}".format(err))
            #    logging.error("rs: {}\ncs: {},\n path_as_npa: {},\n numpy_mask.shape: {},\n fill: {},\n perimeter: {},\n val: {},\n deal_with_points_outside_array: {},\n line_type: {}".format(
            #                    rs,     cs,       path_as_npa,       numpy_mask.shape,       fill,       perimeter,       val,       deal_with_points_outside_array,       line_type))            
            #    perimeter = False
                    
            if(len(rs)>2):
                rr, cc = polygon_perimeter(rs, cs)
                    
                if(line_type == "."):
                    del_range = np.arange(0, rr.size, 2)
                    rr = np.delete(rr, del_range)
                    cc = np.delete(cc, del_range)
                elif(line_type == "-"):
                    del_range0 = np.arange(0, rr.size, 5)
                    del_range1 = np.arange(1, rr.size, 5)
                    del_range2 = np.arange(2, rr.size, 5)
                    del_range = np.hstack([del_range0, del_range1, del_range2])
                    rr = np.delete(rr, del_range)
                    cc = np.delete(cc, del_range)

            elif(len(rs)>1):
               
                rr, cc = line(rs[0], cs[0], rs[1], cs[1])
            else:
                rr, cc = [rs], [cs]
            
            if deal_with_points_outside_array:
                for i in range(len(rr)-1, -1, -1):
                    # nie shape tylko shape + poczatek wyciecia (org_point)
                    bound_shape = numpy_mask.shape

                    out = (rr[i] < 0) or (rr[i] >= bound_shape[0]) or (cc[i] < 0) or (cc[i] >= bound_shape[1])
                    if out:
                        rr = np.delete(rr,i)
                        cc = np.delete(cc,i)
                perimeter = len(rr) != 0
        if perimeter:
            numpy_mask[rr, cc] = val_perimeter  #numpy h,w (y,x) not w,h (x,y) !!!

##############################################################################
# MAIN
##############################################################################
def main():
    import json
    import os, sys
    curr_script_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(curr_script_path)
    from as_json import jsonUpdate, jsonDumpSafe
                
    #----------------------------------------------------------------------------
    # initialize logging 
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    initial_log_fn = "_v_contour.log"
    logging.basicConfig(level=logging.INFO, format = log_format, handlers=[logging.FileHandler(initial_log_fn, mode='w'), logging.StreamHandler(sys.stdout)])
    
    #----------------------------------------------------------------------------
    my_path =[[ 5,  5],
              [ 6, 12],
              [13, 12],
              [11,  6]]
    my_hole =[[ 8,  8],
              [ 8, 10],
              [10, 10],
              [10,  8]]
    logging.info("v_contour from list of points")
    my_contour = v_contour(my_path)
    my_contour["mass_center"] = [10,10] #manual
    my_contour.set_mass_center([10,10]) #manual2
    my_contour.update_mass_center() #auto
    logging.info(my_contour)

    logging.info("cast to dictionary")
    my_dict =my_contour.as_dict()
    logging.info(my_dict)
    
    
    logging.info("cast to dictionary and save to json")
    meta_path = os.path.normpath('x_shape_contour.json')
    jsonDumpSafe(meta_path, my_contour.as_dict())
   

    logging.info("read from json file")
    with open (meta_path) as f:
        contour_dict_data= json.load(f)
    my_read_contour = v_contour(contour_dict_data)
    logging.info(my_read_contour)


if __name__ == '__main__':
    main()