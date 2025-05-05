#!/usr/bin/env python
# coding: utf-8

# pip/conda libs:
import os, sys, io
import pathlib
import time
import numpy as np
import logging
import torch
import copy
from PIL import Image
import json
from argparse import ArgumentParser
from collections import namedtuple
from pandas.core.common import flatten

import multiprocessing
from multiprocessing import Process, Queue
from torch.utils.data import Dataset, DataLoader

import subprocess
#-----------------------------------------------------------------------------------------
curr_script_path = os.path.dirname(os.path.abspath(__file__))
flexnet_path = os.path.normpath(os.path.join(curr_script_path, ".."))
flexnet_host_path = os.path.normpath(os.path.join(flexnet_path, ".."))
sys.path.append(flexnet_host_path)
#-----------------------------------------------------------------------------------------
from v_utils.v_contour import *
from v_utils.v_polygons import *
from v_utils.v_json import jsonUpdate, jsonDumpSafe
from v_utils.v_dataset import MRIDataset
from flexnet.utils.gen_unet_utils import load_model, load_model_from_state_dict, get_cuda_mem_used_free, get_gen_mem_used_free, try_parse_dicts_from_file
from v_utils.v_arg import arg2boolAct, print_cfg_list, print_cfg_dict
from v_utils.v_arg import convert_dict_to_cmd_line_args, convert_cmd_line_args_to_dict, convert_cfg_files_to_dicts
from flexnet.model.pytorch_resnet_flex_unet import ResNetFlexUNet
import flexnet.utils.img_helper as img_helper
from flexnet.evaluation.v_classification_stat import classification_stats_for_single_img

#----------------------------------------------------------------------------
# use_model_on_dataset_images
def use_model_on_dataset_images(model, device, myMRISegmentsDataset, sufix, work_dir,  
                                export_labels       = True,  
                                export_polygons     = True, 
                                export_box          = True,
                                export_prob         = False, 
                                export_prob_nl      = False,
                                export_masks        = False,
                                export_clasStats    = True, 
                                export_clasStatsPng = False,  
                                export_dbg_raw      = False, 
                                export_dbg          = False, 
                                export_pngs_cropped = False,
                                limit_polygons_num  = 0, 
                                fill_polygons_holes = False, 
                                fill_labels_holes   = False, 
                                fill_masks_holes    = False, 
                                export_clss_list    = None,
                                out_queue           = None, 
                                threshold_level     = 0.5,
                                status_log_dict     = None,
                                status_log_pth      = None, 
                                do_empty_cache      = False, 
                                useClamp01NotSigmoid= False):

    dataloader = DataLoader(myMRISegmentsDataset,    batch_size=1         , shuffle=False, num_workers=0)
    
    base_dataset = myMRISegmentsDataset #should work for a single evaluation processes that get the whole database
    if(type(myMRISegmentsDataset) is torch.utils.data.dataset.Subset):
        base_dataset = myMRISegmentsDataset.dataset #should work for multiple evaluation processes that get a subset of the database
        if(type(base_dataset) is torch.utils.data.dataset.Subset):
            base_dataset = base_dataset.dataset #should work for multiple evaluation processes that get a subset of the database

    dataset_has_ref_masks = base_dataset.has_ref_masks 
    out_box = []
    batch_id = 0
    prev_per = 0

    for images_in, cls_masks_ref, test_envs in dataloader:
        logging.debug("@{}: New input image {}".format(os.getpid(), [x for x in test_envs['src_comps_path_l'] if x!='']))
        if (device.type=='cuda'):
            logging.debug('@{}:     mem allocated in GB: {}'.format(os.getpid(),torch.cuda.memory_allocated(device.index) / 1024**3))
            logging.debug('@{}: max mem allocated in GB: {}'.format(os.getpid(),torch.cuda.max_memory_allocated(device.index) / 1024**3))
 
        #finished = False
        #while(not finished):
        #    finished = True
        #try:
        logging.debug("@{}:  Cast input data to {} device format".format(os.getpid(),device))
        images_in_c        = images_in.to(device)
        cls_masks_ref_c    = cls_masks_ref.to(device)
        
        # prediction using the model
        try:
            logging.debug("@{}:  Prediction using model...".format(os.getpid()))
            cls_preds_out_l = model(images_in_c)
            logging.debug("@{}:   done".format(os.getpid()))
        except RuntimeError as err:
            logging.error("@{}:  Error during prediction: {}".format(os.getpid(),err))
            return err

        if(useClamp01NotSigmoid):
            logging.debug("@{}:  Use clamp on the prediction output".format(os.getpid()))
            cls_preds_out_l = cls_preds_out_l.clamp(min=0, max=1.0)
        else:
            logging.debug("@{}:  Use sigmoid on the prediction output".format(os.getpid()))
            cls_preds_out_l = torch.sigmoid(cls_preds_out_l)
        cls_preds_out_l = cls_preds_out_l.data.cpu().numpy()
            
        for image_id in range(cls_preds_out_l.shape[0]): # should be just one because batch_size=1
            if (export_dbg_raw or export_dbg or export_clasStats or export_clasStatsPng ):
                cls_masks_ref = cls_masks_ref_c[image_id].cpu().numpy()
            cls_preds_out = cls_preds_out_l[image_id]
            #general_work_file_pattern = test_envs['general_work_file_pattern'][image_id]
            w = cls_preds_out[0].shape[1]
            h = cls_preds_out[0].shape[0]
            fn_pattern = test_envs['file_name_pattern'][image_id]
            ssd = test_envs['session_sub_dir'][image_id]
            session_file_dir = os.path.normpath(os.path.join(work_dir, ssd))

            if(image_id == 0):
                # dump description.json file with info about translation from the original dicom images
                try:
                    if not os.path.isdir(session_file_dir):
                        pathlib.Path(session_file_dir).mkdir(mode=0o775, parents=True, exist_ok=True)
                        logging.debug("Created dir %s"%(session_file_dir))
                except Exception as err:
                    logging.error("Creating dir (%s) IO error: %s"%(session_file_dir, err))
                    sys.exit(1)
                description_fn = os.path.normpath(os.path.join(session_file_dir, "description.json"))
                src_comps_translated_pxpy = [int(x) for x in test_envs['src_comps_translated_pxpy']]
                src_comps_size = [int(x) for x in test_envs['src_comps_size']]
                jsonDumpSafe(description_fn, {'crop_roi_size': src_comps_size,
                                              'crop_roi_pos': src_comps_translated_pxpy})
            

            for cls_id in range(len(cls_preds_out)):
                                
                logging.debug("@{}:  Read envs".format(os.getpid()))
                cls_envs_dict = test_envs['cls_envs_list'][cls_id]
                cls_name = cls_envs_dict['cls_name'][image_id]
                cls_work_file_dir = os.path.normpath(os.path.join(work_dir, ssd, cls_name))
                logging.debug("@{}:  Work dir for class: {}".format(os.getpid(), cls_work_file_dir))
                
                if(not export_clss_list is None) and (not cls_name in export_clss_list):
                    logging.debug("@{}:   skip operation for {} due to export_clss_list contains only {}".format(os.getpid(), cls_name, export_clss_list))
                    continue

                try:
                    if not os.path.isdir(cls_work_file_dir):
                        pathlib.Path(cls_work_file_dir).mkdir(mode=0o775, parents=True, exist_ok=True)
                        logging.debug("Created dir %s"%(cls_work_file_dir))
                except Exception as err:
                    logging.error("Creating dir (%s) IO error: %s"%(cls_work_file_dir, err))
                    sys.exit(1)
                                                        
                cls_work_file_pattern = os.path.normpath(os.path.join(cls_work_file_dir, fn_pattern + "_{}".format(cls_name)))

                #Generate json file with polygon of the the predicted class mask
                cls_json_fn = cls_work_file_pattern + "{}_polygons.json".format(sufix)
                logging.debug("@{}:  Generate json file with polygon of the the predicted class mask: {}...".format(os.getpid(), cls_json_fn))

                cls_pred_out = cls_preds_out[cls_id]
                
                cls_pred_out_float = cls_pred_out.copy()
                
                # as <0-255> image

                if((cls_pred_out.dtype is np.dtype('float32')) or (cls_pred_out.dtype is np.dtype('float16'))):
                    cls_pred_out = (cls_pred_out * 255.999).astype(np.uint8)

                # applay tresholded to the mask - create binary mask
                logging.debug("@{}:   applay tresholded to the mask - create binary mask...".format(os.getpid()))
                cls_mask_out_binary = np.zeros(cls_pred_out.shape, dtype = np.uint8)
                threshold_uint8 = int(round(threshold_level*255)) 
                cls_mask_out_binary[cls_pred_out[:,:] >= threshold_uint8] = 255

                need_polygons_out = export_polygons or export_labels or export_dbg_raw or export_dbg or export_box or export_clasStats or export_clasStatsPng or (export_masks and fill_masks_holes)
                need_polygons_wh_holes = (export_masks and fill_masks_holes) or  (export_labels and fill_labels_holes) or (export_polygons and fill_polygons_holes) 
                if(need_polygons_out):
                    #get polygons from the binary mask
                    logging.debug("@{}:   get polygons from the binary mask...".format(os.getpid()))
                    cls_polygons_out = v_polygons()
                    cls_polygons_out._mask_ndarray_to_polygons(cls_mask_out_binary, background_val = 0, limit_polygons_num = limit_polygons_num)

                    cls_polygons_out_sh = cls_polygons_out

                    if (np.array(test_envs['crop_box']) != np.array([0,0, *[x.item() for x in test_envs['org_size']]])).any():
                        #reverse cropping operation
                        logging.debug("@{}:   reverse cropping operation...".format(os.getpid()))
                        crop_box = test_envs['crop_box']
                        org_point = [int(crop_box[0][image_id].item()), int(crop_box[1][image_id].item())]
                        # do not overwrite cls_polygons_out because it may be needed fo PNGs export when cropped ones are to be exported
                        #  instead deepcopy cls_polygons_out
                        if(export_pngs_cropped):
                            cls_polygons_out_sh = copy.deepcopy(cls_polygons_out)
                        cls_polygons_out_sh.move2point(org_point)
                    
                    if(export_polygons):
                        logging.debug("@{}:   dump class polygons_dict...".format(os.getpid()))
                        if(fill_polygons_holes):
                            poly_cpy = cls_polygons_out_sh.copy()
                            poly_cpy.remove_inners()
                            cls_polygons_dict = poly_cpy.as_dict()
                        else:
                            cls_polygons_dict = cls_polygons_out_sh.as_dict()
                        jsonDumpSafe(cls_json_fn, cls_polygons_dict)
                    if(need_polygons_wh_holes):
                        cls_polygons_out_sh_wh_holes = cls_polygons_out_sh.copy()
                        cls_polygons_out_sh_wh_holes.remove_inners()
                
                do_export_png = export_labels or export_prob or export_prob_nl or export_masks or export_dbg_raw or export_dbg
                if(do_export_png):
                    png_w = w
                    png_h = h

                    #reverse cropping if needed
                    if( (not export_pngs_cropped) and (np.array(test_envs['crop_box']) != np.array([0,0, *[x.item() for x in test_envs['org_size']]])).any()):
                        
                        logging.debug("@{}:    reverse cropping...".format(os.getpid()))
                        crop_box = test_envs['crop_box']
                        org_size = test_envs['org_size']
                                                
                        out_size = [x.item() for x in test_envs["src_comps_size"]]

                        png_w = out_size[0]
                        png_h = out_size[1]
                        
                if export_labels:
                    # save PNG with :
                    # - RGB mask of polygons. Each polygon has separate nonzero value (on red layer for now), and 
                    #   each hole has separate nonzero value (on green layer for now)
                    fn = cls_work_file_pattern + "{}_labels.png".format(sufix)
                    logging.debug("@{}:    Creating {} with labels RGB".format(os.getpid(), fn))

                    if(fill_labels_holes):
                        img_Image = cls_polygons_out_sh_wh_holes.as_image(fill = True, w=png_w,h=png_h, force_labelRGB = True)
                    else:
                        img_Image = cls_polygons_out_sh.as_image(fill = True, w=png_w,h=png_h, force_labelRGB = True)
                    img_Image.save(fn)

                if(export_box):
                    logging.debug("@{}:   box for current class {}".format(os.getpid(),cls_polygons_out_sh["box"]))
                    if len(out_box) == 0:
                        out_box.extend(cls_polygons_out_sh["box"])
                    else:
                        try:
                            if(len(cls_polygons_out_sh["box"])==4):
                                out_box[0]= min(out_box[0], cls_polygons_out_sh["box"][0])
                                out_box[1]= min(out_box[1], cls_polygons_out_sh["box"][1])
                                out_box[2]= max(out_box[2], cls_polygons_out_sh["box"][2])
                                out_box[3]= max(out_box[3], cls_polygons_out_sh["box"][3])
                            elif(len(cls_polygons_out_sh["box"])!=0):
                                logging.warning("@{}:   img {}, class {}: box for current class ({}) is not valid!".format(os.getpid(),test_envs['src_comps_path_l'][0][image_id], cls_name, cls_polygons_out_sh["box"]))
                        except:
                            logging.warning(' Could not combine out box {} with a single-image single-class box {}.'.format(out_box, cls_polygons_out_sh["box"]))

                    logging.debug("@{}:   up-to-date box = {}".format(os.getpid(), out_box))
                    
                
                cls_pred_out_sh      = cls_pred_out

                #reverse cropping for prediction output if needed
                if( do_export_png and (not export_pngs_cropped) and ((np.array(test_envs['crop_box']) != np.array([0,0, *[x.item() for x in test_envs['org_size']]])).any())):
                        
                    logging.debug("@{}:    reverse cropping...".format(os.getpid()))
                                            
                    pad_l = int(crop_box[0]              )
                    pad_r = int(org_size[0] - crop_box[2])
                    pad_t = int(crop_box[1]              )
                    pad_b = int(org_size[1] - crop_box[3])
                    cls_pred_out_np = np.array(cls_pred_out)
                    cls_pred_out_np_float = np.array(cls_pred_out_float)
                    if(pad_l >= 0 and pad_r >= 0 and pad_t >= 0 and pad_b >= 0):
                        # padding
                        cls_pred_out_org_size = np.pad(cls_pred_out_np, pad_width = ((pad_t, pad_b),(pad_l, pad_r)), mode='constant', constant_values=0)
                        cls_pred_out_org_size_float = np.pad(cls_pred_out_np_float, pad_width = ((pad_t, pad_b),(pad_l, pad_r)), mode='constant', constant_values=0)
                    elif(pad_l <= 0 and pad_r <= 0 and pad_t <= 0 and pad_b <= 0):
                        # cropping
                        cls_pred_out_org_size = cls_pred_out_np[-pad_t: int(org_size[1])-pad_t, -pad_l: int(org_size[0])-pad_l]
                        cls_pred_out_org_size_float = cls_pred_out_np_float[-pad_t: int(org_size[1])-pad_t, -pad_l: int(org_size[0])-pad_l]
                    else:
                        # combination of padding and cropping needed
                        _pad_l = pad_l if pad_l>=0 else 0
                        _pad_r = pad_r if pad_r>=0 else 0
                        _pad_t = pad_t if pad_t>=0 else 0
                        _pad_b = pad_b if pad_b>=0 else 0
                        tmp = np.pad(cls_pred_out_np, pad_width = ((_pad_t, _pad_b),(_pad_l, _pad_r)), mode='constant', constant_values=0)
                        tmp_float = np.pad(cls_pred_out_np_float, pad_width = ((_pad_t, _pad_b),(_pad_l, _pad_r)), mode='constant', constant_values=0)
                            
                        _pad_l = pad_l if pad_l<=0 else 0
                        _pad_r = pad_r if pad_r<=0 else 0
                        _pad_t = pad_t if pad_t<=0 else 0
                        _pad_b = pad_b if pad_b<=0 else 0
                        cls_pred_out_org_size = tmp[-_pad_t: int(org_size[1])-_pad_t, -_pad_l: int(org_size[0])-_pad_l]
                        cls_pred_out_org_size_float = tmp_float[-_pad_t: int(org_size[1])-_pad_t, -_pad_l: int(org_size[0])-_pad_l]

                    cls_pred_out_sh = cls_pred_out_org_size
                    cls_pred_out_sh_float = cls_pred_out_org_size_float
                        
                if export_prob:
                    # save PNG with :
                    # - the output class mask (not contour, but values <0-255>)
                    fn = cls_work_file_pattern + "{}_prob.png".format(sufix)
                    logging.debug("@{}:    Creating {} with output as grayscale".format(os.getpid(), fn))

                    if((cls_pred_out_sh.dtype is np.dtype('float32')) or (cls_pred_out_sh.dtype is np.dtype('float16'))):
                        cls_pred_out_sh_i = (cls_pred_out_sh * 255.999).astype(np.uint8)
                    else:
                        cls_pred_out_sh_i = cls_pred_out_sh
                    img_Image = Image.fromarray(cls_pred_out_sh_i)
                    #plt.imsave(input_example_png, img_numpy)
                    img_Image.save(fn)
                    
                if export_prob_nl:
                    # save PNG with :
                    # - the output class mask (not contour, but values <0-255>)
                    fn = cls_work_file_pattern + "{}_prob_nl.png".format(sufix)
                    logging.debug("@{}:    Creating {} with nonlinearly processed output as grayscale".format(os.getpid(), fn))
                    
                    # processing 
                    #cls_pred_out_sh_p = np.sqrt(cls_pred_out_sh) # <=== tu mozesz mieszac
                    #cls_pred_out_sh_p = np.minimum((60 * np.log(70*cls_pred_out_sh + 1)) / 255.0, np.ones(cls_pred_out_sh.shape, dtype=cls_pred_out_sh.dtype))
                    
                    cls_pred_out_sh_p = cls_pred_out_sh_float.copy()
                    
                    with np.nditer(cls_pred_out_sh_p, op_flags=['readwrite']) as it:
                        for element_ in it:
                            #if element_ < 1/256.0:
                            #    element_[...] = element_ * 255.0
                            if element_ < 1/1000.0:
                                element_[...] = element_ * 999.0
                            else:
                                element_[...] = 0.999
                    
                    if((cls_pred_out_sh_p.dtype is np.dtype('float32')) or (cls_pred_out_sh_p.dtype is np.dtype('float16'))):
                        cls_pred_out_sh_i = (cls_pred_out_sh_p * 255.999).astype(np.uint8)
                    else:
                        cls_pred_out_sh_i = cls_pred_out_sh_p
                    img_Image = Image.fromarray(cls_pred_out_sh_i)
                    #plt.imsave(input_example_png, img_numpy)
                    img_Image.save(fn)

                if export_masks:
                    # save PNG with :
                    # - the output class mask (not contour, but values <0-255>)
                    fn = cls_work_file_pattern + "{}_mask.png".format(sufix)
                    logging.debug("@{}:    Creating {} with output as b/w".format(os.getpid(), fn))
                    
                    if(fill_masks_holes):
                        img_Image = cls_polygons_out_sh_wh_holes.as_image(fill = True, w=png_w,h=png_h, force_labelRGB = True, val = 255)
                    else:
                        if((cls_pred_out_sh.dtype is np.dtype('float32')) or (cls_pred_out_sh.dtype is np.dtype('float16'))):
                            cls_pred_out_sh = (cls_pred_out_sh * 255.999).astype(np.uint8)
                        
                        cls_mask_out_sh_binary = np.zeros(cls_pred_out_sh.shape, dtype = np.uint8)
                        cls_mask_out_sh_binary[cls_pred_out_sh[:,:] >= threshold_uint8] = 255
                        img_Image = Image.fromarray(cls_mask_out_sh_binary)
                    #plt.imsave(input_example_png, img_numpy)
                    img_Image.save(fn)
                   
                do_generate_in_image = export_dbg_raw or export_dbg
                if(do_generate_in_image):
                    #reverse cropping if needed
                    if( (not export_pngs_cropped) and ((np.array(test_envs['crop_box']) != np.array([0,0, *[x.item() for x in test_envs['org_size']]])).any())):
                        input_image_org = Image.open(test_envs['src_comps_path_l'][0][image_id])
                        input_ndarray_org = np.array(input_image_org)

                        image_in_rgb_sh = input_ndarray_org
                    else:
                        image_in_rgb = base_dataset.reverse_transform(images_in_c[image_id])
                        image_in_rgb_sh  = image_in_rgb

                cls_has_ref_mask = dataset_has_ref_masks and (test_envs["cls_envs_list"][cls_id]["ref_polygon_path"][image_id] != "")
                do_generate_ref_mask = cls_has_ref_mask and (export_dbg_raw or export_dbg                                            )
                do_generate_ref_poly = cls_has_ref_mask and (export_dbg_raw or export_dbg or export_clasStats or export_clasStatsPng )
                
                if do_generate_ref_poly:
                    logging.debug("@{}:   get ref polygons...".format(os.getpid()))
                    cls_mask_in = cls_masks_ref[cls_id]
                    if((cls_mask_in.dtype is np.dtype('float32')) or (cls_mask_in.dtype is np.dtype('float16'))):
                        cls_mask_in = (cls_mask_in * 255.999).astype(np.uint8)

                    #get polygons from the binary mask
                    logging.debug("@{}:     get polygons from the binary mask".format(os.getpid()))
                    cls_polygons_ref = v_polygons()
                    cls_polygons_ref._mask_ndarray_to_polygons(cls_mask_in, background_val = 0)

                    #reverse cropping if needed
                    if( (not export_pngs_cropped) and ((np.array(test_envs['crop_box']) != np.array([0,0, *[x.item() for x in test_envs['org_size']]])).any())):
                        cls_polygons_ref_sh = copy.deepcopy(cls_polygons_ref)
                        cls_polygons_ref_sh.move2point(org_point)
                    else:
                        cls_polygons_ref_sh = cls_polygons_ref
                                        
                if export_dbg_raw or export_dbg:
                    logging.debug("@{}:    create numpy masks...".format(os.getpid()))
                    cls_polygons_out_as_numpy_mask = cls_polygons_out_sh.as_numpy_mask(w=png_w,h=png_h)   
                    if(cls_has_ref_mask):
                        cls_polygons_ref_sh_as_numpy_mask = cls_polygons_ref_sh.as_numpy_mask(w=png_w,h=png_h) 

                if export_dbg_raw:
                    # save PNG with :
                    # - the input image as R channel 
                    # - the contour of the reference class mask as G channel
                    # - the output class mask as B channel (not contour, but values <0-255>)
                    fn = cls_work_file_pattern + "{}_dbg_raw.png".format(sufix)
                    logging.debug("@{}:    Creating {} with stacked masks as RGB layers".format(os.getpid(), fn))
                        
                    #cls_pred_out_sh[cls_polygons_out_as_numpy_mask[:,:] != 0] = 255
                    if(cls_has_ref_mask):
                        img_numpy = img_helper.stack_masks_as_rgb(image_in_rgb_sh, cls_polygons_ref_sh_as_numpy_mask, cls_pred_out_sh)
                    else:
                        dummy_component = np.zeros(cls_pred_out_sh.shape, dtype = np.uint8)
                        img_numpy = img_helper.stack_masks_as_rgb(image_in_rgb_sh,                      dummy_component, cls_pred_out_sh)
                    img_Image = Image.fromarray(img_numpy)
                    #plt.imsave(input_example_png, img_numpy)
                    img_Image.save(fn)
                    
                if(export_dbg):
                    # save PNG with :
                    # - the input image as R channel 
                    # - the contour of the reference class mask as G channel (if no reference is given than G channel is equal to B channel)
                    # - the contour of the output class mask as B channel
                    fn = cls_work_file_pattern + "{}_dbg.png".format(sufix)
                    logging.debug("@{}:    Creating {} with stacked masks as RGB layers".format(os.getpid(), fn))

                    if(cls_has_ref_mask):
                        img_numpy = img_helper.stack_masks_as_rgb(image_in_rgb_sh, cls_polygons_ref_sh_as_numpy_mask, cls_polygons_out_as_numpy_mask)
                    else:
                        dummy_component = np.zeros(cls_pred_out_sh.shape, dtype = np.uint8)
                        img_numpy = img_helper.stack_masks_as_rgb(image_in_rgb_sh,                      dummy_component, cls_polygons_out_as_numpy_mask)
                    img_Image = Image.fromarray(img_numpy)
                    img_Image.save(fn)
                    
                    
                if export_clasStats or export_clasStatsPng:
                    # save statistics that consider 2 mm margin assumed in AS project:
                    if(cls_has_ref_mask):
                        logging.debug("@{}:    Calculate statistics that consider 1.5 mm margin assumed in AS project...".format(os.getpid()))
                        if fill_polygons_holes:
                            cls_polygons_out.remove_inners()
                        stats_dict, stat_img = classification_stats_for_single_img(cls_polygons_out, cls_polygons_ref, write_imgs = export_clasStatsPng)
    
                        if export_clasStatsPng:
                            fn = cls_work_file_pattern + "{}_clasStat.png".format(sufix)
                            logging.debug("@{}:     Saving to {}...".format(os.getpid(), fn))
                            stat_img.save(fn)

                        if export_clasStats:
                            fn = cls_work_file_pattern + "{}_clasStat.json".format(sufix)
                            logging.debug("@{}:     Saving to {}".format(os.getpid(), fn))
                            jsonDumpSafe(fn, stats_dict)
    
        if do_empty_cache:
            if (device.type=='cuda'):
                torch.cuda.empty_cache()
                
        tot = len(dataloader)
        per = int(round((batch_id+1) / tot * 100, 0))
        if((per > (prev_per + 5)) or (per == 100)):
            logging.info("  {:3}% done ({} / {})". format(per, (batch_id+1), tot))
            prev_per = per
            if(not status_log_dict is None) and (not status_log_pth is None):
                status_log_dict["P"] = per
                jsonUpdate(status_log_pth, status_log_dict)

        batch_id += 1

    if not (out_queue is None):
        if(export_box):
            logging.debug("@{}: Put box {} to the out_queue...".format(os.getpid(), out_box)) 
            out_queue.put(out_box)  
        else:
            out_queue.put([])        
    logging.debug("@{}: Done, returning...".format(os.getpid())) 
    return 0


#----------------------------------------------------------------------------
# main

def main():
    
    total_memory = sum(get_gen_mem_used_free())
    
    #----------------------------------------------------------------------------
    # initialize logging 
    script_name = os.path.basename(__file__).split(".")[0]
    from datetime import datetime
    time_str = datetime.utcnow().strftime('%y_%m_%d__%H_%M_%S') #time.strftime("%y_%m_%d__%H_%M_%S.%f")
    initial_log_fn = f"_initial_{script_name}_{time_str}_pid{os.getpid()}.log"
    logging_level = logging.INFO
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging_level, format = log_format, handlers=[logging.FileHandler(initial_log_fn, mode='w'), logging.StreamHandler(sys.stdout)])

    logging.info('*' * 50)
    logging.info(f"script {os.path.basename(__file__).split('.')[0]} start @ {time.ctime()}")
    logging.info("initial log file is {}".format(initial_log_fn))
    
    from v_utils.v_logging_std import bind_std_2_logging
    bind_std_2_logging()
    
    #----------------------------------------------------------------------------
    parser = ArgumentParser()
    logging.info('- ' * 25)
    logging.info("Reading configuration...")
    logging.info("Command line arguments:")
    logging.info(" {}".format(' '.join(sys.argv)))
    cmd_line_args_rem = sys.argv[1:]
    
    model_dict_prs = {}
    cfgfl_dict_prs = {}
    
    #-----------------------------------------------------------
    # get middle priority arguments from config files
    logging.info('- ' * 25)
    logging.info("Try read config files with middle level priority arguments...")    
    cfa = parser.add_argument_group('config_file_arguments')
    cfa.add_argument("--cfg" , default=[], action='append', type=str, nargs='*', required=False, metavar="PATH", help="one or more config json filenames. Further config files have higher priority. Command line arguments have the highest priority.", )
    
    if not(("-h" in sys.argv) or ("--help" in sys.argv)):
        cfg_fns_args, cmd_line_args_rem = parser.parse_known_args(cmd_line_args_rem); # bez error gdy natrafi na nieznany parametr (odwrotnie niż "parse_args()")
            
        cfgs = list(map(str, flatten(cfg_fns_args.cfg)))
        # read dictonaries from config files (create a list of dicts)
        cfg_dicts = convert_cfg_files_to_dicts(cfgs)

        # merge all config dicts - later ones will overwrite entries with the same keys from the former ones
        for cfg_dict in cfg_dicts:
            cfgfl_dict_prs.update(cfg_dict)
    
    #-----------------------------------------------------------
    # get low priority arguments from dicts found in the model file 
    logging.info('- ' * 25)
    logging.info("Try read model file with low level priority arguments...")
    moa = parser.add_argument_group('model arguments')
    moa.add_argument("--model", "-m"                                                , type=str           , required=True , metavar="PATH", help="path to model state dictionary .pth file.", )
    
    if not(("-h" in sys.argv) or ("--help" in sys.argv)):
        mo_args, _ = parser.parse_known_args(copy.copy(cmd_line_args_rem)); # bez error gdy natrafi na nieznany parametr (odwrotnie niż "parse_args()")
        
        model_path = mo_args.model
        if(not os.path.isfile(model_path)):
            logging.error("Could not find model file {}".format(model_path))
            if(model_path != "skip_model"):
                sys.exit(8)
        else:
            model_dir, model_fn = os.path.split(model_path)
            V = model_fn.split("_")[0]
    
            timestamp = time.strftime("%y%m%d%H%M", time.gmtime())
    
            #get reference database description dictionary that comes together with model .pth file
            logging.info("Use model from file {}".format(model_path))
            model_timestamp = model_fn.split('_')[0]
 
            logging.debug("Try reading file {} to find integrated dataset description dict...".format(model_path))
            parsed_dicts = try_parse_dicts_from_file(model_path)
            if(not parsed_dicts is None):
                logging.debug("Found integrated argument dicts!")
                parsed_ds_dict      = parsed_dicts['ds_dict']
                parsed_fn_dict      = parsed_dicts['fn_dict']
                model_state_dict    = parsed_dicts['model_state_dict']

                # merge all config dicts - later ones will overwrite entries with the same keys from the former ones
                for cfg_dict in [parsed_ds_dict, parsed_fn_dict]:
                    model_dict_prs.update(cfg_dict)
        
    #----------------------------------------------------------------------------
    # gather all the arguments
    logging.info('- ' * 25)
    logging.info("Collect all the arguments from the model file, config files and command line...")

    # convert cmd_line_args_rem to dictionary so we can use it to update content of the dictonaries from config files
    cmd_line_args_rem_dict = convert_cmd_line_args_to_dict(cmd_line_args_rem)

    # finally update argument dictionary with the increasing priority:
    # 1) low priority arguments from dicts found in the model file 
    logging.debug(" 1) low priority arguments from dicts found in the model file...")
    params_dict = model_dict_prs
    # 2) delete keys that should use defaut values:
    logging.debug(" 2) delete keys that should use defaut values...")
    params_dict.pop("session_dirs"                  , None)
    params_dict.pop("logging_level"                 , None)
    params_dict.pop("req_refs_level"                , None)
    params_dict.pop("train"                         , None)
    params_dict.pop("train_tr_resize_range"         , None)
    params_dict.pop("skip_inputs_withouts_all_comps", None)
    params_dict.pop("crop_size"                     , None)
    params_dict.pop("descr_fn"                      , None)
    params_dict.pop("translated_pxpy"               , None)
    params_dict.pop("ds_polyRef_descr_fn"           , None)
    params_dict.pop("ds_polyRef_translated_pxpy"    , None)
    # 3) middle priority arguments from config files 
    logging.debug(" 3) middle priority arguments from config files ...")
    params_dict.update(cfgfl_dict_prs)
    # 4) high priority command line arguments
    logging.debug(" 4) high priority command line arguments ...")
    params_dict.update(cmd_line_args_rem_dict)

    logging.info("Merged arguments:")
    cfg_d = params_dict
    print_cfg_dict(cfg_d, indent = 1, skip_comments = True)

    # parse the merged arguments dictionary
    args_list_to_parse = convert_dict_to_cmd_line_args(params_dict)
    #----------------------------------------------------------------------------
    # add parser parameters that are specific to this script  
    _tf = [True, False]
    _ord_vals = ['no', 'up', 'down']
    _logging_levels = logging._levelToName.keys()

    pla = parser.add_argument_group('platform arguments')
    pla.add_argument("--force_single_thread", "-fs"      , default=False           , action=arg2boolAct,    required=False, metavar='B'   , help="wymuszenia pracy jednowatkowej")
    me_group = pla.add_mutually_exclusive_group()
    me_group.add_argument("--force_cpu",      "-fc"      , default=False           , action=arg2boolAct,    required=False, metavar='B'   , help="wymuszenia obliczen na CPU")
    me_group.add_argument("--force_gpu_id",   "-fg"      , default=None            , type=int,              required=False, metavar="I"   , help="wymuszenia obliczen na GPU o danym id")
 
    ota = parser.add_argument_group('output arguments')
    ota.add_argument("--out_dir",   "-od"                , default="def_out_dir"   , type=str,              required=False, metavar="PATH", help="directory in which class's directorys with output files will be saved")
    ota.add_argument("--logging_level"                   , default=logging.INFO    , type=int,              required=False, choices=_logging_levels,     help="")
    
    ota.add_argument("--useClamp01NotSigmoid"            , default=False           , action=arg2boolAct,    required=False, metavar='B'   , help="Domyslnie wyjscie z sieci przechodzi przez sigmoid. Tym parametrem można to zamienic na ograniczenie zakresu wyjscia do <0, 1.0>"   )
    ota.add_argument("--threshold_level"                 , default=0.5             , type=float,            required=False, metavar='F'   , help="prog decyzyjny dla binaryzacji ciaglej odpowiedzi sieci. Zakres <0.0 - 1.0>, domyslnie 0.5")
        
    ota.add_argument("--limit_polygons_num"              , default=0               , type=int,              required=False, metavar="I"   , help="can be used to limit polygons to I of the largest ones. 0 means no limit.")
    ota.add_argument("--export_labels"                   , default=True            , action=arg2boolAct,    required=False, metavar='B'   , help="eksport PNG rgb z poligonami na skladowej R i dziurami na skladowej G"      )
    ota.add_argument("--export_polygons"                 , default=True            , action=arg2boolAct,    required=False, metavar='B'   , help="eksport pliku json z poligonami"    )
    ota.add_argument("--export_box"                      , default=True            , action=arg2boolAct,    required=False, metavar='B'   , help="eksport pliku box.json z gabarytami dla danej sesji"         )
    ota.add_argument("--export_prob"                     , default=True            , action=arg2boolAct,    required=False, metavar='B'   , help="eksport PNG w skali szarości z odpowiedzia sieci neuronowej gdzie 255 oznacza najwyzsze prawdopodobienstwo wykrycia klasy"        )
    ota.add_argument("--export_prob_nl"                  , default=True            , action=arg2boolAct,    required=False, metavar='B'   , help="eksport PNG w skali szarości z dodatkowo nieliniowo przetworzona odpowiedzia sieci neuronowej gdzie 255 oznacza najwyzsze prawdopodobienstwo wykrycia klasy"        )
    ota.add_argument("--export_masks"                    , default=False           , action=arg2boolAct,    required=False, metavar='B'   , help="eksport PNG carno-biale z progowana odpowiedzia sieci neuronowej"        )
    ota.add_argument("--export_clasStats"                , default=False           , action=arg2boolAct,    required=False, metavar='B'   , help="eksport statystyk klasyfikacji dla obrazow posiadajacych poligony odniesienia. Wersja json"   )
    ota.add_argument("--export_clasStatsPng"             , default=False           , action=arg2boolAct,    required=False, metavar='B'   , help="eksport statystyk klasyfikacji dla obrazow posiadajacych poligony odniesienia. Wersja PNG")
    ota.add_argument("--export_dbg_raw"                  , default=False           , action=arg2boolAct,    required=False, metavar='B'   , help="eksport PNG rgb: R - obraz wejsciowy, G - kontur poligonow odniesienia, B - odpowiedz sieci neuronowej (ciagła)")
    ota.add_argument("--export_dbg"                      , default=False           , action=arg2boolAct,    required=False, metavar='B'   , help="eksport PNG rgb: R - obraz wejsciowy, G - kontur poligonow odniesienia, B - odpowiedz sieci neuronowej (kontur)")
    ota.add_argument("--export_pngs_cropped"             , default=False           , action=arg2boolAct,    required=False, metavar='B'   , help="PNG sa eksportowane z przycieciem wynikajacym z parametru 'crop_size' podanego dla bazy danych")
    ota.add_argument("--fill_polygons_holes"             , default=False           , action=arg2boolAct,    required=False, metavar='B'   , help="przed eksportem wypełnia ewentualne dziury w poligonach")
    ota.add_argument("--fill_labels_holes"               , default=False           , action=arg2boolAct,    required=False, metavar='B'   , help="przed eksportem wypełnia ewentualne dziury w labelach") 
    ota.add_argument("--fill_masks_holes"                , default=False           , action=arg2boolAct,    required=False, metavar='B'   , help="przed eksportem wypełnia ewentualne dziury w maskach") 
   
    ota.add_argument("--export_clss_filter_pass_list"    , default=[]              , type=str,   nargs='*', required=False, metavar='cname',help="lista nazw klas dla ktorych     BEDA wyeksportowane pliki wyjsciowe zgodnie z ustawieniami przelacznikow export_###. Pusta lista oznacza ze plik wygenerowane bedą dla wszystkich klas wyliczanych w danym modelu .pth")
    ota.add_argument("--export_clss_filter_stop_list"    , default=[]              , type=str,   nargs='*', required=False, metavar='cname',help="lista nazw klas dla ktorych NIE BEDA wyeksportowane pliki wyjsciowe")
   
    if (("-h" in sys.argv) or ("--help" in sys.argv)):
        # help
        logging.info("Params for apply_model_to_session:")
        logging.info(parser.format_help())
        logging.info("Params for MRIDataset:")
        logging.info(MRIDataset.parse_arguments("--help"))
        logging.info("Params for ResNetFlexUNet:")
        logging.info(ResNetFlexUNet.parse_arguments("--help"))
        sys.exit(1)
    else:
        # get evaluation arguments
        logging.info('- ' * 25)
        logging.info("Parse the arguments...")
        args, rem_args = parser.parse_known_args(args_list_to_parse)
    #----------------------------------------------------------------------------
    # redirect logging file to work directory 
    logging.info('-' * 50)
    work_dir = os.path.normpath(args.out_dir)
    log_dir = os.path.normpath(os.path.join(args.out_dir, "_log"))
    logging.info(f'Redirect logging file to {log_dir} directory')
    # create work dir
    try:
        if not os.path.isdir(log_dir):
            pathlib.Path(log_dir).mkdir(mode=0o775, parents=True, exist_ok=True)
            logging.info("Created dir {}".format(log_dir))
    except Exception as err:
        logging.error("Creating dir ({}) IO error: {}".format(log_dir, err))
        sys.exit(1)
        
        
    try:
        if not os.path.isdir(log_dir):
            pathlib.Path(log_dir).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        logging.error('creating "%s" directory failed, error "%s"'%(log_dir, err))
        exit(1)    
        
    # new logging file handler
    log_fn = f"{log_dir}/_dbg_{script_name}_{time_str}_pid{os.getpid()}.log"    
    fileh = logging.FileHandler(log_fn, 'w')
    fileh.setFormatter(logging.Formatter(log_format))

    log = logging.getLogger()  # root logger
    logging_level = args.logging_level
    log.setLevel(logging_level)
    for hdlr in log.handlers[:]:  # remove all old handlers
        if(type(hdlr) is logging.FileHandler):
            old_log_fn = hdlr.baseFilename 
            hdlr.close()
            log.removeHandler(hdlr)
            lines = open(old_log_fn, 'r').read()
            os.remove(old_log_fn)
            fileh.stream.writelines(lines)
    log.addHandler(fileh)      # set the new handler

    # start new logging
    logging.info("Redirect log stream to {}".format(fileh.baseFilename))
          
    logging.info('-' * 50)
        
    #----------------------------------------------------------------------------
    # DATASET arguments
    logging.info(" Parse dataset configuration arguments...")
    ds_args, rem_args = MRIDataset.parse_arguments(rem_args)
        
    #----------------------------------------------------------------------------
    # Model arguments
    logging.info(" Parse Flexnet configuration arguments...")
    fn_args, rem_args = ResNetFlexUNet.parse_arguments(rem_args)

    #----------------------------------------------------------------------------    
    logging.info("-" * 50)

    eval_output_files_sufix = ""#"_unetV{}".format(out_ver_mark)

    # print configuration:
    logging.info("Read configuration:")
    logging.info(" model_state_dict_path: {}".format(mo_args.model))
    logging.info(" output dir:            {}".format(work_dir))
    logging.info("                          ") 
    logging.info(" force_cpu:             {}".format(args.force_cpu))
    logging.info(" force_gpu:             {}".format(args.force_gpu_id))
    logging.info(" outputs sufix:         {}".format(eval_output_files_sufix))
    logging.info("                          ") 
    logging.info(" export_labels          {}".format(args.export_labels      ))
    logging.info(" export_polygons        {}".format(args.export_polygons    ))
    logging.info(" export_box             {}".format(args.export_box         ))
    logging.info(" export_prob            {}".format(args.export_prob        ))
    logging.info(" export_prob_nl         {}".format(args.export_prob_nl     ))
    logging.info(" export_masks           {}".format(args.export_masks       ))
    logging.info(" export_clasStats       {}".format(args.export_clasStats   ))
    logging.info(" export_clasStatsPng    {}".format(args.export_clasStatsPng))
    logging.info(" export_dbg_raw         {}".format(args.export_dbg_raw     ))
    logging.info(" export_dbg             {}".format(args.export_dbg         ))
    logging.info(" export_pngs_cropped    {}".format(args.export_pngs_cropped))
    logging.info("                          ") 

    if len(args.export_clss_filter_pass_list) == 0:
        args.export_clss_filter_pass_list = copy.deepcopy(ds_args.ds_polygon_clss)
    logging.info(" export_clss_filter_pass_list {}".format(args.export_clss_filter_pass_list ))
    for t in args.export_clss_filter_pass_list:
        if(not t in ds_args.ds_polygon_clss):
            logging.warning(" class '{}' is in export_clss_filter_pass_list but the model ({}) does not classify this class (ds_args.ds_polygon_clss = {}). I will skip this one".format(t, args.model, ds_args.ds_polygon_clss))
            args.export_clss_filter_pass_list.remove(t)
            logging.info(" export_clss_filter_pass_list {}".format(args.export_clss_filter_pass_list ))

    logging.info(" export_clss_filter_stop_list{}".format(args.export_clss_filter_stop_list))
    for t in args.export_clss_filter_stop_list:
        if(not t in ds_args.ds_polygon_clss):
            logging.warning(" class '{}' is in export_clss_filter_stop_list but the model ({}) does not classify this class (ds_args.ds_polygon_clss = {}). I will skip this one".format(t, args.model, ds_args.ds_polygon_clss))
            args.export_clss_filter_stop_list.remove(t)
            logging.info(" export_clss_filter_stop_list{}".format(args.export_clss_filter_stop_list))
    
    export_clss_list = copy.deepcopy(args.export_clss_filter_pass_list)
    if(len(args.export_clss_filter_stop_list) != 0):
        export_clss_list = [x for x in export_clss_list if not (x in args.export_clss_filter_stop_list)]
    logging.info(" export_clss_list {}".format(export_clss_list))
    if(len(export_clss_list) == 0):
        logging.error("After parsing export_clss_filter_pass_list and export_clss_filter_stop_list, the result export_clss_list has no entries! Nothing to export!")
        sys.exit(9)

    if(args.limit_polygons_num > 0):
        logging.info(" limit polygons number per class: {}".format(args.limit_polygons_num))
    logging.info("                          ") 
    logging.info(" dataset cfg:")
    print_cfg_dict(vars(ds_args), indent = 2, skip_comments = True)
    logging.info(" flexnet cfg:")
    print_cfg_dict(vars(fn_args), indent = 2, skip_comments = True)
    
    
    #----------------------------------------------------------------------------
    obsolete_parser = ArgumentParser()
    obsolete_parser.add_argument("--translated_pxpy"                               , default=None                , type=int,   nargs=2  , required=False, metavar='I'   , help="Zamiast tego uzyj --descr_fn i --ds_polyRef_descr_fn")
    obsolete_args, rem_args = obsolete_parser.parse_known_args(rem_args)
    
    #----------------------------------------------------------------------------
    if(len(vars(obsolete_args)) != 0):
        logging.warning("After all modules parsed own arguments some obsolete arguments are left")
        logging.info(" obsolete args:")
        print_cfg_dict(vars(obsolete_args), indent = 2, skip_comments = True)
    #----------------------------------------------------------------------------
    if(len(rem_args) != 0):
        logging.error("After all modules parsed own arguments some arguments are left: {}.".format(rem_args))
        sys.exit(10)
    #----------------------------------------------------------------------------
    # platform specific features
    
    logging.info("-" * 50)
    logging.info("Platform specific features...")
    if args.force_cpu:
        device = torch.device('cpu')
    elif (not args.force_gpu_id is None):
        list_of_gpus = [torch.cuda.get_device_name(cid) for cid in range(torch.cuda.device_count())]
        gpu_id = args.force_gpu_id
        if(torch.cuda.is_available() and torch.cuda.device_count() > gpu_id):
            device = torch.device('cuda:{}'.format(gpu_id))
            name_device = torch.cuda.get_device_name(gpu_id)
            logging.info("Calculations forced on GPU ({}).".format(name_device))
        elif not torch.cuda.is_available():
            logging.warning("Calculations forced on GPU{}, but no CUDA dev is available. Change to CPU".format(gpu_id))
            device = torch.device('cpu')
        else:
            logging.error("Calculations forced on GPU{}, but GPU{} is no pressent. Available CUDA GPUs: {}".format(gpu_id, gpu_id, list_of_gpus))
            system.exit(-1)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info("Device: {}".format(device))

    system_name = os.name
    if system_name != 'nt': #not Windows - require to specify that process start with spawn method
        multiprocessing.set_start_method('spawn')

    if(args.force_single_thread):
        use_multiprocessing = False
    elif (not ds_args.force_order is None) and (ds_args.force_order != "no"):
        logging.info("Order of dataset processing is set to {}. Disable multiprocessing".format(ds_args.force_order))
        use_multiprocessing = False
    elif(device.type=='cpu'):
        if 'pydevd_concurrency_analyser.pydevd_thread_wrappers' in sys.modules:
            logging.info("Running in Visual Studio. Disable multiprocessing")
            use_multiprocessing = False
        else:
            logging.info("Running outside Visual Studio. Can use multiprocessing")
            use_multiprocessing = True#False
        total_memory = sum(get_gen_mem_used_free())
    else: #if(device.type=='cuda'):
        use_multiprocessing = False#True
        cuda_total_memory = [torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())]
    logging.info("Use multiprocessing = {}".format(use_multiprocessing))

    #-----------------------------------------------------------
    logging.info("-"*50)
    logging.info("Creating dataset...")
    whole_set = MRIDataset(ds_args)
    logging.info("- "*25)

    dataset_entries_without_comp_fn = "{}/_entries_without_comp.json".format(work_dir)
    len_entries_without_comp = len(whole_set._entries_without_comp)
    if(len_entries_without_comp != 0):
        logging.info("Found {} inputs during dataset preparation that lack at least one of the input components.".format(len_entries_without_comp))
        logging.info(" detail on those can be found in file \"{}\"".format(dataset_entries_without_comp_fn))
        jsonDumpSafe(dataset_entries_without_comp_fn, whole_set._entries_without_comp)
    
    #dataset_entries_without_ref_fn = "{}/_entries_without_ref.json".format(work_dir)
    #len_entries_without_ref = len(whole_set._entries_without_ref)
    #if(len_entries_without_ref != 0):
    #    logging.info("Found {} inputs during dataset preparation that lack at least one of the reference polygon file.".format(len_entries_without_ref))
    #    logging.info(" detail on those can be found in file \"{}\"".format(dataset_entries_without_ref_fn))
    #    jsonDumpSafe(dataset_entries_without_ref_fn, whole_set._entries_without_ref)

    dataset_len = len(whole_set)
    if(dataset_len == 0):
        logging.error("Dataset is empty. Stop execution and exit.")
        sys.exit(1)
    else:
        logging.info("Dataset has {} entries.".format(dataset_len))
        dataset_paths_valid_fn = "{}/_paths_valid.json".format(work_dir)
        logging.info(" detail on those can be found in file \"{}\"".format(dataset_paths_valid_fn))
        jsonDumpSafe(dataset_paths_valid_fn, whole_set.paths)
        
    #----------------------------------------------------------------------------
    # status json
    status_log_dict = {
        "t": round(time.time() * 1000),
        "P": 0,
        "N": dataset_len,
        }
    status_log_fn = "_job_status.json"
    logging.info(" Create job status json {}".format(status_log_fn))
    status_log_pth = os.path.normpath(os.path.join(work_dir, status_log_fn))
    jsonDumpSafe(status_log_pth, status_log_dict)
    #----------------------------------------------------------------------------
    # MODEL
    logging.info('-' * 50)
    logging.info("Creating model {}...".format("ResNetFlexUNet"))

    model = ResNetFlexUNet(fn_args).to(device)

    #from torchsummary import summary
    logging.info("Network summary (from torchsummary):")
    dx, dy = whole_set.paths[0]['src_comps_size']
    num_in_comp = len(whole_set.paths[0]['src_comps_path_l'])

    old_stdout = sys.stdout # Memorize the default stdout stream
    sys.stdout = buffer = io.StringIO()

    #if(device.type=='cuda'):
    #    if(device.index == 0):
    #        summary(model, input_size=(num_in_comp, dx, dy), device = "cuda") 
    #    else:
    #        logging.info("function torchvision.summary() does not work for GPU other than 0. Simply print model structure using logging.info(model)")
    #        logging.info(model)
    #elif(device.type=='cpu'):
    #    summary(model, input_size=(num_in_comp, dx, dy), device = "cpu") 

    sys.stdout = old_stdout # Put the old stream back in place
    logging.info(buffer.getvalue()) # Return a str containing the

   
    #----------------------------------------------------------------------------
    # read model
    
    logging.info('=' * 50)
    logging.info("Try initialize model with model_state_dict from  {}...".format(mo_args.model))
    
    if(mo_args.model != "skip_model"):
        #load_model(model_state_dict_path, model, device)
        load_model_from_state_dict(model_state_dict, model)
    
    # freeze backbone layers
    childeren_layers = list(model.children()) 
    for l in childeren_layers:
        for param in l.parameters():
            param.requires_grad = False
    
    #----------------------------------------------------------------------------
    logging.info('=' * 50)

    out_box = []

    # evaluate on the whole dataset
    since = time.time()
    logging.info("Starting evaluation ...")   

    if not use_multiprocessing:
        pqueue = Queue()
        use_model_on_dataset_images(model, device, whole_set, eval_output_files_sufix, work_dir, 
                                    export_labels       = args.export_labels,      
                                    export_polygons     = args.export_polygons,    
                                    export_box          = args.export_box,         
                                    export_prob         = args.export_prob, 
                                    export_prob_nl      = args.export_prob_nl, 
                                    export_masks        = args.export_masks,       
                                    export_clasStats    = args.export_clasStats,   
                                    export_clasStatsPng = args.export_clasStatsPng,
                                    export_dbg_raw      = args.export_dbg_raw,     
                                    export_dbg          = args.export_dbg,         
                                    export_pngs_cropped = args.export_pngs_cropped,
                                    limit_polygons_num  = args.limit_polygons_num, 
                                    fill_polygons_holes = args.fill_polygons_holes, 
                                    fill_labels_holes   = args.fill_labels_holes  , 
                                    fill_masks_holes    = args.fill_masks_holes   , 
                                    export_clss_list = export_clss_list,
                                    out_queue           = pqueue,
                                    threshold_level     = args.threshold_level,
                                    useClamp01NotSigmoid= args.useClamp01NotSigmoid,
                                    status_log_dict = status_log_dict,
                                    status_log_pth  = status_log_pth)
        logging.debug(" done.")
        if(args.export_box):
            logging.debug("Get box from queue...")
            out_box = pqueue.get()
            logging.info(" done ({})". format(out_box))
                        
    else:
        num_threads = min(int(multiprocessing.cpu_count()-1), int(len(whole_set)/4)) 
        len_per_thread = int(len(whole_set) / num_threads)
        div_table = [len_per_thread for x in range(num_threads)]
        rem_len = len(whole_set)-(num_threads)*(len_per_thread)
        for i in range(rem_len):
            div_table[i%num_threads] = div_table[i%num_threads] + 1
            
        logging.debug(" Divide dataset between threads: {}".format(div_table))                              
        list_of_dss = torch.utils.data.random_split(whole_set, div_table)
        processes = []
                        
        if(device.type=='cuda'):
            memory_used_b, memory_free_b = get_cuda_mem_used_free(device.index) 
            torch.cuda.empty_cache()
            memory_used, memory_free = get_cuda_mem_used_free(device.index)
            logging.info(" Try to free GPU memory before evaluation phase. Allocated memory {:>2.3}->{:>2.3}GB, free memory {:>2.3}->{:>2.3}GB)".format(memory_used_b/1024.0, memory_used/1024.0, memory_free_b/1024.0, memory_free/1024.0))
        else:
            memory_used, memory_free = get_gen_mem_used_free()
            logging.info(" Allocated memory {:>2.3}GB, free memory {:>2.3}GB".format(memory_used/1024.0, memory_free/1024.0))
        
        pqueue = [Queue() for x in range(num_threads)]
        out_box_pr = [[] for x in range(num_threads)]
        for pid, ds in enumerate(list_of_dss):
            my_args = (model, device, ds, eval_output_files_sufix, work_dir, 
                       args.export_labels,      
                       args.export_polygons,    
                       args.export_box,         
                       args.export_prob,       
                       args.export_prob_nl,           
                       args.export_masks,     
                       args.export_clasStats,   
                       args.export_clasStatsPng,
                       args.export_dbg_raw,     
                       args.export_dbg,         
                       args.export_pngs_cropped,
                       args.limit_polygons_num, 
                       args.fill_polygons_holes, 
                       args.fill_labels_holes  , 
                       args.fill_masks_holes   , 
                       export_clss_list,
                       pqueue[pid],
                       args.threshold_level,
                       status_log_dict,
                       status_log_pth)
            process = Process(target=use_model_on_dataset_images, args=my_args, name="Eval_proc_{:02}".format(pid))
            process.daemon = True
            processes.append(process)
        
        min_mem_req = 1000
        max_mem_req = min_mem_req  
        first_try = True
        for process in processes:
            mem_str = ""
            #if(device.type=='cuda'):
            #    for i in range(torch.cuda.device_count()):
            enough_mem = False
            memory_free_prev = 999999999
            while(not enough_mem):
                memory_used, memory_free = get_cuda_mem_used_free(device.index) if(device.type=='cuda') else get_gen_mem_used_free() 
                                         
                if(memory_free > max_mem_req*1.2 or (memory_free_prev<=memory_free and memory_free > max_mem_req)):
                    mem_str = " {} has {:>2.03}GB of free memory.".format(device, memory_free/1024)
                    enough_mem = True
                    memory_free_prev = 999999999
                else:
                    logging.info(" {} has {:>2.03}GB of free memory. Wait...".format(device, memory_free/1024))
                    memory_free_prev = memory_free
                    time.sleep(3)
            logging.info("{} Starting evaluation process {}...".format(mem_str, process.name))#, end='')
            process.start()
            if first_try:#if (device.type=='cuda') and first_try:
                memory_free_lowest = memory_free
                x=0
                while(x<15):
                    x+=1
                    time.sleep(0.1)
                    memory_used_new, memory_free_new = get_cuda_mem_used_free(device.index) if(device.type=='cuda') else get_gen_mem_used_free()
                    if(memory_free_new < memory_free_lowest):
                        memory_free_lowest = memory_free_new
                        max_mem_req = 1.1*(memory_free - memory_free_lowest)
                        if(max_mem_req < min_mem_req): 
                            max_mem_req = min_mem_req  
                        x-=1
                logging.info("  mem required for the first thread = {:>2.03}GB".format((memory_free - memory_free_lowest)/1024))                              
                first_try = False
            else:
                if not (memory_free > (max_mem_req * 2 * num_threads)): #for large memory do not wait (speed up for Tesla with 128GB memory)
                    time.sleep(1)
        for pid, process in enumerate(processes):
            process.join()
            logging.info(" Joined process {} with exit code = {}...".format(process.name, process.exitcode))
            if(process.exitcode != 0):
                logging.error(' Child process returned with error {} (process = {})'.format(process.exitcode, process))
                logging.error(' Stop execution and exit')
                sys.exit(1)
            else:
                logging.debug("  Get box from queue...")
                out_box_pr[pid] = pqueue[pid].get()
                logging.info('   Box {}'.format(out_box_pr[pid]))
        if args.export_box:        
            logging.debug("  Combine boxes from subprocesses...")
            for pid in range(0, num_threads):
                out_box_curr = out_box_pr[pid]
                if(len(out_box) == 0):
                    out_box =  out_box_curr
                else:
                    try:
                        if(len(out_box_curr)==4):
                            out_box[0]= min(out_box[0], out_box_curr[0])
                            out_box[1]= min(out_box[1], out_box_curr[1])
                            out_box[2]= max(out_box[2], out_box_curr[2])
                            out_box[3]= max(out_box[3], out_box_curr[3])
                        else:
                            logging.warning(" box ({}) is not valid!".format(out_box_curr))
                    except:
                        logging.warning(' Could not combine current box {} with a singe subprocess box {}.'.format(out_box, out_box_curr))

    
                
    if(args.export_box):      
        box_fn = os.path.normpath(work_dir + "/box.json")
        box_dict = {"box":out_box}
        logging.info(' Dump session box ({}) to json file {}.'.format(box_dict, box_fn))
        jsonDumpSafe(box_fn, box_dict)
    
    time_elapsed = time.time() - since
    logging.info(' Eval time {:2.0f}m:{:02.0f}s.'.format(time_elapsed // 60, time_elapsed % 60))
    
    status_log_dict["T"] = round(time.time() * 1000)
    jsonUpdate(status_log_pth, status_log_dict)    
    #----------------------------------------------------------------------------
    # final EVALUATION

    logging.info("script ends @ {}".format(time.ctime()))
    logging.info('*' * 50)

if __name__ == '__main__':
    main()
