#!/usr/bin/env python
# coding: utf-8

# pip/conda libs:
import os, sys, io
import pathlib
import time
from datetime import datetime
#import matplotlib.pyplot as plt
import numpy as np
import logging
import torch
from torch.cuda import amp
from torch.utils.data import Dataset, DataLoader
import copy
from PIL import Image, ImageFont, ImageDraw
import json
from argparse import ArgumentParser
from multiprocessing import Process, Queue
import multiprocessing
import itertools
import random
import math
from pandas.core.common import flatten
import shutil
from collections import defaultdict
import re


#-----------------------------------------------------------------------------------------
curr_script_path = os.path.dirname(os.path.abspath(__file__))
flexnet_path = os.path.normpath(os.path.join(curr_script_path, ".."))
flexnet_host_path = os.path.normpath(os.path.join(flexnet_path, ".."))
sys.path.append(flexnet_host_path)
#-----------------------------------------------------------------------------------------
# flexnet libs:
from flexnet.model.pytorch_resnet_flex_unet import ResNetFlexUNet
from v_utils.v_dataset import MRIDataset, expand_session_dirs
import flexnet.utils.img_helper as img_helper
from flexnet.training.loss import calc_loss
from flexnet.utils.gen_unet_utils import load_model, get_cuda_mem_used_free, get_gen_mem_used_free, save_model_with_cfgs, save_checkpoint, load_checkpoint
from v_utils.v_arg import print_cfg_list, print_cfg_dict, arg2boolAct
from v_utils.v_arg import convert_dict_to_cmd_line_args, convert_cmd_line_args_to_dict, convert_cfg_files_to_dicts
from v_utils.v_contour import *
from v_utils.v_polygons import *
from v_utils.v_json import jsonUpdate, jsonDumpSafe
from flexnet.evaluation.v_classification_stat import classification_stats_for_single_img, classification_stats_for_single_img_np

#----------------------------------------------------------------------------
# fast evaluation using global structures
sample_cls_masks_ref_rgb   = []
sample_cls_masks_pred      = []
sample_images_in_rgb       = []
#----------------------------------------------------------------------------
# TRAINING FUNCTIONs


def sprint_metrics(metrics):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] ))

    out_str = " {}".format(", ".join(outputs))
    return out_str

        
def train_single_epoch(model, work_dir, device, optimizer, scheduler, dataloaders, phases = ['train', 'val'],
                       class_num_loss_weights = None, class_pos_loss_weights = None,
                       margin_loss_weight = 1.0, scalerAMP = None, _epoch_id = -1, do_dbg_nans = False, 
                       frame_loss_weights = [],
                       export_bin_imgs = False, bin_images_subdir = "", _tot_F1 = None, 
                       binarization_level = 0.5):

    global sample_cls_masks_pred

    dataset = dataloaders['train'].sampler.data_source.dataset
    cls_num = dataset.num_class
    clss_names = [x['cls_name'] for x in dataset.paths[0]['cls_envs_list']]
    train_state = {}
    mem_cuda_save_after_batch = False
    mem_cuda_save_after_train = True
    is_reeval = False

    # Each curr_epoch has a training and validation phase
    while len(phases) != 0:  
        phase, phases = phases[0], phases[1:]
        num_minibatches = len(dataloaders[phase])
        logging.info("  phase {}, {} minibatch(es) of size {}:".format(phase, num_minibatches, dataloaders[phase].batch_size))
        # Set model to training or evaluation mode
        if phase == 'train':
            model.train()
        else:
            model.eval()

        metrics = defaultdict(float)
        epoch_imgs = 0
        epoch_refs = 0
        epoch_valid_refs_per_cls_name = {}
        for cls_name in clss_names:
            epoch_valid_refs_per_cls_name[cls_name] = 0
        epoch_valid_refs_total = 0
        if (phase=='val'):
            metrics_bins_dict = {}
            for cls_name in clss_names:
                metrics_bins_dict[cls_name] = []
        minibatch_id = 0
        old_perc_done = 0
        for images_in, cls_masks_ref, envs in dataloaders[phase]:
            
            minibatch_id += 1
            
            images_in = images_in.to(device)
            cls_masks_ref = cls_masks_ref.to(device) 

            #check if all images have all required reference polygons
            # first build flattened list of reference paths:
            cls_ref_pth_mb_l = list(itertools.chain(*[ cls_envs['ref_polygon_path'] for cls_envs in envs['cls_envs_list']]))
            #check if those are not empty strings
            has_refs = np.array([ref_path!= "" for ref_path in cls_ref_pth_mb_l])
            minibatch_imgs_num = images_in.shape[0]
            minibatch_valid_refs_total = 0
            minibatch_valid_refs_per_cls_name = {}
            for cls_envs in envs['cls_envs_list']:
                minibatch_valid_refs_per_cls_name[cls_envs["cls_name"][0]] = np.sum([ref_pth != '' for ref_pth in cls_envs['ref_polygon_path']])
                minibatch_valid_refs_total += minibatch_valid_refs_per_cls_name[cls_envs["cls_name"][0]]
            #logical and
            has_all_refs = has_refs.all()
            
            if(not class_num_loss_weights is None):
                cls_num_loss_weightsT = torch.as_tensor(class_num_loss_weights, dtype=torch.float16, device=device)
            else:
                cls_num_loss_weightsT = None
                
            if(not class_pos_loss_weights is None):
                cls_pos_loss_weightsT = torch.as_tensor(class_pos_loss_weights, dtype=torch.float16, device=device)
            else:
                cls_pos_loss_weightsT = None

            if(not has_all_refs):
                cls_valid_ref_np = np.ones(cls_masks_ref.shape[:2], dtype=np.int8)
                for img_id in range(minibatch_imgs_num):
                    for cls_id in range(cls_num):
                        if envs['cls_envs_list'][cls_id]['ref_polygon_path'][img_id] == '':
                            cls_valid_ref_np[img_id, cls_id] = 0
                        else:
                            cls_valid_ref_np[img_id, cls_id] = 1
                cls_valid_refT = torch.as_tensor(cls_valid_ref_np, dtype=torch.int8, device=device)
            else:
                cls_valid_refT = None

            if((margin_loss_weight != 1.0) or ((phase == 'train') and (len(frame_loss_weights) != 0))):
                #logging.warning("Not all items in minibatch has corresponding ref masks. Use masked loss calculation")
                #maskT = torch.ones(cls_masks_ref.shape, dtype=torch.float16, device=device)
                mask_np = np.ones(cls_masks_ref.shape, dtype=np.float16)
                for img_id in range(minibatch_imgs_num):
                    for cls_id in range(cls_num):
                        if envs['cls_envs_list'][cls_id]['ref_polygon_path'][img_id] == '':
                            mask_np[img_id, cls_id, :, :] = 0
                        else:
                            if(margin_loss_weight != 1.0):
                                mmpd = 0.5
                                margin_mm = 1.5
    
                                margin_points = margin_mm/mmpd/2.0
                                margin_points_ceil = np.int16(np.ceil(margin_points))
                                
                                #with open (envs['cls_envs_list'][cls_id]['ref_polygon_path'][img_id]) as f:
                                #    ref_polygons_dict= json.load(f)
                                #ref_polygons = v_polygons.from_dict(ref_polygons_dict)
                                #h, w = cls_masks_ref.shape[-2:]
                                #
                                ## create polygons at a margin of the reference poygons - in that area we DO NOT require match between estimated and reference results
                                #ref_polygons_margin    = v_polygons.from_polygons_borders(ref_polygons, dilation_radius = margin_points)
                                #ref_polygons_margin_np = ref_polygons_margin.as_numpy_mask(fill=True, w=w, h=h, val=1, masks_merge_type='or')
                                
                                cls_ref_polygons_np = cls_masks_ref[img_id, cls_id, :, :].data.cpu().numpy()
                                cls_ref_polygons_np = (cls_ref_polygons_np*255).astype(np.uint8)
                                disc = v_polygons.disk_fr(float_radius=margin_points)
                                polygon_dil_np = skimage.morphology.dilation(cls_ref_polygons_np, disc)
                                polygon_ero_np = skimage.morphology.erosion (cls_ref_polygons_np, disc)
                                ref_polygons_margin_np = np.where((polygon_dil_np != 0) & (polygon_ero_np == 0), np.uint8(255), np.uint8(0))
                                
                                #img_Image = Image.fromarray(cls_ref_polygons_np)
                                #img_Image.save("mask_in.png")
                                #img_Image = Image.fromarray(ref_polygons_margin_np)
                                #img_Image.save("mask_out.png")

                                mask_np[img_id, cls_id, :, :] = np.where((ref_polygons_margin_np != 0), margin_loss_weight, 1.0)
                if ((phase == 'train') and (len(frame_loss_weights) != 0)):
                    l = len(frame_loss_weights)
                    mask_s = mask_np.shape[2:]
                    frame_loss_weights_r = list(reversed(frame_loss_weights))
                    mask_np[:, :,   :  ,  0: l] *= frame_loss_weights
                    mask_np[:, :,   :  , -l:  ] *= frame_loss_weights_r
                    mask_np[:, :,  0: l,   :  ] *= np.array(frame_loss_weights  )[:, np.newaxis]
                    mask_np[:, :, -l:  ,   :  ] *= np.array(frame_loss_weights_r)[:, np.newaxis]

                loss_maskT = torch.as_tensor(mask_np, dtype=torch.float16, device=device)
            else:
                loss_maskT = None

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                epoch_imgs += minibatch_imgs_num
                epoch_refs += minibatch_imgs_num * cls_num
                for cn in clss_names:
                    epoch_valid_refs_per_cls_name[cn] += minibatch_valid_refs_per_cls_name[cn]
                epoch_valid_refs_total += minibatch_valid_refs_total
                
                with torch.cuda.amp.autocast(enabled = (not scalerAMP is None)):
                    cls_preds_out_l = model(images_in)#.squeeze()
                    
                    # wylowienie wyniku rysowanego na "_out_e###.png"
                    if(phase=='val'):
                        if (minibatch_id <= len(sample_cls_masks_pred)):
                            sample_pred_dev = torch.sigmoid(cls_preds_out_l)
                            sample_pred = sample_pred_dev.data.cpu().numpy()[0]
                            del sample_pred_dev
                            sample_cls_masks_pred[minibatch_id-1] = sample_pred
                            
                    if (phase=='train'):
                        loss = calc_loss(cls_preds_out_l, cls_masks_ref, metrics, out_weight_mask = loss_maskT, cls_valid_ref = cls_valid_refT,
                                         cls_num_loss_weights = cls_num_loss_weightsT, 
                                         cls_pos_loss_weights = cls_pos_loss_weightsT,
                                         clss_names = clss_names, do_dbg_nans = do_dbg_nans)

                    elif (phase=='val'): 
                        
                        cls_preds_out_l_sigT = torch.sigmoid(cls_preds_out_l.to(dtype=torch.float32))
                        cls_preds_out_l_sig_np = cls_preds_out_l_sigT.data.cpu().numpy()
                        del cls_preds_out_l_sigT
                        for image_id in range(cls_preds_out_l_sig_np.shape[0]): # should be just one because batch_size=1
                            cls_masks_ref_np = cls_masks_ref[image_id].cpu().numpy()
                            cls_preds_out_np = cls_preds_out_l_sig_np[image_id]
                            #general_work_file_pattern = envs['general_work_file_pattern'][image_id]
                            w = cls_preds_out_np[0].shape[1]
                            h = cls_preds_out_np[0].shape[0]
                            fn_pattern = envs['file_name_pattern'][image_id]
                            ssd = envs['session_sub_dir'][image_id]
                            session_file_dir = os.path.normpath(os.path.join(work_dir, ssd))

                            for cls_id in range(len(cls_preds_out_np)):
                                
                                if(cls_valid_refT is None) or (cls_valid_refT[image_id,cls_id]):
                                    cls_envs_dict = envs['cls_envs_list'][cls_id]
                                    cls_name = cls_envs_dict['cls_name'][image_id]

                                    cls_mask_ref_np = cls_masks_ref_np[cls_id]
                                    cls_pred_out_np = cls_preds_out_np[cls_id]
                                    # as <0-255> image
                                    if((cls_pred_out_np.dtype is np.dtype('float32')) or (cls_pred_out_np.dtype is np.dtype('float16'))):
                                        cls_pred_out_np = (cls_pred_out_np * 255.999).astype(np.uint8)

                                    # applay tresholded to the mask - create binary mask
                                    cls_pred_out_binary_np = np.where(cls_pred_out_np[:,:] > int(round(255*binarization_level,0)), np.uint8(255), np.uint8(0))

                                    metrics_bin, metric_bin_img = classification_stats_for_single_img_np(cls_pred_out_binary_np, cls_mask_ref_np, write_imgs = export_bin_imgs)
                                    metrics_bins_dict[cls_name].append(metrics_bin)

                                    if(export_bin_imgs):
                                        cls_work_file_dir = os.path.normpath(os.path.join(work_dir, bin_images_subdir, ssd, cls_name))
                                        #logging.info("@{}:  Work dir for class: {}".format(os.getpid(), cls_work_file_dir))
                         
                                        if not os.path.isdir(cls_work_file_dir):
                                            pathlib.Path(cls_work_file_dir).mkdir(mode=0o775, parents=True, exist_ok=True)
                                            logging.info("Created dir %s"%(cls_work_file_dir))

                                        cls_work_file_pattern = os.path.normpath(os.path.join(cls_work_file_dir, "_cs_" + fn_pattern + "_{}".format(cls_name)))
                                        #logging.info("Binary prediction metrics: {}".format(metric_bin_img))
                                        
                                        draw = ImageDraw.Draw(metric_bin_img)
                                        fsize = 16
                                        curr_script_path = os.path.dirname(os.path.abspath(__file__))
                                        font = ImageFont.truetype(os.path.join(curr_script_path, "consola.ttf"), fsize)
                                        width, height = metric_bin_img.size
                                        text = "Best F1 @e{}\n {:1.03f} (c{:1.03f})".format(_epoch_id, _tot_F1, metrics_bin["F1"])
                                        draw.text((            2,                    2), text,(0,0,255), stroke_width=1, stroke_fill=(0,0,0), font=font)
                                        draw.text((width-2*fsize, height - 2 - 1*fsize), "TP",(0,190,0), stroke_width=1, stroke_fill=(0,0,0), font=font)
                                        draw.text((width-2*fsize, height - 2 - 2*fsize), "FN",(190,0,0), stroke_width=1, stroke_fill=(0,0,0), font=font)
                                        draw.text((width-2*fsize, height - 2 - 3*fsize), "FP",( 90,0,0), stroke_width=1, stroke_fill=(0,0,0), font=font)
                                        #metric_bin_img.save("{}.png".format(cls_work_file_pattern))                         
                                                                                                                            
                                        #combine results into single PNG
                                                                        
                                        img_cmp0_np = (images_in[image_id][0].cpu().numpy()* 255).astype(np.uint8)
                                        img_cmp0_rgb = np.dstack([img_cmp0_np, img_cmp0_np, img_cmp0_np])  # stacks 3 h x w arrays -> h x w x 3
                                        img_cmp0_pil = Image.fromarray(img_cmp0_rgb)
                                        imgs = [img_cmp0_pil, metric_bin_img]
                                        widths, heights = zip(*(i.size for i in imgs))

                                        total_width = sum(widths)
                                        max_height = max(heights)

                                        cmb_img = Image.new('RGB', (total_width, max_height))

                                        x_offset = 0
                                        for im in imgs:
                                          cmb_img.paste(im, (x_offset,0))
                                          x_offset += im.size[0]

                                        cmb_img.save("{}.png".format(cls_work_file_pattern))
                                    
                            
                    if mem_cuda_save_after_batch and not loss_maskT is None:
                        del loss_maskT

                # NaN debugging
                if(do_dbg_nans and math.isnan(metrics['loss'])):
                    logging.error("loss == NaN! Debugging ...")

                    logging.info(" remove all ZIPs from work_dir {}".format(work_dir))
                    remove_all_zips_from_dir(work_dir)

                    try:
                        logging.info("Try to find NaNs source ...")
                        pred_hasNaN = torch.isnan(cls_preds_out_l).any()
                        pred_hasInf = torch.isinf(cls_preds_out_l).any()
                        if(pred_hasNaN or pred_hasInf):
                            # wywalilo sie juz na etapie licznia odpowiedzi sieci - sprawdz na ktorej warstwie
                            with torch.cuda.amp.autocast(enabled = (not scalerAMP is None)):
                                logging.info("Do verbose model prediction using model.forward_anomaly_det function ...")
                                cls_preds_out_l_new = model.forward_anomaly_det(images_in)
                        else: #
                            # jezeli to byla faza walidacji to gradienty byly wylaczone. Zeby zadzialal mechanizm detect_anomaly trzeba wlaczyc gradienty. Dla fazy treningu nie powinno to niczego zmienić
                            #if phase != 'train':
                            with torch.set_grad_enabled(True):
                                with torch.cuda.amp.autocast(enabled = (not scalerAMP is None)):
                                    with torch.autograd.detect_anomaly():
                                        logging.info("Do verbose model prediction using model.forward_anomaly_det function ...")
                                        cls_preds_out_l_new = model.forward_anomaly_det(images_in)
                                        loss = calc_loss(cls_preds_out_l_new, cls_masks_ref, metrics, out_weight_mask = loss_maskT, cls_num_loss_weights = cls_num_loss_weightsT, cls_pos_loss_weights = cls_pos_loss_weightsT, clss_names = clss_names, do_dbg_nans = do_dbg_nans)
                                        
                                        logging.info("Do verbose backward propagation with torch.autograd.detect_anomaly enabled ...")
                                        if scalerAMP is None: 
                                            loss.backward()
                                        else:
                                            scalerAMP.scale(loss).backward()
                    
                    except Exception as err:
                        logging.error("Error while debbuging NaNs:")
                        logging.error(err)

                    logging.info("Dump various output files for current model state...")
                    sufix = "_nan_dbg_e{}_ph_{}".format(_epoch_id, phase)
                    export_labels       = True  
                    export_polygons     = True 
                    export_box          = True
                    export_prob         = True  
                    export_masks        = False  
                    export_dbg_raw      = True 
                    export_dbg          = True 
                    export_pngs_cropped = False
                    limit_polygons_num  = 0 
                    export_clss_list = None
                    test_envs = envs
                    dataset_has_ref_masks = True
                    out_box = []
                    # get <0.0, 1.0> range prediction - is it needed?
                    logging.info("@{}:  Use sigmoid on the prediction output".format(os.getpid()))
                    cls_preds_out_l_notSig = cls_preds_out_l.data.cpu().numpy()

                    cls_preds_out_l_sig = torch.sigmoid(cls_preds_out_l)
                    cls_preds_out_l_sig = cls_preds_out_l_sig.data.cpu().numpy()
            
                    for image_id in range(cls_preds_out_l_sig.shape[0]): # should be just one because batch_size=1
                        if (export_dbg_raw or export_dbg):
                            cls_masks_ref_np = cls_masks_ref.to(device)[image_id].cpu().numpy()
                        cls_preds_out = cls_preds_out_l_sig[image_id]
                        cls_preds_out_notSig = cls_preds_out_l_notSig[image_id]
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
                                
                            logging.info("@{}:  Read envs".format(os.getpid()))
                            cls_envs_dict = test_envs['cls_envs_list'][cls_id]
                            cls_name = cls_envs_dict['cls_name'][image_id]
                            cls_work_file_dir = os.path.normpath(os.path.join(work_dir, ssd, cls_name))
                            logging.info("@{}:  Work dir for class: {}".format(os.getpid(), cls_work_file_dir))
                
                            if(not export_clss_list is None) and (not cls_name in export_clss_list):
                                logging.info("@{}:   skip operation for {} due to export_clss_list contains only {}".format(os.getpid(), cls_name, export_clss_list))
                                continue

                            try:
                                if not os.path.isdir(cls_work_file_dir):
                                    pathlib.Path(cls_work_file_dir).mkdir(mode=0o775, parents=True, exist_ok=True)
                                    logging.info("Created dir %s"%(cls_work_file_dir))
                            except Exception as err:
                                logging.error("Creating dir (%s) IO error: %s"%(cls_work_file_dir, err))
                                sys.exit(1)
                                                        
                            cls_work_file_pattern = os.path.normpath(os.path.join(cls_work_file_dir, fn_pattern + "_{}".format(cls_name)))

                            #Generate json file with polygon of the the predicted class mask
                            cls_json_fn = cls_work_file_pattern + "{}_polygons.json".format(sufix)
                            logging.info("@{}:  Generate json file with polygon of the the predicted class mask: {}...".format(os.getpid(), cls_json_fn))

                            cls_pred_out = cls_preds_out[cls_id]
                            # as <0-255> image
                            if((cls_pred_out.dtype is np.dtype('float32')) or (cls_pred_out.dtype is np.dtype('float16'))):
                                cls_pred_out = (cls_pred_out * 255.999).astype(np.uint8)

                            # applay tresholded to the mask - create binary mask
                            logging.info("@{}:   applay tresholded to the mask - create binary mask...".format(os.getpid()))
                            cls_mask_out_binary = np.zeros(cls_pred_out.shape, dtype = np.uint8)
                            cls_mask_out_binary[cls_pred_out[:,:] > 127] = 255

                            #get polygons from the binary mask
                            logging.info("@{}:   get polygons from the binary mask...".format(os.getpid()))
                            cls_polygons_out = v_polygons()
                            cls_polygons_out._mask_ndarray_to_polygons(cls_mask_out_binary, background_val = 0, limit_polygons_num = limit_polygons_num)

                            cls_polygons_out_sh = cls_polygons_out

                            
                            cls_pred_out_notSig = cls_preds_out_notSig[cls_id]
                            # as <0-255> image
                            cls_pred_out_nan_binary = np.zeros(cls_pred_out_notSig.shape, dtype = np.uint8)
                            cls_pred_out_nan_binary[np.isnan(cls_pred_out_notSig[:,:])] = 255

                            #get polygons from the binary mask
                            logging.debug("@{}:   get polygons from the binary mask...".format(os.getpid()))
                            cls_polygons_nan_out = v_polygons()
                            cls_polygons_nan_out._mask_ndarray_to_polygons(cls_pred_out_nan_binary, background_val = 0, limit_polygons_num = 0)

                            cls_polygons_nan_out_sh = cls_polygons_nan_out

                            if((np.array(test_envs['crop_box']) != np.array([0,0, *[x.item() for x in test_envs['org_size']]])).any()):
                                #reverse cropping operation
                                logging.info("@{}:   reverse cropping operation...".format(os.getpid()))
                                crop_box = test_envs['crop_box']
                                org_point = [int(crop_box[0][image_id].item()), int(crop_box[1][image_id].item())]
                                # do not overwrite cls_polygons_out because it may be needed fo PNGs export when cropped ones are to be exported
                                #  instead deepcopy cls_polygons_out
                                if(export_pngs_cropped):
                                    cls_polygons_out_sh = copy.deepcopy(cls_polygons_out)
                                cls_polygons_out_sh.move2point(org_point)

                                cls_polygons_nan_out_sh.move2point(org_point)
                    
                            if(export_polygons):
                                logging.info("@{}:   dump class polygons_dict...".format(os.getpid()))
                                cls_polygons_dict = cls_polygons_out_sh.as_dict()
                                jsonDumpSafe(cls_json_fn, cls_polygons_dict)
                
                            do_export_png = export_labels or export_prob or export_dbg_raw or export_dbg
                            if(do_export_png):
                                png_w = w
                                png_h = h

                                #reverse cropping if needed
                                if( (not export_pngs_cropped) and ((np.array(test_envs['crop_box']) != np.array([0,0, *[x.item() for x in test_envs['org_size']]])).any())):
                        
                                    logging.info("@{}:    reverse cropping...".format(os.getpid()))
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
                                logging.info("@{}:    Creating {} with labels RGB".format(os.getpid(), fn))

                                img_Image = cls_polygons_out_sh.as_image(fill = True, w=png_w,h=png_h, force_labelRGB = True)
                                img_Image.save(fn)
                                
                                fn = cls_work_file_pattern + "{}_nans.png".format(sufix)
                                logging.info("@{}:    Creating {} with nans RGB".format(os.getpid(), fn))

                                img_Image = cls_polygons_nan_out_sh.as_image(fill = True, w=png_w,h=png_h, force_labelRGB = True)
                                img_Image.save(fn)
                                

                            if(export_box):
                                logging.info("@{}:   box for current class {}".format(os.getpid(),cls_polygons_out_sh["box"]))
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

                                logging.info("@{}:   up-to-date box = {}".format(os.getpid(), out_box))
                    
                
                            cls_pred_out_sh      = cls_pred_out

                            #reverse cropping for prediction output if needed
                            if( do_export_png and (not export_pngs_cropped) and ((np.array(test_envs['crop_box']) != np.array([0,0, *[x.item() for x in test_envs['org_size']]])).any())):
                        
                                logging.info("@{}:    reverse cropping...".format(os.getpid()))
                                            
                                pad_l = int(crop_box[0]              )
                                pad_r = int(org_size[0] - crop_box[2])
                                pad_t = int(crop_box[1]              )
                                pad_b = int(org_size[1] - crop_box[3])
                                cls_pred_out_np = np.array(cls_pred_out)
                                if(pad_l >= 0 and pad_r >= 0 and pad_t >= 0 and pad_b >= 0):
                                    # padding
                                    cls_pred_out_org_size = np.pad(cls_pred_out_np, pad_width = ((pad_t, pad_b),(pad_l, pad_r)), mode='constant', constant_values=0)
                                elif(pad_l <= 0 and pad_r <= 0 and pad_t <= 0 and pad_b <= 0):
                                    # cropping
                                    cls_pred_out_org_size = cls_pred_out_np[-pad_t: int(org_size[1])-pad_t, -pad_l: int(org_size[0])-pad_l]
                                else:
                                    # combination of padding and cropping needed
                                    _pad_l = pad_l if pad_l>=0 else 0
                                    _pad_r = pad_r if pad_r>=0 else 0
                                    _pad_t = pad_t if pad_t>=0 else 0
                                    _pad_b = pad_b if pad_b>=0 else 0
                                    tmp = np.pad(cls_pred_out_np, pad_width = ((_pad_t, _pad_b),(_pad_l, _pad_r)), mode='constant', constant_values=0)
                            
                                    _pad_l = pad_l if pad_l<=0 else 0
                                    _pad_r = pad_r if pad_r<=0 else 0
                                    _pad_t = pad_t if pad_t<=0 else 0
                                    _pad_b = pad_b if pad_b<=0 else 0
                                    cls_pred_out_org_size = tmp[-_pad_t: int(org_size[1])-_pad_t, -_pad_l: int(org_size[0])-_pad_l]

                                cls_pred_out_sh = cls_pred_out_org_size
                        
                            if export_prob:
                                # save PNG with :
                                # - the output class mask (not contour, but values <0-255>)
                                fn = cls_work_file_pattern + "{}_prob.png".format(sufix)
                                logging.info("@{}:    Creating {} with output as grayscale".format(os.getpid(), fn))

                                if((cls_pred_out_sh.dtype is np.dtype('float32')) or (cls_pred_out_sh.dtype is np.dtype('float16'))):
                                    cls_pred_out_sh = (cls_pred_out_sh * 255.999).astype(np.uint8)
                                img_Image = Image.fromarray(cls_pred_out_sh)
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
                                    image_in_rgb = dataset.reverse_transform(images_in[image_id])
                                    image_in_rgb_sh  = image_in_rgb

                            cls_has_ref_mask = dataset_has_ref_masks and (test_envs["cls_envs_list"][cls_id]["ref_polygon_path"][image_id] != "")
                            do_generate_ref_mask = cls_has_ref_mask and (export_dbg_raw or export_dbg  )
                            do_generate_ref_poly = cls_has_ref_mask and (export_dbg_raw or export_dbg  )
                
                            if do_generate_ref_poly:
                                logging.info("@{}:   get ref polygons...".format(os.getpid()))
                                cls_mask_in = cls_masks_ref_np[cls_id]
                                if((cls_mask_in.dtype is np.dtype('float32')) or (cls_mask_in.dtype is np.dtype('float16'))):
                                    cls_mask_in = (cls_mask_in * 255.999).astype(np.uint8)

                                #get polygons from the binary mask
                                logging.info("@{}:     get polygons from the binary mask".format(os.getpid()))
                                cls_polygons_ref = v_polygons()
                                cls_polygons_ref._mask_ndarray_to_polygons(cls_mask_in, background_val = 0)

                                #reverse cropping if needed
                                if( (not export_pngs_cropped) and ((np.array(test_envs['crop_box']) != np.array([0,0, *[x.item() for x in test_envs['org_size']]])).any())):
                                    cls_polygons_ref_sh = copy.deepcopy(cls_polygons_ref)
                                    cls_polygons_ref_sh.move2point(org_point)
                                else:
                                    cls_polygons_ref_sh = cls_polygons_ref
                                        
                            if export_dbg_raw or export_dbg:
                                logging.info("@{}:    create numpy masks...".format(os.getpid()))
                                cls_polygons_out_as_numpy_mask = cls_polygons_out_sh.as_numpy_mask(w=png_w,h=png_h)   
                                if(cls_has_ref_mask):
                                    cls_polygons_ref_sh_as_numpy_mask = cls_polygons_ref_sh.as_numpy_mask(w=png_w,h=png_h) 

                            if export_dbg_raw:
                                # save PNG with :
                                # - the input image as R channel 
                                # - the contour of the reference class mask as G channel
                                # - the output class mask as B channel (not contour, but values <0-255>)
                                fn = cls_work_file_pattern + "{}_dbg_raw.png".format(sufix)
                                logging.info("@{}:    Creating {} with stacked masks as RGB layers".format(os.getpid(), fn))
                        
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
                                logging.info("@{}:    Creating {} with stacked masks as RGB layers".format(os.getpid(), fn))

                                if(cls_has_ref_mask):
                                    img_numpy = img_helper.stack_masks_as_rgb(image_in_rgb_sh, cls_polygons_ref_sh_as_numpy_mask, cls_polygons_out_as_numpy_mask)
                                else:
                                    dummy_component = np.zeros(cls_pred_out_sh.shape, dtype = np.uint8)
                                    img_numpy = img_helper.stack_masks_as_rgb(image_in_rgb_sh,                      dummy_component, cls_polygons_out_as_numpy_mask)
                                img_Image = Image.fromarray(img_numpy)
                                img_Image.save(fn)
                    
                        
                    logging.error("NaN debugging ends. Exit.\n\n")
                    sys.exit(7)

                if mem_cuda_save_after_batch and not images_in is None:
                    del images_in
                if mem_cuda_save_after_batch and not cls_masks_ref is None:
                    del cls_masks_ref
                if mem_cuda_save_after_batch and not envs is None:
                    del envs
                if mem_cuda_save_after_batch and not cls_preds_out_l is None:
                    del cls_preds_out_l
                
                # backward + optimize only if in training phase
                if phase == 'train':
                    
                    if scalerAMP is None: 
                        loss.backward()
                        optimizer.step()

                    else:
                        scalerAMP.scale(loss).backward()
                        scalerAMP.step(optimizer)
                        scalerAMP.update()
                if phase == 'train' and mem_cuda_save_after_batch:
                    del loss # mem saving
                if(mem_cuda_save_after_batch and device.type == 'cuda'):
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()

            new_perc_done = int((minibatch_id*100+0.5)//num_minibatches)
            if(new_perc_done == 100 or (new_perc_done - old_perc_done) > 10):
                old_perc_done = new_perc_done
                logging.info("   {}% done".format(new_perc_done))

        if(phase == 'train' and mem_cuda_save_after_train):      
            if not images_in is None:
                del images_in
            if not cls_masks_ref is None:
                del cls_masks_ref
            if not envs is None:
                del envs
            if not cls_preds_out_l is None:
                del cls_preds_out_l
            if not loss_maskT is None:
                del loss_maskT
            del loss # mem saving
            if(device.type == 'cuda'):
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()

        # statistics
                
        for k in metrics.keys():
            if(k=="loss"):
                metrics[k] = metrics[k] / epoch_imgs
            else:
                ss = k.split("loss_")
                if(len(ss) > 1 and ss[1] in epoch_valid_refs_per_cls_name.keys()):
                    cname = ss[1]
                    metrics[k] = metrics[k] /epoch_valid_refs_per_cls_name[cname]
        
        out_str = sprint_metrics(metrics)
        logging.info("  {}".format(out_str))
            
        if(epoch_valid_refs_total == 0):
            logging.warning("No minibatches have been processed!")
        if(phase=='train'):
            for k in metrics.keys():
                param_name = "{}_{}".format(k, phase)
                if(epoch_valid_refs_total == 0):
                    train_state[param_name] = 1e10
                else:
                    train_state[param_name] = metrics[k]
        elif(phase=='val'):
            for mn in ["PR", "RC", "F1", "AC", "TP", "FP", "FN", "TN"]:
                param_name = "{}_{}".format(mn, phase)
                acc = []
                #!!!val_clss_weights = class_pos_loss_weights / class_pos_loss_weights.mean()
                for cid in range(len(clss_names)):
                    cls_name = clss_names[cid]
                    metrics_bins = metrics_bins_dict[cls_name]
                    param_name_cls = "{}_{}_{}".format(mn, cls_name, phase)
                    if(epoch_valid_refs_total == 0):
                        train_state[param_name_cls] = -1
                    else:
                        ms = np.array([d[mn] for d in metrics_bins])
                        ma = np.average(ms)
                        train_state[param_name_cls] = ma
                        #acc += ma*class_num_loss_weights[cid] if (not class_num_loss_weights is None) else ma
                        acc.append(ma)
                    # loss = 1 - F1
                    if mn == "F1":
                        loss_name_cls = "{}_{}_{}".format("loss", cls_name, phase)
                        train_state[loss_name_cls] = 1 - train_state[param_name_cls]
                # total result as a geometrical mean of the classes results
                #train_state[param_name] = np.array(acc).prod()**(1.0/len(acc)) if (len(acc) > 0) else 0.0
                # total result as a mean of the classes results
                train_state[param_name] = np.mean(acc) if (len(acc) > 0) else 0.0
                # loss = 1 - F1
                if mn == "F1":
                    F1_name = "{}_{}".format("F1", phase)
                    loss_name = "{}_{}".format("loss", phase)
                    train_state[loss_name] = 1 - train_state[F1_name]

    return train_state 

def parse_pass_id(fn):
    m = re.search('_pass_id(\d+)', fn)
    try:
        return int(m.group(1))
    except:
        return -1

def parse_pass_id(fn):
    m = re.search('_pass_id(\d+)', fn)
    try:
        return int(m.group(1))
    except:
        return -1

def find_all_zips_from_dir(dir, pass_id_filter = None, required_string = None):
    pths = [os.path.normpath(os.path.join(dir, x)) for x in os.listdir(dir) if (os.path.isfile(os.path.normpath(os.path.join(dir, x))) and (x.find(".pth.zip") != -1))]
    if not pass_id_filter is None:
        pths = [p for p in pths if (parse_pass_id(p) == pass_id_filter)]
    if not required_string is None:
        pths = [p for p in pths if (p.find(required_string) != -1)]
    return pths

def remove_all_zips_from_dir(dir, pass_id_filter = None, required_string = None):
    pths = find_all_zips_from_dir(dir, pass_id_filter, required_string)
    for prev_best_model_path in pths:
        os.remove(prev_best_model_path)

def train_model(model, work_dir, device, optimizer, scheduler, dataloaders, train_state, 
                do_save_best_models=True, do_save_checkpoints=True, 
                dicts_to_dump = None, 
                class_num_loss_weights = None, 
                class_pos_loss_weights = None,
                margin_loss_weight = 1.0,
                frame_loss_weights = [],
                scalerAMP = None,
                do_dbg_nans = False,
                binarization_level = 0.5):    
    
    #try:
    #    work_dir = dataloaders['val'].dataset.work_dir #should work for a single evaluation processes that get the whole database
    #except AttributeError:
    #    work_dir = dataloaders['val'].dataset.dataset.work_dir #should work for multiple evaluation processes that get a subset of the database

    logging.info('train_model():')

    for train_state["curr_epoch"] in range(train_state["curr_epoch"]+1, train_state["planed_epochs_num"]):

        for param_group in optimizer.param_groups:
            LR = param_group['lr']
        logging.info(' start work in epoch id{} ({}/{}), LR{:.6}'.format(train_state["curr_epoch"], train_state["curr_epoch"]+1, train_state["planed_epochs_num"], LR))

        since = time.time()
        state_dict_file_name = ""
        
        curr_train_state = train_single_epoch(model, work_dir, device, optimizer, scheduler, dataloaders, ['train', 'val'],
                                              class_num_loss_weights, class_pos_loss_weights, margin_loss_weight, 
                                              scalerAMP, _epoch_id = train_state["curr_epoch"], 
                                              do_dbg_nans = do_dbg_nans, frame_loss_weights = frame_loss_weights,
                                              export_bin_imgs=False, bin_images_subdir = "", _tot_F1 = 0,
                                              binarization_level = binarization_level)
        
        is_best_val_loss = curr_train_state["loss_val"] <= train_state["best_loss_val"]

        #if(is_best_val_loss):
        #    logging.info("  New best model. Recalculate evaluation phase with result images dumping...")
        #    train_single_epoch(model, work_dir, device, optimizer, scheduler, dataloaders, ['val'],
        #                                      class_num_loss_weights, class_pos_loss_weights, margin_loss_weight, 
        #                                      scalerAMP, _epoch_id = train_state["curr_epoch"], 
        #                                      do_dbg_nans = do_dbg_nans, frame_loss_weights = frame_loss_weights,
        #                                      export_bin_imgs=True, 
        #                                      bin_images_subdir = "loss_imgs", _tot_F1 = curr_train_state["F1_val"],
        #                                      binarization_level = binarization_level)
            
        if(is_best_val_loss):
            train_state["best_F1_val"   ] = curr_train_state["F1_val"  ]
            train_state["best_loss_val" ] = curr_train_state["loss_val"]
            train_state["best_epoch"    ] = train_state["curr_epoch"   ]

            if do_save_best_models:

                # remove previous best models
                remove_all_zips_from_dir(work_dir, pass_id_filter = (train_state["pass_id"]), required_string = "_loss")

                logging.info("  Dump state dict and config dicts to file...")
                clss_str = ""
                sample_cls_env_list = dataloaders['whole'].dataset.paths[0]['cls_envs_list']
                cls_name_l = [ cls_envs['cls_name'] for cls_envs in sample_cls_env_list]
                for tis in cls_name_l:
                    clss_str += tis + "_"

                best_model_file_name = "{}_{}msd_cfgs_e{:03}_loss{:.4}_pass_id{}.pth.zip".format(train_state["timestamp"], clss_str, train_state["curr_epoch"], train_state["best_loss_val"], train_state["pass_id"])
                best_model_path = os.path.normpath(os.path.join(work_dir, best_model_file_name))
                save_model_with_cfgs(best_model_path, model, dicts_to_dump)
                train_state["best_model_fn"    ] = best_model_path

            
        # according to https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        # LR step should be done after training and evaluation
        scheduler.step()

        #checkpoint
        if(do_save_checkpoints):
            chp_fn = os.path.normpath(os.path.join(work_dir, "{}_checkpoint_e{:03}.tar".format(train_state["timestamp"], train_state["curr_epoch"])))
            logging.info("Checkpoint saving to: {}".format(chp_fn))
            save_checkpoint(chp_fn, model, optimizer, scheduler, train_state, scalerAMP)

        # Logging and statistics collection
        time_elapsed = time.time() - since

        #log
        if(train_state["curr_epoch"] == 0):
            train_state["mean_epoch_time_s"] = time_elapsed
            train_state["epochs"] = []
        else:
            prev_w = train_state["curr_epoch"]
            train_state["mean_epoch_time_s"] = (float(train_state["mean_epoch_time_s"]) * prev_w + time_elapsed) / (prev_w + 1)

        log_dict_entry = {
            "epoch_id"           : train_state["curr_epoch"],
            "LR"                 : LR,
            "epoch_time_s"       : "{:4.0f}".format(time_elapsed),
            "measures"           : curr_train_state,
            "epochs_since_best"  : train_state["curr_epoch"] - train_state["best_epoch"], 
            "best_model_fn"      : train_state["best_model_fn"],
            "checkpoint_fn"      : chp_fn if(do_save_checkpoints) else ""
            }
        if (device.type=='cuda'):
            memory_used, memory_free = get_cuda_mem_used_free(device.index)
            mem_alloc = torch.cuda.memory_allocated(device.index) / 1024**2
            process_gpu_mem_stats = torch.cuda.memory_stats(device.index)
            mem_alloc_max    = process_gpu_mem_stats['allocated_bytes.all.peak']/(1024.0**2)
            mem_reserved_max = process_gpu_mem_stats['reserved_bytes.all.peak' ]/(1024.0**2)
            log_dict_entry.update({
                "total_gpu_mem_used_MB"          : int(round(memory_used     , 0)),
                "total_gpu_mem_free_MB"          : int(round(memory_free     , 0)),
                "process_gpu_mem_alloc_MB"       : int(round(mem_alloc       , 0)),
                "process_gpu_mem_alloc_max_MB"   : int(round(mem_alloc_max   , 0)),
                "process_gpu_mem_reserved_max_MB": int(round(mem_reserved_max, 0)),
                })
        else:
            memory_used, memory_free = get_gen_mem_used_free() 
            log_dict_entry.update({
                "total_cpu_memory_used_MB"   : int(round(memory_used     , 0)),
                "total_cpu_memory_free_MB"   : int(round(memory_free     , 0)),
                })
        
        train_state["epochs"].append(log_dict_entry)

        if("loss_log_file" in train_state.keys()):
            jsonUpdate(train_state["loss_log_file"], train_state)

        # evaluate on sample images
        output_img_file_sufix = "_out_e{:03}".format(train_state["curr_epoch"])
        dump_epoch_sample_imgs(output_img_file_sufix, work_dir, train_state_dict = train_state)
        shutil.copy(os.path.join(work_dir, output_img_file_sufix+".png"), os.path.join(work_dir, "_out_elast.png"))

        #print statistics to screen
        logging.info(' time {:2.0f}m:{:02.0f}s. Loss val epoch {:.5}, best: {:.5}. F1 val epoch: {:.3}, best: {:.3}.'.format(time_elapsed // 60, time_elapsed % 60, curr_train_state["loss_val"], train_state["best_loss_val"], curr_train_state["F1_val"],  train_state["best_F1_val"]))
        
        if (device.type=='cuda'):
            logging.info(' cuda_memory: used {:>2.3}, free {:>2.3}; torch mem alloc: current {:>2.3}, max {:>2.3} [GB]'.format(memory_used/1024.0, memory_free/1024.0, mem_alloc/1024.0, mem_alloc_max/1024.0))
        
        if(train_state["curr_epoch"] != train_state["planed_epochs_num"]-1):
            logging.info('-' * 50)

    # load best model weights
    #model.load_state_dict(best_model_state_dict)
    
#----------------------------------------------------------------------------
# fast evaluation using global structures

# Change channel-order and make 3 channels for matplot
def dump_epoch_sample_imgs(sufix, work_dir, train_state_dict = None):
    global sample_cls_masks_ref_rgb
    global sample_cls_masks_pred
    global sample_images_in_rgb    

    # Map each channel (i.e. class) to each color
    cls_masks_pred_rgb = [img_helper.masks_to_colorimg(x) for x in sample_cls_masks_pred]

    fileName = os.path.normpath(os.path.join(work_dir, "{}.png".format(sufix)))
    img_helper.plot_side_by_side([sample_images_in_rgb, sample_cls_masks_ref_rgb, cls_masks_pred_rgb], fileName, state_dict=train_state_dict)
    
    #del pred_rgb
    
def move_files(src_dir, dst_dir, skip_pattern = None, overwrite = False):

    try:
        if os.path.isdir(src_dir):
            try:
                if not os.path.isdir(dst_dir):
                    pathlib.Path(dst_dir).mkdir(mode=0o775, parents=True, exist_ok=True)
            except Exception as err:
                logging.error('creating "{}" directory failed, error "%s"'.format(dst_dir, err))
                return 1
        else:
            logging.error('cannot find path "{}"'.format(src_dir))

        files = os.listdir(src_dir)
        # remove dirs from list
        files = [fn for fn in files if os.path.isfile(os.path.join(src_dir, fn))]
        # remove files following the skip_pattern
        if(not skip_pattern is None):
            files = [fn for fn in files if (fn.find(skip_pattern) == -1)]
        for fn in files:
            src = os.path.normpath(os.path.join(src_dir, fn))
            dst = os.path.normpath(os.path.join(dst_dir, fn))
            if os.path.isfile(dst):
                if overwrite:
                    os.remove(dst)
                    dest = shutil.move(src, dst) 
                else:
                    os.remove(src)
            else:
                dest = shutil.move(src, dst)  

        return(0)

    except:

        return(1)

def reset_seeds(seed):
    
    # seed for pseudo-random images flipping in the training part of the dataset
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
#----------------------------------------------------------------------------
# main
def main():

    global sample_cls_masks_ref_rgb
    global sample_cls_masks_pred
    global sample_images_in_rgb
        
    #----------------------------------------------------------------------------
    # initialize logging 
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    time_str = datetime.utcnow().strftime('%y_%m_%d__%H_%M_%S') #time.strftime("%y_%m_%d__%H_%M_%S.%f")
    initial_log_fn = "_initial_training_{}_pid{}.log".format(time_str, os.getpid())
    logging.basicConfig(level=logging.INFO, format = log_format, handlers=[logging.FileHandler(initial_log_fn, mode='w'), logging.StreamHandler(sys.stdout)])

    logging.info('*' * 50)
    logging.info(f"script {os.path.basename(__file__).split('.')[0]} start @ {time.ctime()}")
    logging.info("initial log file is {}".format(initial_log_fn))
    logging.info("*" * 50)
    
    from v_utils.v_logging_std import bind_std_2_logging
    bind_std_2_logging()
    #----------------------------------------------------------------------------
    logging.info("Reading configuration...")
    parser = ArgumentParser()
    logging.info(' -' * 25)
    logging.info(" Command line arguments:\n  {}".format(' '.join(sys.argv)))

    cfa = parser.add_argument_group('config_file_arguments')
    cfa.add_argument("--cfg" , default=[], action='append', type=str, nargs='*', required=False, metavar="PATH", help="one or more config json filenames. Further config files have higher priority. Command line arguments have the highest priority.", )
    
    if not(("-h" in sys.argv) or ("--help" in sys.argv)):
        cfg_fns_args, cmd_line_args_rem = parser.parse_known_args(); # bez error gdy natrafi na nieznany parametr (odwrotnie niż "parse_args()")
        cfgs = list(map(str, flatten(cfg_fns_args.cfg)))
        # read dictonaries from config files (create a list of dicts)
        cfg_dicts = convert_cfg_files_to_dicts(cfgs)

        # convert cmd_line_args_rem to dictionary so we can use it to update content of the dictonaries from config files
        cmd_line_args_rem_dict = convert_cmd_line_args_to_dict(cmd_line_args_rem)
        
        logging.info('-' * 50)
        logging.info(" Merge config files arguments with command line arguments...")
        # merge all config dicts - later ones will overwrite entries with the same keys from the former ones
        cfg_dict_pr = {}
        for cfg_dict in cfg_dicts:
            cfg_dict_pr.update(cfg_dict)
        # finally update with the command line arguments dictionary
        cfg_dict_pr.update(cmd_line_args_rem_dict)
        
        logging.info(" Merged arguments:")
        cfg_d = cfg_dict_pr
        print_cfg_dict(cfg_d, indent = 1, skip_comments = True)

        # parse the merged dictionary
        args_list_to_parse = convert_dict_to_cmd_line_args(cfg_dict_pr)

    _tf = [True, False]
    _logging_levels = logging._levelToName.keys()
    _continue_modes = ["try_next_pass", "force_pass_0", "try_load_checkpoint"]

    tca = parser.add_argument_group('training_config_arguments')
    tca.add_argument("--batch_size"                      , default=1               , type=int,              required=False, metavar='I'   , help="ile obrazow na batch. Ustaw na max ktory nie wywala uczenia z powodu braku pamieci. Dla Tesli to np. 9.")
    tca.add_argument("--train_tile_size"                 , default=None            , type=int,   nargs=2,   required=False, metavar='I'   , help="Domyslnie jest jednak ustawiony na najwiekszy rozmiar w zbiorze. [0,0] wymusza wyrownanie rozmiarow wszystkich obrazow poprzez ich rozszerzenie i jest domyslne gdy \"batch_size\" > 1")
    tca.add_argument("--val_crop_size"                   , default=None            , type=int,   nargs=2,   required=False, metavar='I'   , help="Domyslnie jest jednak ustawiony na najwiekszy rozmiar w zbiorze. [0,0] wymusza wyrownanie rozmiarow wszystkich obrazow poprzez ich rozszerzenie i jest domyslne gdy \"batch_size\" > 1")
    tca.add_argument("--train_tile_weights"              , default=[]              , type=float, nargs='*', required=False,                 help=" wagi dla loss z kolejnych punktow zaczynajac od brzegu kafelka. Dla punktow dla ktorych nie podano wagi, stosuje wage 1.0.")
    tca.add_argument("--margin_loss_weight"              , default=1.0             , type=float,            required=False, metavar='F'   , help="")
    tca.add_argument("--planed_epochs_num"               , default=5               , type=int,              required=False, metavar='N'   , help="Ile epok")
    tca.add_argument("--dataset_train_part_ratio"        , default=0.8             , type=float,            required=False, metavar='F'   , help="Podzial zioru na train i eval. Domyslnie 80%% zbioru idzie na trenowanie")
    tca.add_argument("--dataset_val_session_dirs"        , default=[]              , type=str,   nargs='*', required=False,                 help="lista user/session brana do ewaluacji. Wymusza ignorowanie dataset_train_part_ratio") 
    tca.add_argument("--allow_missing_ref"               , default=True            , action=arg2boolAct,    required=False, metavar='B'   , help="Czy dopuscic obrazy bez wszystkich ref")
    tca.add_argument("--freez_resnet_backbone"           , default=False           , action=arg2boolAct,    required=False, metavar='B'   , help="Czy zablokowac mozliwosc zmiany parametrow ResNet18 dla decymacyjnej galezi UNet")
    tca.add_argument("--train_continue_mode"             , default="try_next_pass" , type=str,              required=False, choices=_continue_modes,     help="'try_next_pass' - search the out_dir for previous pass results, and initialize network with the found pth.zip file, 'try_load_checkpoint' - continue from checkpoint")
    tca.add_argument("--do_balance_ref_classes"          , default=False           , action=arg2boolAct,    required=False, metavar='B'   , help="")
    tca.add_argument("--use_amp"                         , default=True            , action=arg2boolAct,    required=False, metavar='B'   , help="Czy uzywac f16 zamiast f32 przy uczeniu na CUDA")
    tca.add_argument("--force_deterministic"             , default=False           , action=arg2boolAct,    required=False, metavar='B'   , help="Wymuszenie powtarzalnych operacji. Wymusza konfiguracje --upsampling_mode 'nearest'")
    tca.add_argument("--dataset_seed"                    , default=1               , type=int,              required=False, metavar='I'   , help="Intiger seed inicjujacy generatory liczb pseudolosowych")
    tca.add_argument("--seed"                            , default=1               , type=int,              required=False, metavar='I'   , help="Intiger seed inicjujacy generatory liczb pseudolosowych")
    
    pla = parser.add_argument_group('platform_arguments')
    pla.add_argument("--force_single_thread", "-fs"      , default=False           , action=arg2boolAct,    required=False, metavar='B'   , help="wymuszenia pracy jenowatkowej")
    me_group = parser.add_mutually_exclusive_group()
    me_group.add_argument("--force_cpu",      "-fc"      , default=False           , action=arg2boolAct,    required=False, metavar='B'   , help="wymuszenia obliczen na CPU")
    me_group.add_argument("--force_gpu_id",   "-fg"      , default=None            , type=int,              required=False, metavar="I"   , help="wymuszenia obliczen na GPU o danym id")
    pla.add_argument("--limit_train_len"                 , default=None            , type=int,              required=False, metavar='I'   , help="Ograniczenie liczebnosci zbiorow uczacego i ewaluacyjnego dla przyspieszenia obliczen, np. na cpu")
    pla.add_argument("--limit_val_len"                   , default=None            , type=int,              required=False, metavar='I'   , help="Ograniczenie liczebnosci zbiorow uczacego i ewaluacyjnego dla przyspieszenia obliczen, np. na cpu")
    pla.add_argument("--sample_test_len"                 , default=3               , type=int,              required=False, metavar='I'   , help="")
   
    eva = parser.add_argument_group('evaluation_arguments')
    eva.add_argument("--threshold_level"                 , default=0.5             , type=float,            required=False, metavar='F'   , help="prog decyzyjny dla binaryzacji ciaglej odpowiedzi sieci. Zakres <0.0 - 1.0>, domyslnie 0.5")
      
    lra = parser.add_argument_group('LR_config_arguments')
    lra.add_argument("--learning_rate_start"             , default=5e-4            , type=float,            required=False, metavar='F'   , help="")
    lra.add_argument("--learning_rate_step_len"          , default=10              , type=int,              required=False, metavar='I'   , help="")
    lra.add_argument("--learning_rate_step_factor"       , default=0.5             , type=float,            required=False, metavar='F'   , help="")
    lra.add_argument("--learning_rate_saw_depth"         , default=0.1             , type=float,            required=False, metavar='F'   , help="")
    lra.add_argument("--learning_rate_warmup_epochs"     , default=0               , type=int,              required=False, metavar='I'   , help="")
    
    ota = parser.add_argument_group('output_arguments')
    ota.add_argument("--out_dir",   "-od"                , default="def_train_dir" , type=str,              required=False, metavar="PATH", help="directory in which class's directorys with output files will be saved")
    ota.add_argument("--do_save_checkpoints"             , default=True            , action=arg2boolAct,    required=False, metavar='B'   , help="Uczenie moze zapisac checkpoint co epoke i mozna sprobowac go wczytac i kontynuowac uczenie")
    ota.add_argument("--logging_level"                   , default=logging.INFO    , type=int,              required=False, choices=_logging_levels,     help="")
    ota.add_argument("--do_dbg_nans"                     , default=True            , action=arg2boolAct,    required=False, metavar='B'   , help="czy wyrzucic pliki z uzycia modelu jezeli w wyniku uczenia lub ewaluacji modelu wartosc loss bedzie rowna NaN" )
    ota.add_argument("--comment"                         , default=[]              , type=str,   nargs='*', required=False, metavar='STR' , help="komentarz")
    
    obsolete_parser = ArgumentParser()
    obsolete_parser.add_argument("--do_evaluate_all_after_each_epoch"                                     , required=False, help="JUZ NIEUZYWANY! Uzycie modelu na zbiorze ewaluacyjnym wraz z generacja png z podgladem efektow")
    obsolete_parser.add_argument("--skip_evaluation_for_n_epochs"                                         , required=False, help="JUZ NIEUZYWANY! ")
    obsolete_parser.add_argument("--pretrained_resnet_backbone"                                           , required=False, help="JUZ NIEUZYWANY! ")
    obsolete_parser.add_argument("--use_sigmoid_on_loss"                                                  , required=False, help="JUZ NIEUZYWANY! ")
    obsolete_parser.add_argument("--use_sigmoid_on_pred"                                                  , required=False, help="JUZ NIEUZYWANY! ")
    obsolete_parser.add_argument("--try_load_checkpoint"                                                  , required=False, help="JUZ NIEUZYWANY! Uzyj opcji 'train_continue_mode' z wartoscia 'try_load_checkpoint'")
    obsolete_parser.add_argument("--pass_id"                                                              , required=False, help="JUZ NIEUZYWANY! Numer pass_id jest ustawiany automatycznie")
    obsolete_parser.add_argument("--translated_pxpy"                                                      , required=False, help="JUZ NIEUZYWANY! ")
    obsolete_parser.add_argument("--loss_bce_weight"                                                      , required=False, help="JUZ NIEUZYWANY! Zawsze jest używana miara soft F1 i nie ma juz wazenia pomiedzy dice_loss i bce_loss roznymi miarami")
    obsolete_parser.add_argument("--export_metric_F1"                                                     , required=False, help="JUZ NIEUZYWANY! Miara F1 jest używana w ewaluacji do wyznaczenia loss (eval_loss = (1 - eval_F1))")
    obsolete_parser.add_argument("--export_metric_per_case"                                               , required=False, help="JUZ NIEUZYWANY! ")
    
    if not(("-h" in sys.argv) or ("--help" in sys.argv)): 
        # get training arguments
        tr_args, rem_args = parser.parse_known_args(args_list_to_parse)
        tr_ob_args, rem_args = obsolete_parser.parse_known_args(rem_args)
        
        logging.info(" Parsed training configuration arguments:")
        tr_args_d = vars(tr_args)
        print_cfg_dict(tr_args_d, indent = 1, skip_comments = True)
        
        obsole_args = [{k:vars(tr_ob_args)[k]} for k in vars(tr_ob_args).keys() if (not vars(tr_ob_args)[k] is None)]
        for a in obsole_args:
            logging.warning(" Found obsolate training argument {}".format(a))
    else: 
        # help
        logging.info("Params for pytorch_model_train:")
        logging.info(parser.format_help())
        logging.info("Params for pytorch_model_train (obsolate):")
        logging.info(obsolete_parser.format_help())
        logging.info("Params for MRIDataset:")
        logging.info(MRIDataset.parse_arguments("--help"))
        logging.info("Params for ResNetFlexUNet:")
        logging.info(ResNetFlexUNet.parse_arguments("--help"))
        sys.exit(1)
        
    #----------------------------------------------------------------------------
    if(tr_args.force_deterministic):
        # seed for pseudo-random images flipping in the training part of the dataset
        torch.backends.cudnn.deterministic = True
        torch.set_deterministic(True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.set_deterministic(False)

    reset_seeds(tr_args.dataset_seed)
    #----------------------------------------------------------------------------
    # create work dir
    logging.info("-" * 50)
    work_dir                = tr_args.out_dir     

    try:
        if not os.path.isdir(work_dir):
            pathlib.Path(work_dir).mkdir(mode=0o775, parents=True, exist_ok=True)
            logging.info("Created work dir {}".format(work_dir))
    except Exception as err:
        logging.error("Creating dir ({}) IO error: {}".format(work_dir, err))
        sys.exit(1)
    
    #----------------------------------------------------------------------------
    # passes continuation
    pass_id = 0
    nn_init_pth = None
    if (tr_args.train_continue_mode == 'try_next_pass'):
        pths = find_all_zips_from_dir(work_dir, pass_id_filter = None)
        pass_ids = [(parse_pass_id(fn), fn) for fn in pths]
        if(len(pass_ids) > 0):
            max_pass, nn_init_pth = max(pass_ids, key=lambda p: p[0])
            if max_pass != -1:
                pass_id = max_pass + 1
                logging.info(" Found previous results for pass {}.".format(max_pass))
                
                prev_pass_dir_name = "bkp_pass_{}".format(max_pass)
                prev_pass_dir_pth = os.path.normpath(os.path.join(work_dir, prev_pass_dir_name))
                logging.info(" Move pass {} data to {}.".format(max_pass, prev_pass_dir_pth))
                move_files(work_dir, prev_pass_dir_pth, skip_pattern = ".pth.zip", overwrite = False)
                
                work_root, work_dir_name = os.path.split(work_dir)
                pass_dir_link_name = work_dir_name + "_pass[{}]".format(max_pass)
                pass_dir_link_path = os.path.abspath(os.path.normpath(os.path.join(work_root, pass_dir_link_name)))
                if not os.path.islink(pass_dir_link_path):
                    logging.info(" Create link {} to {}.".format(pass_dir_link_path, prev_pass_dir_pth))
                    try:
                        os.symlink(os.path.realpath(prev_pass_dir_pth), pass_dir_link_path, target_is_directory = True)
                    except OSError:
                        logging.warning(" On newer versions of Windows 10, unprivileged accounts can create symlinks if Developer Mode is enabled. When Developer Mode is not available/enabled, the SeCreateSymbolicLinkPrivilege privilege is required, or the process must be run as an administrator.")


                logging.info(" Continue with pass {} and initializing pth {}".format(pass_id, nn_init_pth))
                rem_args.extend(['--pretrained_model_state_dict_path', nn_init_pth])
   
    #----------------------------------------------------------------------------
    # redirect logging file to work directory 

    # new logging file handler
    filehc = logging.FileHandler(work_dir+"/_training_{}.log".format(time_str), 'w')
    filehc.setFormatter(logging.Formatter(log_format))
    filehl = logging.FileHandler(work_dir+"/_training_last.log", 'w')
    filehl.setFormatter(logging.Formatter(log_format))

    log = logging.getLogger()  # root logger
    logging_level = tr_args.logging_level
    log.setLevel(logging_level)
    for hdlr in log.handlers[:]:  # remove all old handlers
        if(type(hdlr) is logging.FileHandler):
            old_log_fn = hdlr.baseFilename 
            hdlr.close()
            log.removeHandler(hdlr)
            with open(old_log_fn, 'r') as f:
                lines = f.read()
            os.remove(old_log_fn)
            filehc.stream.writelines(lines)
            filehl.stream.writelines(lines)
    log.addHandler(filehc)      # set the new handler
    log.addHandler(filehl)      # set the new handler

    # start new logging
    logging.info("change log file to {} and {}".format(filehc.baseFilename, filehl.baseFilename))
    
    #----------------------------------------------------------------------------
    # platform specific features
    logging.info("-" * 50)
    logging.info("platform specific features")
    
    if tr_args.force_cpu:
        device = torch.device('cpu')
    elif (not tr_args.force_gpu_id is None):
        list_of_gpus = [torch.cuda.get_device_name(cid) for cid in range(torch.cuda.device_count())]
        gpu_id = tr_args.force_gpu_id
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

    if device.type == 'cuda':
        # set default device to cuda with the given ID
        torch.cuda.device(device)

    logging.info("device: {}".format(device))

    system_name = os.name
#    if system_name != 'nt': #not Windows - require to specify that process start with spawn method
#        multiprocessing.set_start_method('spawn')

    if(tr_args.force_single_thread):
        logging.info("Forced single thread operation. Disable multiprocessing")
        use_multiprocessing = False
    elif system_name != 'nt': #not Windows - require to specify that process start with spawn method
        multiprocessing.set_start_method('spawn')
        
        
    elif(device.type=='cpu'):
        if 'pydevd_concurrency_analyser.pydevd_thread_wrappers' in sys.modules:
            logging.info("Running in Visual Studio. Disable multiprocessing")
            use_multiprocessing = False
        else:
            logging.info("Running outside Visual Studio. Can use multiprocessing")
            use_multiprocessing = True#False
    else: #if(device.type=='cuda'):
        use_multiprocessing = True
        cuda_total_memory = torch.cuda.get_device_properties(device.index).total_memory

    if (device.type=='cpu') and (tr_args.use_amp):
        logging.warning("use_amp is {} but device.type is {}. Amp is only valid for CUDA. I disable amp.".format(tr_args.use_amp, device.type))
        tr_args.use_amp = False
    
    #----------------------------------------------------------------------------
    # DATASET 
    logging.info("-"*30)
    logging.info("Creating dataset...")
    
    # inferred parameters
    crop_size = None
    if ((tr_args.train_tile_size is None) and (tr_args.batch_size != 1)):
        logging.info("Change crop_size from \"{}\" to [0, 0] (\"auto\") aiming at equal size of all input images. Reason for that is the batch_size != 1 ({}) that may result in images with different sizes in a single minibatch.".format(tr_args.train_tile_size, tr_args.batch_size))
        crop_size           = [0, 0]
    elif not(tr_args.train_tile_size is None):
        crop_size           = tr_args.train_tile_size
    train                   = False
    req_refs_level          = "some" if tr_args.allow_missing_ref else "all"
    unet_max_div            = 32 # TODO zmien na to co wynika z parsowania flexnet - problem w tym ze parametry Flexnet sa parsowane pozniej

    ds_cfg_inferred_params = [
        "--train"                       , train                 ,
        "--req_refs_level"              , req_refs_level        ,
        "--force_size_is_multiple_of"   , unet_max_div          ,
        ]
    if(not crop_size is None):
        ds_cfg_inferred_params.extend([
        "--crop_size"                   , crop_size             ,])
    ds_cfg_inferred_params_flat_str = list(map(str, flatten(ds_cfg_inferred_params)))
    rem_args.extend(ds_cfg_inferred_params_flat_str)

    logging.info(" Parse dataset configuration arguments:")
    ds_args, rem_args = MRIDataset.parse_arguments(rem_args)

    cfg_d = vars(ds_args)
    print_cfg_dict(cfg_d, indent = 1, skip_comments = True)

    whole_set = MRIDataset( ds_args)
    
    logging.info("-"*30)

    dataset_entries_without_comp_fn = "{}/_entries_without_comp.json".format(work_dir)
    len_entries_without_comp = len(whole_set._entries_without_comp)
    if(len_entries_without_comp != 0):
        logging.info("Found {} inputs during dataset preparation that lack at least one of the input components.".format(len_entries_without_comp))
        logging.info(" detail on those can be found in file \"{}\"".format(dataset_entries_without_comp_fn))
        jsonDumpSafe(dataset_entries_without_comp_fn, whole_set._entries_without_comp)
    
    dataset_entries_without_ref_fn = "{}/_entries_without_ref.json".format(work_dir)
    len_entries_without_ref = len(whole_set._entries_without_ref)
    if(len_entries_without_ref != 0):
        logging.info("Found {} inputs during dataset preparation that lack at least one of the reference polygon file.".format(len_entries_without_ref))
        logging.info(" detail on those can be found in file \"{}\"".format(dataset_entries_without_ref_fn))
        jsonDumpSafe(dataset_entries_without_ref_fn, whole_set._entries_without_ref)
    
    dataset_empty_refs_fn = "{}/_empty_refs.json".format(work_dir)
    len_empty_refs = len(whole_set._empty_refs)
    if(len_empty_refs != 0):
        logging.info("Found {} reference polygon file without any poligons.".format(len_empty_refs))
        logging.info(" detail on those can be found in file \"{}\"".format(dataset_empty_refs_fn))
        jsonDumpSafe(dataset_empty_refs_fn, whole_set._empty_refs)
        
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
    # Dividing dataset into parts 
    logging.info(" -" * 25)
    logging.info("Dividing dataset into parts...")

    if(len(tr_args.dataset_val_session_dirs) != 0):
        logging.warning("dataset_val_session_dirs overwrite dataset_train_part_ratio settings.")
        # solve multi-selection filters with "*" / "?"
        ds_imgDir_root = whole_set.paths[0]['src_comps_dict_l'][0]['imgDir_root']
        user_session_dirs = expand_session_dirs(tr_args.dataset_val_session_dirs, ds_imgDir_root)

        logging.info(" Forced user/session dirs for evaluation:")            
        val_set_idxs = []
        for session_dir in user_session_dirs:
            matching_idxs = [idx for idx in range(len(whole_set.paths)) if (whole_set.paths[idx]['session_sub_dir'] == session_dir)]
            logging.info("   - {} with {} entries".format(session_dir, len(matching_idxs)))
            val_set_idxs.extend(matching_idxs)
        # pytorch seed for splitting a dataset so that it is possible to reproduce the test results
        reset_seeds(tr_args.dataset_seed)
        train_set_idxs = list(np.delete(np.array(range(len(whole_set))), val_set_idxs))
        val_len   = len (val_set_idxs)
        train_len = len (train_set_idxs)

        # limit lens
        if (tr_args.limit_train_len  is not None) and (tr_args.limit_train_len  != -1) and (tr_args.limit_train_len < train_len):train_len   = tr_args.limit_train_len
        if (tr_args.limit_val_len    is not None) and (tr_args.limit_val_len    != -1) and (tr_args.limit_val_len   < val_len  ):val_len     = tr_args.limit_val_len
        val_set_idxs    = random.sample(val_set_idxs,     val_len)
        train_set_idxs  = random.sample(train_set_idxs, train_len)
        
        val_set   = torch.utils.data.Subset(whole_set, val_set_idxs)
        train_set = torch.utils.data.Subset(whole_set, train_set_idxs)

    else:
        train_len = int(tr_args.dataset_train_part_ratio* (dataset_len))
        val_len = dataset_len - train_len
        if (tr_args.limit_train_len  is not None) and (tr_args.limit_train_len  != -1) and (tr_args.limit_train_len < train_len):train_len   = tr_args.limit_train_len
        if (tr_args.limit_val_len    is not None) and (tr_args.limit_val_len    != -1) and (tr_args.limit_val_len   < val_len  ):val_len     = tr_args.limit_val_len
        _unused_len = dataset_len - (train_len + val_len)
        # pytorch seed for splitting a dataset so that it is possible to reproduce the test results
        reset_seeds(tr_args.dataset_seed)
        train_set, val_set, _ = torch.utils.data.random_split(whole_set, [train_len, val_len, _unused_len])
    
    # need a copy of dataset for train dataset because it uses other transforms and transforms trans_eval were used to construct whole_set
    train_set.dataset = copy.copy(whole_set)
    train_set.dataset.train = True
    #for validation set disable cropping
    val_set.dataset.general_crop_size = tr_args.val_crop_size
    val_set.dataset.crop_type = 'random' # algorytm widzi ze to jest DS ewaluacyjny i bedzie wycinal losowo dla danego obrazka, ale dla danego obrazka zawsze tak samo 
    sample_test_len = min(tr_args.sample_test_len, val_len)

    dataloaders = {
        'train' : DataLoader(train_set,  batch_size=tr_args.batch_size,    shuffle=True,  num_workers=0),
        'val'   : DataLoader(val_set,    batch_size=1,                     shuffle=False, num_workers=0),
        'whole' : DataLoader(whole_set,  batch_size=1,                     shuffle=False, num_workers=0)
    }
    
    #----------------------------------------------------------------------------
    # print size of the datasets
    image_datasets = {
        'train': train_set, 'val': val_set, 'whole': whole_set
    }
    dataset_sizes = {
        x: len(image_datasets[x]) for x in image_datasets.keys()
    }
    
    logging.info(" dataset sizes: {}".format(dataset_sizes))
    
    #----------------------------------------------------------------------------
    logging.info(" -" * 25)
    logging.info("Check balans between the clss representation in the train dataset...")
    if(tr_args.do_balance_ref_classes):

        sample_cls_env_list = train_set.dataset.paths[0]['cls_envs_list']
        cls_name_l = [ cls_envs['cls_name'] for cls_envs in sample_cls_env_list]
        cls_stat = {}
        for cls_name in cls_name_l:
            cls_stat[cls_name] = {'num':0, 'pos_area':0, 'tot_area': 0} 

        indices = train_set.indices + val_set.indices
        num_cases = len(indices)
        for case_id in indices:
            cls_envs_list = whole_set.paths[case_id]['cls_envs_list']
            cmp_size = whole_set.paths[case_id]['src_comps_size']
            for cls_id, cls_name in enumerate(cls_name_l):
                cls_path = cls_envs_list[cls_id]['ref_polygon_path']
                if cls_path != '':
                    cls_stat[cls_name]['num' ] += 1
                    with open (cls_path) as f:
                        poly_dict_data= json.load(f)
                    my_polygons = v_polygons.from_dict(poly_dict_data)
                    polygons_npm = my_polygons.as_numpy_mask(fill=True, val=1)
                    area = np.sum(polygons_npm)
                    cls_stat[cls_name]['pos_area'] += area
                    cls_stat[cls_name]['tot_area'] += cmp_size[0] * cmp_size[1]
                
        for cls_name in cls_name_l:
            cls_stat[cls_name]['pos_area_per_case'] = cls_stat[cls_name]['pos_area'] / cls_stat[cls_name]['num']
            cls_stat[cls_name]['pos2total_ratio'  ] = cls_stat[cls_name]['pos_area'] / cls_stat[cls_name]['tot_area']
        tot_area_per_case = sum([ cls_stat[key]['pos_area_per_case'] for key in cls_stat])
        tot_num           = sum([ cls_stat[key]['num'              ] for key in cls_stat])

        logging.info("Stats for clss:")
        for cls_name in cls_name_l:
            logging.info(" {}: {}".format(cls_name, cls_stat[cls_name]))

            class_num_loss_weights  = [num_cases/cls_stat[key]['num']       for key in cls_stat]
            class_num_loss_weights  = class_num_loss_weights/np.mean(class_num_loss_weights)
            class_pos_loss_weights  = [1.0/cls_stat[key]['pos2total_ratio'] for key in cls_stat]
            logging.info("Classes balancing weights chosen based on num of classes (increase over 1.0 in order to mimic that we have ref for all cases):")
            for class_id, cls_name in enumerate(cls_name_l):
                logging.info(" {}: {}".format(cls_name, class_num_loss_weights[class_id]))
            logging.info("Classes positive weights chosen based on num of positive samples for classes:")
            for class_id, cls_name in enumerate(cls_name_l):
                logging.info(" {}: {}".format(cls_name, class_pos_loss_weights[class_id]))
    else:
        class_num_loss_weights = None
        class_pos_loss_weights = None
        logging.info("Classes balancing weights are disabled by reseting 'do_balance_ref_classes' flag")
    #----------------------------------------------------------------------------
    # print size of a batch of training dataset
    images_in, masks, paths = next(iter(dataloaders['train']))
    
    logging.info(" -" * 25)
    logging.info("shape of batch of images_in for training : {}, min  = {:.3}, max = {:.3}".format(images_in.shape, images_in.numpy().min(), images_in.numpy().max()))
    logging.info("shape of batch of masks     for training : {}, mean = {:.3}, std = {:.3}".format(masks.shape, masks.numpy().mean(), masks.numpy().std()))
        
    del images_in
    del masks
    del paths
    #----------------------------------------------------------------------------
    #  TEST SAMPLES prepare and globals initialization
    sample_id = 0
    sample_cls_masks_ref = []
    for s_sample_images_in, s_sample_cls_masks_ref, s_sample_test_envs in dataloaders["val"]:
        if sample_id >= sample_test_len:
            break
        s_sample_images_in_shape = s_sample_images_in.shape
        sample_image_in_rgbX_np   = whole_set.reverse_transform(s_sample_images_in[0])
        del s_sample_images_in
        s_sample_cls_masks_ref_np = s_sample_cls_masks_ref[0].cpu().numpy()
        del s_sample_cls_masks_ref
        if(sample_image_in_rgbX_np.shape[2] == 1):
            #sample_image_in_rgb_np   = np.repeat(sample_image_in_rgbX_np, 3, axis=2)
            sample_image_in_rgb_np   = np.dstack([sample_image_in_rgbX_np[:, :, 0], sample_image_in_rgbX_np[:, :, 0], sample_image_in_rgbX_np[:, :, 0]])
        elif(sample_image_in_rgbX_np.shape[2] == 2):
            sample_image_in_rgb_np   = np.dstack([sample_image_in_rgbX_np[:, :, 0], sample_image_in_rgbX_np[:, :, 1], sample_image_in_rgbX_np[:, :, 0]])
        elif(sample_image_in_rgbX_np.shape[2] == 3):
            sample_image_in_rgb_np    = sample_image_in_rgbX_np
        elif(sample_image_in_rgbX_np.shape[2] > 3):
            sample_image_in_rgb_np    = sample_image_in_rgbX_np[:,:,0:3]
        sample_images_in_rgb.append(sample_image_in_rgb_np)
        sample_cls_masks_ref.append(s_sample_cls_masks_ref_np) 
        sample_cls_masks_pred.append(np.zeros(s_sample_cls_masks_ref_np.shape))
        logging.info("Sample {} img {} (rgb{} T{}), ref {} (T{})".format(sample_id, s_sample_test_envs['src_comps_path_l'][0][0], sample_image_in_rgb_np.shape, s_sample_images_in_shape, s_sample_test_envs['cls_envs_list'][0]['ref_polygon_path'][0], s_sample_cls_masks_ref_np.shape))
        sample_id += 1
    sample_cls_masks_ref_rgb  = [img_helper.masks_to_colorimg(x) for x in sample_cls_masks_ref ]
    
    #----------------------------------------------------------------------------
    # Train state init
    train_state= {}
    train_state["pass_id"          ] = 0
    train_state["planed_epochs_num"] = tr_args.planed_epochs_num
    train_state["seed"             ] = tr_args.seed
    train_state["gpu_id"           ] = "cuda:{}".format(device.index) if (device.type=='cuda') else "cpu"
    train_state["curr_epoch"       ] = -1
    train_state["best_epoch"       ] = -1
    train_state["best_loss_val"    ] = 1e10
    train_state["best_F1_epoch"    ] = -1
    train_state["best_F1_val"      ] = -1
    train_state["timestamp"        ] = time.strftime("%y%m%d%H%M", time.gmtime())
    train_state["ds_val_len"       ] = len( dataloaders['val'  ] ) * dataloaders['val'  ].batch_size
    train_state["ds_train_len"     ] = len( dataloaders['train'] ) * dataloaders['train'].batch_size
    train_state["ds_whole_len"     ] = len( dataloaders['whole'] ) * dataloaders['whole'].batch_size

    loss_log_file_name = "loss_log.json"
    loss_log_file_name_full = os.path.normpath(os.path.join(work_dir, loss_log_file_name))
    train_state["loss_log_file"] = loss_log_file_name_full

    #----------------------------------------------------------------------------
    # MODEL
    logging.info("-"*30)

    logging.info("Creating model {}...".format("ResNetFlexUNet"))
    
    reset_seeds(tr_args.seed)
    
    # inferred parameters
    num_in_comp = len(whole_set.paths[0]['src_comps_path_l'])
    flexnet_cfg_list_size_params = [
        "--n_class"               , whole_set.num_class,
        "--in_channels"           , num_in_comp,
        ]
    rem_args.extend(list(map(str, flatten(flexnet_cfg_list_size_params))))
    
    # passes continuation
    if (tr_args.train_continue_mode == 'try_next_pass'):
        if(pass_id != 0):
            train_state["pass_id"] = pass_id
            logging.info(" Continue with pass {} and initializing pth {}".format(train_state["pass_id"], nn_init_pth))
            rem_args.extend(['--pretrained_model_state_dict_path', nn_init_pth])
   
    logging.info(" Parse Flexnet configuration arguments:")
    fn_args, rem_args = ResNetFlexUNet.parse_arguments(rem_args)
    

    print_cfg_dict(vars(fn_args), indent = 1, skip_comments = True)

    if(tr_args.force_deterministic and (fn_args.upsampling_mode != 'nearest')):
        logging.warning(" Deterministic operations are forced - force upsampling_mode for flexnet to 'nearest'")
        fn_args.upsampling_mode = 'nearest'

    model = ResNetFlexUNet(fn_args).to(device)
    
    from torchsummary import summary
    logging.info("Network summary (from torchsummary):")
    dc, dx, dy = s_sample_images_in_shape[1:]

    old_stdout = sys.stdout # Memorize the default stdout stream
    sys.stdout = buffer = io.StringIO()

    if(device.type=='cuda'):
        if(device.index == 0):
            summary(model, input_size=(dc, dx, dy), device = "cuda") 
        else:
            logging.info("function torchvision.summary() does not work for GPU other than 0. Simply print model structure using logging.info(model)")
            logging.info(model)
    elif(device.type=='cpu'):
        summary(model, input_size=(dc, dx, dy), device = "cpu") 

    sys.stdout = old_stdout # Put the old stream back in place
    logging.info(buffer.getvalue()) # Return a str containing the

    # reset seeds - torch summary is performed only when GPU0 is used and it advances pseudo random generators 
    # and, as a result further calculations on GPU0 will be different 
    # than the same operations on other GPUs therefore reset seeds 
    # so state of random generators is the same no mather witch GPU is used.
    reset_seeds(tr_args.seed)

    if(tr_args.freez_resnet_backbone):
        # freeze backbone layers
        # Comment out to finetune further
        for l in model.base_layers:
            for param in l.parameters():
                param.requires_grad = False
        
    lr_start = tr_args.learning_rate_start
    if(tr_args.learning_rate_warmup_epochs != 0):
        lr_start = tr_args.learning_rate_start / (tr_args.learning_rate_warmup_epochs+1)

    optimizer_ft = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = lr_start)
    
    scalerAMP = None
    if tr_args.use_amp:
        scalerAMP = amp.GradScaler()

    def dec_saw_with_warmup_fun(epoch, step_len = tr_args.learning_rate_step_len, step_factor = tr_args.learning_rate_step_factor, saw_depth = tr_args.learning_rate_saw_depth, warm_up_size = tr_args.learning_rate_warmup_epochs):
        ep_since_step = (epoch - warm_up_size) % step_len
        saw_gamma = 1.0 - saw_depth
        saw_mul = math.pow(saw_gamma, 1/(step_len-1))
        if epoch <= warm_up_size:
            return (epoch+1) / (epoch+0)
        elif (ep_since_step == 0):
            return 1/saw_gamma * step_factor 
        else:
            return saw_mul
    #lr = 0.25
    #for e in range(1, 100):
    #    m = dec_saw_with_warmup_fun(e)
    #    lr = lr * m
    #    print("{:3}: lr{} m{}".format(e, lr, m))
    exp_lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer_ft, lr_lambda=dec_saw_with_warmup_fun)
        
    logging.info('=' * 50)

    #----------------------------------------------------------------------------
    if(len(rem_args) != 0):
        logging.error("After all modules parsed own arguments some arguments are left: {}.".format(rem_args))
        logging.info('=' * 50)
        sys.exit(10)
        
    #----------------------------------------------------------------------------
    jsonDumpSafe(train_state["loss_log_file"], {'time_start': time.ctime()})

    #----------------------------------------------------------------------------
    # Try find the last checkpoint
    if tr_args.train_continue_mode != 'try_load_checkpoint':
        checkpoint_epoch = -1
    else:
        checkpoint_epoch = tr_args.planed_epochs_num - 1
        while (checkpoint_epoch>-1):
            checkpoint_fn_pattern  = work_dir + '/*checkpoint_e{}*'.format(checkpoint_epoch)
            checkpoint_fn_pattern  = os.path.normpath(checkpoint_fn_pattern)
            checkpoint_fp_l = glob.glob(checkpoint_fn_pattern)
            if(len(checkpoint_fp_l)>0):
                fn = checkpoint_fp_l[0]
                logging.info("Checkpoint file found for epoch {} ({})...".format(checkpoint_epoch, fn))
                try:
                    load_checkpoint(fn, model, optimizer_ft, exp_lr_scheduler, train_state, scalerAMP)
                    logging.info(" Load successful. Skipp training up to this epoch".format(checkpoint_epoch, fn))
                    break
                except FileNotFoundError:
                    logging.warning(" Load unsuccessful. Try next...")
                    checkpoint_epoch = checkpoint_epoch-1 
            else:
                logging.info("Checkpoint not found for epoch {} ({}). Try next...".format(checkpoint_epoch, fn))
                checkpoint_epoch = checkpoint_epoch-1

    # dump dataset description and training description files for reference. Just now because checkpoint loading can retore train_state["timestamp"]
    training_time_stamp = train_state["timestamp"]
    dataset_desc_dict_fn = "{}/{}_dsd.json".format(work_dir, training_time_stamp)
    logging.info("saving dataset description dict to file \"{}\"".format(dataset_desc_dict_fn))
    jsonDumpSafe(dataset_desc_dict_fn, vars(ds_args))
    train_desc_dict_fn = "{}/{}_td.json".format(work_dir, training_time_stamp)
    logging.info("saving training description dict to file \"{}\"".format(train_desc_dict_fn))
    jsonDumpSafe(train_desc_dict_fn, vars(tr_args))
    flexnet_desc_dict_fn = "{}/{}_fn.json".format(work_dir, training_time_stamp)
    logging.info("saving flexnet description dict to file \"{}\"".format(flexnet_desc_dict_fn))
    jsonDumpSafe(flexnet_desc_dict_fn, vars(fn_args))
    dicts_to_dump = {
        "tr_dict": vars(tr_args),
        "ds_dict": vars(ds_args),
        "fn_dict": vars(fn_args),
    }

    if(not dataloaders["val"].dataset.dataset.has_ref_masks):
        raise Exception('Database don\'t has reference masks! Cannot train models using this database ("ds_polyRefDirs_root" not given?).')
            
    train_model(model, work_dir, device, optimizer_ft, exp_lr_scheduler, dataloaders, train_state, 
                do_save_checkpoints=tr_args.do_save_checkpoints, 
                dicts_to_dump = dicts_to_dump, 
                class_num_loss_weights = class_num_loss_weights, 
                class_pos_loss_weights = class_pos_loss_weights, 
                margin_loss_weight = tr_args.margin_loss_weight,
                scalerAMP = scalerAMP,
                do_dbg_nans = tr_args.do_dbg_nans,
                frame_loss_weights = tr_args.train_tile_weights,
                binarization_level = tr_args.threshold_level)
    
        
    jsonUpdate(train_state["loss_log_file"], {'time_end': time.ctime()})
    #----------------------------------------------------------------------------
    # final EVALUATION
    output_img_file_sufix = "_out_e{:03}_final_eval".format(train_state["curr_epoch"])
    dump_epoch_sample_imgs(output_img_file_sufix, work_dir, train_state_dict = train_state)

    logging.info("best_loss_val {}".format( train_state["best_loss_val"]  ))
    logging.info("best_epoch    {}".format( train_state["best_epoch"]     ))
    logging.info("script ends @ {}".format(time.ctime()))
    logging.info('*' * 50)

if __name__ == '__main__':
    main()
    