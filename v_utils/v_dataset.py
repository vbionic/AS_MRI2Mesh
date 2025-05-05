#-----------------------------------------------------------------------------------------
#from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
#The dataset should inherit from the standard torch.utils.data.Dataset class, and implement __len__ and __getitem__.
#The only specificity that we require is that the dataset __getitem__ should return:
#
#image: a PIL Image of size (H, W) 
#target: a dict containing the following fields 
#boxes (FloatTensor[N, 4]): the coordinates of the N bounding boxes in [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H
#labels (Int64Tensor[N]): the label for each bounding box
#image_id (Int64Tensor[1]): an image identifier. It should be unique between all the images in the dataset, and is used during evaluation
#area (Tensor[N]): The area of the bounding box. This is used during evaluation with the COCO metric, to separate the metric scores between small, medium and large boxes.
#iscrowd (UInt8Tensor[N]): instances with iscrowd=True will be ignored during evaluation.
#(optionally) masks (UInt8Tensor[N, H, W]): The segmentation masks for each one of the objects
#(optionally) keypoints (FloatTensor[N, K, 3]): For each one of the N objects, it contains the K keypoints in [x, y, visibility] format, defining the object. visibility=0 means that the keypoint is not visible. Note that for data augmentation, the notion of flipping a keypoint is dependent on the data representation, and you should probably adapt references/detection/transforms.py for your new keypoint representation
#If your model returns the above methods, they will make it work for both training and evaluation, and will use the evaluation scripts from pycocotools.
#
#Additionally, if you want to use aspect ratio grouping during training (so that each batch only contains images with similar aspect ratio), then it is recommended to also implement a get_height_and_width method, which returns the height and the width of the image. If this method is not provided, we query all elements of the dataset via __getitem__ , which loads the image in memory and is slower than if a custom method is provided.
#-----------------------------------------------------------------------------------------

import os, sys
import glob 
import numpy as np
import math
import logging
import torch
import json
from PIL import Image, ImageEnhance
import random
import re
from argparse import ArgumentParser, SUPPRESS
import time
#-----------------------------------------------------------------------------------------
#curr_script_path = os.path.dirname(os.path.abspath(__file__))
#flexnet_path = os.path.normpath(os.path.join(curr_script_path, ".."))
#flexnet_host_path = os.path.normpath(os.path.join(flexnet_path, ".."))
#sys.path.append(flexnet_host_path)
#-----------------------------------------------------------------------------------------
from v_utils.v_contour import *
from v_utils.v_polygons import *
#-----------------------------------------------------------------------------------------
from v_utils.v_arg import arg2boolAct, print_cfg_dict
#-----------------------------------------------------------------------------------------

class MRIDataset(object):
    _min_help_comps = 3

    @staticmethod
    def parse_arguments(argv):
        
        # possible choices:
        _crop_types = ['center', 'random']
        _req_refs_vals = ['none', 'all', 'some']
        _logging_levels = logging._levelToName.keys()
        _force_order_vals = ['no', 'up', 'down']
        _component_convert_mode_vals = ["ToGrayscale", "ToGrayscaleFromLabels",] 
        _component_missing_action_vals = ["skip", "copy_cmp0", "fill_zeros", 'try_reload_label_else_fill_zeros']
        _tf = [True, False]
        _polyRef_resample_vals = ['nearest', 'bilinear']
        
        cmp_ids = [(int(s[5]) if s[6]=="_" else int(s[5:7]))  for s in argv if (type(s) is str and (s.find("--cmp")==0) and len(s) > 7)]
        max_cmp_id_found = max(cmp_ids) if len(cmp_ids)!=0 else -1

        parser = ArgumentParser(prog = "MRIDataset", argument_default=SUPPRESS)
        #sys.argv

        gna = parser.add_argument_group('general arguments')
        gna.add_argument   ("-ses" , "--session_dirs"                   , default=['*/*']             , type=str,   nargs='*', required=False,                 help="lista user/session") 
        gna.add_argument   ("--crop_size"                               , default=None                , type=int,   nargs=2  , required=False, metavar='I'   , help="Domyslnie jest jednak ustawiony na najwiekszy rozmiar w zbiorze. [0,0] wymusza wyrownanie rozmiarow wszystkich obrazow poprzez ich rozszerzenie i jest domyslne gdy \"batch_size\" > 1")
        gna.add_argument   ("--do_pre_trans_crop"                       , default=True                , action=arg2boolAct   , required=False, metavar='B'   , help="True: zamiast jednej operacji crop, po transformacjach, wykonywane sa dwie, przy czym pierwsza to tzw. pre-crop ktora zmniejsza obraz wejsciowy jeszcze przed transformacjami co moze je przyspieszyc gdy obrazy wejsciowe sa bardzo duze. Drugi crop w polaczeniu z pierwszym daje ten sam efekt co wczesniejszy pojedynczy crop.")
        gna.add_argument   ("--crop_type"                               , default="center"            , type=str,              required=False, choices=_crop_types,         help="sposob wycinania crop_size. 'random' wycina obszar w losowym miejscu a 'center' ze srodka obrazu. 'random' dla 'train'==True jest losowany dla kazdego powtorzenia, dla 'train'==False jest zawsze taki sam dla danego obrazu (generator jest inicjowany sciezka obrazu)")
        gna.add_argument   ("--force_size_is_multiple_of"               , default=32                  , type=int,              required=False, metavar='I'   , help="wymuszenie wyrowanania rozmiaru obrazu, przez rozszerzenie, do danej wartosci w X i Y. Wymaga tego Flexnet ktory decymuje obraz, np. 32 razy.")
        gna.add_argument   ("--force_order"                             , default='up'                , type=str,              required=False, choices=_force_order_vals,   help="")
        gna.add_argument   ("--logging_level"                           , default=logging.WARNING     , type=int,              required=False, choices=_logging_levels,     help="")
        gna.add_argument   ("--do_report_dataset_getitem_time"          , default=False               , action=arg2boolAct   , required=False, metavar='B'   , help="True: Raportuje czas wczytywania komponentow, labeli i ich transformacji.")
        

        cma = parser.add_argument_group('components arguments')
        for cid in range(max(MRIDataset._min_help_comps, max_cmp_id_found+1)):
            cma.add_argument   ("--cmp{}_imgDir_root".format(cid)                                     , type=str             , required=cid==0, metavar='PATH',help="folder bedacy korzeniem dla folderow z obrazami wejsciowymi")
            cma.add_argument   ("--cmp{}_imgDir_sufix".format(cid)                                    , type=str             , required=False,  metavar='STR', help="jezeli wewnatrz user/session jest jeszcze jeden folder np. upsampled to nalezy to podac tu")
            cma.add_argument   ("--cmp{}_img_sufix".format(cid)                                       , type=str             , required=cid==0, metavar='STR', help="sufix wejsciowych obrazow")
            cma.add_argument   ("--cmp{}_convert_mode".format(cid)                                    , type=str             , required=False, choices=_component_convert_mode_vals, help="")
            cma.add_argument   ("--cmp{}_img_idx_offset".format(cid)                                  , type=int             , required=False, metavar='I'   , help="")
            cma.add_argument   ("--cmp{}_missing_action".format(cid)                                  , type=str             , required=False, choices=_component_missing_action_vals, help="")
        cma.add_argument   ("--skip_inputs_withouts_all_comps"          , default=True                , action=arg2boolAct   , required=False, metavar='B'   , help="True: Obraz wejsciowy trafia do bazy tylko gdy wszystkie komponenty sa dostepne. False: brakujace skladowe zastepowane sa przez skladowa R (0).")
        cma.add_argument   ("--descr_fn"                                , default='description.json'  , type=str,              required=False, metavar='FILENAME',  help="nazwa pliku z folderu sesji ktore w szczegolnosci zawiera info o przesuniecie [X,Y] obrazow wejsciowych wzgledem oryginału z dicom, liczone w punktach (pole crop_roi_pos)")
        cma.add_argument   ("--do_set_id_zero_if_fn_parse_fails"        , default=False               , action=arg2boolAct   , required=False, metavar='B'   , help="dopuszcza pliki ktorych nazwa nie zaczyna się od numeru ciecia. Dla nich przyjmuje numer ciecia 0.")
          

        rfa = parser.add_argument_group('references arguments')
        rfa.add_argument   ("--req_refs_level"                          , default='none'              , type=str,              required=False, choices=_req_refs_vals,      help="")
        rfa.add_argument   ("--ds_polygon_clss", "--ds_polygon_types"                                 , type=str,   nargs='*', required=True,          metavar='STR' ,      help="lista klas segmentow. Jednoczesnie nazwa folderow z referencyjnymi poligonami") 
        #me_group = rfa.add_mutually_exclusive_group()
        #me_group.add_argument   ("--ds_polygon_clss", "--ds_polygon_types"                                                       , type=str,   nargs='*', required=False,  metavar='STR' ,      help="lista klas segmentow. Jednoczesnie nazwa folderow z referencyjnymi poligonami") 
        #me_group.add_argument   ("--ds_polygon_types"                                                      , type=str,   nargs='*', required=False,  metavar='STR' ,      help="lista klas segmentow. Jednoczesnie nazwa folderow z referencyjnymi poligonami") 
        rfa.add_argument   ("--ds_polyRefDirs_root"                                                   , type=str,              required=False, metavar='PATH',      help="folder bedacy korzeniem dla folderow z poligonami referencyjnymi")
        rfa.add_argument   ("--ds_polyRef_sufix"                        , default='_polygons.json'    , type=str,              required=False, metavar='STR' ,      help="sufix wejsciowych plikow referencyjnych poligonow")
        rfa.add_argument   ("--ds_polyRef_descr_fn"                     , default='description.json'  , type=str,              required=False, metavar='FILENAME',  help="nazwa pliku z folderu sesji ktore w szczegolnosci zawiera info o przesuniecie [X,Y] poligonow z dataset wzgledem oryginału z dicom, liczone w punktach (pole crop_roi_pos)")
        
        tra = parser.add_argument_group('training arguments')
        tra.add_argument   ("--train"                                   , default=False               , action=arg2boolAct   , required=False, metavar='B'   , help="fukcja transform bedzie wykonywac losowe transformacje, tj. hflip, vflip, skalowanie obrazu i rotacja")
        tra.add_argument   ("--train_tr_resize_range"                   , default=None                , type=float,   nargs=2, required=False, metavar='F'   , help="skalowanie rozmiaru o wartosci losowanej z przedzialu <F0, F1>")
        tra.add_argument   ("--train_tr_resize_xy_ind"                  , default=False               , action=arg2boolAct   , required=False, metavar='B'   , help="skala losowana niezaleznie dla x i y")
        tra.add_argument   ("--train_rotation_range", "--train_tr_rotation_range", default=0.0        , type=float           , required=False, metavar='F'   , help="rotacja o losowy kat z zakrsu <-F, +F> stopni")
        tra.add_argument   ("--train_flips"                             , default=True                , action=arg2boolAct   , required=False, metavar='B'   , help="fukcja transform bedzie wykonywac losowo hflip, vflip")
        tra.add_argument   ("--train_polyRef_resample"                  , default='nearest'           , type=str,              required=False, choices=_polyRef_resample_vals,      help="Resampling mode <'nearest', 'bilinear'> for trnsformed reference polygons (for input components'bilinear' is always used)")
        
        
        obsolete_parser = ArgumentParser()
        # wzor jak dodawac argumenty obsolete. Ten ponizej nie jest obsolete (jest rowniez aliasem dla nowego parametru train_rotation_range) a jedynie jest wzorcem dla przyszlych parametrow obsolete
        obsolete_parser.add_argument("--train_tr_rotation_range"                                              , required=False, help="JUZ NIEUZYWANY! Uzyj train_rotation_range")
    
        if not(("-h" in argv) or ("--help" in argv)): 
            # get training arguments
            ds_args, rem_args = parser.parse_known_args(argv)
            ob_args, rem_args = obsolete_parser.parse_known_args(rem_args)

            obsole_args = [{k:vars(ob_args)[k]} for k in vars(ob_args).keys() if (not vars(ob_args)[k] is None)]
            for a in obsole_args:
                logging.warning(" Found obsolate training argument {}".format(a))

            return ds_args, rem_args
        else: 
            # help
            return parser.format_help()

    def __init__(self,
                 dataset_desc
                 ):

        self.force_order                    = dataset_desc.force_order
        self.train                          = dataset_desc.train
        self.train_tr_resize_range          = dataset_desc.train_tr_resize_range
        self.train_polyRef_resample         = Image.NEAREST if (dataset_desc.train_polyRef_resample == 'nearest') else Image.BILINEAR
        self.train_tr_resize_xy_ind         = dataset_desc.train_tr_resize_xy_ind
        self.train_rotation_range           = dataset_desc.train_rotation_range
        self.train_flips                    = dataset_desc.train_flips
        self.req_refs_level                 = dataset_desc.req_refs_level
        self.skip_inputs_withouts_all_comps = dataset_desc.skip_inputs_withouts_all_comps
        self.logging_level                  = dataset_desc.logging_level
        self.do_pre_trans_crop              = dataset_desc.do_pre_trans_crop
        self.save_logging_level()
        self.warnings_once = []

        self.num_class = len(dataset_desc.ds_polygon_clss)
        ref_data_given = (hasattr(dataset_desc, 'ds_polyRefDirs_root') and not(dataset_desc.ds_polyRefDirs_root is None))

        if((self.req_refs_level != 'none') and not ref_data_given):
            logging.error("Param 'req_refs_level' is set to {} but 'ds_polyRefDirs_root' is not given. Set req_refs_level to 'none' or set 'ds_polyRefDirs_root'. Exit execution".format(self.req_refs_level))
            self.restore_logging_level()
            sys.exit(40)

        self.has_ref_masks = ref_data_given 

        self.paths = []
        self._entries_without_ref = []
        self._entries_without_comp = []
        self._empty_refs = []
        
        self.raport_time = dataset_desc.do_report_dataset_getitem_time
        if self.raport_time:
            self.tr_time_cmps_read = 0
            self.tr_time_labs_read = 0
            self.tr_time_transform = 0
            self.tr_time_nums      = 0
            self.ev_time_cmps_read = 0
            self.ev_time_labs_read = 0
            self.ev_time_transform = 0
            self.ev_time_nums      = 0

        # find number of componets
        max_cmp_id = MRIDataset.find_max_cmp_id(dataset_desc)
            
        # create components describtion dicts and infer parameters that were not given directly
        cmp_dicts = []
        dataset_dict = vars(dataset_desc)
        for cid in range(max_cmp_id+1):
            cmp_dicts.append({})
            cmp_dicts[cid][   "imgDir_root"] = dataset_dict["cmp{}_imgDir_root".format(cid)   ] if ("cmp{}_imgDir_root".format(cid)    in dataset_dict.keys()) else cmp_dicts[cid-1]["imgDir_root" ] 
            cmp_dicts[cid][  "imgDir_sufix"] = dataset_dict["cmp{}_imgDir_sufix".format(cid)  ] if ("cmp{}_imgDir_sufix".format(cid)   in dataset_dict.keys()) else(cmp_dicts[cid-1]["imgDir_sufix"] if cid>0 else "")
            cmp_dicts[cid][     "img_sufix"] = dataset_dict["cmp{}_img_sufix".format(cid)     ] if ("cmp{}_img_sufix".format(cid)      in dataset_dict.keys()) else cmp_dicts[cid-1]["img_sufix"   ] 
            cmp_dicts[cid]["img_idx_offset"] = dataset_dict["cmp{}_img_idx_offset".format(cid)] if ("cmp{}_img_idx_offset".format(cid) in dataset_dict.keys()) else 0
            cmp_dicts[cid][  "convert_mode"] = dataset_dict["cmp{}_convert_mode".format(cid)  ] if ("cmp{}_convert_mode".format(cid)   in dataset_dict.keys()) else "ToGrayscale"
            cmp_dicts[cid]["missing_action"] = dataset_dict["cmp{}_missing_action".format(cid)] if ("cmp{}_missing_action".format(cid) in dataset_dict.keys()) else None 
            if cmp_dicts[cid]["missing_action"] is None:
                if cid == 0: 
                    if self.skip_inputs_withouts_all_comps:
                        cmp_dicts[cid]["missing_action"] = "skip" 
                    else:
                        cmp_dicts[cid]["missing_action"] = "skip_all"
                else:
                    if self.skip_inputs_withouts_all_comps:
                        cmp_dicts[cid]["missing_action"] = "skip"  
                    elif cmp_dicts[cid]["convert_mode"] == "ToGrayscaleFromLabels":
                        cmp_dicts[cid]["missing_action"] = "try_reload_label_else_fill_zeros"  
                    else:
                        cmp_dicts[cid]["missing_action"] = "copy_cmp0"
   
        # solve multi-selection filters with "*" / "?"
        ds_imgDir_root = cmp_dicts[0]["imgDir_root"]
        user_session_dirs = expand_session_dirs(dataset_desc.session_dirs, ds_imgDir_root)

        logging.info("Search user/session dirs:")
        for session_dir in user_session_dirs:
            logging.info("   - {}".format(session_dir))
            
        for session_dir in user_session_dirs:
            
            # find original translation - required for spatial matching of input images with reference polygons
            #translation_for_src_comps = [0, 0]
            size_of_src_comps = None
            for cidx, cmp_dict in enumerate (cmp_dicts):
                descrDir = os.path.normpath(os.path.join(cmp_dict["imgDir_root"], session_dir))
                descr_fn = os.path.normpath(os.path.join(descrDir, dataset_desc.descr_fn))
                try:
                    with open (descr_fn) as f:
                        descr_dict_data= json.load(f)
                        if(cidx == 0):
                            size_of_src_comps = descr_dict_data["crop_roi_size"]
                        if not(descr_dict_data["crop_roi_pos"] is None):
                            #translation_for_src_comps   = descr_dict_data["crop_roi_pos"]
                            cmp_dict["translated_pxpy"] = descr_dict_data["crop_roi_pos"]
                        else:
                            cmp_dict["translated_pxpy"] = [0, 0]
                except:
                    cmp_dict["translated_pxpy"] = [0, 0]
                    wrn_type = "Component {} without translation info".format(cidx)
                    if not(wrn_type in self.warnings_once):
                        self.warnings_once.append(wrn_type)
                        logging.warning("Expected to find {} file with translation info for componet {} but did not. Report just once".format(descr_fn, cidx))
                        

            # gather all files for all components into a dictionary
            imgs_comps_dict = {}
            for cidx, cmp_dict in enumerate (cmp_dicts):
                
                imgDir = os.path.normpath(os.path.join(cmp_dict["imgDir_root"], session_dir, cmp_dict["imgDir_sufix"]))
                
                iname_pattern  = imgDir + '/*{}*'.format(cmp_dict["img_sufix"])
                iname_pattern  = os.path.normpath(iname_pattern)

                cmpImg_paths_s = glob.glob(iname_pattern)

                cmpImg_paths_s.sort()
                for cmpImg_file_id, cmpImg_file_path in enumerate(cmpImg_paths_s):

                    cmpImg_file_name = os.path.basename(cmpImg_file_path)
                    file_id, file_id_len, file_id_start, file_id_last = MRIDataset._get_img_id_from_filename(cmpImg_file_name, do_set_zero_if_parse_fails = dataset_desc.do_set_id_zero_if_fn_parse_fails)

                    cmp_idx_offset = cmp_dicts[cidx]["img_idx_offset"]

                    dst_image_idx = file_id-cmp_idx_offset
                    if not(dst_image_idx in imgs_comps_dict.keys()):
                        imgs_comps_dict[dst_image_idx] = [{} for comp in cmp_dicts]
                        #file_name_pattern = os.path.splitext(cmpImg_file_name)[0]
                        id_frm_str = "{{:0{}}}".format(file_id_len)
                        file_name_pattern = cmpImg_file_name[0:file_id_start] + id_frm_str.format(dst_image_idx)
                        imgs_comps_dict[dst_image_idx][0]["file_name_pattern"] = file_name_pattern
                        imgs_comps_dict[dst_image_idx][0]["session_sub_dir"  ] = session_dir
                        
                    imgs_comps_dict[dst_image_idx][cidx]["src_comp_path_l"  ] = cmpImg_file_path
                    imgs_comps_dict[dst_image_idx][cidx]["src_image_id"     ] = file_id

            # from the imgs_comps_dict create a list of valid images (with required components) 
            session_path_dict_l = []
            for img_id in imgs_comps_dict.keys():
                img_comps_dict = imgs_comps_dict[img_id]

                has_comps = [(len(img_comp_dict) != 0) and ("src_comp_path_l" in img_comp_dict.keys()) for img_comp_dict in img_comps_dict]

                # non empty component 0 (R) is always required
                if(has_comps[0]):  
                    # check if all components are required 
                    if(not self.skip_inputs_withouts_all_comps) or (np.array(has_comps).all()):

                        contours_l = []
                        for ds_polygon_cls_idx, ds_polygon_cls in enumerate(dataset_desc.ds_polygon_clss):
                            contours_l.append({
                                "cls_name"                      : ds_polygon_cls,
                                "ref_polygon_path"              : ""
                                })
                        src_comps_path_l            = [(comp["src_comp_path_l"] if ("src_comp_path_l" in comp.keys()) else '') for comp in img_comps_dict ]
                        session_path_dict_l.append({"src_comps_path_l"              : src_comps_path_l, 
                                                    "src_comps_dict_l"              : [copy.copy(cmp_dicts[comp_id]) for comp_id, comp in enumerate(img_comps_dict) ], 
                                                    "src_image_id"                  : img_id, 
                                                    "src_comps_size"                : size_of_src_comps, 
                                                    "file_name_pattern"             : img_comps_dict[0]["file_name_pattern"],
                                                    "session_sub_dir"               : img_comps_dict[0]["session_sub_dir"  ], 
                                                    "cls_envs_list"                 : contours_l,
                                                    "src_comps_translated_pxpy"     : cmp_dicts[0]["translated_pxpy"]
                                                    })
                    else:
                        src_comps_path_l            = [(comp["src_comp_path_l"] if ("src_comp_path_l" in comp.keys()) else '') for comp in img_comps_dict ]
                        wrn_type = "Input without all components"
                        if not(wrn_type in self.warnings_once):
                            self.warnings_once.append(wrn_type)
                            logging.warning("Input for component 0 found but misssing for other. Full list: {}. Report just once. Full list is in ._entries_without_comp".format(src_comps_path_l))
                        
                        self._entries_without_comp.append({"cmp0 path" : src_comps_path_l[0], "other components paths": src_comps_path_l[1:]})
                   
            #if input polygons are given for the dataset than add them to the session_path_dict_l[polygon_file_id]["cls_envs_list"] dictionary
            if(self.has_ref_masks):
                
                # find original translation - required for spatial matching of input images with reference polygons
                translation_for_refs = [0, 0]
                descrDir = os.path.normpath(os.path.join(dataset_desc.ds_polyRefDirs_root,    session_dir))
                descr_fn      = os.path.normpath(os.path.join(descrDir, dataset_desc.ds_polyRef_descr_fn))
                try:
                    with open (descr_fn) as f:
                        descr_dict_data= json.load(f)
                        if("translated_pxpy" in descr_dict_data.keys()): # backward compatibility
                            translation_for_refs = descr_dict_data["translated_pxpy"]
                        else:
                            translation_for_refs = descr_dict_data["crop_roi_pos"] # newer version
                except:
                    wrn_type = "Reference polygons for {} without translation info".format(session_dir)
                    if not(wrn_type in self.warnings_once):
                        self.warnings_once.append(wrn_type)
                        logging.warning("Expected to find {} file with translation info (the crop_roi_pos field) for session {} but did not. Use [0, 0]. Report just once".format(descr_fn, session_dir))

                for ds_polygon_cls_idx, ds_polygon_cls in enumerate(dataset_desc.ds_polygon_clss):
                    polyDir      = os.path.normpath(os.path.join(dataset_desc.ds_polyRefDirs_root,    session_dir, ds_polygon_cls))
                    
                    cname_pattern       = polyDir + '/*{}'.format(dataset_desc.ds_polyRef_sufix)
                    cname_pattern       = os.path.normpath(cname_pattern)

                    polygon_paths= glob.glob(cname_pattern)
                    polygon_paths.sort()
                    
                    for polygon_file_id, polygon_file_path in enumerate(polygon_paths):
                        polygon_file_path_base = os.path.basename(polygon_file_path)
                        img_id, img_id_len, img_id_start, img_id_last = MRIDataset._get_img_id_from_filename(polygon_file_path_base, do_set_zero_if_parse_fails = dataset_desc.do_set_id_zero_if_fn_parse_fails)
                        # try to find the corresponding image_file
                        for path_dict in session_path_dict_l:
                            if(img_id == path_dict["src_image_id"]):
                                path_dict.update({"ref_polygon_translated_pxpy"   : translation_for_refs})
                                path_dict["cls_envs_list"][ds_polygon_cls_idx].update({"ref_polygon_path"               : polygon_file_path})
                                check_if_ref_is_not_empty = True
                                if(check_if_ref_is_not_empty):
                                    if(polygon_file_path.find(".png") != -1):
                                        #logging.debug("read from png file for a single class")                    
                                        #label_img = Image.open(polygon_file_path)
                                        empty = False
                                    else:
                                        with open (polygon_file_path) as f:
                                            contours_dict_data= json.load(f)
                                        my_polygons = v_polygons.from_dict(contours_dict_data)
                                        empty = len(my_polygons["polygons"]) == 0
                                    if(empty):
                                        wrn_type = "Empty polygon"
                                        if not(wrn_type in self.warnings_once):
                                            self.warnings_once.append(wrn_type)
                                            logging.warning(" Reference polygon {} is empty. Report just once. Full list is in ._empty_refs".format(polygon_file_path))
                                        self._empty_refs.append(polygon_file_path)
                        
                                #break
                            if(img_id < path_dict["src_image_id"]):
                                break
        
            if(self.force_order == "down"):
                session_path_dict_l.reverse()
            self.paths.extend(session_path_dict_l)
            
        if(len(self.paths) == 0):
            logging.warning("Dataset has no src img! Please check dataset description and check if definition of components is correct. Return.")   
            self.restore_logging_level() 
            return

        if(self.has_ref_masks):
            has_some_ref_num = 0
            has_no_ref_num = 0
            has_all_ref_num = 0
            for path_dict_id in range(len(self.paths)-1, -1, -1): 
                path_dict = self.paths[path_dict_id]
                has_all_refs = True
                missing_ref_clss = []
                has_some_ref = False
                for ds_polygon_cls_idx, ds_polygon_cls in enumerate(dataset_desc.ds_polygon_clss): 
                    if path_dict["cls_envs_list"][ds_polygon_cls_idx]["ref_polygon_path"] == "":
                        has_all_refs = False
                        missing_ref_clss.append(ds_polygon_cls)
                        wrn_type = "Flag has_ref_masks but no ref mask has been found"
                        if not (self.req_refs_level == 'none') and not(wrn_type in self.warnings_once):
                            self.warnings_once.append(wrn_type)
                            logging.warning("'req_refs_level' is set to {} but but no corresponding {} ref mask has been found for src img {}. Report just once. Full list is in ._entries_without_ref".format(self.req_refs_level, ds_polygon_cls, path_dict["src_comps_path_l"][0]))
                        
                    else:
                        has_some_ref = True
                if(len(missing_ref_clss) != 0):
                    self._entries_without_ref.append({"cmp0 path" : path_dict["src_comps_path_l"][0], "missing ref type": missing_ref_clss})
                if(has_some_ref and not has_all_refs):
                    has_some_ref_num += 1
                elif(has_all_refs):
                    has_all_ref_num += 1
                elif(not has_some_ref):
                    has_no_ref_num += 1

                if(self.req_refs_level == 'some') and not has_some_ref:
                    wrn_type = "Remove due to not meeting the (req_refs_level == 'some') condition"
                    if not(wrn_type in self.warnings_once):
                        self.warnings_once.append(wrn_type)
                        logging.warning("'req_refs_level' is set to 'some' but no corresponding {} ref mask has been found for src img {}. Report just once. Full list is in ._entries_without_ref".format(missing_ref_clss, path_dict["src_comps_path_l"][0]))
                        logging.warning("'req_refs_level' is set to 'some', therefore I remove src img {} from the dataset. Report just once. Full list is in ._entries_without_ref".format(path_dict["src_comps_path_l"][0]))
                    self.paths.pop(path_dict_id)
                elif (self.req_refs_level == 'all') and not has_all_refs:
                    wrn_type = "Remove due to not meeting the (req_refs_level == 'all') condition"
                    if not(wrn_type in self.warnings_once):
                        self.warnings_once.append(wrn_type)
                        logging.warning("'req_refs_level' is set to 'all' but one corresponding {} ref mask(s) has been found for src img {}. Report just once. Full list is in ._entries_without_ref".format(missing_ref_clss, path_dict["src_comps_path_l"][0]))
                        logging.warning("'req_refs_level' is set to 'all', therefore I remove src img {} from the dataset. Report just once. Full list is in ._entries_without_ref".format(path_dict["src_comps_path_l"][0]))
                    self.paths.pop(path_dict_id)
                    
            total_num = has_no_ref_num+has_all_ref_num+has_some_ref_num
            logging.info(" {} / {} src img have 0 refs,  {} / {} src img have some refs,   {} / {} src img have all refs".format(has_no_ref_num, total_num, has_some_ref_num, total_num, has_all_ref_num, total_num))
            if (self.req_refs_level != 'none'):
                    logging.warning("'req_refs_level' is set to '{}', therefore {} objects are in the dataset ".format(self.req_refs_level, len(self.paths)))
             
            if (len(self.paths) == 0) and (self.req_refs_level != 'none'):
                logging.error("'req_refs_level' is set to '{}' and the dataset is empty.".format(self.req_refs_level))
                 
                
        for path in self.paths:
            if path['src_comps_size'] is None:
                for cmpFile_path in path["src_comps_path_l"]:
                    if(cmpFile_path != ''):
                        try:
                            with Image.open(cmpFile_path) as img:
                                path['src_comps_size'] = img.size
                        except:
                            path['src_comps_size'] = [0,0]
                        break

        if(type(dataset_desc.crop_size) is list and (dataset_desc.crop_size[0] == 0) and (dataset_desc.crop_size[1] == 0)):
            max_w = 0
            max_h = 0
            for path in self.paths:
                w, h = path['src_comps_size']
                max_w = max(max_w, w)
                max_h = max(max_h, h)
            crop_size = [max_w, max_h]
            logging.info("crop_size automatically set to [max_w, max_h] from dataset {}".format(crop_size) )
        else:
            crop_size = dataset_desc.crop_size
                
        self.general_crop_size = np.array(crop_size) if not(crop_size is None) else None
        self.force_size_is_multiple_of = dataset_desc.force_size_is_multiple_of
        if not (self.general_crop_size is None):
            changed = False
            new_size = copy.copy(self.general_crop_size)
            for size_id, size in enumerate(self.general_crop_size):
                if((not (size is None)) and (size//self.force_size_is_multiple_of) != (size/self.force_size_is_multiple_of)):
                    new_size[size_id] = (size//self.force_size_is_multiple_of+1) * self.force_size_is_multiple_of
                    changed = True
            if(changed):
                logging.warning("crop_size {} is not a multiple of {}. Change it to {}".format(self.general_crop_size, self.force_size_is_multiple_of, new_size) )
                self.general_crop_size = new_size

        self.crop_type = dataset_desc.crop_type

        self.restore_logging_level()

    def find_max_cmp_id(dataset_desc):
        if(type(dataset_desc) == dict):
            dataset_dict = dataset_desc
        else:
            dataset_dict = vars(dataset_desc)
        max_cmp_id = -1
        for key in dataset_dict.keys():
            if(key.find("cmp") == 0):
                id_end = key.find('_')
                cmp_id = int(key[3:id_end])
                if(cmp_id > max_cmp_id):
                    max_cmp_id = cmp_id
        return max_cmp_id


    def save_logging_level(self):
        if not(self.logging_level is None):
            logger = logging.getLogger()
            self.org_logging_levels = [logger.level] #logging.DEBUG
            if(not self.logging_level is None):
                logger.setLevel(self.logging_level)
            for handler in logger.handlers:
                self.org_logging_levels.append(handler.level)
                handler.setLevel(self.logging_level)
                
    def restore_logging_level(self):
        if not(self.logging_level is None):
            logger = logging.getLogger()
            logger.setLevel(self.org_logging_levels[0])
            for handler_id, handler in enumerate(logger.handlers):
                if(len(self.org_logging_levels) > handler_id):
                    handler.setLevel(self.org_logging_levels[handler_id+1])

    def _get_img_id_from_filename(fname, do_set_zero_if_parse_fails = False):
        digits_pattern = r'(\d{1,})' #match 1 or more digits
        match_obj = re.search(r'^\d{1,}', fname)
        if((match_obj is None) or (len(match_obj.regs) != 1)):
            if do_set_zero_if_parse_fails:
                img_id_start = 0
                img_id_last  = 0
                img_id_str   = ""
                img_id       = 0
                img_id_len   = 0
            else:
                logging.error("Could no parse image id in file name {}".format(fname))
                sys.exit(1)
        else:
            img_id_start = match_obj.regs[0][0]
            img_id_last  = match_obj.regs[0][1]-1
            img_id_str   = match_obj.string[img_id_start : img_id_last+1]
            img_id = int(img_id_str)
            img_id_len = len(img_id_str)

        return img_id, img_id_len, img_id_start, img_id_last

    def find_valid_file_path(self, cmp_dict, src_image_id, file_name_pattern, session_dir):
        srch_img_id = src_image_id + cmp_dict['img_idx_offset']
                  
        imgDir = os.path.normpath(os.path.join(cmp_dict["imgDir_root"], session_dir, cmp_dict["imgDir_sufix"]))
                
        iname_pattern  = imgDir + '/*{}*'.format(cmp_dict["img_sufix"])
        iname_pattern  = os.path.normpath(iname_pattern)

        cmpImg_paths_s = glob.glob(iname_pattern)

        for cmpImg_file_id, cmpImg_file_path in enumerate(cmpImg_paths_s):
            cmpImg_file_name = os.path.basename(cmpImg_file_path)
            file_id, file_id_len, file_id_start, file_id_last = MRIDataset._get_img_id_from_filename(cmpImg_file_name)

            if(srch_img_id == file_id):
                return cmpImg_file_path

        return ""
    

    def __getitem__(self, idx):
                
        if self.raport_time:
            start_time = time.time()

        paths = self.paths[idx]
        # load images as masks
        imgs_path_l = paths["src_comps_path_l"]
        
        polygon_path_dicts =paths["cls_envs_list"]
        comp_img_l =[]
        
        for cmp_id in range(0, len(paths["src_comps_path_l"])):
            if(paths["src_comps_path_l"][cmp_id] == ''):
                if(paths['src_comps_dict_l'][cmp_id]['missing_action'] == "try_reload_label_else_fill_zeros"):
                    found_comp_path = self.find_valid_file_path(paths['src_comps_dict_l'][cmp_id], paths["src_image_id"], paths["file_name_pattern"], paths["session_sub_dir"])
                    if(found_comp_path==""):
                        paths['src_comps_dict_l'][cmp_id]['missing_action'] = "fill_zeros"
                        logging.info(" not found file for comp {} of image {} during reload routine: fill with zeros".format(cmp_id, paths["src_image_id"]))
                    else:
                        paths["src_comps_path_l"][cmp_id] = found_comp_path
                        logging.info(" did found file for comp {} of image {} during reload routine: {}".format(cmp_id, paths["src_image_id"], found_comp_path))
                        
            if(paths["src_comps_path_l"][cmp_id] == ''):
                if(paths['src_comps_dict_l'][cmp_id]['missing_action'] == "copy_cmp0"):
                    comp_new = copy.deepcopy(comp_img_l[0])
                elif(paths['src_comps_dict_l'][cmp_id]['missing_action'] == "fill_zeros"):
                    w, h = comp_img_l[0].size
                    if(paths['src_comps_dict_l'][cmp_id]['convert_mode'] == 'ToGrayscaleFromLabels'):
                        comp_new = Image.new('RGB', (w, h), (0,0,0))
                    else:
                        comp_new = Image.new('L', (w, h), 0)
            else:
                comp_new = Image.open(paths["src_comps_path_l"][cmp_id])
                if cmp_id > 0:
                    # check if translation between components differs
                    cmpC_tr = paths["src_comps_dict_l"][cmp_id]["translated_pxpy"]
                    cmp0_tr = paths["src_comps_dict_l"][     0]["translated_pxpy"]
                    cmp0_w, cmp0_h = comp_img_l[     0].size
                    cmpC_w, cmpC_h = comp_new.size
                    if((cmp0_tr[0] != cmpC_tr[0]) or (cmp0_tr[1] != cmpC_tr[1]) or (cmp0_w != cmpC_w) or (cmp0_h != cmpC_h)):
                        # translate polygons according to translated_pxpy info for src_cmp[0] and current src_cmp so both match spatialy
                        dt = [cmpC_tr[0] - cmp0_tr[0], cmpC_tr[1] - cmp0_tr[1]]
                        cmp0_w, cmp0_h = comp_img_l[     0].size
                        cmpC_w, cmpC_h = comp_new.size
                        #my_polygons.move2point(dt)
                        pad_l = int(dt[0]              )
                        pad_r = int(cmp0_w - (cmpC_w + dt[0]))
                        pad_t = int(dt[1]              )
                        pad_b = int(cmp0_h - (cmpC_h + dt[1]))
                        comp_new_np = np.array(comp_new)
                        has_more_cmps = (len(comp_new_np.shape) > 2) and (comp_new_np.shape[2] > 1)
                        if(pad_l >= 0 and pad_r >= 0 and pad_t >= 0 and pad_b >= 0):
                            # padding
                            padding = ((pad_t, pad_b),(pad_l, pad_r), (0,0)) if has_more_cmps else ((pad_t, pad_b),(pad_l, pad_r))
                            comp_new_np_tr = np.pad(comp_new_np, pad_width = padding, mode='constant', constant_values=0)
                        elif(pad_l <= 0 and pad_r <= 0 and pad_t <= 0 and pad_b <= 0):
                            # cropping
                            comp_new_np_tr = comp_new_np[-pad_t: int(cmp0_h)-pad_t, -pad_l: int(cmp0_w)-pad_l]
                        else:
                            # combination of padding and cropping needed
                            _pad_l = pad_l if pad_l>=0 else 0
                            _pad_r = pad_r if pad_r>=0 else 0
                            _pad_t = pad_t if pad_t>=0 else 0
                            _pad_b = pad_b if pad_b>=0 else 0
                            padding = ((_pad_t, _pad_b),(_pad_l, _pad_r), (0,0)) if has_more_cmps else ((_pad_t, _pad_b),(_pad_l, _pad_r))
                            tmp = np.pad(comp_new_np, pad_width = padding, mode='constant', constant_values=0)
                            
                            _pad_l = pad_l if pad_l<=0 else 0
                            _pad_r = pad_r if pad_r<=0 else 0
                            _pad_t = pad_t if pad_t<=0 else 0
                            _pad_b = pad_b if pad_b<=0 else 0
                            comp_new_np_tr = tmp[-_pad_t: int(cmp0_h)-_pad_t, -_pad_l: int(cmp0_w)-_pad_l]
                        comp_new = Image.fromarray(comp_new_np_tr)

            

            convert_mode = paths['src_comps_dict_l'][cmp_id]['convert_mode']
            if(convert_mode == 'ToGrayscale'):
                if(comp_new.mode != 'L'):
                    comp_new = comp_new.convert('L')
            elif(convert_mode == 'ToGrayscaleFromLabels'):
                #comp_new.save("convertedToGrayscaleComp_org.png")
                if(comp_new.mode == 'RGB'):
                    R,G,B = comp_new.split()
                elif(comp_new.mode == 'RGBA'):
                    R,G,B,A = comp_new.split()
                else:
                    logging.error("file {} for component {} has convertion mode specified to {} but it is {}, RGB or RGBA expected!".format(paths["src_comps_path_l"][cmp_id], cmp_id, convert_mode, comp_new.mode))
                    sys.exit(41)
                labels_L_np = np.where((np.array(R) != 0) | (np.array(B) != 0), np.uint8(255), np.uint8(0))
                comp_new = Image.fromarray(labels_L_np)
                #comp_new.save("convertedToGrayscaleComp_out.png")
            else:
                logging.error("Unknown convertion mode {} specified for component {} for file {}!".format(convert_mode, cmp_id, paths["src_comps_path_l"][cmp_id]))
                sys.exit(42)
                #wrn_type = "Comp {} was RGB and I converted".format(cmp_id)
                #if not(wrn_type in self.warnings_once):
                #    self.warnings_once.append(wrn_type)
                #    logging.warning("file {} for component {} is of mode {}. I am converting it to grayscale. Warning only once!".format(paths["src_comps_path_l"][cmp_id], cmp_id, comp_new.mode))
            comp_img_l.append(comp_new)
        
        if self.raport_time:
            elapsed_time = time.time() - start_time
            if(self.train):
                self.tr_time_cmps_read += elapsed_time
                self.tr_time_nums += 1    
                logging.info(f"  1. cmps   read {round(elapsed_time, 5)} s, aver {round(self.tr_time_cmps_read/self.tr_time_nums, 5)} s")
            else:
                self.ev_time_cmps_read += elapsed_time
                self.ev_time_nums += 1 
                logging.info(f"                                             1. cmps   read {round(elapsed_time, 5)} s, aver {round(self.ev_time_cmps_read/self.ev_time_nums, 5)} s") 
            start_time = time.time() 
        
        w, h = comp_img_l[0].size
                    
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        cls_id           = 1
        cls_ids          = []
        #cls_numpy_masks  = []
        cls_pil_masks    = []
        #cls_labels       = []
        cls_boxes        = []

        for polygon_path_dict in polygon_path_dicts:
            if(polygon_path_dict["ref_polygon_path"] == ""):
                cls_pil_masks.append(None)
            elif(polygon_path_dict["ref_polygon_path"].find(".png") != -1):
                logging.debug("read from png file for a single class")                    
                label_new = Image.open(polygon_path_dict["ref_polygon_path"])
                
                st = paths["src_comps_dict_l"][     0]["translated_pxpy"]
                rt = paths["ref_polygon_translated_pxpy"]
                if((st[0] != rt[0]) or (st[1] != rt[1])):
                    dt = [rt[0] - st[0], rt[1] - st[1]]
                    my_polygons.move2point(dt)
                    
                # translate polygons according to translated_pxpy info for src_img and ref_polygons so both match spatialy
                cmpC_tr = paths["ref_polygon_translated_pxpy"]
                cmp0_tr = paths["src_comps_dict_l"][     0]["translated_pxpy"]
                cmp0_w, cmp0_h = comp_img_l[     0].size
                cmpC_w, cmpC_h = label_new.size
                if((cmp0_tr[0] != cmpC_tr[0]) or (cmp0_tr[1] != cmpC_tr[1]) or (cmp0_w != cmpC_w) or (cmp0_h != cmpC_h)):
                    # translate polygons according to translated_pxpy info for src_cmp[0] and current src_cmp so both match spatialy
                    dt = [cmpC_tr[0] - cmp0_tr[0], cmpC_tr[1] - cmp0_tr[1]]
                    cmp0_w, cmp0_h = comp_img_l[     0].size
                    cmpC_w, cmpC_h = label_new.size
                    #my_polygons.move2point(dt)
                    pad_l = int(dt[0]              )
                    pad_r = int(cmp0_w - (cmpC_w + dt[0]))
                    pad_t = int(dt[1]              )
                    pad_b = int(cmp0_h - (cmpC_h + dt[1]))
                    comp_new_np = np.array(label_new)
                    has_more_cmps = (len(comp_new_np.shape) > 2) and (comp_new_np.shape[2] > 1)
                    if(pad_l >= 0 and pad_r >= 0 and pad_t >= 0 and pad_b >= 0):
                        # padding
                        padding = ((pad_t, pad_b),(pad_l, pad_r), (0,0)) if has_more_cmps else ((pad_t, pad_b),(pad_l, pad_r))
                        comp_new_np_tr = np.pad(comp_new_np, pad_width = padding, mode='constant', constant_values=0)
                    elif(pad_l <= 0 and pad_r <= 0 and pad_t <= 0 and pad_b <= 0):
                        # cropping
                        comp_new_np_tr = comp_new_np[-pad_t: int(cmp0_h)-pad_t, -pad_l: int(cmp0_w)-pad_l]
                    else:
                        # combination of padding and cropping needed
                        _pad_l = pad_l if pad_l>=0 else 0
                        _pad_r = pad_r if pad_r>=0 else 0
                        _pad_t = pad_t if pad_t>=0 else 0
                        _pad_b = pad_b if pad_b>=0 else 0
                        padding = ((_pad_t, _pad_b),(_pad_l, _pad_r), (0,0)) if has_more_cmps else ((_pad_t, _pad_b),(_pad_l, _pad_r))
                        tmp = np.pad(comp_new_np, pad_width = padding, mode='constant', constant_values=0)
                            
                        _pad_l = pad_l if pad_l<=0 else 0
                        _pad_r = pad_r if pad_r<=0 else 0
                        _pad_t = pad_t if pad_t<=0 else 0
                        _pad_b = pad_b if pad_b<=0 else 0
                        comp_new_np_tr = tmp[-_pad_t: int(cmp0_h)-_pad_t, -_pad_l: int(cmp0_w)-_pad_l]
                    label_new = Image.fromarray(comp_new_np_tr)

            

                #convert_mode = paths['src_comps_dict_l'][cmp_id]['convert_mode']
                #if(convert_mode == 'ToGrayscale'):
                #    if(label_new.mode != 'L'):
                #        label_new = label_new.convert('L')
                #elif(convert_mode == 'ToGrayscaleFromLabels'):
                if True:
                    #comp_new.save("convertedToGrayscaleComp_org.png")
                    if(label_new.mode == 'RGB'):
                        R,G,B = label_new.split()
                    elif(label_new.mode == 'RGBA'):
                        R,G,B,A = label_new.split()
                    else:
                        logging.error("file {} for component {} has convertion mode specified to {} but it is {}, RGB or RGBA expected!".format(paths["src_comps_path_l"][cmp_id], cmp_id, convert_mode, comp_new.mode))
                        sys.exit(41)
                    labels_L_np = np.where((np.array(R) != 0) | (np.array(B) != 0), np.uint8(255), np.uint8(0))
                    label_new = Image.fromarray(labels_L_np)

                cls_pil_masks.append(label_new)

                cls_ids.append(cls_id)
                cls_id +=1
            else: #if(polygon_path_dict["ref_polygon_path"] != ""):
                logging.debug("read from json file for a single class")
                with open (polygon_path_dict["ref_polygon_path"]) as f:
                    contours_dict_data= json.load(f)
                logging.debug("Input dict: {}".format(contours_dict_data))
                my_polygons = v_polygons.from_dict(contours_dict_data)
                logging.debug("casted to v_polygons: {}".format(my_polygons.to_indendent_str()))

                # translate polygons according to translated_pxpy info for src_img and ref_polygons so both match spatialy
                st = paths["src_comps_dict_l"][     0]["translated_pxpy"]
                rt = paths["ref_polygon_translated_pxpy"]
                if((st[0] != rt[0]) or (st[1] != rt[1])):
                    dt = [rt[0] - st[0], rt[1] - st[1]]
                    my_polygons.move2point(dt)

                cls_pil_mask  = my_polygons.as_image(fill=True, val=255, w=w, h=h)
                cls_pil_masks.append(cls_pil_mask)

                cls_ids.append(cls_id)
                cls_id +=1
        
        if self.raport_time:
            elapsed_time = time.time() - start_time
            if(self.train):
                self.tr_time_labs_read += elapsed_time
                logging.info(f"  3. labels read {round(elapsed_time, 5)} s, aver {round(self.tr_time_labs_read/self.tr_time_nums, 5)} s")
            else:
                self.ev_time_labs_read += elapsed_time
                logging.info(f"                                             3. labels read {round(elapsed_time, 5)} s, aver {round(self.ev_time_labs_read/self.ev_time_nums, 5)} s") 
            start_time = time.time()

        tr_image, tr_masks = self.transform(comp_img_l, cls_pil_masks, paths)

        if self.raport_time:
            elapsed_time = time.time() - start_time
            if(self.train):
                self.tr_time_transform += elapsed_time
                logging.info(f"  3. transf in {round(elapsed_time, 5)} s, aver {round(self.tr_time_transform/self.tr_time_nums, 5)} s")
            else:
                self.ev_time_transform += elapsed_time
                logging.info(f"                                             3. transf in {round(elapsed_time, 5)} s, aver {round(self.ev_time_transform/self.ev_time_nums, 5)} s") 
            start_time = time.time()

        return tr_image, tr_masks, paths


    def transform(self, comp_img_l, masks, paths):
    
        width, height = comp_img_l[0].size
        tr_dbg = False and self.train
        
        # even if no crop is performed, the masks still needs to be expanded to the size of the input image and crop operation does it also
        crop_start = np.array([0,0])
        crop_end = np.array(comp_img_l[0].size)
        crop_box = [int(x) for x in (crop_start[0], crop_start[1], crop_end[0], crop_end[1])]
        for mask_id in range(len(masks)):
            if not(masks[mask_id] is None):
                if masks[mask_id].size != comp_img_l[0].size:
                    masks[mask_id] = masks[mask_id].crop(crop_box)

        # PREPARATION OF TRANSFORM PARAMETERS for train transformations: scale, rotation, flips 
        if(self.train): 
            smin, smax = self.train_tr_resize_range if not (self.train_tr_resize_range is None) else (1.0, 1.0)
            sx = smin + random.random() * (smax-smin)
            sy = smin + random.random() * (smax-smin) if self.train_tr_resize_xy_ind else sx
            s_w = int(np.round(sx * width))
            s_h = int(np.round(sy * height))

            rmin, rmax = (-self.train_rotation_range, self.train_rotation_range) if not (self.train_rotation_range is None) else (0.0, 0.0)
            rr = rmin + random.random() * (rmax-rmin)

            flip_x = self.train_flips and (random.random() > 0.5)
            flip_y = self.train_flips and (random.random() > 0.5)
                
            org_masks_size = []
            for mask_id in range(len(masks)):
                if not (masks[mask_id] is None):
                    org_masks_size.append(masks[mask_id].size)
        
            
        # calculate expected size of images after transformations - used for the first crop, used to speed up further transformations
        if(self.train): 
            exp_trans_size = scale_and_rotate_image_predict_new_size(comp_img_l[0].size, sx, sy, rr)
            exp_trans_width, exp_trans_height = exp_trans_size
        else:   
            exp_trans_size = width, height
            exp_trans_width, exp_trans_height = exp_trans_size
            
        # calculate crop parameters for the transformed image
        do_crop = False 
        if (not self.general_crop_size is None):
            crop_size = np.array([exp_trans_size[id] if self.general_crop_size[id] is None else self.general_crop_size[id] for id in range(2)])
            if (crop_size != exp_trans_size).any():
                do_crop = True
                if self.crop_type == 'center':
                    roi_box = [exp_trans_width//2, exp_trans_height//2, exp_trans_width//2+1, exp_trans_height//2+1]
                    roi_box_center = [(roi_box[0] + roi_box[2])/2, (roi_box[1] + roi_box[3])/2]
                    crop_start = np.round(roi_box_center - crop_size/2)
                elif self.crop_type == 'random':
                    hr = exp_trans_width  - crop_size[0]
                    vr = exp_trans_height - crop_size[1]
                    if(self.train):
                        chr = random.random()
                        cvr = random.random()
                    else:
                        # for validation use repetitable approach
                        rs = random.getstate()
                        random.seed(paths["src_image_id"])
                        chr = random.random()
                        cvr = random.random()
                        random.setstate(rs)
                    ho = int(chr * hr) if (hr > 0) else 0
                    vo = int(cvr * vr) if (vr > 0) else 0
                    crop_start = np.array([ho, vo])
            crop_end = crop_start + crop_size
        
        if (not self.force_size_is_multiple_of is None) and (self.force_size_is_multiple_of != 1):
            prev_size = np.array(crop_end - crop_start)
            for size_id, size in enumerate(prev_size):
                if((size//self.force_size_is_multiple_of) != (size/self.force_size_is_multiple_of)):
                    new_size = (size//self.force_size_is_multiple_of+1) * self.force_size_is_multiple_of #padding 
                    size_dif = new_size - prev_size[size_id]
                    crop_end[size_id] += size_dif
                    do_crop = True

        paths["org_size"] = exp_trans_size
        crop_box = [int(x) for x in (*crop_start, *crop_end)]
        paths["crop_box"] = crop_box
        
        if self.do_pre_trans_crop and (self.train) and do_crop: 
            # pre-transformations-crop - can speed up calculations of transformations by cropping image to only this parth that will be inside the final cropping box
            crop_size = crop_end - crop_start
            crop_center = crop_start + crop_size/2
            crop_center_pr = crop_center / exp_trans_size

            inv_trans_size = scale_and_rotate_image_predict_new_size(crop_size, 1/sx, 1/sy, rr)
            inv_trans_width, inv_trans_height = inv_trans_size
            
            # perform pre crop
            if do_crop and np.any(np.array(inv_trans_size) < (width, height)):  
                pre_crop_center = np.round((width, height) * crop_center_pr)
                pre_crop_start  = np.round(pre_crop_center - np.array(inv_trans_size)/2)
                pre_crop_end    = pre_crop_start + inv_trans_size
                pre_crop_box = [int(x) for x in (*pre_crop_start, *pre_crop_end)]
                for idx, image in enumerate(comp_img_l):
                    cropped_img = comp_img_l[idx].crop(pre_crop_box)
                    comp_img_l[idx] = cropped_img
                    if tr_dbg:
                        cropped_img.save("{}_1_pre_croped_img_{}_{}x{}.png".format(paths["src_image_id"], idx, comp_img_l[idx].width, comp_img_l[idx].height))

                for mask_id in range(len(masks)):
                    if not(masks[mask_id] is None):
                        cropped_mask = masks[mask_id].crop(pre_crop_box)
                        check_if_cut_through_poly = True
                        if(check_if_cut_through_poly):
                            mask = np.array(cropped_mask)
                            first_row   = (pre_crop_box[0] >                       0) and mask[                     0,                    :].any()           
                            last_row    = (pre_crop_box[2] < masks[mask_id].height  ) and mask[cropped_mask.height -1,                    :].any()
                            first_col   = (pre_crop_box[1] >                       0) and mask[                     :,                    0].any()            
                            last_col    = (pre_crop_box[3] < masks[mask_id].width   ) and mask[                     :, cropped_mask.width-1].any() 
                            if(first_row or last_row or first_col or last_col):
                                try:
                                    before_resize_size = org_masks_size[mask_id]
                                except:
                                    before_resize_size = '"no resize"'
                                logging.debug("Polygon from {} has been cut during pre_cropping to box {}! Before resize size = {}, before crop size= {}, out size = {}.".format(paths['cls_envs_list'][mask_id]['ref_polygon_path'], pre_crop_box, before_resize_size, masks[mask_id].size, cropped_mask.size))
                        masks[mask_id] = cropped_mask
                        if tr_dbg:
                            masks[mask_id].save("{}_1_pre_croped_msk_{}x{}.png".format(paths["src_image_id"], masks[mask_id].width, masks[mask_id].height))

                # what is left for post-crop
                exp_trans_size = scale_and_rotate_image_predict_new_size(comp_img_l[0].size, sx, sy, rr)
                exp_trans_width, exp_trans_height = exp_trans_size
                exp_trans_center = np.array(exp_trans_size)//2
                post_crop_size = crop_end - crop_start
                crop_start = np.round(exp_trans_center - post_crop_size/2) 
                crop_end   = crop_start + post_crop_size
                crop_box = [int(x) for x in (*crop_start, *crop_end)]
        else:   
            inv_trans_size = width, height
            inv_trans_width, inv_trans_height = inv_trans_size

        # train transformations: scale, rotation, flips- PERFORM TRANSFORMATION
        if(self.train):                 
            for idx, image in enumerate(comp_img_l):
                if tr_dbg:
                    comp_img_l[idx].save("{}_2_input_img_{}_{}x{}.png".format(paths["src_image_id"], idx, comp_img_l[idx].width, comp_img_l[idx].height))
                comp_img_l[idx] = scale_and_rotate_image(comp_img_l[idx], sx, sy, rr, resample=Image.BILINEAR, flip_x = flip_x, flip_y = flip_y)
                if tr_dbg:
                    comp_img_l[idx].save("{}_3_affin_img_{}_{}x{}_sx{:.2f}_sy{:.2f}_r{:.2f}_fx{}_fy{}.png".format(paths["src_image_id"], idx, comp_img_l[idx].width, comp_img_l[idx].height, sx, sy, rr, flip_x, flip_y))
                
            for mask_id in range(len(masks)):
                if not (masks[mask_id] is None):
                    if tr_dbg:
                        masks[mask_id].save("{}_2_input_msk_{}x{}.png".format(paths["src_image_id"], masks[mask_id].width, masks[mask_id].height))
                    masks[mask_id] = scale_and_rotate_image(masks[mask_id], sx, sy, rr, resample=self.train_polyRef_resample, flip_x = flip_x, flip_y = flip_y)
                    if tr_dbg:
                        masks[mask_id].save("{}_3_affin_msk_{}x{}_sx{:.2f}_sy{:.2f}_r{:.2f}_fx{}_fy{}.png".format(paths["src_image_id"], masks[mask_id].width, masks[mask_id].height, sx, sy, rr, flip_x, flip_y))
        

        # perform crop (or post-crop if a pre-crop was already done)
        if do_crop:    
            for idx, image in enumerate(comp_img_l):
                cropped_img = comp_img_l[idx].crop(crop_box)
                comp_img_l[idx] = cropped_img
                if tr_dbg:
                    cropped_img.save("{}_4_pst_croped_img_{}_{}x{}_sx{:.2f}_sy{:.2f}_r{:.2f}_fx{}_fy{}.png".format(paths["src_image_id"], idx, comp_img_l[idx].width, comp_img_l[idx].height, sx, sy, rr, flip_x, flip_y))

            for mask_id in range(len(masks)):
                if not(masks[mask_id] is None):
                    cropped_mask = masks[mask_id].crop(crop_box)
                    check_if_cut_through_poly = True
                    if(check_if_cut_through_poly):
                        mask = np.array(cropped_mask)
                        first_row   = (crop_box[0] >                       0) and mask[                     0,                    :].any()           
                        last_row    = (crop_box[2] < masks[mask_id].height  ) and mask[cropped_mask.height -1,                    :].any()
                        first_col   = (crop_box[1] >                       0) and mask[                     :,                    0].any()            
                        last_col    = (crop_box[3] < masks[mask_id].width   ) and mask[                     :, cropped_mask.width-1].any() 
                        if(first_row or last_row or first_col or last_col):
                            try:
                                before_resize_size = org_masks_size[mask_id]
                            except:
                                before_resize_size = '"no resize"'
                            logging.debug("Polygon from {} has been cut during cropping to box {}! Before resize size = {}, before crop size= {}, out size = {}.".format(paths['cls_envs_list'][mask_id]['ref_polygon_path'], crop_box, before_resize_size, masks[mask_id].size, cropped_mask.size))
                    masks[mask_id] = cropped_mask
                    if tr_dbg:
                        masks[mask_id].save("{}_4_pst_croped_msk_{}x{}_sx{:.2f}_sy{:.2f}_r{:.2f}_fx{}_fy{}.png".format(paths["src_image_id"], masks[mask_id].width, masks[mask_id].height, sx, sy, rr, flip_x, flip_y))
                           
        # Transform to tensor
        comps_npa_l = [np.array(comp_img) for comp_img in comp_img_l]
        comps_npa = np.array(comps_npa_l)

        imageT = torch.from_numpy(comps_npa)# convert from (w, h) to (3, h, w)
        # backward compatibility
        if isinstance(imageT, torch.ByteTensor):
            imageT = imageT.float().div(255)

        msks_np_l = [np.array(m, dtype=np.float32)/255.0 if (not m is None) else np.zeros((comp_img_l[0].size[1], comp_img_l[0].size[0]), dtype=np.float32) for m in masks]
        msks_np = np.array(msks_np_l)
        try:
            masksT = torch.from_numpy(msks_np)
        except Exception as err:
            logging.error("Error while converting numpy mask to torch.Tensor:")
            logging.error(err)
            logging.error("input numpy msks_np: {}".format(msks_np))
            sys.exit(43)
        
        return imageT, masksT
        
    def __len__(self):
        return len(self.paths)

    def reverse_transform(self, inp):
        inp = inp.cpu().numpy()
        inp = inp.transpose((1, 2, 0))
        inp = np.clip(inp, 0, 1)
        inp = (inp * 255).astype(np.uint8)

        return inp

# https://stackoverflow.com/questions/17056209/python-pil-affine-transformation
def scale_and_rotate_image(im, sx, sy, deg_cw, resample=Image.BILINEAR, flip_x = False, flip_y = False):

    if (sx==1.0) and (sy==1.0) and (deg_cw==0.0) and (not flip_x) and (not flip_y):
        return im

    im_orig = im
    im = Image.new('L', im_orig.size, 0)
    im.paste(im_orig)

    w, h = im.size
    deg_ccw = -deg_cw
    angle = math.radians(-deg_ccw)

    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)

    scaled_w, scaled_h = w * sx, h * sy

    new_w = int(math.ceil(math.fabs(cos_theta * scaled_w) + math.fabs(sin_theta * scaled_h)))
    new_h = int(math.ceil(math.fabs(sin_theta * scaled_w) + math.fabs(cos_theta * scaled_h)))

    cx = w / 2.
    cy = h / 2.
    tx = new_w / 2.
    ty = new_h / 2.

    
    # Wolfram Alpha 
    #   {{1,0,c1}, {0,1,c2}, {0,0,1}}*{{cos(a),sin(a),0}, {-sin(a),cos(a),0}, {0,0,1}}*{{1/s1,0,0}, {0,1/s2,0}, {0,0,1}}*{{f1,0,0}, {0,f2,0}, {0,0,1}}*{{1,0,-t1}, {0,1,-t2}, {0,0,1}}
    # = 
    #   (f1 cos(a))/s1 | (f2 sin(a))/s2 | -(f1 t1 cos(a))/s1 - (f2 t2 sin(a))/s2 + c1
    #  -(f1 sin(a))/s1 | (f2 cos(a))/s2 |  (f1 t1 sin(a))/s1 - (f2 t2 cos(a))/s2 + c2
    #                0 |              0 |                                           1
    if   not flip_x and not flip_y:
        (f1, f2) = ( 1, 1)
    elif not flip_x and     flip_y:
        (f1, f2) = (1, -1)
    elif     flip_x and not flip_y:
        (f1, f2) = (-1,  1)
    elif     flip_x and     flip_y:
        (f1, f2) = (-1, -1)

    (a, b, c) = ( f1*cos_theta/sx, f2*sin_theta/sy, -(f1*tx*cos_theta/sx) - (f2*ty*sin_theta/sy) + cx)
    (d, e, f) = (-f1*sin_theta/sx, f2*cos_theta/sy,  (f1*tx*sin_theta/sx) - (f2*ty*cos_theta/sy) + cy)

    return im.transform((new_w, new_h), Image.AFFINE, (a, b, c, d, e, f), resample = resample)
    
def scale_and_rotate_image_predict_new_size(im_size, sx, sy, deg_cw):

    w, h = im_size
    if (sx==1.0) and (sy==1.0) and (deg_cw==0.0) and (not flip_x) and (not flip_y):
        return (w, h)

    deg_ccw = -deg_cw
    angle = math.radians(-deg_ccw)

    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)

    scaled_w, scaled_h = w * sx, h * sy

    new_w = int(math.ceil(math.fabs(cos_theta * scaled_w) + math.fabs(sin_theta * scaled_h)))
    new_h = int(math.ceil(math.fabs(sin_theta * scaled_w) + math.fabs(cos_theta * scaled_h)))
    
    return (new_w, new_h)

def expand_session_dirs(session_dir_l, root_dir):
    # solve multi-selection filters with "*" / "?"
    ret_dir_l = []
    if (not type(session_dir_l) is list) and (not type(session_dir_l) is tuple):
        session_dir_l = [session_dir_l]
    for curr_dir in session_dir_l:
        if(curr_dir.find("*")==-1 and curr_dir.find("?")==-1):
            ret_dir_l.append(curr_dir)
        else:
            #logging.debug("parsing  user and session folder with multi-selection mark ({}).".format(curr_dir))
            top_dir, rem_dir = (curr_dir, None)
            if(curr_dir.find("/") != -1):
                top_dir, rem_dir = curr_dir.split("/", maxsplit=1)
            elif(curr_dir.find("\\\\") != -1):
                top_dir, rem_dir = curr_dir.split("\\\\", maxsplit=1)
            elif(curr_dir.find("\\") != -1):
                top_dir, rem_dir = curr_dir.split("\\", maxsplit=1)

            if (top_dir.find("*") == -1) and (top_dir.find("?") == -1):
                top_dirs = [top_dir]
            else:
                user_paths_pattern = os.path.normpath(os.path.join(root_dir, top_dir))
                user_paths = glob.glob(user_paths_pattern)
                user_paths.sort()
                top_dirs = [os.path.basename(user_path) for user_path in user_paths]

            for top_dir in top_dirs:
                if (rem_dir is None):
                    found_path = os.path.normpath(os.path.join(root_dir, top_dir))
                    if(os.path.isdir(found_path)):
                        ret_dir_l.append(top_dir)
                else:
                    root_dir_p1 = os.path.normpath(os.path.join(root_dir, top_dir))
                    ret_dir_p1_l = expand_session_dirs([rem_dir], root_dir_p1)
                    for rem_dir_p1 in ret_dir_p1_l:
                        found_path = os.path.normpath(os.path.join(top_dir, rem_dir_p1))
                        ret_dir_l.append(found_path)

    unique_entries_set = set(ret_dir_l)
    for entry in unique_entries_set:
        if entry.find("_log")!=-1:
            unique_entries_set.remove(entry)
    ret_dir_l = list(unique_entries_set)
    ret_dir_l.sort()

    return ret_dir_l
