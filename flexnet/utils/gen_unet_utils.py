#!/usr/bin/env python
# coding: utf-8

# pip/conda libs:
import os
import pathlib
import sys
import logging
import torch
import psutil
import subprocess
import json
import zipfile
import shutil
import time
import copy
#-----------------------------------------------------------------------------------------
curr_script_path = os.path.dirname(os.path.abspath(__file__))
flexnet_path = os.path.normpath(os.path.join(curr_script_path, ".."))
flexnet_host_path = os.path.normpath(os.path.join(flexnet_path, ".."))
sys.path.append(flexnet_host_path)
#-----------------------------------------------------------------------------------------
from v_utils.v_json import jsonDumpSafe

#----------------------------------------------------------------------------
# TRAINING FUNCTIONs
def check_is_flex_unet(model_state_dict):
    keys_l = list(model_state_dict.keys())
    for key in keys_l:
        if ( key.find('layers.0.') != -1):
            return True
    return False

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), path))
  
def save_model_with_cfgs(zip_file_path, model, dicts_to_dump = None):
    
    dir_out_name = os.path.splitext(zip_file_path)[0]
    try:
        if not os.path.isdir(dir_out_name):
            pathlib.Path(dir_out_name).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        logging.error(' Creating {} directory failed, error {}'.format(dir_out_name, err))
        exit(1)

    best_model_state_dict = model.state_dict()
    model_fn = "model_state_dict.pth"
    file_path = os.path.join(dir_out_name, model_fn)
    logging.info("  Saving state dict for a new best model to file \"{}\"".format(file_path))
    torch.save(best_model_state_dict, file_path)
    del best_model_state_dict

    if(not dicts_to_dump is None):
        dicts_names = dicts_to_dump.keys()
        for dict_n in dicts_names:
            dict_fn = "{}.json".format(dict_n)
            file_path = os.path.join(dir_out_name, dict_fn)
            logging.info("  Saving {} dict for a new best model to file \"{}\"".format(dict_n, file_path))
            jsonDumpSafe(file_path, dicts_to_dump[dict_n])

    zipf = zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_STORED)#zipfile.ZIP_DEFLATED)
    logging.info("  Zip {} dict to file \"{}\"".format(dir_out_name, zip_file_path))
    zipdir(dir_out_name, zipf)
    zipf.close()

    logging.info("  Delete {} dir".format(dir_out_name))
    shutil.rmtree(dir_out_name)
    
def try_parse_dicts_from_file(dict_file_path, device = torch.device('cpu')):
        
    if(dict_file_path.find(".zip") != -1):
        return try_parse_dicts_from_zip_file(dict_file_path, device)
    elif(dict_file_path.find(".pth") != -1):
        return try_parse_dicts_from_pth_file(dict_file_path, device)
    else:
        return None

def try_parse_dicts_from_pth_file(dict_file_path, device = torch.device('cpu')):
    
    dict_to_parse = torch.load(dict_file_path, map_location=device)
    if(not 'content_type' in dict_to_parse.keys()):
        logging.info(" 'content_type' key not found in the file. Unrecognized format.")
        return None
    
    logging.info(" 'content_type' found in the file: {}".format(dict_to_parse['content_type']))
    
    return dict_to_parse

def try_parse_dicts_from_zip_file(zip_file_path, device = torch.device('cpu')):

    time_str = time.strftime("%y_%m_%d__%H_%M_%S", time.gmtime())
    dir_out_root = f"_tmp_PID{os.getpid()}__{time_str}"
    logging.info(" Unzip file to {} temporary dir".format(dir_out_root))
    dir_out_name = f"{dir_out_root}/{os.path.splitext(os.path.basename(zip_file_path))[0]}"
    try:
        if not os.path.isdir(dir_out_name):
            pathlib.Path(dir_out_name).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        logging.error(' Creating {} directory failed, error {}'.format(dir_out_name, err))
        exit(1)

    zipf = zipfile.ZipFile(zip_file_path, 'r')
    zip_file_path
    zipf.extractall(dir_out_name)
    parsed_dicts = {}
    found_pth = False
    for f in zipf.filelist:
        dict_path = os.path.join(dir_out_name, f.filename)
        fn = os.path.basename(f.filename)
        dn, ext = os.path.splitext(fn)
        if (ext == '.pth'):
            parsed_dicts[dn] = torch.load(dict_path, map_location=device)
            found_pth = True
        elif(ext == ".json"):
            with open(dict_path, 'r') as f:
                parsed_dicts[dn] = json.load(f)
        else:
            logging.warning("Unrecognized dict named '{}' in zip file '{}'.".format(f.filename, zip_file_path))
    
    logging.info(" dicts found in the file: {}".format(parsed_dicts.keys()))
    if(not found_pth):
        logging.warning(" Did not found .pth file inside the zip file '{}'. Unrecognized format.".format(zip_file_path))
        return None
    
    logging.info("  Delete {} dir".format(dir_out_root))
    shutil.rmtree(dir_out_root)
    
    return parsed_dicts
    
def load_model(model_state_dict_path, model, device = torch.device('cpu'), strict = True):

    if(model_state_dict_path.find(".zip") != -1):
        parse_dicts = try_parse_dicts_from_zip_file(model_state_dict_path, device)
        model_state_dict = parse_dicts["model_state_dict"]
    elif(model_state_dict_path.find(".pth") != -1):
        parse_dicts = try_parse_dicts_from_pth_file(model_state_dict_path, device)
        if(parse_dicts is None):
            model_state_dict = torch.load(model_state_dict_path, map_location=device)
        else:
            model_state_dict = parse_dicts["model_state_dict"]

    load_model_from_state_dict(model_state_dict, model, strict = strict)
        
def load_model_from_state_dict(model_state_dict, model, strict = True):

    if('conv_original_size0.0.weight' in model_state_dict.keys()):
        if  (model_state_dict['conv_original_size0.0.weight'].shape[1] == model.get_in_channels_num()):
            logging.info(" model and state dict input size match")
        elif(model_state_dict['conv_original_size0.0.weight'].shape[1] != model.get_in_channels_num()):
            logging.warning(" model and state dict input sizes do NOT match! model_state_dict['conv_original_size0.0.weight'].shape[1]={}, model.get_in_channels_num() = {}". format(model_state_dict['conv_original_size0.0.weight'].shape[1], model.get_in_channels_num()))
            if(strict):
                logging.error(" Strict match between model and state dict is required therefore I can not proceed. Exit")
                sys.exit(1)
            
    if('conv_last.bias' in model_state_dict.keys()):
        if  (model_state_dict['conv_last.bias'].shape == model.conv_last.bias.shape):
            logging.info(" model and state dict output size match")
        elif(model_state_dict['conv_last.bias'].shape != model.conv_last.bias.shape):
            logging.warning(" model and state dict output sizes do NOT match! model_state_dict['conv_last.bias'].shape={}, model.conv_last.bias.data.shape = {}". format(model_state_dict['conv_last.bias'].shape, model.conv_last.bias.data.shape))
            if(strict):
                logging.error(" Strict match between model and state dict is required therefore I can not proceed. Exit")
                sys.exit(1)
        
    unexpected_keys, missing_keys = [], []  
    try:
       unexpected_keys, missing_keys = model.load_state_dict       (model_state_dict, strict = strict)   
    except Exception as err:
        if strict:
            logging.error("Error while reading model dict: {}".format(err))
            sys.exit(1)
        else:
            logging.info("Error while reading model dict: {}".format(err))

    if len(unexpected_keys) > 0:
        logging.info('Unexpected key(s) in state_dict: {}. '.format(', '.join('"{}"'.format(k) for k in unexpected_keys)))
    if len(missing_keys) > 0:
        logging.info('Missing key(s) in state_dict: {}. '.format(', '.join('"{}"'.format(k) for k in missing_keys)))


    #Remember that you must call model.eval() to set dropout and batch normalization layers to evaluation mode 
    #before running inference. Failing to do this will yield inconsistent inference results. 
    #If you wish to resuming training, call model.train() to ensure these layers are in training mode.
    model.eval()
    
#----------------------------------------------------------------------------
# zapis punktu przywracania i odczyt punktu przywracania zgodnie z 
#  https://pytorch.org/tutorials/beginner/saving_loading_models.html
def save_checkpoint(checpoint_file_path, model, optimizer_ft, scheduler, train_state, scalerAMP = None):
    torch.save({
        'model_state_dict'      : model.state_dict(),
        'optimizer_state_dict'  : optimizer_ft.state_dict(),
        'scheduler_state_dict'  : scheduler.state_dict(),
        'train_state_dict'      : train_state,
        'scalerAMP_state_dict'  : scalerAMP
        }, checpoint_file_path)
        
def load_checkpoint(checpoint_file_path, model, optimizer_ft, scheduler, train_state, scalerAMP = None):

    checkpoint = torch.load(checpoint_file_path)
    
    model.load_state_dict       (checkpoint['model_state_dict'      ])
    optimizer_ft.load_state_dict(checkpoint['optimizer_state_dict'  ])
    scheduler.load_state_dict   (checkpoint['scheduler_state_dict'  ])
    # wczytujac slownik nalezy uzyc update(). Przu uzyciu przypisania (=) tworzony jest nowy obiekt i na zewnatrz tej funkcji (w oryginalnym slowniku) zmiana nie bedzie widoczna
    #train_state=checkpoint['train_state_dict'      ]
    train_state.update          (checkpoint['train_state_dict'      ])
    if((not scalerAMP is None) and ('scalerAMP_state_dict' in checkpoint.keys())):
        scalerAMP.load_state_dict   (checkpoint['scalerAMP_state_dict'  ])

    #Remember that you must call model.eval() to set dropout and batch normalization layers to evaluation mode 
    #before running inference. Failing to do this will yield inconsistent inference results. 
    #If you wish to resuming training, call model.train() to ensure these layers are in training mode.
    #model.eval()
    model.train()
    
#----------------------------------------------------------------------------
def get_cuda_mem_used_free(dev_id = 0):
    try:
        ret = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.free', '--format=csv,noheader,nounits', '--id={}'.format(dev_id)])
    except Exception as err:
        logging.error("Error in get_cuda_mem_used_free: {}".format(err))
        return 0, 0

    try:
        vals = ret[0:-1].decode().split(',')
    except:
        vals = ret[0:-1].split(',')
    try:
        memory_used = int(vals[0])
        memory_free = int(vals[1])
        return memory_used, memory_free
    except Exception as err:
        logging.error("Error in get_cuda_mem_used_free: {}".format(err))
        return 0, 0

#----------------------------------------------------------------------------
def get_gen_mem_used_free():
    try:
        mem_info = psutil.virtual_memory()
        memory_used = int(mem_info.used/1024/1024)
        memory_free = int(mem_info.available/1024/1024)
        return memory_used, memory_free
    except Exception as err:
        logging.error("Error in get_gen_mem_used_free: {}".format(err))
        return 0, 0
