import sys, os
import pathlib
#-----------------------------------------------------------------------------------------
sys.path.append(os.getcwd())
#-----------------------------------------------------------------------------------------
import os
import glob
import tracemalloc
import multiprocessing as mp
import timeit
import time
import shutil
import logging
import json 
import cv2
from columnar import columnar  
#-----------------------------------------------------------------------------------------
from v_utils.v_polygons import *
from v_utils.v_json import jsonUpdate, jsonDumpSafe
from v_utils.v_dataset import expand_session_dirs
#---------------------------------------------------------
from argparse           import ArgumentParser
#--------------------------------------------------------
input_dir       = 'as_input'
stage00_dir     = 'unk'
stage01_dir     = 'as_data/st01_dicom_selecting'
stage02_dir     = 'as_data/st02_roi'
stage03_dir     = 'as_data/st03_preprocessed_images'
#--------------------------------------------------------
dicom_dir       = 'images'
nrrd_dir        = 'nrrd'
upsampled_dir   = ''
upsampled_rectified_dir   = 'upsampled_rectified'
rectified_dir   = ''
superpixel_dir  = ''
superpixel_rectified_dir  = ''
brightness_dir  = 'brightness'
#--------------------------------------------------------
run_list        = []
err_list        = []
#--------------------------------------------------------

def log_err(name, sesID):
    err             = {}
    err['plugin']   = name
    err['sesion']   = sesID
        
    err_list.append(err)

def log_run(name, sesID):
    run             = {}
    run['plugin']   = name
    run['sesion']   = sesID
        
    run_list.append(run)

#-----------------------------------------------------------

mrtype = []

def Run_crop_images(dicomDir, roiDir, outDir, ses, pad_with_black_margins = True):

    global mrtype

    log_run ('CROP %s', ses)

    #----------------------------------------------------

    roi_outDir = os.path.normpath(os.path.join(outDir, "roi"))
    try:
        if not os.path.isdir(os.path.normpath(roi_outDir)):
            pathlib.Path(roi_outDir).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        logging.error('creating "%s" directory failed, error "%s"'%(roi_outDir,err))
        exit(1)

    dicom_dir 		= dicomDir + '/images/*dicom.json'
    dicom_dir		= os.path.normpath(dicom_dir)
    dicom_list  	= glob.glob(dicom_dir)

    if len(dicom_list)==0:
        log_err ('CROP %s:%s',userID,sesID)
        return

    try:
        dicomf 	= open(dicom_list[0],"r");     
    except:
        log_err ('CROP', ses)
        return

    sdf_path 	= dicomDir + '/upsampled/set_data.json'
    sdf_path	= os.path.normpath(sdf_path)

    try:
        sdf 	= open(sdf_path,"r");     
    except:
        log_err ('CROP', ses)
        return

    try:
        sdfx 	= open(sdf_path,"r");     
    except:
        log_err ('CROP', ses)
        return

    stf_path 	= dicomDir + '/upsampled/sorted.json'
    stf_path	= os.path.normpath(stf_path)

    try:
        stf 	= open(stf_path,"r");     
    except:
        log_err ('CROP', ses)
        return

    # jsons 

    #----------------------------------------------------

    jdicom          = json.load(dicomf); 
    jdx	 	        = json.load(sdfx)

    margin_mm = 5
    logging.info(' pixel spacing      : %.2fmm'%jdx["pixel_spacing_x"])
    margin 	    = int(margin_mm/jdx["pixel_spacing_x"]); 
    logging.info(f' margin             : {margin_mm}mm -> {margin} points')
    
    #----------------------------------------------------

    boxf_path 	= roiDir + '/box.json'
    boxf_path	= os.path.normpath(boxf_path)

    try:
        boxf 	= open(boxf_path,"r");     
    except:
        logging.error("cannot open box.json file : {}".format(boxf_path))
        log_err ('CROP',ses)
        return

    boxj		= json.load(boxf)

    px 			= boxj['box'][0] - margin
    py 			= boxj['box'][1] - margin
    ex 			= boxj['box'][2] + margin
    ey 			= boxj['box'][3] + margin

    if not pad_with_black_margins:
        if px<0:
            px = 0
        if py<0:
            py = 0

    lsif_dir 	= dicomDir + '/upsampled/*lsi*.png'
    lsif_dir	= os.path.normpath(lsif_dir)
    lsif_list 	= glob.glob(lsif_dir)
    
    dicomf_dir 	= dicomDir + '/upsampled/*dicom.json'
    dicomf_dir	= os.path.normpath(dicomf_dir)
    dicomf_list = glob.glob(dicomf_dir)
    dicomf_list.sort()

    gsif_dir 	= dicomDir + '/upsampled/*gsi*.png'
    gsif_dir	= os.path.normpath(gsif_dir)
    gsif_list 	= glob.glob(gsif_dir)

    nsif_dir 	= dicomDir + '/upsampled/*nsi*.png'
    nsif_dir	= os.path.normpath(nsif_dir)
    nsif_list 	= glob.glob(nsif_dir)

    csif_dir 	= dicomDir + '/upsampled/*csi*.png'
    csif_dir	= os.path.normpath(csif_dir)
    csif_list 	= glob.glob(csif_dir)

    fsif_dir 	= dicomDir + '/upsampled/*fsi*.png'
    fsif_dir	= os.path.normpath(fsif_dir)
    fsif_list 	= glob.glob(fsif_dir)

    roif_dir 	= roiDir + '/roi/*_roi_labels.png'
    roif_dir	= os.path.normpath(roif_dir)
    roif_list 	= glob.glob(roif_dir)

    roip_dir 	= roiDir + '/roi/*unet*polygons.json'
    roip_dir	= os.path.normpath(roip_dir)
    roip_list 	= glob.glob(roip_dir)

    img_tmp 	= cv2.imread(lsif_list[0], cv2.IMREAD_UNCHANGED)

    if not pad_with_black_margins:
        if ex>=img_tmp.shape[1]:
            ex = img_tmp.shape[1]-1
        if ey>=img_tmp.shape[0]:
            ey = img_tmp.shape[0]-1

    logging.info(f"input image shape           : {img_tmp.shape}") 
    logging.info(f"box of ROI                  : {boxj['box']}") 
    logging.info(f"box of ROI with {margin_mm} mm margins: [{px}, {py}, {ex}, {ey}]") 
    crop_box = [int(x) for x in (px, py, ex+1, ey+1)]
    logging.info(f"crop box                    : {crop_box}") 
    
    coord_list = {}

    for name in lsif_list:
        img = Image.open(name)

        if img is not None:
            if(img.mode != 'L'):
                img = img.convert('L')

            cropper_img = img.crop(crop_box)
            out_path = outDir + '/' + os.path.basename(name)
            out_path = os.path.normpath(out_path)
            cropper_img.save(out_path)
            
    for name in dicomf_list:
        dicom_file 	= open(name,"r")
        dicom_data	= json.load(dicom_file)
        dicom_file.close()
        
        for i in range (0,100):
            try:
                del dicom_data['Unknown%d'%i]
            except:
                pass
        
        Ix = dicom_data["ImageOrientationPatient"][0]
        Iy = dicom_data["ImageOrientationPatient"][1]
        Iz = dicom_data["ImageOrientationPatient"][2]
        Jx = dicom_data["ImageOrientationPatient"][3]
        Jy = dicom_data["ImageOrientationPatient"][4]
        Jz = dicom_data["ImageOrientationPatient"][5]
        
        X0 = dicom_data["ImagePositionPatient"][0]
        Y0 = dicom_data["ImagePositionPatient"][1]
        Z0 = dicom_data["ImagePositionPatient"][2]
        
        Sx = dicom_data["PixelSpacing"][0]
        Sy = dicom_data["PixelSpacing"][1]
        
        X0 = X0 + px*Sx*Ix + py*Sy*Jx
        Y0 = Y0 + px*Sx*Iy + py*Sy*Jy
        Z0 = Z0 + px*Sx*Iz + py*Sy*Jz
        
        dicom_data["ImagePositionPatient"][0] = X0
        dicom_data["ImagePositionPatient"][1] = Y0
        dicom_data["ImagePositionPatient"][2] = Z0
        
        dicom_data["Columns"] = ex+1-px
        dicom_data["Rows"] = ey+1-py
        
        out_path = outDir + '/' + os.path.basename(name)
        out_path = os.path.normpath(out_path)
        
        jsonDumpSafe(out_path, dicom_data)
        
        filename_core = os.path.basename(name).split('_',1)[0]
        coord_list[filename_core]=[X0, Y0, Z0]

    for name in roif_list:
        img = Image.open(name)

        if img is not None:
            
            cropper_img = img.crop(crop_box)
            out_path = roi_outDir + '/' + os.path.basename(name)
            out_path = os.path.normpath(out_path)
            cropper_img.save(out_path)

    for name in gsif_list:
        img = Image.open(name)
        
        if img is not None:
            if(img.mode != 'L'):
                img = img.convert('L')

            cropper_img = img.crop(crop_box)
            out_path = outDir + '/' + os.path.basename(name)
            out_path = os.path.normpath(out_path)
            cropper_img.save(out_path)

    for name in nsif_list:
        img = Image.open(name)
        
        if img is not None:
            if(img.mode != 'L'):
                img = img.convert('L')

            cropper_img = img.crop(crop_box)
            out_path = outDir + '/' + os.path.basename(name)
            out_path = os.path.normpath(out_path)
            cropper_img.save(out_path)

    for name in csif_list:
        img = Image.open(name)
        
        if img is not None:
            if(img.mode != 'L'):
                img = img.convert('L')

            cropper_img = img.crop(crop_box)
            out_path = outDir + '/' + os.path.basename(name)
            out_path = os.path.normpath(out_path)
            cropper_img.save(out_path)

    org_img 		= cv2.imread(lsif_list[0], cv2.IMREAD_UNCHANGED)

    jdata 			= json.load(sdf)
    jdata['order']  = json.load(stf)

    dicom_info 					    = {}
    try:
        dicom_info['SequenceName'] 	= jdicom['SequenceName']
    except:
        dicom_info['SequenceName'] 	= 'unknown'

    dicom_info['ProtocolName']      = jdicom['ProtocolName'] if 'ProtocolName' in jdicom else "NONE"
    
    jdata['dicom']       		    = dicom_info
    
    jdata['upsampled_size'] 		= [ org_img.shape[1],   org_img.shape[0] ]
    jdata['crop_roi_pos'] 			= [ px,                 py ]
    jdata['crop_roi_size'] 			= [ ex-px+1,            ey-py+1 ]    
    
    jdata['coordinates'] = coord_list

    outj_path = outDir + '/description.json'
    jsonDumpSafe(outj_path, jdata)

    for i in range (0,100):
        try:
            del jdicom['Unknown%d'%i]
        except:
            pass

    outdicom_path = outDir + '/dicom.json'
    jsonUpdate(outdicom_path,jdicom)
    
    mrtype.append([ses,jdicom['ProtocolName'] if 'ProtocolName' in jdicom else "NONE", '%2.2f' % jdata['distance_between_slices']])

#---------------------------------------------------------
# main
#---------------------------------------------------------
def main():

    parser          = ArgumentParser()
    ops         = ['crop', 'rectify-brightness']

    parser.add_argument("-ses",    "--session_id"       , default = "*/*"   ,help="session id", nargs='+',	metavar="PATH",	required=False)
    parser.add_argument("-v",      "--verbose"          , dest="verbose" , default="off",  help="verbose level <off, on/dbg>",                      required=False)

    args            = parser.parse_args()
    #---------------------------------------------------------------------------------------------------------------------------------------
    # initialize logging 
    
    data_dir = stage03_dir
    script_name = os.path.basename(__file__).split(".")[0]
    from datetime import datetime
    time_str = datetime.utcnow().strftime('%y_%m_%d__%H_%M_%S') #time.strftime("%y_%m_%d__%H_%M_%S.%f")
    log_dir = f"{data_dir}/_log"
    try:
        if not os.path.isdir(log_dir):
            pathlib.Path(log_dir).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        logging.error('creating "%s" directory failed, error "%s"'%(log_dir, err))
        exit(1)
    initial_log_fn = f"{log_dir}/_dbg_{script_name}_{time_str}_pid{os.getpid()}.log"
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    logging_level = logging.INFO if ((args.verbose is None) or (args.verbose == "off")) else logging.DEBUG
    logging.basicConfig(level=logging_level, format = log_format, handlers=[logging.FileHandler(initial_log_fn, mode='w'), logging.StreamHandler(sys.stdout)])
    
    try:
        os.chmod(initial_log_fn, 0o666)
    except:
        logging.warning(f"Could not change log file permitions {initial_log_fn}. I'm not the owner of the file?")
    
    
    from v_utils.v_logging_std import bind_std_2_logging
    bind_std_2_logging()
    
    logging.info('*' * 50)
    logging.info(f"script {os.path.basename(__file__).split('.')[0]} start @ {time.ctime()}")
    #---------------------------------------------------------------------------------------------------------------------------------------

    session_l = expand_session_dirs(args.session_id, stage02_dir)

    #---------------------------------------------------------------------------------------------------------------------------------------

    logging.info('      > -----------------------------------------------------')
    logging.info('      > sessions list to process:')
    for ses in session_l:
        logging.info('      >    '+ ses)
    logging.info('      > -----------------------------------------------------')

    for ses in session_l:
        logging.info('      > starting to process : '+ ses)

        dicomDir    = os.path.normpath(os.path.join(stage01_dir, ses))
        boxDir    	= os.path.normpath(os.path.join(stage02_dir, ses)) 
        outDir      = os.path.normpath(os.path.join(stage03_dir, ses))

        try:
            if not os.path.isdir(outDir):
                pathlib.Path(outDir).mkdir(mode=0o775, parents=True, exist_ok=True)
        except Exception as err:
            logging.error('creating "%s" directory failed, error "%s"'%(user_path,err))
            exit(1)

        logging.info('=' * 50)
        logging.info(' working set   : %s'%(ses))
        logging.info(' current dir   : %s'%os.getcwd())
        logging.info(' subdirectory  :')
        logging.info('    input dicoms               : {}'.format( dicomDir))
        logging.info('    box dir                    : {}'.format( boxDir))
        logging.info('    output images and metadata : {}'.format( outDir))
        logging.info('-' * 50)

            #------------------------------------------------------------------------------------------------------------------------------
        if not os.path.isdir(outDir):
            logging.info('directory "%s" does not exists, propably scan orinetation or type was wrong'%outDir)
            continue

        Run_crop_images         (dicomDir, boxDir, outDir, ses)
        
    logging.info('-' * 50)
    logging.info("RUNS   : {}".format(len(run_list)))
    logging.info("ERRORS : {}".format(len(err_list)))
    if len(err_list):
        logging.error("ERROR LIST : ")
        for e in err_list:
            logging.error(e)
    logging.info('*' * 50)
    if len(err_list):
        exit(-1)

#-----------------------------------------------------------
    
if __name__ == '__main__':
    main()
