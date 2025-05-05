import sys, os
import pathlib
#-----------------------------------------------------------------------------------------
sys.path.append(os.getcwd())
#-----------------------------------------------------------------------------------------
import os
import glob
import tracemalloc
import multiprocessing as mp
from multiprocessing import Process, Queue
import timeit
import time
import shutil
import logging
import json

import numpy as np
import cv2 as cv
import math
from scipy.interpolate import interpn
from logging.handlers import QueueListener, QueueHandler

from columnar import columnar
#-----------------------------------------------------------------------------------------
from v_utils.v_polygons import *
from v_utils.v_json import jsonUpdate, jsonDumpSafe
from v_utils.v_dataset import expand_session_dirs
#---------------------------------------------------------
from argparse           import ArgumentParser
#--------------------------------------------------------
input_dir       = 'as_input'
stage00_dir     = 'as_data/st00_dicom_processing'
stage01_dir     = 'as_data/st01_dicom_selecting'
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

def calculate_3D_point(transform, point):
    a = np.matmul(transform,point)
    return [float(a[0]), float(a[1]), float(a[2])]
    
def get_transform_matrix(dataset):
    Sx = dataset["ImagePositionPatient"][0]
    Sy = dataset["ImagePositionPatient"][1]
    Sz = dataset["ImagePositionPatient"][2]
    
    Xx = dataset["ImageOrientationPatient"][0]
    Xy = dataset["ImageOrientationPatient"][1]
    Xz = dataset["ImageOrientationPatient"][2]
    Yx = dataset["ImageOrientationPatient"][3]
    Yy = dataset["ImageOrientationPatient"][4]
    Yz = dataset["ImageOrientationPatient"][5]
    
    [Xx, Xy, Xz] = [Xx, Xy, Xz]/np.linalg.norm([Xx, Xy, Xz])
    [Yx, Yy, Yz] = [Yx, Yy, Yz]/np.linalg.norm([Yx, Yy, Yz])
    
    Di = dataset["PixelSpacing"][0]
    Dj = dataset["PixelSpacing"][1]
    
    transform_matrix = [[Xx*Di, Yx*Dj ,0, Sx],[Xy*Di, Yy*Dj, 0, Sy],[Xz*Di, Yz*Dj, 0, Sz],[0, 0, 0, 1]]
    return transform_matrix
    
def transf_image_point(point, angle, ofs):
    nx = (point[0])*math.cos(angle)-(point[1])*math.sin(angle) - ofs[0]
    ny = (point[0])*math.sin(angle)+(point[1])*math.cos(angle) - ofs[1]
    return [nx,ny,0,1]

def log_err(name, ses):
    err 			= {}
    err['plugin'] 	= name
    err['session'] 	= ses
        
    return err

def log_run(name, ses):
    run 			= {}
    run['plugin'] 	= name
    run['session'] 	= ses
        
    return run

#-----------------------------------------------------------

def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        pathlib.Path(dst).mkdir(mode=0o775, parents=True, exist_ok=True)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=symlinks, ignore=ignore)



def rotate_images(q, input_dir, iop, logging_data):
    log = None
    err = None
    
    qlog = logging_data[0]
    log_format = logging_data[1]
    logging_level = logging_data[2]
    
    logger_thr = logging.getLogger(__name__+" logger")
    handler = QueueHandler(qlog)
    handler.setFormatter(logging.Formatter(log_format))
    logger_thr.addHandler(handler)
    logger_thr.setLevel(logging_level)
    
    
    #perform rotation to align horizontal axis of the image with the machine table surface (y coordinate of the first IOP vector = 0)
    #img_dir 	= input_dir + '/upsampled/*lsi.png'
    img_dir 	= input_dir + '/upsampled/*.png'
    img_dir		= os.path.normpath(img_dir)
    
    img_list  	= glob.glob(img_dir)
    img_list.sort()
    
    
    vecA = np.array(iop[0:3])
    vecB = np.array(iop[3:6])
    
    
    logger_thr.info("Vectors initial: {} {}".format(vecA, vecB))
    
    #check whether there is anything to be done
    if (vecA[1] != 0):
        if (vecB[1] != 0):  #very unlikely to happen otherwise 
            
            #calculate the new pair of vectors
            new_vecA = vecA - vecA[1]/vecB[1]*vecB
            new_vecB = vecB + vecA[1]/vecB[1]*vecA
            
            logger_thr.info("  New vectors (pre-norm):   {} {}".format(new_vecA, new_vecB))
            
            #calculate the value of the angle of rotation (direction is unknown yet)
            alfa = math.acos(np.linalg.norm(np.dot(new_vecA,vecA))/np.linalg.norm(new_vecA))*180/math.pi
            
            #normalize new vectors
            new_vecA = new_vecA/np.linalg.norm(new_vecA)
            new_vecB = new_vecB/np.linalg.norm(new_vecB)
            
            logger_thr.info("  New vectors:   {} {}".format(new_vecA, new_vecB))
            
            #calculate the direction of rotation (CW, CCW)
            k1 = np.cross(new_vecA,vecA)
            k1 = k1/np.linalg.norm(k1)
            k2 = np.cross(k1,new_vecA)
            
            #k2[1]/new_vecB[1] is negative for clockwise rotation of the image
            #alfa must be positive for clockwise rotation
            
            #the sign of alfa depends also on the imaging direction (HF*, FF*) (?)
            
            #alfa = - alfa * k2[1]/new_vecB[1]
            alfa =  alfa * k2[1]/new_vecB[1]
            
            logger_thr.info("  Angle: {}".format(alfa))
        else:
            err = "Y coordinate of vector {} == 0, unable to rotate images".format(vecB)
            logger_thr.error(err)
            q.put((log, err))
            return 
        
        print_info = True
        for image_fn in img_list:
            logger_thr.info("File {}".format(image_fn))
            if image_fn.rsplit('_',1)[1] == 'gsi.png':
                continue
            #cur_img = cv.imread(image_fn, cv.IMREAD_GRAYSCALE)
            cur_img = cv.imread(image_fn, cv.IMREAD_UNCHANGED)
            #print(cur_img.shape)
            X = list(range(cur_img.shape[1]))
            Y = list(range(cur_img.shape[0]))
            new_img = np.zeros(cur_img.shape)
            
            alfa_rad = -alfa/180*math.pi
            
            #get the extreme coordinates of the rotated image in the original coordinates
            nx = (cur_img.shape[1]-1)*math.cos(alfa_rad)-(cur_img.shape[0]-1)*math.sin(alfa_rad)
            ny = (cur_img.shape[1]-1)*math.sin(alfa_rad)+(cur_img.shape[0]-1)*math.cos(alfa_rad)
            #new image center will be half the way (one of the corners is always in (0;0))
            
            #calculate the center coordinates of the new image
            centerx_new = nx/2.0
            centery_new = ny/2.0
            
            #center of the original image
            centerx = (cur_img.shape[1]-1)/2.0
            centery = (cur_img.shape[0]-1)/2.0
            
            #calculate offset in pixels
            ofsx = centerx_new - centerx
            ofsy = centery_new - centery
            
            #ofsx = 50
            #ofsy = -10
            #alfa_rad = 0
            
            #for linia in range(cur_img.shape[0]):
            #    #wspolrzedne = []
            #    # maska = []
            #    #for kolumna in range(cur_img.shape[1]):
            #    #    nx = (kolumna)*math.cos(alfa_rad)-(linia)*math.sin(alfa_rad) - ofsx
            #    #    ny = (kolumna)*math.sin(alfa_rad)+(linia)*math.cos(alfa_rad) - ofsy
            #    #    wspolrzedne.append([ny, nx])
            #        
            #    wspolrzedne = [[(kolumna)*math.sin(alfa_rad)+(linia)*math.cos(alfa_rad) - ofsy ,(kolumna)*math.cos(alfa_rad)-(linia)*math.sin(alfa_rad) - ofsx] for kolumna in range(cur_img.shape[1])]
            #    #assert all([wspolrzedne[i] == wspolrzedne2[i] for i in range(cur_img.shape[1])])
            #    new_img[linia,:] = interpn((Y,X),cur_img, wspolrzedne, method='splinef2d', bounds_error = False, fill_value = 0) #* maska
            
            wspolrzedne = [[[(kolumna)*math.sin(alfa_rad)+(linia)*math.cos(alfa_rad) - ofsy ,(kolumna)*math.cos(alfa_rad)-(linia)*math.sin(alfa_rad) - ofsx] for kolumna in range(cur_img.shape[1])] for linia in range(cur_img.shape[0])]
            new_img[:,:] = interpn((Y,X),cur_img, wspolrzedne, method='splinef2d', bounds_error = False, fill_value = 0)
            
            
            #cv.imwrite(image_fn.rsplit('_',1)[0]+'r_'+image_fn.rsplit('_',1)[1],new_img)
            cv.imwrite(image_fn.rsplit('.',1)[0]+'.png',new_img)
            
            if image_fn.rsplit('_',1)[1] == 'lsi.png':
                #dicom data is modified only once, I choose lsi image to trigger the update
                dicom_h = open(image_fn.rsplit('_',1)[0]+'_dicom.json')
                dicom_data = json.load(dicom_h)
                dicom_h.close()
                matrix_old = get_transform_matrix(dicom_data)
                
                #update the image position in 3D space according to the calculated offset
                oy = (-ofsx*math.sin(alfa_rad)+ofsy*math.cos(alfa_rad))
                ox = ((oy*math.sin(alfa_rad)+ofsx)/math.cos(alfa_rad))
                
                ofs3D = new_vecA*ox*dicom_data['PixelSpacing'][0] + new_vecB*oy*dicom_data['PixelSpacing'][1]
                
                IPP = dicom_data['ImagePositionPatient']
                IPP_new = IPP - ofs3D
                
                #logging.info("Old IPP: {}, new IPP: {}".format(IPP, IPP_new))
                dicom_data['ImagePositionPatient'] = IPP_new.tolist()
                IOP = dicom_data['ImageOrientationPatient']
                #update image orientation
                dicom_data['ImageOrientationPatient'] = [*new_vecA, *new_vecB]
                matrix_new = get_transform_matrix(dicom_data)
                
                #jsonDumpSafe(image_fn.rsplit('_',1)[0]+'r_dicom.json', dicom_data)
                jsonDumpSafe(image_fn.rsplit('_',1)[0]+'_dicom.json', dicom_data)
            
                if print_info:
                    print_info = False
                    logger_thr.info("Image 0, Old c: {}, new c: {}, ofs: {}".format([centerx, centery], [centerx_new, centery_new], [ofsx, ofsy]))
                    logger_thr.info("Calculated offset: {}".format([ox,oy]))
                    logger_thr.info("Offset 3D: {}".format(ofs3D))
                    logger_thr.info("IPP: {}".format(IPP))
                    logger_thr.info("IPP_new: {}".format(IPP_new))
                    logger_thr.info("IOP: {}".format(IOP))
                    logger_thr.info("IOP_new: {}".format(dicom_data['ImageOrientationPatient']))

                    logger_thr.info("Test:")
                    logger_thr.info("Mat_old: {}".format(matrix_old))
                    logger_thr.info("Mat_new: {}".format(matrix_new))
                    z1 = transf_image_point([0,0],     alfa_rad, [ofsx, ofsy])
                    z2 = transf_image_point([300,300], alfa_rad, [ofsx, ofsy])
                    logger_thr.info("{}: {}={}".format(z1, calculate_3D_point(matrix_old,z1), calculate_3D_point(matrix_new,[0,0,0,1])))
                    logger_thr.info("{}: {}={}".format(z2, calculate_3D_point(matrix_old,z2), calculate_3D_point(matrix_new,[300,300,0,1])))
    log = "done"
    q.put((log, err))
    return             
                

#---------------------------------------------------------
# main
#---------------------------------------------------------
def main():
    global err_list
    global run_list

    parser          = ArgumentParser()

    parser.add_argument("-ses",    "--session_id"       , default = "*/*"   ,help="session id", nargs='+',	metavar="PATH",	required=False)
                                                                                                                               
    parser.add_argument("-op",     "--operation"        , dest="op"      ,help="operation",                                        required=False)
    parser.add_argument("-th",     "--threads", type = int, default = -2 ,help="Number of simultaneous processes",       required=False)
    parser.add_argument("-ft",     "--filter_type"      , dest="ft"      ,help="filter seq type <T1 (def), T2, PD, 3D, all>",      required=False)
    parser.add_argument("-fo",     "--filter_orientation",   dest="fo"   ,help="filter seq orientation plane <ax (def), nax, all>",required=False)
    parser.add_argument("-fss",    "--filter_slice_spacing", dest="fss"  ,help="filter seq to those with slice spacing in the given interval <MIN, MAX> in mm. MIN can be ommited", nargs='+', required=False)
    parser.add_argument("-v",      "--verbose"          , dest="verbose" ,help="verbose level <off, on/dbg>",                      required=False)

    args            = parser.parse_args()

    #---------------------------------------------------------------------------------------------------------------------------------------
    # initialize logging 
    data_dir = stage01_dir
    script_name = os.path.basename(__file__).split(".")[0]
    from datetime import datetime
    time_str = datetime.utcnow().strftime('%y_%m_%d__%H_%M_%S') #time.strftime("%y_%m_%d__%H_%M_%S.%f")
    log_dir = f"{data_dir}/_log"
    try:
        if not os.path.isdir(log_dir):
            pathlib.Path(log_dir).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        #logging.error('creating "%s" directory failed, error "%s"'%(log_dir, err))
        print(logging.error('creating "%s" directory failed, error "%s"'%(log_dir, err)))
        exit(1)
    initial_log_fn = f"{log_dir}/_dbg_{script_name}_{time_str}_pid{os.getpid()}.log"
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    logging_level = logging.INFO if ((args.verbose is None) or (args.verbose == "off")) else logging.DEBUG
    #logging.basicConfig(level=logging_level, format = log_format, handlers=[logging.FileHandler(initial_log_fn, mode='w'), logging.StreamHandler(sys.stdout)])

    qlog = Queue()
    
    ql = QueueListener(qlog, logging.StreamHandler(sys.stdout), logging.FileHandler(initial_log_fn, mode='w'))
    ql.start()
    
    try:
        os.chmod(initial_log_fn, 0o666)
    except:
        logging.warning(f"Could not change log file permitions {initial_log_fn}. I'm not the owner of the file?")
    
    
    logger = logging.getLogger(__name__ + "root")
    handler = QueueHandler(qlog)
    handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(handler)
    logger.setLevel(logging_level)



    logger.info('*' * 50)
    logger.info("script start @ {}".format(time.ctime()))

    #from v_utils.v_logging_std import bind_std_2_logging
    #bind_std_2_logging()
    
    #---------------------------------------------------------------------------------------------------------------------------------------

    session_l = expand_session_dirs(args.session_id, stage00_dir)
    
    #---------------------------------------------------------------------------------------------------------------------------------------
    op_l = ['all']
    ft_l = ["T1", "T2", "PD", "3D", "all"]
    fo_l = ["ax", "nax", "all"]

    verbose         = 'off'             if args.verbose is None else args.verbose
    op              = 'all'             if args.op      is None else args.op
    fo              = 'ax'              if args.fo      is None else args.fo
    ft              = 'T1'              if args.ft      is None else args.ft
    fss             = [0.0, 4.0]        if args.fss     is None else args.fss
    
    if not (op in op_l):
        logger.error("operation filter {} does not match any of expected vals: {}".format(op, op_l))
        exit(1)
    if not (ft in ft_l):
        logger.error("type filter {} does not match any of expected vals: {}".format(ft, ft_l))
        exit(1)
    if not (fo in fo_l):
        logger.error("orientation filter {} does not match any of expected vals: {}".format(fo, fo_l))
        exit(1)
    if (len(fss) > 2 or len(fss)==0):
        logger.error("slice spacing filter {} has wrong number of parameters. One or two expected (\"-fss MIN MAX\" or \"-fs MAX\")".format(fss))
        exit(1)
    if (len(fss) == 2 and (fss[1] < fss[0])):
        logger.error("slice spacing filter {} range has the second value lower than the first one.".format(fss))
        exit(1)
        
    threads = args.threads 
    if args.threads <= 0:
        threads = max(1, (os.cpu_count() - abs(args.threads)))

    #---------------------------------------------------------------------------------------------------------------------------------------

    logger.info('      > -----------------------------------------------------')
    logger.info('      > sessions list to process:')
    for ses in session_l:
        logger.info('      >    '+ ses)
    logger.info('      > -----------------------------------------------------')

    axial_list = []
    ptab_list = []
    for ses in session_l:
        logger.info('      > starting to process : '+ ses)
        
        do_rotate = False
        
        #checking the session specific parameters
        try:
            conffn = ses.replace('/','_').replace('\\','_') + ".cfg"
            conffh = open(os.path.join("as_cfg",conffn))
            configuration = json.load(conffh)
            conffh.close()
        except Exception as err:
            logger.info("Configuration file IO error: {}".format(err))
            logger.info("Proceeding with default settings")
            configuration = None
        if (configuration is not None):
            if ("st01" in configuration):
                #we have specific settings
                if "fo" in configuration["st01"]:
                    fo = configuration["st01"]["fo"]
                    logger.info(f"specific settings found: fo = {fo}")
                if "ft" in configuration["st01"]:
                    ft = configuration["st01"]["ft"]
                    logger.info(f"specific settings found: ft = {ft}")
                if "fss" in configuration["st01"]:
                    fss = configuration["st01"]["fss"]
                    logger.info(f"specific settings found: fss = {fss}")
                if "perform_rot" in configuration["st01"]:
                    do_rotate = configuration["st01"]["perform_rot"]
                    logger.info(f"specific settings found: perform_rot = {do_rotate}")
        

        dicomDir        = os.path.normpath(os.path.join(stage00_dir, ses))
        outDir 		    = os.path.normpath(os.path.join(stage01_dir, ses))

        logger.info('=' * 50)
        logger.info(' working set   : %s'%(ses))
        logger.info(' current dir   : %s'%os.getcwd())
        logger.info(' process       : %s'%op)
        logger.info(' f. type       : %s'%ft)
        logger.info(' f. orientation: %s'%fo)
        logger.info(' f. sl. spacing: %s'%fss)
        logger.info(' verbose level : %s'%verbose)
        logger.info(' subdirectory  :')
        logger.info('    input dicoms               : {}'.format( dicomDir))
        logger.info('    output images and metadata : {}'.format( outDir))
        logger.info('-' * 50)

        dicom_dir 		= dicomDir + '/images/*dicom.json'
        dicom_dir		= os.path.normpath(dicom_dir)

        dicom_list  	= glob.glob(dicom_dir)

        if len(dicom_list)==0:
            err_list.append(log_err ('ST01:dicom->list', ses))
            continue

        try:
            dicomf 	= open(dicom_list[0],"r");     
        except:
            err_list.append(log_err ('ST01:dicom->open', ses))
            continue

        jdicom          = json.load(dicomf); 
        #------------------------------------------------------------------------------------------------------------------------------
        isCT = ('Modality' in jdicom.keys()) and (jdicom['Modality'] == 'CT')
        if(isCT):
            logger.warning("CT scan detected!")
            if(ft != 'all'):
                logger.warning("Disable MRI type filtering")
                ft = 'all'
                logger.info(' f. type       : %s'%ft)
        #------------------------------------------------------------------------------------------------------------------------------
        # image orientation plane filtering
        
        isAxial = False
        
        if(fo != "all"):
            ptab 					= jdicom['ImageOrientationPatient']
            [cx,cy,cz, rx,ry,rz] 	= ptab

            cz_dom 			= abs(cx) < abs(cz) and abs(cy) < abs(cz)
            rz_dom 			= abs(rx) < abs(rz) and abs(ry) < abs(rz)
            isAxial     	= (not cz_dom) and (not rz_dom) 
    
            if (fo == "ax"): 
                if not isAxial:
                    logger.info("not axial, skipping sequence")
                    continue      
                else:             
                    logger.info("axial, processing sequence")
            elif (fo == "nax"):   
                if isAxial:       
                    logger.info("axial, skipping sequence")
                    continue      
                else:             
                    logger.info("non axial, processing sequence")

        #------------------------------------------------------------------------------------------------------------------------------
        # sequence type filtering
        if((ft != "all") and (not isCT)):
            # T1 or PD sequence should fulfill both conditions 1) and 2)
            # T1:
            #1) RT = < 250, 750>
            #2) ET = <  8,   40>
            # T2:    
            #1) RT = <1000, ...>
            #2) ET = (  40, ...>
            # PD:    
            #1) RT = <1000, ...>
            #2) ET = <   8,  40>
            # 3D:    
            #1) RT = < ...,  30>
            #2) ET = < ..., ...>
            RT 				= jdicom['RepetitionTime']
            if('EchoTime' in jdicom.keys()):
                ET 				= jdicom['EchoTime']
            elif('EffectiveEchoTime' in jdicom.keys()):
                ET 				= jdicom['EffectiveEchoTime']

            isT1 			= (RT >=  250) and (RT <= 750) and (ET >=  5) and (ET <= 40) 
            isT2 			= (RT >= 1000)                 and (ET >  40)   
            isPD 			= (RT >= 1000)                 and (ET >=  5) and (ET <= 40) 
            is3D 			=                  (RT <=  30)                               
    
            #if (ses.find("B000018/000005")==0):
            #    logger.info(f"{ft}, RT={RT}, ET={ET} processing sequence by forced exception. ")
            if ((ft == "T1") and not isT1) or \
                 ((ft == "T2") and not isT2) or \
                 ((ft == "PD") and not isPD) or \
                 ((ft == "3D") and not is3D):
                logger.info("not {}, skipping sequence".format(ft))
                continue
            else:
                logger.info("{}, processing sequence".format(ft))

        #------------------------------------------------------------------------------------------------------------------------------
        # slice spacing filter
        if not(fss is None):
            sdf_path 	= dicomDir + '/upsampled/set_data.json'
            sdf_path	= os.path.normpath(sdf_path)

            try:
                sdf 	= open(sdf_path,"r");     
            except:
                logger.error(" Could not read {} file!".format(sdf_path))
                sys.exit(1)

            jdata 			= json.load(sdf)

            SS = jdata['distance_between_slices']

            #SS 					= jdicom['SpacingBetweenSlices']
                
            minSS = float(fss[0]) if len(fss) > 1 else 0.0
            maxSS = float(fss[1]) if len(fss) > 1 else float(fss[0])
            #print("min:{} max:{} SS:{}".format(minSS, maxSS,SS))
            isInSS 			= (minSS <=  SS) and (maxSS >= SS)
    
            if not isInSS:
                logger.info("Spacing between slices {:.2} is not in a required range {}, skipping sequence".format(SS, fss))
                continue
            else:
                logger.info("Spacing between slices = {:.2}, processing sequence".format(SS))

        #------------------------------------------------------------------------------------------------------------------------------

        try:
            if not os.path.isdir(outDir):
                pathlib.Path(outDir).mkdir(mode=0o775, parents=True, exist_ok=True)
        except Exception as err:
            logger.error('creating "%s" directory failed, error "%s"'%(outDir,err))
            exit(1)
        logger.info('COPY to %s'%outDir)
        if isAxial:
            axial_list.append(outDir)
            ptab_list.append(ptab)
        copytree(dicomDir, outDir)
    logger.info(axial_list)
    logger.info(ptab_list)
    
    if do_rotate:
        logger.info("===========================ROTATING=============================")
        pid = 0
        process_waiting_l = []
        for i,ax in enumerate(axial_list):
            logger.info(ax)
            p_name=f"{ax}_rotate"
            queue = Queue()
            
            logging_data = [qlog, log_format, logging_level]
            
            p = Process(target=rotate_images, args=(queue, ax, ptab_list[i], logging_data), name=p_name, daemon = True)
            process_waiting_l.append({  "pid": pid, "process": p, "queue": queue})
            pid+=1
            #rotate_images(ax, ptab_list[i])
        
        #---------------------------------------------------------------------------------------------------------------------------------------
        th_to_start_num = len(process_waiting_l)
        logger.info(f'      > total processes {th_to_start_num}:')
        for proces_dict in process_waiting_l:
            logger.info(f'      >    {proces_dict["pid"]}: {proces_dict["process"].name}')
        logger.info(f'      > -----------------------------------------------------')
        
        #---------------------------------------------------------------------------------------------------------------------------------------
        th_pending_num = 0
        th_done_num    = 0 
        th_started_num = 0
        process_pending_l        = []
        process_done_l           = []
        
        while (th_done_num < th_to_start_num):
        
            changed = False
            
            # check if started processes have finished 
            for l_idx, proces_dict in enumerate(process_pending_l):
            
                p = proces_dict["process"]
                p.join(timeout=0)
                #logger.info(f'checking    {proces_dict["pid"]}: {proces_dict["process"].name}')
                if not p.is_alive():
                    q = proces_dict["queue"]
                    (log, err) = q.get()
                    q.close()
                    run_list.append(log)
                    if not err is None:
                        err_dict={"process_name":p.name, "msg":err}
                        err_list.append(err_dict)
                        
                    process_done_dict = process_pending_l.pop(l_idx)
                    process_done_l.append(process_done_dict)
                    
                    changed = True
                    th_pending_num -= 1
                    th_done_num    += 1
                    
                    logger.info(f' done {proces_dict["pid"]}: {proces_dict["process"].name}')
                    p.close()
                    break
            
            # start a new process
            if (th_started_num < th_to_start_num) and (th_pending_num < threads):
                    
                new_process_dict = None
                for l_idx, proces_dict in enumerate(process_waiting_l):
                    new_process_dict = process_waiting_l.pop(l_idx)
                    break
                    
                if not new_process_dict is None:
                    logger.info(f'          > -----------------------------------------------------')
                    logger.info(f'starting    {new_process_dict["pid"]}: {new_process_dict["process"].name}')
                        
                    #if dbg:
                    #    new_process_dict["process"] = Process(target=dummy_wait, args=(new_process_dict["queue"], new_process_dict["process"].name), name=new_process_dict["process"].name, daemon = True)
                    
                    p = new_process_dict["process"]
                    p.start()
                    
                    process_pending_l.append(new_process_dict)
                    
                    th_started_num += 1
                    th_pending_num += 1
                    changed = True
                
            if not changed:
                time.sleep(1)

    logger.info(f"RUNS   : {len(run_list)}")
    logger.info(f"ERRORS : {len(err_list)}")
    if len(err_list):
        logger.info("LIST OF SETS ENDED WITH ERRORS: ")
        for task_descr in err_list:
            logger.info(f'{task_descr}')
    ql.stop()
    if len(err_list):
        exit(-1)
#-----------------------------------------------------------
    
if __name__ == '__main__':
    main()
