import sys, getopt
import numpy as np
import json 
import os
import pathlib
import cv2 as cv
import glob
import tracemalloc
import multiprocessing
from scipy import ndimage, misc
import logging
#-----------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib as mpl
#-----------------------------------------------------------------------------------------
from PIL import Image
from argparse import ArgumentParser
#-----------------------------------------------------------------------------------------
# AS libs
#-----------------------------------------------------------------------------------------
sys.path.append(os.getcwd())
#-----------------------------------------------------------------------------------------
from v_utils.v_polygons import *
from v_utils.v_json import jsonUpdate, jsonDumpSafe
#-----------------------------------------------------------------------------------------

class listaobiektow:
    def __init__(self):
        self.data = []
    
    def getRange2(self, i, x, y):
        return (self.data[i][1]-x)*(self.data[i][1]-x)+(self.data[i][2]-y)*(self.data[i][2]-y)
    
    def findClosest(self, x,y):
        closest_num = 0
        closest_range = 999999999999999
        for i in range(0,len(self.data)):
            if self.getRange2(i,x,y)<closest_range:
                closest_range = self.getRange2(i,x,y)
                closest_num = i
        return self.data[closest_num][0]
            
    def add(self, label, x, y):
        self.data.append([label, x, y])
        
        

def process_dir(img_path, lbl_path, roi_path, destination_lbl_path, FilterSize, Threshold, verbose):

    logging.info('FilterSize: {} Threshold: {}'.format(FilterSize, Threshold))

    im16  = cv.imread(img_path)
    logging.info(img_path)

    imin = im16.min()
    imax = im16.max()

    im = im16.copy()[:,:,0]
    #logging.info("min={}, max={}".format(imin,imax))

    im8_scal = im16.copy()[:,:,0]
    im16_roi = im16.copy()[:,:,0]
    
    labels_img = cv.imread(lbl_path)#, cv.IMREAD_ANYDEPTH)
    destination_labels_img = cv.imread(destination_lbl_path)#, cv.IMREAD_ANYDEPTH)
    logging.info(lbl_path)
    
    roi = cv.imread(roi_path)
    
    roi[:,:,0] = roi[:,:,0] + roi[:,:,1] + roi[:,:,2]
    roi = roi[:,:,0]
    im16_roi[roi==0] = 0
    
    histogram = cv.calcHist([im16_roi.astype(np.uint8)],[0],None,[256],[0,255])
    calka_hist = np.zeros(histogram.shape)
    
    for i in range(1, len(histogram)):
        calka_hist[i] = calka_hist[i-1]+histogram[i]
    calka_hist = 100*calka_hist/np.max(calka_hist)
    
    prog_automatyczny = 0
    while calka_hist[prog_automatyczny] < 25:
        prog_automatyczny += 1
    
    if verbose:
        plt.figure('Histogram')
        plt.plot(range(1, len(histogram)), histogram[1:], range(0, len(histogram)), calka_hist)
        plt.show()

    
    # number = np.max(labels) + 1
    # for n in range(0,number):
        # suma = np.sum(im16[labels==n])
        # temp = im16.copy()
        # temp[labels==n]=1
        # temp[labels!=n]=0
        # licznosc = np.sum(temp)
        # if licznosc != 0:
            # im[labels==n] = int(suma/licznosc)
        # else:
            # im[labels==n] = 0

    logging.info("in shape {}".format(im.shape))
    if verbose:
        plt.figure('Obraz')
        plt.imshow(im)
        plt.show()

    ret,thresh = cv.threshold(im, Threshold, 255, cv.THRESH_BINARY_INV)
    #ret,thresh = cv.threshold(im, prog_automatyczny, 255, cv.THRESH_BINARY_INV)
    thresh = thresh.astype(np.uint8)
    if verbose:
        plt.figure('Progowanie')
        plt.imshow(thresh)
        plt.show()

    ret, labels1 = cv.connectedComponents(thresh)
    if verbose:
        plt.figure('Etkiety 1')
        plt.imshow(labels1)
        plt.show
    
    lista = listaobiektow()
    
    FilterSize_close_big_circle = 40
    kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(FilterSize_close_big_circle, FilterSize_close_big_circle))
    resultClose1 = np.zeros(labels1.shape)
    if ret>2:
        for currentLabel in range(2,ret):
            labelsTemp = labels1.copy()
            labelsTemp[labels1!=currentLabel]=0
            labelsTemp[labels1==currentLabel]=1
            
            M = cv.moments(labelsTemp.astype(np.uint8))
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                lista.add(currentLabel, cX, cY)
            
            morph1 = cv.morphologyEx(labelsTemp.astype(np.uint8), cv.MORPH_CLOSE, kernel1)
            resultClose1 = resultClose1 + morph1
    resultClose1[resultClose1>0] = 255
    if verbose:
        plt.figure('Po pierwszym zamknieciu')
        plt.imshow(resultClose1)
        plt.show()

    FilterSize_delete_small = 20
    kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(FilterSize_delete_small, FilterSize_delete_small))
    morph1 = cv.morphologyEx(resultClose1, cv.MORPH_OPEN, kernel1)
    if verbose:
        plt.figure('Po pierwszym otwarciu')
        plt.imshow(morph1)
        plt.show()
    logging.info("morph1 shape {}".format(morph1.shape))
    
    kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5, 5))
    morph1 = cv.morphologyEx(morph1, cv.MORPH_ERODE, kernel1)
    
    
    ret, labels2 = cv.connectedComponents(morph1.astype(np.uint8), connectivity=4)
    for curr_label in range(1,ret):
        labelsTemp = labels2.copy()
        labelsTemp[labels2!=curr_label]=0
        labelsTemp[labels2==curr_label]=1
        M = cv.moments(labelsTemp.astype(np.uint8))
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            numer = lista.findClosest(cX, cY)
            if numer > 0:
                labels2[labels1 == numer] = 5

    if verbose:
        plt.figure('Etkiety koncowe')
        plt.imshow(labels2)
        plt.show()

    label_numbers = np.zeros(destination_labels_img.shape[0:1])
    label_numbers = destination_labels_img[:,:,0]*256+destination_labels_img[:,:,1]
    
    labels_covered = label_numbers.copy()
    labels_covered[labels2 == 0] = 0
    if verbose:
        plt.figure('label_numbers')
        plt.imshow(label_numbers)
        plt.show()
    if verbose:
        plt.figure('labels_covered')
        plt.imshow(labels_covered)
        plt.show()

    bones_img = np.zeros(labels2.shape)
    labels_covered_unique = []
    for curr_label in np.nditer(labels_covered):
        if curr_label > 0:
            if curr_label not in labels_covered_unique:
                labels_covered_unique.append(curr_label)

    
    for curr_label in labels_covered_unique:
        bones_img[label_numbers==curr_label] = 1
    #bones_img = (bones_img>0)                

    
    numlabels,labels3 = cv.connectedComponents(bones_img.astype(np.uint8), connectivity=4)
    if verbose:
        plt.figure('Kosci')
        plt.imshow(bones_img.astype(np.uint8))
        plt.show()
    
    use_as_functions = True

    if not use_as_functions:

        bones = []
        # wyniki w nowej formie
        bones_polygons = v_polygons()
    
        for label in range(numlabels-1):
            labels2copy = (labels3==(label+1))
            #labels2copy1 = cv.morphologyEx(labels2copy.astype(np.uint8), cv.MORPH_CLOSE, kernel2)
            #contours, hierarchy = cv.findContours(labels2copy1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            contours, hierarchy = cv.findContours(labels2copy.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #        cv.drawContours(result_images[-1], contours, -1, (0,1.0,0), 1)
            #M = cv.moments(labels2copy.astype(np.uint8))
            #if M["m00"] > 0:
            #    cX = int(M["m10"] / M["m00"])
            #    cY = int(M["m01"] / M["m00"])
     #      #     cv.circle(result_images[-1], (cX, cY), 3, (1.0, 0.3, 0.3), -1)
            ##bones.append([contours, [cX, cY]])
            #bones.append(contours)

            # wyniki w nowej formie
            for contour in contours:
                if(contour.shape[1]==1):
                    squized_contour = np.squeeze(contour, axis=1)
                else:
                    squized_contour = contour
                bones_polygons.add_polygon_from_paths(outer_path = squized_contour, holes_paths = []) 

    else: # use v_polygons functions with holes detection
        a_max = labels3.max()
        a_min = labels3.min()
        bones_polygons = v_polygons.from_ndarray(labels3)

    #if verbose:
    #    for k in bones:
    #        cv.drawContours(im, k, -1, (500,500,500), 1)
    #    plt.imshow(im)
    #    plt.show()
    #
    ##process contour
    #cont = []
    #for bone in bones:
    #    for co in bone:
    #        singlecontour = []
    #        for T in co:
    #            x = T[0][0]
    #            y = T[0][1]
    #                
    #            singlecontour.append([int(x),int(y)])
    #        cont.append(singlecontour)
    #
    #
    #im8_mask = im8_scal.copy()
    #im8_mask[:,:] = 0
    #
    #logging.info(len(cont))
    #if len(cont)>0:
    #    for i in cont:
    #        cv.fillPoly(im8_mask, pts =[np.array(i)], color=(255))
    #        
    ##return(cont,im8_scal,im8_mask,im8_scal,[min_x,min_y,max_x,max_y,avg_x,avg_y], bones_polygons)
    #return(im8_scal,im8_mask,im8_scal, bones_polygons)
    return bones_polygons, im8_scal.shape

def process_dir_old(img_path, lbl_path, FilterSize, Threshold, verbose):

    im16  = cv.imread(img_path)
    logging.info(img_path)

    imin = im16.min()
    imax = im16.max()

    im = im16.copy()[:,:,0]
    #logging.info("min={}, max={}".format(imin,imax))

    im8_scal = im16.copy()[:,:,0]
    
    labels_img = cv.imread(lbl_path, cv.IMREAD_ANYDEPTH)
    logging.info(lbl_path)
    

    
    # number = np.max(labels) + 1
    # for n in range(0,number):
        # suma = np.sum(im16[labels==n])
        # temp = im16.copy()
        # temp[labels==n]=1
        # temp[labels!=n]=0
        # licznosc = np.sum(temp)
        # if licznosc != 0:
            # im[labels==n] = int(suma/licznosc)
        # else:
            # im[labels==n] = 0

    logging.info("in shape {}".format(im.shape))
    if verbose:
        plt.figure('Obraz')
        plt.imshow(im)
        plt.show()

    ret,thresh = cv.threshold(im, Threshold, 255, cv.THRESH_BINARY_INV)
    thresh = thresh.astype(np.uint8)
    if verbose:
        plt.figure('Progowanie')
        plt.imshow(thresh)
        plt.show()

    kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(FilterSize, FilterSize))
    logging.info("thresh shape {}".format(thresh.shape))
    morph1 = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel1)
    if verbose:
        plt.figure('Po otwarciu')
        plt.imshow(morph1)
        plt.show()

    logging.info("morph1 shape {}".format(morph1.shape))
    ret, labels = cv.connectedComponents(morph1)
    if verbose:
        plt.figure('Etkiety')
        plt.imshow(labels)
        plt.show

    morph1a = morph1.copy()
    morph1a[labels==1]=0
    kernel3 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(4,4))
    thresh2 = cv.morphologyEx(morph1a, cv.MORPH_OPEN, kernel1)
    thresh2 = thresh2.astype(np.uint8)
    if verbose:
        plt.figure('Po 2 progowaniu')
        plt.imshow(thresh2)
        plt.show()
#    numlabels,labels2 = cv.connectedComponents(thresh2)
# znajdz suoerpiksele pokrywajace sie z wyznaczonymi koscmi
    
    #bones_img = thresh2.copy()
    bones_img = np.zeros(thresh2.shape)
    kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(10,10))
    labels2copy = (thresh2>0)
    labels2copy1 = cv.morphologyEx(labels2copy.astype(np.uint8), cv.MORPH_ERODE, kernel2)

    #bones_img = labels2copy1
    labels_covered = labels_img[labels2copy1!=0]
    
    labels_covered_unique = []
    for curr_label in labels_covered:
        if curr_label not in labels_covered_unique:
            labels_covered_unique.append(curr_label)
    
    for curr_label in labels_covered_unique:
        bones_img[labels_img==curr_label] = 1
    #bones_img = (bones_img>0)                
    
    
    numlabels,labels2 = cv.connectedComponents(bones_img.astype(np.uint8))
    
    use_as_functions = True

    if not use_as_functions:

        bones = []
        # wyniki w nowej formie
        bones_polygons = v_polygons()
    
        for label in range(numlabels-1):
            labels2copy = (labels2==(label+1))
            #labels2copy1 = cv.morphologyEx(labels2copy.astype(np.uint8), cv.MORPH_CLOSE, kernel2)
            #contours, hierarchy = cv.findContours(labels2copy1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            contours, hierarchy = cv.findContours(labels2copy.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #        cv.drawContours(result_images[-1], contours, -1, (0,1.0,0), 1)
            #M = cv.moments(labels2copy.astype(np.uint8))
            #if M["m00"] > 0:
            #    cX = int(M["m10"] / M["m00"])
            #    cY = int(M["m01"] / M["m00"])
     #      #     cv.circle(result_images[-1], (cX, cY), 3, (1.0, 0.3, 0.3), -1)
            ##bones.append([contours, [cX, cY]])
            #bones.append(contours)

            # wyniki w nowej formie
            for contour in contours:
                if(contour.shape[1]==1):
                    squized_contour = np.squeeze(contour, axis=1)
                else:
                    squized_contour = contour
                bones_polygons.add_polygon_from_paths(outer_path = squized_contour, holes_paths = []) 

    else: # use v_polygons functions with holes detection
        a_max = labels2.max()
        a_min = labels2.min()
        bones_polygons = v_polygons.from_ndarray(labels2)

    #if verbose:
    #    for k in bones:
    #        cv.drawContours(im, k, -1, (500,500,500), 1)
    #    plt.imshow(im)
    #    plt.show()
    #
    ##process contour
    #cont = []
    #for bone in bones:
    #    for co in bone:
    #        singlecontour = []
    #        for T in co:
    #            x = T[0][0]
    #            y = T[0][1]
    #                
    #            singlecontour.append([int(x),int(y)])
    #        cont.append(singlecontour)
    #
    #
    #im8_mask = im8_scal.copy()
    #im8_mask[:,:] = 0
    #
    #logging.info(len(cont))
    #if len(cont)>0:
    #    for i in cont:
    #        cv.fillPoly(im8_mask, pts =[np.array(i)], color=(255))
    #        
    ##return(cont,im8_scal,im8_mask,im8_scal,[min_x,min_y,max_x,max_y,avg_x,avg_y], bones_polygons)
    #return(im8_scal,im8_mask,im8_scal, bones_polygons)
    return bones_polygons, im8_scal.shape



def try_get_key_frame(csid, prev_cset, cset_path):
    iskey = 0
    try:
        csetf = open(cset_path,"r");     
    except:
        if csid==0:
            cset = {
                        "plugin": 
                        {
                            "name" : "pat-bones2"
                        },
                        "variable": 
                        {
                            "FilterSize": 
                            {
                                "id" : "FS0",
                                "description"   :   "rozmiar maski filtru morfologicznego NxN",
                                "value"         :   6,
                                "format"        :   "{value}x{value}",
                                "min"           :   3,
                                "max"           :   55,
                                "step"          :   2,
                                "ctrl_type"     :   "slider"
                            },
                            "Threshold": 
                            {
                                "id" : "Th0",     
                                "description"   :   "wartość progu progowania binarnego",
                                "value"         :   30,
                                "format"        :   "{value}",
                                "min"           :   10,
                                "max"           :   100,
                                "step"          :   1,
                                "ctrl_type"     :   "slider"
                            }
                        }
                    }
  
            jsonDumpSafe(cset_path, cset)
            logging.info('creating cset file : {}'.format(cset_path))
        else:
            cset = prev_cset
    else:
        cset    = json.load(csetf)
        iskey   = 1

    return(cset,iskey)

#---------------------------------------------------------
# main
#---------------------------------------------------------

def main():

    parser = ArgumentParser()

    parser.add_argument("-dpDir",   "--dp_dir",         dest="dp_dir",      help="input png directory",             metavar="PATH", required=True)
    parser.add_argument("-lbDir",   "--lb_dir",         dest="lb_dir",      help="input labels directory",          metavar="PATH", required=True)
    parser.add_argument("-aidsDir", "--aids_dir",       dest="aids_dir",    help="IA dataset directory",            metavar="PATH", required=True)
    parser.add_argument("-ns",      "--name_sufix",     dest="ns",          help="name sufix (nsi,lsi,gsi, ...)",                   required=False)
    parser.add_argument("-v",       "--verbose",        dest="verbose",     help="verbose level",                                   required=False)

    args = parser.parse_args()

    verbose = 'off'                 if args.verbose is None else args.verbose

    dpDir   = args.dp_dir
    aiDir   = args.aids_dir
    lbDir   = args.lb_dir
    ns      = 'lsi' if args.ns is None else args.ns

    dpDir = os.path.normpath(dpDir)
    aiDir = os.path.normpath(aiDir)
    lbDir = os.path.normpath(lbDir)
    #bnDir = os.path.normpath(aiDir + '/pat_bones_new')
    
    #par = '_' + dpDir.rsplit('_',1)[1]

   

    # if not os.path.isdir(aiDir):
        # logging.error("Output AI dir (%s) not found"%aiDir)
        # sys.exit(1)

    try:
        if not os.path.isdir(aiDir):
            pathlib.Path(aiDir).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        print("Creating bones pattern dir (%) IO error: {}"%isDir,format(err))
        sys.exit(1)
        
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    initial_log_fn = aiDir+"/as_bones_new.log"
    logging.basicConfig(level=logging.INFO, format = log_format, handlers=[logging.FileHandler(initial_log_fn, mode='w'), logging.StreamHandler(sys.stdout)])
    
    from v_utils.v_logging_std import bind_std_2_logging
    bind_std_2_logging()
    
    if not os.path.isdir(dpDir):
        logging.error('Input directory (%s) with PNG files not found !',dpDir)
        exit(1)
    
    if not os.path.isdir(lbDir):
        logging.error('Input directory (%s) with label files not found !',dpDir)
        exit(1)

    tc          = multiprocessing.cpu_count()

    logging.info('      > -----------------------------------------------------')
    logging.info(' 0000 > Detected threads      :%d '% tc)

    at          = int((3*tc)/4)
    cv.setNumThreads(at)

    logging.info('cv2 threads           :%d ( min(%d, 8))'%(at,tc))
    logging.info('-----------------------------------------------------')
    logging.info('Verbose level         :%s '% verbose)
    logging.info('      > -----------------------------------------------------')

    gname       = dpDir + '/*_' + ns + '_superpixel_avg.png'
    gname       = os.path.normpath(gname)

    images      = glob.glob(gname)
    imid        = 0

    if images == []:
        logging.error('invalid file name or path (%s)'%gname)

    images.sort()

    points = []

    geometry = []
    status_path	= os.path.normpath(aiDir+'/'+'status.json')
    status_data = {}

    logging.info('LOOP  ---- marking bones')

    imid        = 0
    cset        = {}

    status_data = {}

    for iname in images:

        xname           = os.path.basename(iname)
        fname, fext     = os.path.splitext(xname)
        logging.info(fname)
        fname, fsuf     = fname.split('_',1)

        #if imid!= 0:
            #logging.info('NEXT      > -----------------------------------------------------')

        logging.info('file name     : {}'.format(fname))

        npy_path    = os.path.normpath(dpDir+'/'+fname+'_'+ns+'_superpixel_avg.png')
        lbl_path    = os.path.normpath(lbDir+'/'+fname+'_'+ns+'_superpixel_labels.png')
        roi_path    = os.path.normpath(lbDir+'/'+fname+'_roi_labels.png')
        #destination_lbl_path = os.path.normpath(lbDir+'/'+fname+'_nsi_050110'+'_superpixel_labels.png')
        #destination_lbl_path = os.path.normpath(lbDir+'/'+fname+'_nsi_100120'+'_superpixel_labels.png')
        destination_lbl_path = lbl_path
        cset_path   = os.path.normpath(aiDir+'/'+fname+'_'+ns+'_shape_bones_new_cset.json')

        logging.info(' 0101 > cset json     : {}'.format(cset_path))
   
        cset, iskey = try_get_key_frame(imid, cset, cset_path)

        FilterSize = cset['variable']['FilterSize']['value']
        Threshold  = cset['variable']['Threshold']['value']

        logging.info(' 0103 > FilterSize    : {}'.format( FilterSize))
        logging.info(' 0104 > Threshold     : {}'.format( Threshold ))

        imid += 1
    #0
        bones_v_polygons, img_shape = process_dir(npy_path, lbl_path, roi_path, destination_lbl_path, FilterSize, Threshold, verbose=="on")

        # wyniki w nowej wersji: 
        meta_path = os.path.normpath(aiDir + '/' + fname + '_bones_+00_' + ns + '_algV0_polygons.json')
        jsonDumpSafe(meta_path, bones_v_polygons.as_dict())

        img_path = os.path.normpath(aiDir + '/' + fname + '_bones_+00_' + ns + '_algV0_labels.png'); 
        he, wi = img_shape
        
        logging.info('wymiary obrazu maski: {}x{}'.format(wi,he))
        
        bones_img = bones_v_polygons.as_image(fill = True, w = wi, h = he, force_labelRGB = True)
        bones_img.save(img_path)

    
    # #5    
        # bones_v_polygons, img_shape = process_dir(npy_path, lbl_path, roi_path, destination_lbl_path, FilterSize, Threshold+5, verbose=="on")
        
        # # wyniki w nowej wersji: 
        # meta_path = os.path.normpath(aiDir + '/' + fname + '_bones_+05_' + ns + '_algV0_polygons.json')
        # jsonDumpSafe(meta_path, bones_v_polygons.as_dict())

        # img_path = os.path.normpath(aiDir + '/' + fname + '_bones_+05_' + ns + '_algV0_labels.png')
        # he, wi = img_shape
        # bones_img = bones_v_polygons.as_image(fill = True, w = wi, h = he, force_labelRGB = True)
        # bones_img.save(img_path)
    # #10
        # bones_v_polygons, img_shape = process_dir(npy_path, lbl_path, roi_path, destination_lbl_path, FilterSize, Threshold+10, verbose=="on")
        
        # # wyniki w nowej wersji: 
        # meta_path = os.path.normpath(aiDir + '/' + fname + '_bones_+10_' + ns + '_algV0_polygons.json')
        # jsonDumpSafe(meta_path, bones_v_polygons.as_dict())

        # img_path = os.path.normpath(aiDir + '/' + fname + '_bones_+10_' + ns + '_algV0_labels.png')
        # he, wi = img_shape
        # bones_img = bones_v_polygons.as_image(fill = True, w = wi, h = he, force_labelRGB = True)
        # bones_img.save(img_path)

    # #15
        # bones_v_polygons, img_shape = process_dir(npy_path, lbl_path, roi_path, destination_lbl_path, FilterSize, Threshold+15, verbose=="on")
        
        # # wyniki w nowej wersji: 
        # meta_path = os.path.normpath(aiDir + '/' + fname + '_bones_+15_' + ns + '_algV0_polygons.json')
        # jsonDumpSafe(meta_path, bones_v_polygons.as_dict())

        # img_path = os.path.normpath(aiDir + '/' + fname + '_bones_+15_' + ns + '_algV0_labels.png')
        # he, wi = img_shape
        # bones_img = bones_v_polygons.as_image(fill = True, w = wi, h = he, force_labelRGB = True)
        # bones_img.save(img_path)

    # #-5    
        # bones_v_polygons, img_shape = process_dir(npy_path, lbl_path, roi_path, destination_lbl_path, FilterSize, Threshold-5, verbose=="on")

        # # wyniki w nowej wersji: 
        # meta_path = os.path.normpath(aiDir + '/' + fname + '_bones_-05_' + ns + '_algV0_polygons.json')
        # jsonDumpSafe(meta_path, bones_v_polygons.as_dict())

        # img_path = os.path.normpath(aiDir + '/' + fname + '_bones_-05_' + ns + '_algV0_labels.png')
        # he, wi = img_shape
        # bones_img = bones_v_polygons.as_image(fill = True, w = wi, h = he, force_labelRGB = True)
        # bones_img.save(img_path)

    # #-10    
        # bones_v_polygons, img_shape = process_dir(npy_path, lbl_path, roi_path, destination_lbl_path, FilterSize, Threshold-10, verbose=="on")

        # # wyniki w nowej wersji: 
        # meta_path = os.path.normpath(aiDir + '/' + fname + '_bones_-10_' + ns + '_algV0_polygons.json')
        # jsonDumpSafe(meta_path, bones_v_polygons.as_dict())

        # img_path = os.path.normpath(aiDir + '/' + fname + '_bones_-10_' + ns + '_algV0_labels.png')
        # he, wi = img_shape
        # bones_img = bones_v_polygons.as_image(fill = True, w = wi, h = he, force_labelRGB = True)
        # bones_img.save(img_path)
    
    # #-15    
        # bones_v_polygons, img_shape = process_dir(npy_path, lbl_path, roi_path, destination_lbl_path, FilterSize, Threshold-15, verbose=="on")
        
        # # wyniki w nowej wersji: 
        # meta_path = os.path.normpath(aiDir + '/' + fname + '_bones_-15_' + ns + '_algV0_polygons.png')
        # jsonDumpSafe(meta_path, bones_v_polygons.as_dict())

        # img_path = os.path.normpath(aiDir + '/' + fname + '_bones_-15_' + ns + '_algV0_labels.png')
        # he, wi = img_shape
        # bones_img = bones_v_polygons.as_image(fill = True, w = wi, h = he, force_labelRGB = True)
        # bones_img.save(img_path)
        #writing image   

        #img_path = os.path.normpath(aiDir + '/' + fname + '_shape_lsi.bmp')
        #cv.imwrite(img_path,timg8)

        #img_path = os.path.normpath(aiDir + '/' + fname + '_shape_gsi.bmp')
        #cv.imwrite(img_path,tscal)

        tmps = {}
        tmps['errors'] = 0
        tmps['keyframe'] = iskey

        status_data[fname] = tmps 

        jsonDumpSafe(status_path, status_data)

#-----------------------------------------------------------------------------------------
    
#----------------------------------------------------------------------------
main()