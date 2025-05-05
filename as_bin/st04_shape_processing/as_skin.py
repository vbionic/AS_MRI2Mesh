import sys, os
import pathlib
#-----------------------------------------------------------------------------------------
sys.path.append(os.getcwd())
#-----------------------------------------------------------------------------------------
import getopt
import logging
import numpy as np
import json 
import cv2
import glob
import tracemalloc
import multiprocessing
import math as mt
#-----------------------------------------------------------------------------------------
from   scipy import ndimage, misc
from skimage.draw import line
#-----------------------------------------------------------------------------------------
from skimage.segmentation   import slic
from skimage.segmentation   import mark_boundaries
from skimage.util           import img_as_float
from skimage.color          import rgb2gray
#-----------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import ndimage, misc
#-----------------------------------------------------------------------------------------
from PIL import Image
#from pydicom.tag import Tag
#from pydicom.datadict import keyword_for_tag
from argparse import ArgumentParser
#-----------------------------------------------------------------------------------------
from v_utils.v_contour  import *
from v_utils.v_polygons import *
from v_utils.v_json import *
#-----------------------------------------------------------------------------------------

skin_border = []
skin_coeff  = []

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    mask = np.uint8(mask)
    #print(mask)
    return mask

def process_shape(images, labels, superpixels, sk_dbg_pth, sk_dbg_stack, arm_pth):

    global skin_border
    global skin_coeff

    csi         = images[3]
    nsi         = images[2]
 
    roi         = labels[0][:,:,2]

    kernel      = create_circular_mask(5,5)

    op   		= cv2.morphologyEx(roi, cv2.MORPH_DILATE, kernel)
    cl          = cv2.morphologyEx(roi, cv2.MORPH_HITMISS, kernel)

    mask 	= op-cl

    [yc,xc] 	= ndimage.measurements.center_of_mass(roi)

    if mt.isnan(xc) or mt.isnan(yc):
        roi = 0
        return([roi,roi])
    
    yc = int(yc)
    xc = int(xc)
  
    roix = roi.copy()
    roiz = roi.copy()
    roiy = roi.copy()
    
    roix[:,:] = 0
    roiz[:,:] = 0
    roiy[:,:] = 0

    roix[yc:yc+3,xc:xc+3] = 128

    r 	= 255
    k   = (2*mt.pi*r)

    xm = roi.shape[1]
    ym = roi.shape[0]

    sinn = 16
    sout = 16

    skin_len = int(k+1)
    skin_h   = sinn + sout

    sw = skin_len
    sh = skin_h

    skin_img = np.zeros([skin_h,skin_len])

    arm_img  = np.zeros([r+1,skin_len])

    csimaxX = csi.shape[1]
    csimaxY = csi.shape[0]

    for n in range(0,skin_len):

        xe = int(0.5 + r * mt.sin( 2*mt.pi * (n/(skin_len)) ))
        ye = int(0.5 + r * mt.cos( 2*mt.pi * (n/(skin_len)) ))

        dl = list(zip(*line(*[yc,xc], *[yc+ye,xc+xe])))
        dl.reverse()

#        print(r,len(dl))

        ena = 0

        ury = 0
        for i in range(0,len(dl)):
            px = dl[i][1]
            py = dl[i][0]
            if py>=0 and px>=0 and py<ym and px<xm:
                if(op[py,px]>0) or ena==1:
                    arm_img[ury,n] = csi[py,px]
                    ury+=1
                    roix[py,px] = 128
                    if ena==0:
                        roiz[py,px] = 128
                        qmax = len(dl)
                        for q in range(-sout,sinn-1):
                           ridx = (i+q) 
                           if ridx >= 0 and ridx < qmax and dl[ridx][0]<csimaxY and dl[ridx][1]<csimaxX and dl[ridx][0]>=0 and dl[ridx][1]>=0:
                               skin_img[q+sout,n] = csi[dl[ridx][0],dl[ridx][1]]
                    ena = 1


            else:
               arm_img[i,n] = 0
          


#    op   	= cv2.morphologyEx(roix, cv2.MORPH_DILATE, kernel)
#    cl           = roix #cv2.morphologyEx(roix, cv2.MORPH_HITMISS, kernel)

    cv2.imwrite(sk_dbg_pth, skin_img)
    cv2.imwrite(arm_pth, arm_img)

    #process skin 

    window_w = 64
    ww       = 64

    tile = np.zeros([sh,window_w])

    LFat   = []
    LSkin  = []

    for i in range(0,sw):

        beg = i - int(window_w/2)
        end = i + int(window_w/2)
           
        n   = 0

        for t in range(beg,end):
            k = t 
            if k<0:
                tile[:,n] = skin_img[:,sw+k]
            elif k>=sw:
                tile[:,n] = skin_img[:,k-sw-1]
            else:
                tile[:,n] = skin_img[:,k]
            n = n+1

        Tmin = 255 

        for j in range(-7,8):
            for k in range(0,(sh>>1)):
                v = tile[k,(ww>>1)+j]
                if Tmin>v:
                    Tmin = v

        for k in range((sh>>1)+2,0,-1):
            avg = 0
            for j in range(-3,4):
                avg += tile[k,(ww>>1)+j] 
            avg /= 7
            if avg<(8+Tmin*1.5):
                x = k-(sh>>1)
                if x<4:
                    x=4
                LSkin.append(x)
                tile[k,(ww>>1)] = 255
                break


    for n in range(0,skin_len):

        xe = int(0.5 + r * mt.sin( 2*mt.pi * (n/(k*360)) ))
        ye = int(0.5 + r * mt.cos( 2*mt.pi * (n/(k*360)) ))

        dl = list(zip(*line(*[yc,xc], *[yc+ye,xc+xe])))
        dl.reverse()
 
        ena = 0

        gr = 3
        for i in range(0,len(dl)):
            px = dl[i][1]
            py = dl[i][0]
            if py>=0 and px>=0 and py<ym and px<xm:
                if(roi[py,px]>0):
                    roix[py,px] = 255
                    for k in range(0,gr): #range(0,LSkin[i]):
                        nx = dl[i-k][1]
                        ny = dl[i-k][0]
                        if np.sqrt((px-nx)*(px-nx) + (py-ny)*(py-ny))<gr:
                            if ny>=0 and nx>=0 and ny<ym and nx<xm:
                                 roiy[ny,nx]= 255
                    break

    ker 		= np.array([[1,1,1],[1,1,1],[1,1,1]]).astype(np.uint8)
    

    roiy 	    = cv2.dilate(roiy,ker,1)
    roiy 	    = cv2.erode(roiy,ker,1)


#    roiy 	    = cv2.medianBlur(roiy,3)
#    ker = np.array([[0,1,0],[1,0,1],[0,1,0]]).astype(np.uint8)
#    roiy          = cv2.morphologyEx(roiy, cv2.MORPH_HITMISS, ker)

    skin_points = cv2.bitwise_and(roiy,csi)

    n_tot 		= np.sum(roiy)/255
    n_sum 		= np.sum(skin_points)

    coeff 		= 100*(n_sum/(128*n_tot))

    skin_coeff.append(coeff)

    print("wsp. wypełnienia : %d %%"%coeff)

#    op   		= cv2.morphologyEx(roiy, cv2.MORPH_DILATE, kernel).astype(np.int32)
#    cl          = roiy.astype(np.int32)#cv2.morphologyEx(roi, cv2.MORPH_HITMISS, kernel)

#    mask 	 		= op-cl

#    plt.imshow(skin_points,cmap='gray')
#    plt.show()

    return([roiy,skin_img])

#    plt.imshow(skin_img,cmap='gray') 
#    plt.show() 

    ax1 = plt.subplot(141)
    ax1.imshow(roi,cmap='gray')

    ax2 = plt.subplot(142,sharex=ax1,sharey=ax1)
    ax2.imshow(roix,cmap='gray')

    ax3 = plt.subplot(143,sharex=ax1,sharey=ax1)
    ax3.imshow(roiz,cmap='gray')

    ax4 = plt.subplot(144,sharex=ax1,sharey=ax1)
    ax4.imshow(mask,cmap='gray')

    plt.show() 

    return([mask,sk_dbg_stack])

#-----------------------------------------------------------------------------------------
def main():

    parser = ArgumentParser()

    parser.add_argument("-imgDir",  "--img_dir",    dest="img_dir",    help="input png directory",            metavar="PATH",required=True)
    parser.add_argument("-labDir",  "--labels_dir", dest="labels_dir", help="labels png directory",           metavar="PATH",required=True)
    parser.add_argument("-skinDir", "--skin_dir",   dest="skin_dir",   help="output skin shape directory",    metavar="PATH",required=True)
    parser.add_argument("-v",       "--verbose",    dest="verbose",    help="verbose level",                                 required=False)

    args = parser.parse_args()

    verbose     = 'off'                 if args.verbose is None else args.verbose
    imgDir  	= args.img_dir
    labDir  	= args.labels_dir
    skinDir  	= args.skin_dir

    imgDir = os.path.normpath(imgDir)
    labDir = os.path.normpath(labDir)
    skinDir = os.path.normpath(skinDir)

    if not os.path.isdir(imgDir):
        logging.error('Error : Input directory (%s) with PNG files not found !'%imgDir)
        exit(1)

    try:
        if not os.path.isdir(skinDir):
            pathlib.Path(skinDir).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        logging.error("Creating shapem pattern dir (%s) IO error: %s"%(skinDir,err))
        sys.exit(1)

    logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s",handlers=[logging.FileHandler(skinDir+"/skin_shape.log",mode='w'),logging.StreamHandler(sys.stdout)])

    from v_utils.v_logging_std import bind_std_2_logging
    bind_std_2_logging()
    
    tc          = multiprocessing.cpu_count()

    logging.info('> -----------------------------------------------------')
    logging.info('> Detected threads      :%d '% tc)

    at          = int((3*tc)/4)
    cv2.setNumThreads(at)

    logging.info('> CV2 threads           :%d ( min(%d, 8))'%(at,tc))
    logging.info('> -----------------------------------------------------')
    logging.info('> Verbose level         :%s '% verbose)
    logging.info('> -----------------------------------------------------')

    gname       = imgDir + '/*_nsi.png'
    gname       = os.path.normpath(gname)

    images      = glob.glob(gname)
    imid        = 0

    if images == []:
        logging.error('> invalid file name or path (%s)'%gname)
        exit(1) 

    images.sort()

    skin_dbg_stack = None

    imid        = 0

    all_polygons = v_polygons()

    #images = images[58:]

    for iname in images:

        xname           = os.path.basename(iname)
        fname, fext     = os.path.splitext(xname)
        fname, fsuf     = fname.split('_')

        if imid!= 0:
            logging.info('> -----------------------------------------------------')

        logging.info('> file name     : ' + fname)

        lsi_path 	    = os.path.normpath(imgDir+'/'+fname+'_lsi.png')
        gsi_path 	    = os.path.normpath(imgDir+'/'+fname+'_gsi.png')
        nsi_path 	    = os.path.normpath(imgDir+'/'+fname+'_nsi.png')
        csi_path 	    = os.path.normpath(imgDir+'/'+fname+'_csi.png')

        slx_path 	    = os.path.normpath(imgDir+'/'+fname+'*slic*_labels.png')
        spx_path 	    = os.path.normpath(imgDir+'/'+fname+'*super*_labels.png')
        lab_path_nsi	    = os.path.normpath(labDir+'/'+fname+'_nsi_roi_unetV0_labels.png')
        lab_path_wo 	    = os.path.normpath(labDir+'/'+fname+'*roi_labels.png')

        out_mask_path 	    = os.path.normpath(skinDir+'/'+fname+'_skin_labels.png')
        out_arm_path 	    = os.path.normpath(skinDir+'/'+fname+'_unreel.png')
        out_poly_path 	    = os.path.normpath(skinDir+'/'+fname+'_skin_polygons.json')

        out_skin_path 	    = os.path.normpath(skinDir+'/'+fname+'_skin_dbg.png')

        imid += 1

        #images --------------------------------------------

        lsi = cv2.imread(lsi_path,cv2.IMREAD_GRAYSCALE)
        gsi = cv2.imread(gsi_path,cv2.IMREAD_GRAYSCALE)
        nsi = cv2.imread(nsi_path,cv2.IMREAD_GRAYSCALE)
        csi = cv2.imread(csi_path,cv2.IMREAD_GRAYSCALE)

        images      = [lsi,gsi,nsi,csi]

        logging.info("INFO  > MR images loaded, nsi,gsi,nsi,csi files")

        #labels --------------------------------------------

        labels      = []
        labels_list1  = glob.glob(lab_path_nsi)
        labels_list2  = glob.glob(lab_path_wo)

        labels_list   = labels_list1 if len(labels_list1)>0 else labels_list2

        for lpth in labels_list:      
           img = cv2.imread(lpth)
           labels.append(img)

        logging.info("INFO  > roi labesl loaded, %d file(s)"%len(labels))

        #superpixels --------------------------------------------

        superpixels = []

        slic_list    = glob.glob(spx_path)
        for spth in slic_list:      
           img = cv2.imread(spth)
           superpixels.append(img)

        slic_list    = glob.glob(slx_path)
        for spth in slic_list:      
           img = cv2.imread(spth)
           superpixels.append(img)

        logging.info("INFO  > superpixels loaded, %d file(s)"%len(superpixels))
             
        #start processing ---------------------------------------

        [mask,skin_dbg_stack] 	= process_shape(images, labels, superpixels,out_skin_path,skin_dbg_stack,out_arm_path)

        cv2.imwrite(out_mask_path, mask)

#        creating file with metadata (box & contour)

        mypolygons = v_polygons.from_ndarray(mask)
        jsonDumpSafe(out_poly_path, mypolygons.as_dict())

#        out_skin_st_path 	    = os.path.normpath(skinDir+'/'+fname+'_skin_stack.png')
#        cv2.imwrite(out_skin_st_path, skin_dbg_stack)

    print(len(skin_coeff))
    for c in skin_coeff:
        print("skin coeff : %d %%"%c)

    
    print("avg coeff : %.2f %%"%(np.sum(skin_coeff)/len(skin_coeff)))
   

    return
        
#-----------------------------------------------------------------------------------------
   

if __name__ == '__main__':
    main()
