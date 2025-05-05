import sys, getopt, shutil
import pydicom
import numpy as np
import json
import os
import cv2
import math
import glob
import tracemalloc
import multiprocessing
import mapbox_earcut as earcut

#-----------------------------------------------------------------------------------------
from scipy import ndimage, misc
from skimage import data, img_as_float
from skimage import exposure
#-----------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits import mplot3d
#-----------------------------------------------------------------------------------------
from argparse import ArgumentParser
import logging
#-----------------------------------------------------------------------------------------
sys.path.append(os.getcwd())
#-----------------------------------------------------------------------------------------
from v_utils.v_contour import *
from v_utils.v_polygons import *
from v_utils.v_json import jsonUpdate, jsonDumpSafe

#-----------------------------------------------------------------------------------------

def add_polygons(polygons,xyz,slice_dist):
    for p in polygons:
        vpath = p["outer"]["path"]
        for v in vpath:
            xyz.append([v[0]*0.5,v[1]*0.5,pid*slice_dist])    

    return(xyz)

#-----------------------------------------------------------------------------------------

def add_one_polygon(polygon,z,slice_dist):
    p 		= polygon
    xyz 	= []
    vpath 	= p
    for i in range(0,len(polygon["poly"])):
        v = polygon["poly"][i]
        xyz.append([v[0]*0.5,v[1]*0.5,z*slice_dist])

    vset = {"poly":xyz,"cent":polygon["cent"]*0.5}
    return(vset)
    
#-----------------------------------------------------------------------------------------

def load_json(pol_path):

    try:
        pol_file 	= open(pol_path,"r");     
        pol_dict	= json.load(pol_file)
        logging.info('loading %s file'%(os.path.basename(pol_path)))
        return(pol_dict)
    except:
        logging.error('cannot open %s file !!!'%(fname))
        exit(1)

#-----------------------------------------------------------------------------------------

def create_lines(a,r):

    r_line = np.arange(0,r)
    lines = []
    normals = []
    for s in range(0,a):
        wc = math.cos(2*math.pi*s/a)
        ws = math.sin(2*math.pi*s/a)
        line = []
        for rr in range(2*(r-1),-1,-1):
            px = (rr/2.0)*wc
            py = (rr/2.0)*ws
            line.append([px,py])
        lines.append(np.array(line))
        
        normals.append([wc,ws])

    return(lines,normals)

#-----------------------------------------------------------------------------------------

def get_polygon(pols):

    rd 			= {}
    rd["ski"] 	= []

    logging.debug(">beg")

    lst_pos     = -1
    cur_pol 	= []
    cur_color 	= []
    cur_len =  0
    cur_cen =  [0,0]

    for pol in pols: 
        point_list	= pol["outer"]["path"]
        lenght   	= len(pol["outer"]["path"])
        centroid 	= np.round(np.mean(point_list,axis=0),2)
        
        logging.debug("%6.2f\t%6.2f\t len = %d"%(centroid[0],centroid[1],lenght))
        
        if cur_len<lenght:
            cur_pol = pol["outer"]["path"]
            cur_len = lenght
            cur_cen = centroid

    logging.debug("%6.2f\t%6.2f\t len = %d"%(cur_cen[0],cur_cen[1],cur_len))
    
    return(cur_pol, cur_len, cur_cen)

def get_multi_polygon(pols):

    logging.debug(">beg")

    lst_pos     = -1
    cur_pol 	= []
    cur_color 	= []
    cur_len =  []
    cur_cen =  []

    logging.debug("+")

    for pol in pols: 
        point_list	= pol["outer"]["path"]
        lenght   	= len(pol["outer"]["path"])
        centroid 	= np.round(np.mean(point_list,axis=0),2)

        cur_pol.append(pol["outer"]["path"])
        cur_len.append(lenght)
        cur_cen.append(centroid)
        
        logging.debug(" %6.2f\t%6.2f\t len = %d"%(centroid[0],centroid[1],lenght))
    
    return(cur_pol, cur_len, cur_cen)

#-----------------------------------------------------------------------------------------

def rebuild_skin(file_path,poly,centerT): 

    global param_a
    global circ_lines
    
    img         = cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
    img[:,:]    = 0
    
    if(len(poly)>3):
        cv2.fillPoly(img, [poly], (255)) 


    xmax    = img.shape[1]-1
    ymax    = img.shape[0]-1

    center  = np.round(np.mean(poly,axis=0),2)

    cont    = []
    crgb    = []
    lastc   = center

    img[:,:]    = img[:,:]!=0
    found 	    = 0
 
    if len(poly)!=0:

        lost = []
        
        for a in range(0,param_a):

            line    = circ_lines[a]
            found   = 0

            for v in line:
                pos = v+center
                if pos[0]<1 or pos[1]<1 or pos[0]>=xmax or pos[1]>=ymax:
                    continue
                
                px = int(np.round(pos[0]))
                py = int(np.round(pos[1]))

                x0 = img[py-1:py+2,px-1:px+2].mean()

                if x0 >= 0.5: 
                    cont.append([pos[0],pos[1]])
                    found 	= 1
                    break

            if not found:
                print("for angle %d no contour was found (polygon length : %d, center : "%(a,len(poly)),center,")")
                print(poly)
                print(center)
                lost = a
                cont.append(lastc)
                
        cur_pol = {"pnum":1, "center":[center], "vertices":[np.array(cont)]} 

    else:
    
        cur_pol = {"pnum":0, "center":[], "vertices":[]} 

    return(cur_pol)

#-----------------------------------------------------------------------------------------

def group_load_polygons(inputdir, name, iter_list):

    tmp = []
    for xpath in iter_list:

        xname           	= os.path.basename(xpath)
        fname, fext     	= os.path.splitext(xname)
        fname, ftis, fsuf   = fname.split('_')

        poly_path           = inputdir + "/" + name + "/" + fname + "_" + name + "_polygons.json"

        poly_dict           = load_json(poly_path)

        tmp.append(poly_dict["polygons"])

    return(tmp)  

#-----------------------------------------------------------------------------------------

def group_convs_polygons(poly_list):

    print(">> gen_convs_polygons")

    t_poly 	        = []
    t_cent  	    = []

    for poly in poly_list:

        cp,cl,cc                  = get_polygon(poly)

        t_poly.append( { "pnum":1, "center":[cc], "vertices":[cp]})

    return(t_poly)

#-----------------------------------------------------------------------------------------

def group_convm_polygons(poly_list):

    print(">> gen_convm_polygons")

    t_poly 	        = []
    t_cent  	    = []

    for poly in poly_list:

        cp,cl,cc                  = get_multi_polygon(poly)

        for n in range(len(cp)):
           cp[n].reverse()
           
        t_poly.append( { "pnum":len(cc), "center":cc, "vertices":cp})

    return(t_poly)

#-----------------------------------------------------------------------------------------

def group_gen_cpath(poly_list):

    print(">> gen_path_of_centers")
    
    pnum    = len(poly_list)
    pcent   = []
    
    for n in range(pnum):

        clast = [0,0]
        csum  = [0,0]
        cnum  = 0

        for i in range(n-3,n+4):
           if i>=0 and i<pnum:
          
               clen = len(poly_list[n]["vertices"])
               cent = poly_list[n]["center"]
       
               if clen>12:
                  csum += cent
                  cnum += 1
               
        if cnum > 0:
            csum /= cnum              
        else:
            csum = clast

        clast = csum
        pcent.append(csum)

    qcen = [0.0,0.0]
    qnum = 0

    for n in range(pnum):
        #print("center : ",poly_list[n]["center"]," acc_cent : ", qcen)
        qcen += np.array(poly_list[n]["center"])
        qnum +=1

    qcen /= qnum
    #print("average center : ",qcen)

    return(pcent, qcen)

#-----------------------------------------------------------------------------------------

def group_get_dir(poly_list,coordinates):

    spos = []
    
    for poly_path in poly_list:

        xname           	= os.path.basename(poly_path)
        fname, fext     	= os.path.splitext(xname)
        fname, ftis, fsuf   = fname.split('_')

        spos.append(np.array(coordinates[fname]))

    sdir = -1 if (spos[1][2] - spos[0][2]) < 0 else 1

    return(sdir)

#-----------------------------------------------------------------------------------------

def group_add_sz_pos(p_slice, p_center, spacing):

    pp = []
    
    for n in range(len(p_slice)):

        if len(p_slice[n]["vertices"])>0:
        
            cpol = p_slice[n]["vertices"][0]
            cent = p_slice[n]["center"][0]
            
            xpol    = []
            xcent   = []
            
            #print("polygon length : ",len(cpol))
            for i in range(len(cpol)):
                xpol.append([cpol[i][0]*0.5, cpol[i][1]*0.5, spacing*n])
#                xpol.append([cpol[i][0]*0.5, (-cpol[i][1])*0.5, spacing*n])
            
            xpol.reverse()
            xcent = [cent[0]*0.5, cent[1]*0.5, spacing*n]
#            xcent = [cent[0]*0.5, (p_center[0][1] - cent[1])*0.5, spacing*n]

            pp.append({ "pnum":1, "center":[xcent], "vertices":[xpol]})

    return(pp)

#-----------------------------------------------------------------------------------------

def group_add_mz_pos(p_slice, p_center, spacing):

    pp = []
    
    for n in range(len(p_slice)):

        xp = []
        xc = []
        
        for k in range(len(p_slice[n]["vertices"])):
            if len(p_slice[n]["vertices"][k])>0:
            
                cpol = p_slice[n]["vertices"][k]
                cent = p_slice[n]["center"][k]
                
                xpol    = []
                xcent   = []
                
                #print("polygon length : ",len(cpol))
                for i in range(len(cpol)):
                    xpol.append([cpol[i][0]*0.5, cpol[i][1]*0.5, spacing*n])
#                   xpol.append([cpol[i][0]*0.5, (p_center[0][1] - cpol[i][1])*0.5, spacing*n])
                
                xpol.reverse()

                xp.append(xpol)  
                xc.append([cent[0]*0.5, (p_center[0][1] - cent[1])*0.5, spacing*n])

        pp.append({ "pnum":len(p_slice[n]["vertices"]), "center":xc, "vertices":xp})

    return(pp)
    
#-----------------------------------------------------------------------------------------

def group_write_single_poly_vertexes(out_file,start_idx, vlist):
    
    sid = 0
    vid = start_idx

    for vset in vlist:

        if len(vset["vertices"])>0:

            print("#slice {0} with {1} vertices".format(sid,len(vset["vertices"])))
            
            verts 	= vset["vertices"][0]
            centr 	= vset["center"][0]
         
            out_file.write("#slice {0} with {1} vertices\n".format(sid,len(verts)))
            for item in verts:
                out_file.write("v %.3f %.3f %.3f\n"%(item[0],item[1],item[2]))
                vid += 1 

        sid += 1         

    return(vid,sid)

#-----------------------------------------------------------------------------------------

def group_write_poly_vertexes_wd(out_file,start_idx, vlist, Zdist):
    
    sid = 0
    vid = start_idx

    for vset in vlist:

        for vx in vset["vertices"]:
          
            if len(vx)>0:
                vx.reverse()
                print("#slice {0} with {1} vertices".format(sid,len(vx)))
                
                out_file.write("#slice {0} with {1} vertices\n".format(sid,len(vx)))
                for item in vx:
                    out_file.write("v %.3f %.3f %.3f\n"%(item[0],item[1],item[2]))
                    vid += 1 
                for item in vx:
                    out_file.write("v %.3f %.3f %.3f\n"%(item[0],item[1],item[2]+Zdist))
                    vid += 1 
        sid += 1         

    return(vid,sid)


def filter_poly(slices):

    return(slices)

#-----------------------------------------------------------------------------------------

def gen_normals(vlist,norm):

    nsize = len(norm)
    print("nsize : ",nsize)

    nlist = []
    ntlist = []

    px0  = []
    py0  = []

    xlist = []
    xlist.append(vlist[-1])
    for v in vlist : 
      xlist.append(v)
    xlist.append(vlist[0])

    for n in range(1,len(xlist)-1):
        xL = xlist[n-1][0]
        yL = xlist[n-1][1]
        x0 = xlist[n  ][0]
        y0 = xlist[n  ][1]
        xR = xlist[n+1][0]
        yR = xlist[n+1][1]
    
        xa = -(y0-yL)
        ya =  (x0-xL)

        xb = -(yR-y0)
        yb =  (xR-x0)

        x = xa+xb
        y = ya+yb

        if y>=0 and x>=0:   
            a = math.pi/2 if x == 0 else math.atan(y/x)
            id = int((2*a/math.pi)*(nsize/4))

        elif y<0 and x>=0:   
            a = -math.pi/2 if x == 0 else math.atan(y/x)
            id = int((nsize-1)+(2*a/math.pi)*(nsize/4))

        elif y<0 and x<=0:   
            a = math.pi/2 if x == 0 else math.atan(y/x)
            id = int(((nsize/4)*3)-(2*(a)/math.pi)*int(nsize/4))

        else:   
            a = -math.pi/2 if x == 0 else math.atan(y/x)
            id = int(((nsize/4)*2)+(2*(a)/math.pi)*int(nsize/4))

        px0.append(x0)
        py0.append(y0)
 
        nlist.append(id)
        ntlist.append((id+90)%180)

        #print(xL,yL,"<",xa,ya,">",x0,y0,"<",xb,yb,">",xR,yR,">>",x,y,">alpha>",a,"=", norm[id])
     
    #plt.plot(px0,py0)


    verts3d = np.array(vlist).reshape(-1, 3)
    logging.debug(verts3d)
    verts2d = np.delete(verts3d, 2, 1)
    logging.debug(verts2d)
    verts2f = verts2d.copy().flatten()
    logging.debug(verts2f)
    rings = np.array([len(verts3d)])
    result = earcut.triangulate_float32(verts2d, rings)
    logging.debug(result.dtype)
    logging.debug(result.shape)
    result = result.reshape(-1, 3)
    logging.debug(result)
    return(nlist,ntlist,result)

#-----------------------------------------------------------------------------------------

parser = ArgumentParser()

parser.add_argument("-iDir",  "--in_dir",      dest="in_dir",   help="poligons input directory",   metavar="PATH",required=True)
parser.add_argument("-sDir",  "--skin_dir",    dest="skin_dir",   help="poligons input directory",   metavar="PATH",required=True)
parser.add_argument("-oDir",  "--out_dir",     dest="out_dir",  help="mesh output directory",      metavar="PATH",required=True)
parser.add_argument("-dDir",  "--desc_dir",    dest="desc_dir", help="description json file path", metavar="PATH",required=True)
parser.add_argument("-nSfx",  "--name_sfx",    dest="name_sfx", help="name sufix",                                required=True)
parser.add_argument("-v",     "--verbose",     dest="verbose", help="verbose level",                              required=False)

args = parser.parse_args()

verbose 	= 'off'                 if args.verbose is None else args.verbose
inDir  		= args.in_dir
skinDir     = args.skin_dir
outDir  	= args.out_dir
descDir  	= args.desc_dir
name_sfx    = args.name_sfx

iDir 		= os.path.normpath(inDir)
skinDir 	= os.path.normpath(skinDir)
oDir 		= os.path.normpath(outDir)
dDir 		= os.path.normpath(descDir)

logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(message)s",handlers=[logging.FileHandler("gen_mesh.log",mode='w'),logging.StreamHandler(sys.stdout)])    
    
if not os.path.isdir(iDir):
    logging.error('Error : Input directory (%s) with polygons files not found !'%iDir)
    exit(1)
logging.info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
logging.info("START : as_gen_mesh.py")
logging.info("in    : "+iDir)
logging.info("out   : "+oDir)

np.set_printoptions(linewidth=128)

tc          = multiprocessing.cpu_count()

logging.info('-----------------------------------------------------')
logging.info('Detected threads      :%d '% tc)

at          = int((3*tc)/4)
cv2.setNumThreads(at)

logging.info('CV2 threads           :%d ( min(%d, 8))'%(at,tc))
logging.info('-----------------------------------------------------')
logging.info('Verbose level         :%s '% verbose)
logging.info('-----------------------------------------------------')

gname           = iDir + '/skin/*_polygons.json'
gname           = os.path.normpath(gname)
gname 		    = gname.replace('[','[[]')

skin_poly_list  = glob.glob(gname)
imid            = 0

if len(skin_poly_list)==0:
    logging.error('invalid file name or path (%s)'%gname)
    exit(1)

skin_poly_list.sort()

#----------------------------------------------------
# slice spacing
#----------------------------------------------------

desc_path           = dDir + "/" + "description.json"
desription          = load_json(desc_path)

slice_dist          = desription["distance_between_slices"]
logging.info("slice distance : %d"%slice_dist)

#----------------------------------------------------
#paramerty globalne
#----------------------------------------------------

param_a         = 90
param_r         = 256

scan            =  {}
scan["skin" ]   =  {}
scan["bones"]   =  {}
scan["vessels"] =  {}
scan["fat"]     =  {}
scan["muscles"] =  {}

#----------------------------------------------------
# wczytanie jsonow z poligonami
#----------------------------------------------------

scan["skin"   ]["src"]  = group_load_polygons(skinDir,"skin"   ,skin_poly_list)
scan["fat"    ]["src"]  = group_load_polygons(skinDir,"fat"    ,skin_poly_list)
scan["muscles"]["src"]  = group_load_polygons(skinDir,"muscles",skin_poly_list)
scan["bones"  ]["src"]  = group_load_polygons(skinDir,"bones"  ,skin_poly_list)
scan["vessels"]["src"]  = group_load_polygons(skinDir,"vessels",skin_poly_list)

#----------------------------------------------------
#stworzenie 360/"param_a" lini o dlugosci "param_r" 
#----------------------------------------------------

circ_lines,circ_norm = create_lines(param_a,param_r)

#----------------------------------------------

scan["skin"   ]["slice"] = group_convs_polygons(scan["skin"   ]["src"])
scan["fat"    ]["slice"] = group_convm_polygons(scan["fat"    ]["src"])
scan["muscles"]["slice"] = group_convm_polygons(scan["muscles"]["src"])
scan["bones"  ]["slice"] = group_convm_polygons(scan["bones"  ]["src"])
scan["vessels"]["slice"] = group_convm_polygons(scan["vessels"]["src"])

#----------------------------------------------
#dodatkowe parametry
#----------------------------------------------

total_p         = len(scan["skin"]["slice"])

#----------------------------------------------

cpath, glb_center               = group_gen_cpath(scan["skin" ]["slice"])

scan["skin" ]["center_path"]    = cpath
scan["skin" ]["center"]         = glb_center

#----------------------------------------------

#scan_dir                        = group_get_dir(skin_poly_list,desription["coordinates"])
#
#if scan_dir==-1:
#    skin_poly_list.reverse()
#    scan["skin" ]["center_path"].reverse()
#    scan["skin" ]["slice"].reverse()
#    
#    scan["fat" ]["slice"].reverse()
#    scan["muscles" ]["slice"].reverse()
#    scan["bones" ]["slice"].reverse()
#    scan["vessels" ]["slice"].reverse()

#----------------------------------------------

scan["bones" ]["slice"] = filter_poly(scan["bones" ]["slice"]) 

#----------------------------------------------

print("total slices : ",total_p)
#print("direction : ",scan_dir)

#----------------------------------------------

for n in range(total_p):

    xname           	= os.path.basename(skin_poly_list[n])
    fname, fext     	= os.path.splitext(xname)
    fname, ftis, fsuf   = fname.split('_')

    labels_path         = iDir + '/skin/' + fname + "_skin_labels.png"
    
    logging.debug(labels_path)
    
    p_slice                     = scan["skin"]["slice"][n]
    p_vertices                  = np.int32(p_slice["vertices"][0]) # Bug with fillPoly, needs explict cast to 32bit
    p_center                    = p_slice["center"][0]
    
    new_skin_poly		        = rebuild_skin(labels_path,p_vertices,p_center) 
    
    scan["skin"]["slice"][n]    = new_skin_poly

#----------------------------------------------

scan["skin" ]["slice"]      = group_add_sz_pos(scan["skin"   ]["slice"], scan["skin"]["center"], slice_dist)
scan["muscles"]["slice"]    = group_add_mz_pos(scan["muscles"]["slice"], scan["skin"]["center"], slice_dist)
scan["bones"]["slice"]      = group_add_mz_pos(scan["bones"  ]["slice"], scan["skin"]["center"], slice_dist)
scan["vessels"]["slice"]    = group_add_mz_pos(scan["vessels"]["slice"], scan["skin"]["center"], slice_dist)

#----------------------------------------------

ses     = os.path.basename(oDir)
user    = os.path.basename(os.path.dirname(oDir))

m3d_name = user + "_" + ses + "_mesh_" + name_sfx
m3d_skin_name = user + "_" + ses + "_skin_" + "mesh_" + name_sfx

skinfile = open(oDir + '/'+m3d_skin_name + '.obj', 'w')
thefile  = open(oDir + '/'+m3d_name + '.obj', 'w')
mtlfile  = open(oDir + '/'+m3d_name + '.obj.mtl', 'w')
jsonfile = open(oDir + '/'+m3d_name + '.json', 'w')

#----------------------------------------------
#json file
#----------------------------------------------

jv = []
for vset in scan["skin"]["slice"]:

    verts 	= vset["vertices"][0]
    centr 	= vset["center"][0]
    v_dict	= {}
    v_dict["center"] 	= [ float(np.round(centr[0],3)), float(np.round(centr[1],3)), float(np.round(centr[2],3)) ]
    v_dict["polygon"] 	= []
    for item in verts:
        v_dict["polygon"].append([ float(np.round(item[0],3)),float(np.round(item[1],3)),float(np.round(item[2],3))])
    jv.append(v_dict) 

finalj = {}
finalj["mesh"] = jv
json.dump(finalj, jsonfile, indent=4)

#----------------------------------------------
#obj file
#----------------------------------------------

thefile.write("#mtl file\n")
thefile.write("mtllib ./"+ m3d_name + ".obj.mtl\n")

skinfile.write("#mtl file\n")
skinfile.write("mtllib ./"+ m3d_name + ".obj.mtl\n")

sid = 0
vid = 0

xv,  xs     = group_write_single_poly_vertexes(thefile,vid,scan["skin"]["slice"])
vid, sid    = group_write_single_poly_vertexes(skinfile,vid,scan["skin"]["slice"])

vertsB = scan["skin"]["slice"][0]["vertices"][0]
centrB = scan["skin"]["slice"][0]["center"][0]

vid     +=1
vidB     = vid

print("#bottom center vertex")
thefile.write("#bottom center vertex\n")
thefile.write("v %.2f %.2f %.2f\n"%(centrB[0],centrB[1],vertsB[0][2]))

skinfile.write("#bottom center vertex\n")
skinfile.write("v %.2f %.2f %.2f\n"%(centrB[0],centrB[1],vertsB[0][2]))

vertsE = scan["skin"]["slice"][-1]["vertices"][0]
centrE = scan["skin"]["slice"][-1]["center"][0]

vid     +=1
vidE   = vid

print("#top center vertex")

thefile.write("#top center vertex\n")
thefile.write("v %.2f %.2f %.2f\n"%(centrE[0],centrE[1],vertsE[0][2]))

skinfile.write("#top center vertex\n")
skinfile.write("v %.2f %.2f %.2f\n"%(centrE[0],centrE[1],vertsE[0][2]))

vid     +=1
bones_vid       = vid
vid, sid        = group_write_poly_vertexes_wd(thefile,vid,scan["bones"]["slice"],slice_dist)

#vid     +=1
vessels_vid     = vid
vid, sid        = group_write_poly_vertexes_wd(thefile,vid,scan["vessels"]["slice"],slice_dist)

#vid     +=1
# muscles_vid     = vid
# vid, sid        = group_write_poly_vertexes_wd(vid,scan["muscles"]["slice"],slice_dist)

print("#vertex normals")
thefile.write("#vertex normals\n")
for j in range(0,param_a):
    thefile.write("vn %.3f %.3f 0.000\n"%(circ_norm[j][0],circ_norm[j][1]))  

for j in range(0,param_a):
    skinfile.write("vn %.3f %.3f 0.000\n"%(circ_norm[j][0],circ_norm[j][1]))  


#--------------------------------
# material for skin 
#--------------------------------

thefile.write("g skin\n")

print("#material for skin")
thefile.write("#material for skin\n")
thefile.write("usemtl material_0\n")
thefile.write("#skin faces\n")

# print("#bottom cap")
# thefile.write("#bottom cap\n")
# for j in range(0,param_a-1):
    # thefile.write("f {0} {1} {2}\n".format(j+2,j+1,vidB))  
# thefile.write("f {0} {1} {2}\n".format(1,param_a,vidB))  

# print("#top cap")
# thefile.write("#top cap\n")
# for j in range(0,param_a-1):
    # thefile.write("f {0} {1} {2}\n".format(vid-param_a+j+1,vid-param_a+j+2,vidE))  
# thefile.write("f {0} {1} {2}\n".format(vid-param_a+param_a,vid-param_a+1,vidE))  

verts = scan["skin"]["slice"]
for i in range(0,len(verts)-1):
    print("#slice {0}-{1} with {2} faces".format(i,i+1,2*param_a))
    thefile.write("#slice {0}-{1} with {2} faces\n".format(i,i+1,2*param_a))
    skinfile.write("#slice {0}-{1} with {2} faces\n".format(i,i+1,2*param_a))

    s0 = param_a*(i + 0) + 1
    s1 = param_a*(i + 1) + 1
    for j in range(0,param_a-1):
#        thefile.write("f {0}//{3} {1}//{4} {2}//{5}\n".format(s0+j,s0+j+1,s1+j  ,j+1,j+2,j+1))  
#        thefile.write("f {0}//{3} {1}//{4} {2}//{5}\n".format(s1+j,s0+j+1,s1+j+1,j+1,j+2,j+2))  

#        skinfile.write("f {0}//{3} {1}//{4} {2}//{5}\n".format(s0+j,s0+j+1,s1+j  ,j+1,j+2,j+1))  
#        skinfile.write("f {0}//{3} {1}//{4} {2}//{5}\n".format(s1+j,s0+j+1,s1+j+1,j+1,j+2,j+2))  
       

        thefile.write("f {0}//{3} {1}//{4} {2}//{5}\n".format(s1+j,s1+j+1,s0+j  ,j+1, j+2, j+1))  
        thefile.write("f {0}//{3} {1}//{4} {2}//{5}\n".format(s1+j+1,s1+j,s0+j  ,(j+90)%180+2, (j+90)%180+1, (j+90)%180+1))  

        thefile.write("f {0}//{3} {1}//{4} {2}//{5}\n".format(s0+j+1,s0+j,s1+j+1,j+1, j+2, j+1))
        thefile.write("f {0}//{3} {1}//{4} {2}//{5}\n".format(s0+j,s0+j+1,s1+j+1,(j+90)%180+1, (j+90)%180+2, (j+90)%180+1))  

        skinfile.write("f {0}//{3} {1}//{4} {2}//{5}\n".format(s1+j,s1+j+1,s0+j  ,j+1, j+2, j+1))  
        skinfile.write("f {0}//{3} {1}//{4} {2}//{5}\n".format(s0+j+1,s0+j,s1+j+1,j+2, j+1, j+2))  
       
#    thefile.write("f {0}//{3} {1}//{4} {2}//{5}\n".format(s0+param_a-1,s0,s1+param_a-1,param_a,1,param_a))  
#    thefile.write("f {0}//{3} {1}//{4} {2}//{5}\n".format(s1+param_a-1,s0,s1          ,param_a,1,1))  

#    skinfile.write("f {0}//{3} {1}//{4} {2}//{5}\n".format(s0+param_a-1,s0,s1+param_a-1,param_a,1,param_a))  
#    skinfile.write("f {0}//{3} {1}//{4} {2}//{5}\n".format(s1+param_a-1,s0,s1          ,param_a,1,1))  

    thefile.write("f {0} {1} {2}\n".format(s1+param_a-1,s1,s0+param_a-1))  
    thefile.write("f {0} {1} {2}\n".format(s1,s1+param_a-1,s0+param_a-1))  
    thefile.write("f {0} {1} {2}\n".format(s0+param_a-1,s0,s1          ))  
    thefile.write("f {0} {1} {2}\n".format(s0,s0+param_a-1,s1          ))  

    skinfile.write("f {0}//{3} {1}//{4} {2}//{5}\n".format(s1+param_a-1,s1 ,s0+param_a-1,param_a,1,param_a))  
    skinfile.write("f {0}//{3} {1}//{4} {2}//{5}\n".format(s0+param_a-1,s1 ,s0          ,param_a,1,1))  

#--------------------------------
# material for bones
#--------------------------------

thefile.write("g bones\n")

thefile.write("#material for bones\n")
thefile.write("usemtl material_bones\n")
thefile.write("#bones faces\n")

verts = scan["bones"]["slice"]

xid = 0

for k in range(0,len(verts)):
    print("#slice {0}-{1} with {2} faces".format(k,k+1,2*param_a))
    thefile.write("#slice {0}-{1} with {2} faces\n".format(k,k+1,2*param_a))

    vset = verts[k]["vertices"]
    
    for vlist in vset:

        vcnt = len(vlist)
        
        s0 = bones_vid + xid  + 0
        s1 = bones_vid + xid  + vcnt
        
        nlist, ntlist, cap = gen_normals(vlist,circ_norm)

        for j in range(0,len(vlist)-1):
            thefile.write("f {0}//{3} {1}//{4} {2}//{5}\n".format(s0+j,s0+j+1,s1+j  ,1+nlist[j+0],1+nlist[j+1],1+nlist[j+0]))  
            thefile.write("f {0}//{3} {1}//{4} {2}//{5}\n".format(s1+j,s0+j+1,s1+j+1,1+nlist[j+0],1+nlist[j+1],1+nlist[j+1]))  

            #thefile.write("f {0}//{3} {1}//{4} {2}//{5}\n".format(s1+j,s1+j+1,s0+j  ,1+nlist[j+0],1+nlist[j+1],1+nlist[j+0]))  
            #thefile.write("f {0}//{3} {1}//{4} {2}//{5}\n".format(s0+j,s1+j+1,s0+j+1,1+ntlist[j+0],1+ntlist[j+1],1+ntlist[j+1]))  

#            thefile.write("f {0} {1} {2}\n".format(s0+j,s0+j+1,s1+j  ))  
#            thefile.write("f {0} {1} {2}\n".format(s1+j,s0+j+1,s1+j+1))  

#            thefile.write("f {0} {1} {2}\n".format(s1+j,s1+j+1,s0+j  ))  
#            thefile.write("f {0} {1} {2}\n".format(s0+j,s1+j+1,s0+j+1))  

        for tr in cap:
            thefile.write("f {0} {1} {2}\n".format(s0+tr[0],s0+tr[1],s0+tr[2]))  
            thefile.write("f {0} {1} {2}\n".format(s0+tr[0],s0+tr[2],s0+tr[1]))  
            thefile.write("f {0} {1} {2}\n".format(s1+tr[0],s1+tr[1],s1+tr[2]))  
            thefile.write("f {0} {1} {2}\n".format(s1+tr[0],s1+tr[2],s1+tr[1]))  

           
        xid += 2*vcnt

#--------------------------------

thefile.write("g vessels\n")

thefile.write("#material for vessels\n")
thefile.write("usemtl material_vessels\n")
thefile.write("#vessels faces\n")

verts = scan["vessels"]["slice"]

xid = 0

for k in range(0,len(verts)):
    print("#slice {0}-{1} with {2} faces".format(k,k+1,2*param_a))
    thefile.write("#slice {0}-{1} with {2} faces\n".format(k,k+1,2*param_a))

    vset = verts[k]["vertices"]
    
    for vlist in vset:

        vcnt = len(vlist)
        
        s0 = vessels_vid + xid  + 0
        s1 = vessels_vid + xid  + vcnt
        
        nlist, ntlist, cap = gen_normals(vlist,circ_norm)

        for j in range(0,len(vlist)-1):
#            thefile.write("f {0}//{3} {1}//{4} {2}//{5}\n".format(s0+j,s0+j+1,s1+j  ,j+1,j+2,j+1))  
#            thefile.write("f {0}//{3} {1}//{4} {2}//{5}\n".format(s1+j,s0+j+1,s1+j+1,j+1,j+2,j+2))  

            thefile.write("f {0} {1} {2}\n".format(s0+j,s0+j+1,s1+j  ))  
            thefile.write("f {0} {1} {2}\n".format(s1+j,s0+j+1,s1+j+1))  

            thefile.write("f {0} {1} {2}\n".format(s0+j,s1+j  ,s0+j+1  ))  
            thefile.write("f {0} {1} {2}\n".format(s1+j,s1+j+1,s0+j+1))  

        for tr in cap:
            thefile.write("f {0} {1} {2}\n".format(s0+tr[0],s0+tr[1],s0+tr[2]))  
            thefile.write("f {0} {1} {2}\n".format(s0+tr[0],s0+tr[2],s0+tr[1]))  
            thefile.write("f {0} {1} {2}\n".format(s1+tr[0],s1+tr[1],s1+tr[2]))  
            thefile.write("f {0} {1} {2}\n".format(s1+tr[0],s1+tr[2],s1+tr[1]))  


        xid += 2*vcnt

# thefile.write("g muscles\n")

# thefile.write("#material for vessels\n")
# thefile.write("usemtl material_muscles\n")
# thefile.write("#muscles faces\n")

# verts = scan["muscles"]["slice"]

# xid = 0

# for k in range(0,len(verts)):
    # print("#slice {0}-{1} with {2} faces".format(k,k+1,2*param_a))
    # thefile.write("#slice {0}-{1} with {2} faces\n".format(k,k+1,2*param_a))

    # vset = verts[k]["vertices"]
    
    # for vlist in vset:

        # vcnt = len(vlist)
        
        # s0 = muscles_vid + xid  + 0
        # s1 = muscles_vid + xid  + vcnt
        
        # for j in range(0,len(vlist)-1):
            # thefile.write("f {0}//{3} {1}//{4} {2}//{5}\n".format(s0+j,s0+j+1,s1+j  ,j+1,j+2,j+1))  
            # thefile.write("f {0}//{3} {1}//{4} {2}//{5}\n".format(s1+j,s0+j+1,s1+j+1,j+1,j+2,j+2))  

        # xid += 2*vcnt


thefile.close()
skinfile.close()

#mtl file

print("mtlfile writing")

mtlfile.write("newmtl material_0\n")
mtlfile.write("Ka 1.00 1.00 1.00\n")
mtlfile.write("Kd 0.55 0.55 0.55\n")
mtlfile.write("Ks 0.40 0.40 0.40\n")
mtlfile.write("Ns   10.00\n")

mtlfile.write("illum 0\n")
mtlfile.write("d     0.3\n")
mtlfile.write("Tr    0.7\n")

mtlfile.write("Tf    1.0\n")
mtlfile.write("Ni    1.50\n")

mtlfile.write("newmtl material_muscles\n")
mtlfile.write("Ka 1.000000 0.200000 1.000000\n")
mtlfile.write("Kd 1.000000 1.000000 1.000000\n")
mtlfile.write("Ks 0.900000 0.900000 0.900000\n")
mtlfile.write("illum 1\n")
mtlfile.write("Tr 0.0\n")
mtlfile.write("d  1.0\n")

mtlfile.write("newmtl material_bones\n")
mtlfile.write("Ka 0.800000 0.800000 0.800000\n")
mtlfile.write("Kd 0.752941 0.752941 0.752941\n")
mtlfile.write("Ks 0.900000 0.900000 0.900000\n")
mtlfile.write("illum 1\n")
mtlfile.write("Tr 0.0\n")
mtlfile.write("d  1.0\n")

mtlfile.write("newmtl material_vessels\n")
mtlfile.write("Ka 1.000000 0.000000 0.000000\n")
mtlfile.write("Kd 1.000000 0.000000 0.100000\n")
mtlfile.write("Ks 0.200000 0.200000 0.200000\n")
mtlfile.write("Ns   10.00\n")

mtlfile.write("illum 1\n")
mtlfile.write("Tr 0.0\n")
mtlfile.write("d  1.0\n")

mtlfile.close()

cmd = 'ctmconv ' + oDir + '/' + m3d_name + '.obj ' + oDir + '/' + m3d_name+'_x.obj' + ' --calc-normals'
print(cmd)
os.system(cmd)

cmd = 'ctmconv ' + oDir + '/' + m3d_name + '.obj ' + oDir + '/' + m3d_name+'.stl'
print(cmd)
os.system(cmd)

cmd = 'ctmconv ' + oDir + '/' + m3d_skin_name + '.obj ' + oDir + '/' + m3d_skin_name + '.stl'
print(cmd)
os.system(cmd)


#-----------------------------------------------------------------------------------------