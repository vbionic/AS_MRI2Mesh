import sys, getopt
#import pydicom
import numpy as np
#from PIL import Image
import json
#import vtkplotlib as vpl
import os
import pathlib
from argparse import ArgumentParser
import glob
#import pyvista as pv
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
#from scipy.spatial import Delaunay
from stl import mesh
import math
from time import gmtime, strftime
import copy
import open3d as o3d

def calcNormals(pairOfNormals):
    pairOfNormals=np.array(pairOfNormals)
    print(pairOfNormals)
    x=average(pairOfNormals[:,0])
    y=average(pairOfNormals[:,1])
    z=average(pairOfNormals[:,2])
    #mod=math.sqrt(pow(x,2)+pow(y,2)+pow(z,2))
    #x/=mod
    #y/=mod
    #z/=mod
    #print([x,y,z])
    return [x,y,z]
def triangulate_between_slices(slice1,slice2):
    
    data=np.zeros(len(slice1["points"])+len(slice2["points"]), dtype=mesh.Mesh.dtype)
    
    vectorsIter=0
    if point_dist_2d(slice1["points"][0],slice1["points"][1])<=point_dist_2d(slice2["points"][0],slice2["points"][1]):
       
        pairOfPointst=[slice1["points"][0],slice2["points"][0]]
        #pairOfNormals=[slice1["normals"][0],slice2["normals"][0]]
        for i in range(1,len(slice1["points"])):

            pairOfPointst.append(slice1["points"][i])
            #pairOfNormals.append(slice1["normals"][i])
            data['vectors'][vectorsIter]=np.array(pairOfPointst)
            vectorsIter+=1
            pairOfPointst.pop(0)
            #pairOfNormals.pop(0)
            pairOfPointst.append(slice2["points"][i])
            #pairOfNormals.append(slice2["normals"][i])
            data['vectors'][vectorsIter]=np.array([pairOfPointst[0],pairOfPointst[2],pairOfPointst[1]])
            vectorsIter+=1
            pairOfPointst.pop(0)
            #pairOfNormals.pop(0)
        pairOfPointst.append(slice1["points"][0])
        #pairOfNormals.append(slice1["normals"][0])
        data['vectors'][vectorsIter]=np.array(pairOfPointst)
        vectorsIter+=1
        pairOfPointst.pop(0)
        #pairOfNormals.pop(0)
        pairOfPointst.append(slice2["points"][0])
        #pairOfNormals.append(slice2["normals"][0])
        data['vectors'][vectorsIter]=np.array([pairOfPointst[0],pairOfPointst[2],pairOfPointst[1]])
        vectorsIter+=1
        pairOfPointst.pop(0)
        #pairOfNormals.pop(0)
    else:
        pairOfPointst=[slice2["points"][0],slice1["points"][0]]
        for i in range(1,len(slice1["points"])):

            pairOfPointst.append(slice2["points"][i])
            data['vectors'][vectorsIter]=np.array([pairOfPointst[0],pairOfPointst[2],pairOfPointst[1]])
            vectorsIter+=1
            pairOfPointst.pop(0)
            pairOfPointst.append(slice1["points"][i])
            data['vectors'][vectorsIter]=np.array(pairOfPointst)
            vectorsIter+=1
            pairOfPointst.pop(0)
        pairOfPointst.append(slice2["points"][0])
        data['vectors'][vectorsIter]=np.array([pairOfPointst[0],pairOfPointst[2],pairOfPointst[1]])
        vectorsIter+=1
        pairOfPointst.pop(0)
        pairOfPointst.append(slice1["points"][0])
        data['vectors'][vectorsIter]=np.array(pairOfPointst)
        vectorsIter+=1
        pairOfPointst.pop(0)
   
    return data

#def calculate_3D_triangles(slice1, slice2):
#    print("calculate_3D_triangles")
#    print("slice1 len:"+str(len(slice1)))
#    print("slice2 len:"+str(len(slice2)))
#    data=np.zeros(len(slice1)+len(slice2)-1,dtype=mesh.Mesh.dtype)
#    for i in range(len(slice1)):
#        if( i==0):
#            data['vectors'][i]=np.array([[0,0,0],#slice1[len(slice1)-1],
#                                        slice1[len(slice1)-1],
#                                        slice1[0]])
#        data['vectors'][i]=np.array([[0,0,0],#[slice1[i-1],
#                                   slice1[i-1],
#                                    slice1[i]])
#    for i in range(len(slice2)):
#        if( i==0):
#            data['vectors'][len(slice1)-1+i]=np.array([[0,0,0],#[slice2[len(slice2)-1],
#                                        slice2[len(slice2)-1],
#                                        slice2[0]])
#        data['vectors'][len(slice1)-1+i]=np.array([[0,0,0],#[slice2[i-1],
#                                   slice2[i-1],
#                                    slice2[i]])
#    print("/calculate_3D_triangles")
#    return data
def point_dist_3d(point1,point2):
    return math.sqrt(math.pow(point1[0]-point2[0],2)+math.pow(point1[1]-point2[1],2)+math.pow(point1[2]-point2[2],2))

def point_dist_2d(point1,point2):
    return math.sqrt(math.pow(point1[0]-point2[0],2)+math.pow(point1[1]-point2[1],2))

def middlePoint(point1,point2):
    return [(point1[0]+point2[0])/2,(point1[1]+point2[1])/2,point1[2]]
def normalize_num_of_points(pointList,NumOfPoints):
    
    listOfDist=[]
    
    #print(len(pointList))
    if(len(pointList)>NumOfPoints):
        return pointList
    i=0
    maxLen=1
    lastMaxLen=math.sqrt(2)
    while len(pointList)!=NumOfPoints:
        if(len(pointList)-1==i):
            if(point_dist_2d(pointList[i],pointList[0])>maxLen):
                pointList.insert(i+1,middlePoint(pointList[i],pointList[0]))
                if(len(pointList)==NumOfPoints):
                    
                    return pointList
            i=0
            tmp=maxLen
            maxLen=lastMaxLen/2
            lastMaxLen=tmp
        if(point_dist_2d(pointList[i],pointList[i+1])>maxLen):
            pointList.insert(i+1,middlePoint(pointList[i],pointList[i+1]))
        i+=1
   
    return pointList



def find_nearest_points(slice1,slice2):
    
    
    
    
    firstDist = point_dist_2d(slice1[0],slice2[0])
    rightSecDist = point_dist_2d(slice1[0],slice2[1])
    leftSecDist = point_dist_2d(slice1[0],slice2[len(slice2)-1])
    if(point_dist_2d(slice1[0],slice2[0])>point_dist_2d(slice1[0],slice2[1]))or(point_dist_2d(slice1[0],slice2[0])>point_dist_2d(slice1[0],slice2[len(slice2)-1])):
        
        if(rightSecDist<leftSecDist):
            for j in range(1,len(slice2)):
                first_dist=point_dist_2d(slice1[0],slice2[j])
                sec_dist=point_dist_2d(slice1[0],slice2[j+1])
               
               
                if(point_dist_2d(slice1[0],slice2[j])<point_dist_2d(slice1[0],slice2[j+1])):
                    return j
        else:
            
            for j in range(len(slice2)-1,-1,-1):
                first_dist=point_dist_2d(slice1[0],slice2[j])
                sec_dist=point_dist_2d(slice1[0],slice2[j-1])
                if(point_dist_2d(slice1[0],slice2[j])<point_dist_2d(slice1[0],slice2[j-1])):
                    return j
        
    else:
        
        return 0

### {plot Func
def plot_slice_normals(points,normals):
    points=np.array(points)
    normals=np.array(normals)

    centerOfSlice=[average(points[:,0]),average(points[:,1])]
    X=points[:,0]
    Y=points[:,1]
    U=normals[:,0]-points[:,0]
    V=normals[:,1]-points[:,1]
    fig1, ax1 = plt.subplots()
    ax1.set_title('Normals of the slice and a middle point')
    ax1.plot(points[:,0],points[:,1],'.' )
    ax1.plot(centerOfSlice[0],centerOfSlice[1],"o")
    Q = ax1.quiver(X, Y, U, V)
    plt.show()

def plot_cloud_of_points(x,y,z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)

    plt.show()
def plot_mesh(mesh):
    figure = plt.figure()
    ax = mplot3d.Axes3D(figure)
    ax.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh.vectors))
    ax.set_xlim(mesh.min_[0],mesh.max_[0])
    ax.set_ylim(mesh.min_[1],mesh.max_[1])
    ax.set_zlim(mesh.min_[2],mesh.max_[2])
    plt.show()

#####}
def mesh_to_png(mesh,fileName):
    print("mesh_to_png")
    figure = plt.figure()
    ax = mplot3d.Axes3D(figure)
    ax.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh.vectors))
    ax.set_xlim(mesh.min_[0],mesh.max_[0])
    ax.set_ylim(mesh.min_[1],mesh.max_[1])
    ax.set_zlim(mesh.min_[2],mesh.max_[2])
    plt.savefig(fileName)
    print("//mesh_to_png")

def create_point_normals(point, centerOfSice):
    x=point[0]-centerOfSice[0]
    y=point[1]-centerOfSice[1]
    x/=math.sqrt(math.pow(x,2)+math.pow(y,2))
    y/=math.sqrt(math.pow(x,2)+math.pow(y,2))
    x+=point[0]
    y+=point[1]
    return [x,y,point[2]]

def average(lst):
    #print(lst)
    return sum(lst) / len(lst)
def save_to_ply(slices,fileName):
    print("save_to_ply")
    xyz=slices[0]["points"]
    for i in range(1,len(slices)):
        xyz=np.concatenate((xyz,slices[i]["points"]),axis=0)
    #print(xyz)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.io.write_point_cloud( fileName, pcd)
    print("/save_to_ply")
def pixel_to_mm(slice,pixSpacing):
    #print(slice)
    #print(pixSpacing[0])
    #print(pixSpacing[1])
    for i in range(len(slice)):
        #print("slice point="+str(slice[i][0]))
        #print("pix spacing="+str(float(pixSpacing[0])*slice[i][0]))
        slice[i][0]=float(pixSpacing[0])*slice[i][0]
        slice[i][1]*=float(pixSpacing[1])
        #print("slice point after="+str(slice[i][0]))
    #print("slice")
    #print(slice)
    return slice

def max_num_of_points(slices):
    maxLen = 512 #len(slices["points"][0])
    for j in range(1,len(slices)):
        print(len(slices[j]["points"]))
        if (len(slices[j]["points"]) > maxLen):
            maxLen=len(slices[j]["points"])
    return maxLen
################
##process_file
################
def process_file(inputFile,numOfIteration,spacing,pixelSpacing):
    
    with open (inputFile) as f:
        data= json.load(f)
    
    slice={
        "points":[],
        "normals":[]

        }

    slice["points"]=pixel_to_mm(data["contours"]['00']['path'],pixelSpacing)
    #print(slice["points"])
    print("#################")
    #slice["points"]=normalize_num_of_points(slice["points"],1024)
    for i in range(len(slice["points"])):
        slice["points"][i].append(spacing*numOfIteration)
    
    points=np.array(slice['points'])

    #print(points)
    centreOfSlice=[average([points[:,0]]),average(points[:,1])]

    #for i in range(len(slice["points"])):
    #    slice["normals"].append(create_point_normals(slice["points"][i],centreOfSlice))
    return slice

def normalize_start_of_list(slices):

    
    for i in range(len(slices)-1):
        numOfShifts=find_nearest_points(slices[i]["points"],slices[i+1]["points"])
        for j in range(numOfShifts):
            slices[i+1]["points"].append(slices[i+1]["points"].pop(0))
            #slices[i+1]["normals"].append(slices[i+1]["normals"].pop(0))
    
    return slices
#############
##proces_dir
#############
def process_dir(dirPoints,dirSpacing, outputdir,filesName, verbose):
    print("process_dir")
    
    STLname = outputdir+"/"+filesName+".stl"

    inputPointsFiles = glob.glob(dirPoints + '/*contour.json')
    inputPointsFiles.sort();    
    
    inputSpacingFiles = glob.glob(dirSpacing + '/set_data.json')
    print(inputSpacingFiles)
    file_list = []
    for filename in inputPointsFiles:
        file_list.append(os.path.basename(filename))
    file_list.sort()
    with open(inputSpacingFiles[0]) as f:
        data=json.load(f)
    
    try:
        spacingBetweenSlices = data["distance_between_slices"]
    except KeyError:
        print("spacing not found")
        print("unknown distance between slices")
        sys.exit(1)
    try:
        pixelSpacing = [data["pixel_spacing_x"],data["pixel_spacing_y"]]
    except KeyError:
        print("spacing not found")
        sys.exit(1)
    print("spacingBetweenSlices = " + str(spacingBetweenSlices)+"pixel x spacing = "+str(pixelSpacing[0])+" pixel y spacing = "+str(pixelSpacing[1]))
    
    slices=[]
    i=0
    print("process_file")
    print(inputPointsFiles[0])
    for file in inputPointsFiles:      #print part of list -> [0:5] from elemet 0 to 4
        slices.append(process_file(file,i,spacingBetweenSlices,pixelSpacing))
        i+=1
    print("/process_file")
    #print(len(slices[1]))
    maxNumOfPoints = max_num_of_points(slices)
    print("max num_ of_points = "+str(maxNumOfPoints))
    for i in range(len(slices)):
        slices[i]["points"]=normalize_num_of_points(slices[i]["points"],maxNumOfPoints)
    slices=normalize_start_of_list(slices)


    save_to_ply(slices,outputdir+"/"+filesName+".ply")

    print("triangulate_between_slices")
    xxx=triangulate_between_slices(slices[0],slices[1])
    for i in range(2,len(slices)):
       xxx=np.concatenate((xxx,triangulate_between_slices(slices[i-1],slices[i])),axis=0)
    print("/triangulate_between_slices")
    print("mesh saving")
    mesho=mesh.Mesh(xxx.copy())
    mesho.save(STLname)
    print("/mesh saving")
    mesh_to_png(mesho,outputdir+"/"+filesName+".png")
    print("process_dir")
    #os.chmod(STLname, 644)


#-----------------------------------------------------------------------------------------
#MAIN
#-----------------------------------------------------------------------------------------
if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument("-shDir",  "--sh_dir",     dest="sh_dir",  help="input json directory",          metavar="PATH",required=True)
    parser.add_argument("-dpDir",  "--dp_dir",     dest="dp_dir",  help="input spacing directory",       metavar="PATH",required=True)
    parser.add_argument("-osDir",  "--os_dir",     dest="os_dir",  help="output stl directory",          metavar="PATH",required=True)
    parser.add_argument("-v",      "--verbose",    dest="verbose", help="verbose level",                              required=False)
    parser.add_argument("-fn",      "--file_name",    dest="file_name", help="output files name",                         required=False)
    
    args = parser.parse_args()
    
    verbose = 'off'                 if args.verbose is None else args.verbose
   
    idDirPoints     = args.sh_dir
    idDirSpacing    = args.dp_dir
    orDir  	= args.os_dir
    filesName =    "3DModel"+strftime("%d%b%Y%H%M", gmtime())                 if args.file_name is None else args.file_name+"_surface"
    
    idDirPoints = os.path.normpath(idDirPoints)
    idDirSpacing = os.path.normpath(idDirSpacing)
    orDir = os.path.normpath(orDir)
    
    if not os.path.isdir(idDirPoints):
        print('Error : Input directory (%s) with DICOM files not found !',idDirPoints)
        exit(-1)
    if not os.path.isdir(idDirSpacing):
        print('Error : Input directory (%s) with DICOM files not found !',idDirSpacing)
        exit(-1)
    try:
    	if not os.path.isdir(orDir):
    	    pathlib.Path(orDir).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
    	print("Output dir IO error: {}".format(err))
    	sys.exit(1)
    
    print("Dir with points input: "+idDirPoints)
    print("Dir with spacing input: "+idDirSpacing)
    print("Dir with output: "+orDir)
    
    process_dir(idDirPoints, idDirSpacing,orDir,filesName, verbose)
    
    
#-----------------------------------------------------------------------------------------
