#pliki z maskamiRGB w png:
#00000010_..._bones_-05_250135_algV<id>_labels.png
#00000010_..._bones_-05_250135_unetV<id>_labels.png
#00000037_..._skin_unetV<id>_labels.png
#00000037_..._skin_algV<id>_labels.png
#
#pliki z poligonami w json:
#00000010_..._bones_-05_250135_algV<id>_polygons.json
#00000010_..._bones_-05_250135_unetV<id>_polygons.json
#00000037_..._skin_unetV<id>_polygons.json
#00000037_..._skin_algV<id>_polygons.json


#                    img_Image = tissue_polygons_out.as_image(fill = True, w=out_w,h=out_h, force_labelRGB = True)
#                    img_Image.save(fn)

#                    org_point = [box[0], box[1]]
#                    tissue_polygons.move2point(org_point)
#                tissue_polygons_dict = tissue_polygons_out_sh.as_dict()
#                jsonDumpSafe(tissue_json_fn, tissue_polygons_dict)





import sys, getopt
import os
import pathlib
#import pydicom
sys.path.append(os.getcwd())

import numpy as np
import json
import matplotlib as plt
import cv2
import glob
import tracemalloc
import multiprocessing
#import imutils
#import seaborn as sns
from scipy import ndimage, misc
import math
import imageio
#from scipy.misc import imshow
#Sfrom plyfile import PlyData, PlyElement
#-----------------------------------------------------------------------------------------
sys.path.append(os.getcwd())
#-----------------------------------------------------------------------------------------
from v_utils.v_contour  import *
from v_utils.v_polygons import *
from v_utils.v_json import jsonUpdate, jsonDumpSafe
#-----------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib as mpl
#-----------------------------------------------------------------------------------------
from PIL import Image
#from pydicom.tag import Tag
#from pydicom.datadict import keyword_for_tag
from argparse import ArgumentParser
#-----------------------------------------------------------------------------------------
#def show_numpy_image(image):
def point_dist_2d(point1,point2):
    return math.sqrt(math.pow(point1[0]-point2[0],2)+math.pow(point1[1]-point2[1],2))
def plt_vectors_polar(vectors):
    ax = plt.subplot(projection='polar')
    print_list(vectors["angle"])
    plt.polar(np.deg2rad(vectors["angle"]), vectors["len"], 'k.', zorder=3)
    #plt.polar(angles, values)
    ax.grid(True)
    plt.show()
def plt_contour_points(contour,title,massCenter=[0,0]):
    #centerOfSlice=[average(points[:,0]),average(points[:,1])]
    #X=points[:,0]
    #Y=points[:,1]
    #U=normals[:,0]-points[:,0]
    #V=normals[:,1]-points[:,1]
    fig1, ax1 = plt.subplots()
    ax1.set_title(title)
    ax1.plot(contour[:,:,0],-contour[:,:,1],'.' )
    ax1.plot(massCenter[0], -massCenter[1], '.')
    ax1.grid(True)
    #ax1.plot(centerOfSlice[0],centerOfSlice[1],"o")
    #Q = ax1.quiver(X, Y, U, V)
    plt.show()
def average(lst):
    #print(lst)
    return sum(lst) / len(lst)
def plt_slice_normals(contour):


    centerOfSlice=[average(contour[:,:,0]),average(-contour[:,:,1])]
    X=centerOfSlice[0]
    Y=centerOfSlice[1]
    U=contour[:,:100,0]
    V=contour[:,:100,1]
    fig1, ax1 = plt.subplots()
    ax1.set_title('Normals of the slice and a middle point')
    #ax1.plot(contour[:,:,0],-contour[:,:,1],'.' )
    ax1.plot(centerOfSlice[0],centerOfSlice[1],"o")
    Q = ax1.quiver(X, Y, U, V)
    plt.show()
def print_list(list):
    print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
    for i in range(len(list)):
        print(str(i)+"# "+str(list[i]))
    print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")

def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled
#def remove_middle(image, contour):
def vect_angle(point,massCenter):
    #print("point[0] = "+str(point[0])+" point[1] = "+str(point[1]))
    #print("point[0] = "+str(point[0]-massCenter[0])+" point[1] = "+str(massCenter[1]-point[1])+" massCenter= "+ str(massCenter))
    #point[0]-=massCenter[0]
    #point[1]-=massCenter[1]
    #print("point = "+str(point))
    angle= math.atan2(abs(massCenter[1]-point[1]),abs(point[0]-massCenter[0]))
    if point[0]-massCenter[0]<0:
        if massCenter[1]-point[1]<0:
            angle+=math.pi
        else:
            angle = math.pi -angle
            
    else :
        if massCenter[1]-point[1]<0:
            angle = 2*math.pi - angle
    #print("angle = "+str(180/math.pi*angle))
    return math.degrees(angle)
            

def compute_vect(contour,massCenter):
    vect =[]
    tempDict={"len":[],
    "angle":[]}
    #print("contour[0]")
    #print(contour[0])
    #massCenter[0],massCenter[1]=massCenter[1],massCenter[0]
    for i in range(len(contour)):
        tempDict["len"].append(point_dist_2d(contour[i][0],massCenter))
        tempDict["angle"].append(vect_angle(contour[i][0],massCenter))
        
        
    return tempDict

def plot_contour(contour):
    fig1, ax1 = plt.subplots()
    #print(contour)
    for i in contour:
        #print("comtour i"+str(i))
        contourToPlt=np.concatenate(i)
        #print(contourToPlt)
        ax1.plot(contourToPlt[:,0],-contourToPlt[:,1],'.' )
    plt.show()
def find_vector_element_pos(vector ,element):
    array=np.array(vector["angle"])
    pos=np.where(array==element)[0]
    print(len(pos))
    if len(pos)>1:
        print("pos = "+str(pos))
        truePos=pos[0]
        for i in range(1,len(pos)):
            if vector["len"][pos[i]]<vector["len"][truePos]:
                truePos=pos[i]
        pos=[truePos]
    pos=pos[0]
    #if min:
    #    while True:
    return pos
def find_max_min_angle(angles):
    angles.sort()
    #print("angles")
    #print_list(angles)
    angleDist=[]
    for i in range(len(angles)-1):
        #print(i)
        angle=abs(angles[i+1]-angles[i])
        if angle<180:
            angleDist.append(angle)
        else:
            angleDist.append(360-angles[i+1]+angles[i])
    angle=angles[len(angles)-1]-angles[0]
    if angle<180:
        angleDist.append(angle)
    else:
        angleDist.append(360-angles[len(angles)-1]+angles[0])
    #angleDist.append(angles[len(angles)-1]-angles[0])
#    print("//////////////////////////")
#    print("len(angleDist)"+str(len(angleDist)))
#    print("len(angles)"+str(len(angles)))
#    print("angle[minPos] = " + str(angles[angleDist.index(max(angleDist))]))
    maxAngle=angles[angleDist.index(max(angleDist))]
    print("maxAngle = " + str(maxAngle))

    

    if angleDist.index(max(angleDist))==(len(angles)-1):
        print("angle[maxPos] = " +str(angles[0]))
        minAngle=angles[0]
    else:
        print("angle[maxPos] = " +str(angles[angleDist.index(max(angleDist))+1]))
        minAngle=angles[angleDist.index(max(angleDist))+1]
    print("fn max angle dist: "+str(max(angleDist)))
    print("//////////////////////////")
    #elementToPop=[]
    #while True:
    #    if(minAngle in holes ):
    #        elementToPop=[minAngle,maxAngle]
    #    elif [maxAngle,minAngle] in holes:
    #        elementToPop=[maxAngle,minAngle]
    #    else:
    #        break
    #    print("holes.index(elementToPop) = "+str(holes.index(elementToPop)))
    #    holes.pop(holes.index(elementToPop))
    #    angleDist[angleDist.index(max(angleDist))]=0
    #    maxAngle=angles[angleDist.index(max(angleDist))]
    #    if angleDist.index(max(angleDist))==(len(angles)-1):
    #        print("angle[maxPos] = " +str(angles[0]))
    #        minAngle=angles[0]
    #    else:
    #        print("angle[maxPos] = " +str(angles[angleDist.index(max(angleDist))+1]))
    #        minAngle=angles[angleDist.index(max(angleDist))+1]

    return minAngle,maxAngle
def add_line(contour,contourPos,startPoint,endPoint):

    print("------------add line contour list-------------------")
    endPointPos=contourPos+1
    ##print_list(contour[:endPointPos])

    #print("contour[endPointPos] = "+str(contour[endPointPos]))
    #print("endPointPos = "+str(endPointPos))
    
    #print("startPoint = "+ str(startPoint))
    #print("endPoint = "+ str(endPoint))
    if(startPoint[0]<endPoint[0]):
        print("addline 0")
        a=(startPoint[1]-endPoint[1])/(startPoint[0]-endPoint[0])
        b=startPoint[1]-(a*startPoint[0])
        #print("a = "+str(a))
        #print("b = "+str(b))
        #print("[[startPoint+i, a*startPoint+b]] = " +str([[startPoint[0], a*startPoint[0]+b]]))
        for i in range(int(startPoint[0]),int(endPoint[0])):
            
            contour=np.concatenate((contour[:endPointPos],[[[startPoint[0]+i-startPoint[0], a*(startPoint[0]+i-startPoint[0])+b]]],contour[endPointPos:]))
            endPointPos+=1
            #print("dupa")
    elif (startPoint[0]>endPoint[0]):
        #print("addline 1")
        a=(startPoint[1]-endPoint[1])/(startPoint[0]-endPoint[0])
        b=startPoint[1]-(a*startPoint[0])
        #print("a = "+str(a))
        #print("b = "+str(b))
        #print("[[startPoint+i, a*startPoint+b]] = " +str([[startPoint[0], a*startPoint[0]+b]]))
        for i in range(int(startPoint[0]-1),int(endPoint[0]+1),-1):
            #print("startPoint[0]-(i-startPoint[0])"+str(startPoint[0]+(i-startPoint[0])))
            contour=np.concatenate((contour[:endPointPos],[[[startPoint[0]+(i-startPoint[0]), a*(startPoint[0]+i-startPoint[0])+b]]],contour[endPointPos:]))
            #print("dupa")
            endPointPos+=1
    else:
        if(startPoint[1]<endPoint[1]):
            for i in range(int(startPoint[1])+1,int(endPoint[1])):
                #print("ADD -> [[startPoint[0], i]] = "+ str([[startPoint[0], i]]))
                #print("i = "+str(i))
                #print("startPoint[1] = "+ str(startPoint[1]) )
                #print("endPoint[1] = "+ str(endPoint[1]) )
                #print("if(startPoint[1]<endPoint[1]):")
                contour=np.concatenate((contour[:endPointPos],[[[startPoint[0], i]]],contour[endPointPos:]))
                
                #plt_contour_points(contour,"add   Line")
        else:
            for i in range(int(endPoint[1])+1,int(startPoint[1])):
                contour=np.concatenate((contour[:endPointPos],[[[startPoint[0], i]]],contour[endPointPos:]))
                #print("ADD -> [[startPoint[0], i]] = "+ str([[startPoint[0], i]]))
                #print("i = "+str(i))
                #print("startPoint[1] = "+ str(startPoint[1]) )
                #print("endPoint[1] = "+ str(endPoint[1]) )
                #print("if(startPoint[1]>endPoint[1]):")
                #plt_contour_points(contour,"add   Line")
        ##print_list(contour)
    return contour

def remove_holes(contour):
    holes=[]
    print("holes() before contour len = "+ str(len(contour)))
    i =0
    while i!=len(contour)-1:
        print("i = " +str(i))
        if point_dist_2d(contour[i][0],contour[i+1][0])>math.sqrt(2):
            print("contour[i][0] = "+str(contour[i][0]))
            print("contour[i+1][0] = "+str(contour[i+1][0]))
            contour = add_line(contour,i,contour[i][0],contour[i+1][0])
        i+=1
            #cv2.waitKey(0)
    #print("HOLES LIST")
    ##print_list(holes)
    print("holes() after contour len = "+ str(len(contour)))
    return contour
def make_line_from_contour(skinContour,contour,massCenter):
    #print("skin contour"+str(skinContour))
    print("contour to proces")
    ##print_list(contour)
    print("++++++++++MAKE LINE++++++++++++++")
    print("massCenter = "+ str(massCenter))
    #plt_contour_points(contour,"before make_line_from_contour",massCenter=massCenter)
    #vectors=compute_vect(contour,massCenter)
    
    #angles= vectors["angle"].copy()

    
    #plt_contour_points(contour,"before remove holes",massCenter=massCenter)
    contour=remove_holes(contour)
    #plt_contour_points(contour,"after remove holes",massCenter=massCenter)
    vectors=compute_vect(contour,massCenter)
    print("massCenter = "+ str(massCenter))
    #plt_vectors_polar(vectors)
    #plt_slice_normals(contour)
    
    angles= vectors["angle"].copy()
    print(" angles len = " + str(len(angles)))
    print(" contour len = " + str(len(contour)))
    minAngle,maxAngle=find_max_min_angle(angles)
    minPos=vectors["angle"].index(minAngle) #find_vector_element_pos(vectors,min(vectors["angle"]))
    maxPos=vectors["angle"].index(maxAngle)#find_vector_element_pos(vectors,max(vectors["angle"]))
    print("len(contour)-1 = "+str(len(contour)-1))
    print("minAngle = "+str(minAngle))
    print("maxAngle = "+str(maxAngle))
    print("min pos = "+str(minPos)+" len = "+ str(vectors["len"][minPos])+ " angle : " +str(vectors["angle"][minPos]))
    print("max pos = "+str(maxPos)+" len = "+ str(vectors["len"][maxPos])+ " angle : " +str(vectors["angle"][maxPos]))
    print("dif max angle dist:"+str(vectors["angle"][maxPos]-vectors["angle"][minPos]))
    #plot_contour(contour)
    print("len of contour before split: "+ str(len(contour)))
    print("contour[minPos] = "+str(contour[minPos]))
    print("contour[maxPos] = "+str(contour[maxPos]))
    tempContour=[]
    tempVectors={"len":[],"angle":[]}
    vectors["len"]=np.array(vectors["len"])
    vectors["angle"]=np.array(vectors["angle"])
    #print(contour)
#    print("len(vectors[angles])"+str(len(vectors["angle"])))
#    print("##### swap #####")
#    #print([contour[minPos:len(contour)-1]])
#   
#    print("vectors[angle][minPos] = "+str(vectors["angle"][minPos]))
#    print("vectors[angle][maxPos] = "+str(vectors["angle"][maxPos]))
#
#    print("vectors[angle][int((minPos+maxPos)/2)] = "+str(vectors["angle"][int((minPos+maxPos)/2)]))
    if vectors["angle"][minPos] > vectors["angle"][maxPos]:
        print("minPos > maxPos")
        if vectors["angle"][int((minPos+maxPos)/2)]>vectors["angle"][maxPos] and vectors["angle"][int((minPos+maxPos)/2)]<vectors["angle"][minPos]:
#            print("swap()")
            minPos,maxPos=maxPos,minPos
    else:
        print("minPos < maxPos")
        if vectors["angle"][int((minPos+maxPos)/2)]>vectors["angle"][maxPos] or vectors["angle"][int((minPos+maxPos)/2)]<vectors["angle"][minPos]:
#            print("swap()")
            minPos,maxPos=maxPos,minPos
#    print("##### /swap #####")
    #    else:
    #        if vectors["angle"][mint((minPos+maxPos)/2)]<vectors["angle"][minPos]:
    #            minPos,maxPos=maxPos,minPos
    #else:
    #    if(minPos > maxPos):
    #        if vectors["angle"][maxPos+int((minPos+maxPos)/2)]<vectors["angle"][maxPos]:
    #            minPos,maxPos=maxPos,minPos
    #    else:
    #        if vectors["angle"][minPos+int((minPos+maxPos)/2)]<vectors["angle"][maxPos]:
    #            minPos,maxPos=maxPos,minPos
    angleMin=vectors["angle"][minPos]
    angleMax =vectors["angle"][maxPos]
    if (minPos==0 and maxPos==len(contour)-1)or(maxPos==0 and minPos==len(contour)-1):
        print("!!!!!!!!!!!!!!!!!!!!!")
        return contour,[0,360, 360]
    if abs(minPos-maxPos) <=1 and len(contour)>5:
        return contour,[0,360,360]

    if minPos > maxPos:
       
        print("minPos > maxPos")

        tempContour=contour[maxPos:minPos+1]
        contour=contour[maxPos:minPos+1]
        tempVectors["len"]=vectors["len"][maxPos:minPos+1]
        tempVectors["angle"]=vectors["angle"][maxPos:minPos+1]
        vectors["len"]=vectors["len"][maxPos:minPos+1]
        vectors["angle"]=vectors["angle"][maxPos:minPos+1]
        if(average(tempVectors["len"])>(average(vectors["len"]))):
            tempVectors,vectors=vectors,tempVectors
            tempContour,contour=contour,tempContour
        #contour=np.concatenate((contour[minPos:len(contour)-1],contour[0:maxPos]),axis=0)
        #vectors["len"]=np.concatenate(( vectors["len"][minPos:len( vectors["len"])-1], vectors["len"][0:maxPos]),axis=0)
        #vectors["angle"]=np.concatenate(( vectors["angle"][minPos:len( vectors["angle"])-1], vectors["angle"][0:maxPos]),axis=0)
    else:
        print("#####minPos < maxPos#####")
        tempContour=contour[minPos:maxPos+1]
        tempVectors["len"]=vectors["len"][minPos:maxPos+1]
        tempVectors["angle"]=vectors["angle"][minPos:maxPos+1]
        contour=np.concatenate((contour[maxPos:len(contour)-1],contour[0:minPos]),axis=0)
        vectors["len"]=np.concatenate((vectors["len"][maxPos:len(vectors["len"])-1],vectors["len"][0:minPos]),axis=0)
        print("git")
        vectors["angle"]=np.concatenate((vectors["angle"][maxPos:len(vectors["angle"])-1],vectors["angle"][0:minPos]),axis=0)
        index = 0
        print("tempVectors[len][int(len(tempVectors[len])/2)] = "+ str(tempVectors["len"][int(len(tempVectors["len"])/2)]))
        print("vectors[len][int(len(vectors[len])/2)] = "+str(vectors["len"][int(len(vectors["len"])/2)]))
        if(len(vectors["len"])>len(tempVectors["len"])):
            index=int(len(tempVectors["len"])/2)
        else:
            index=int(len(vectors["len"])/2)
        if(average(tempVectors["len"])>(average(vectors["len"]))):
            tempVectors,vectors=vectors,tempVectors
            tempContour,contour=contour,tempContour
        #contour=np.delete(contour,[i for i in range(minPos,maxPos)],0)
    #print("contour")
    print("tempContour[0] = "+ str(tempContour[0])+" len = "+ str(tempVectors["len"][0])+ " angle : " +str(tempVectors["angle"][0]))
    print("tempContour[len(tempContour)-1]] = "+ str(tempContour[len(tempContour)-1])+" len = "+ str(tempVectors["len"][len(tempVectors["len"])-1])+ " angle : " +str(tempVectors["angle"][len(tempVectors["angle"])-1]))
    #debug com
    print("len(vectors[ślen]): "+str(len(vectors["len"])))
    print("contour[0] = "+ str(contour[0])+" len = "+ str(vectors["len"][0])+ " angle : " +str(vectors["angle"][0]))
    print("contour[len(contour)-1])] = "+ str(contour[len(contour)-1])+" len = "+ str(vectors["len"][len(vectors["len"])-1])+ " angle : " +str(vectors["angle"][len(vectors["angle"])-1]))
    print("len of sum contours after split: "+ str(len(contour)+len(tempContour)))
    #/debug com
    #plt_contour_points(tempContour,"after a make_line_from_contour",massCenter=massCenter)
    #plot_contour([tempContour,contour])
    if(angleMin>angleMax):
        angleDiff=(360-angleMin)+angleMax
    else:
        angleDiff=angleMax-angleMin
    return tempContour,[angleMin,angleMax, angleDiff]


    #print(contour)
    #print(vectors)
    #for i in range(len(vectors["len"])):
    #    print("len: "+ str(vectors["len"][i]),"angle: "+ str(vectors["angle"][i]))
    
def remove_overlapping_contours(angleList,contours):
    angleList.sort(key=lambda x:x[2])
    print("!!!!!!contours!!!!!")
    print(contours)
    
    print("!!!!!!angleList!!!!!")
    print(" angleList len = " + str(len(angleList)))
    print_list(angleList)
    i=0
    for i in range(len(angleList)):
        iterator =len(angleList)-1-i
        if(iterator<=0):
            break
        print("iterator = "+ str(iterator)+" angleList[iterator]"+str(angleList[iterator]) )
        print_list(angleList)
        if(angleList[iterator][0]>angleList[iterator][1]):
            for j in range(iterator-1,-1,-1):
                print("j = "+ str(j))
                if angleList[j][0]>angleList[j][1]:
                    if(angleList[iterator][0]>angleList[j][0])or(angleList[iterator][1]<angleList[j][1]):
                        angleList.pop(j)
                        iterator-=1
                        print("iterator = "+ str(iterator)+" angleList[iterator]"+str(angleList[iterator]) )
                        #contour=np.concatenate((contour,np.flip(line[angleList[iterator][3]],0)),axis=0)
                        print("0pop("+str(j)+")")
                else:
                    if ((angleList[iterator][0]<angleList[j][0])or(angleList[iterator][1]>angleList[j][1])):
                        angleList.pop(j)
                        iterator-=1
                        print("iterator = "+ str(iterator)+" angleList[iterator]"+str(angleList[iterator]) )
                        print("1pop("+str(j)+")")
                print_list(angleList)
        else:
            for j in range(iterator-1,-1,-1):
                print("j = "+ str(j))
                if angleList[j][0]<angleList[j][1]:
                    if(angleList[iterator][0]<angleList[j][0])and(angleList[iterator][1]>angleList[j][1]):
                        angleList.pop(j)
                        iterator-=1
                        print("iterator = "+ str(iterator)+" angleList[iterator]"+str(angleList[iterator]) )
                        print("2pop("+str(j)+")")
                #
                print_list(angleList)

    return angleList



def create_contour(skinContour,contours,image,massCenter):
    #print("skinContour")
    #print(skinContour)
    line=[]
    #print("contours")
    #print(contours)
    contour=[]
    angleList=[]
    numberOfContour=0
    print("create_contour len(contours) = "+str(len(contours)))
    
    for i in range(len(contours)):
        if(len(contours[i])<4):
            print("<4")
            continue
        _line,angles=make_line_from_contour(skinContour,contours[i],massCenter)
        
        line.append(_line)
        
        angles.append(numberOfContour)
        #angles.sort()
        angleList.append(angles)
        print(angles)
        numberOfContour+=1
        #print("line[i][0][0] = "+str(line[i][0][0]))
        #print("line[i][len(line)-1][0] = "+str(line[i][len(line)-1][0]))
        #angleList.append([vect_angle(line[i][0][0],massCenter),vect_angle(line[i][len(line)-1][0],massCenter)])
    contourIsCircle=False
    x =0
    i=0
    print_list(line)
    print("len(line) = "+str(len(line)))
    if(len(line)==1):
        return [np.array(line[0],dtype='int32')]
    elementToRemove = -1
    print_list("angleList before whir")
    while(i<len(angleList)):             #for i in range(len(line)):
        print(i)
        if(angleList[i][0]==0 and angleList[i][1]==360):
            print_list(angleList)
            print("i =  "+str(i))
            
            #print_list(line[i])
            x+=1
            
            

            if x>1:
                contour=line[angleList[i][3]]
                #plt_contour_points(contour,"contour is circle",massCenter=massCenter)
                #print_list(contour)
                print("!!!!!!!!RETURN CONTOUR!!!!!!!!!!")
                contour=[np.array(contour,dtype='int32')]
                #plt_contour_points(contour,"contour is circle",massCenter=massCenter)
                
                return contour

            else:
                elementToRemove=i
        i+=1
    if(elementToRemove!=-1):
        angleList.pop(elementToRemove)
    print("->contours")
    #print_list(contours)
    print("-> line")
    #print_list(line)
    print("anglelist")   
    print(angleList)
    angleList=remove_overlapping_contours(angleList,contour);   
    print("@!!!@!@!@!@!@!@@!! sort")
    print("angleList:")
    print_list(angleList)
    angleList.sort(key=lambda x:x[0])
    print("angleList:")
    print_list(angleList)
    print("@!!!@!@!@!@!@!@@!! sort")
    #line[angleList[0][2]].reversed()
    
    contour=np.flip(line[angleList[0][3]],0)
    print("range(1,len(angleList)) = "+ str(range(1,len(angleList))))
    for i in range(1,len(angleList)):
        print("angleList[i][3]"+str(angleList[i][3]))
        contour=np.concatenate((contour,np.flip(line[angleList[i][3]],0)),axis=0)
    print("contour : ")    
    print_list(contour)
        ##line[angleList[i][2]].reversed()
        #print("|>>>>>>>>>>>>>>>>>>>>>>>>>|")
        #print("angleList[i-1][0] = "+str(angleList[i-1][0]))
        #print("angleList[i-1][1] = "+str(angleList[i-1][1]))
        #print("angleList[i][0] = "+str(angleList[i][0]))
        #print("angleList[i][1] = "+str(angleList[i][1]))
#
        #if angleList[i][0]>angleList[i][1]:
        #    if angleList[i-1][0]>angleList[i-1][1]:
        #        if(angleList[i][0]<angleList[i-1][0])or(angleList[i][1]>angleList[i-1][1]):
        #            contour=np.concatenate((contour,np.flip(line[angleList[i][3]],0)),axis=0)
        #            print("0concatenate()")
        #    else:
        #        if ((angleList[i][0]>angleList[i-1][0])or(angleList[i][1]<angleList[i-1][1]))or((angleList[i][0]>angleList[i-1][0])or(angleList[i][1]<angleList[i-1][1])):
        #            contour=np.concatenate((contour,np.flip(line[angleList[i][3]],0)),axis=0)
        #            print("0concatenate()")
        #    #contour=np.concatenate((contour,line[angleList[i][2]]),axis=0)
        #else:
        #    if(angleList[i][0]>angleList[i-1][1])or(angleList[i][1]>angleList[i-1][1]):
        #        contour=np.concatenate((contour,np.flip(line[angleList[i][3]],0)),axis=0)
        #        print("1concatenate()")
        #    if angleList[i][0]<angleList[i][1]:
                
        #    else:
        #ś        contour=np.concatenate((contour,line[angleList[i][2]]),axis=0)
    #contour=line[0]
    #print("contour")
    #print(contour)
    #print("contour[0]")
    #print(contour[0])
    #for i in range(1,len(line)):
        
    
    #contourVects=compute_vect(contour,massCenter)
    #contourVectsZipped=list(zip(contour,contourVects["angle"]))
    #print("sort(key)")
    #print(contourVectsZipped[0][1])
    #contourVectsZipped.sort(key= lambda x: x[1])
    #for i in range(len(contourVectsZipped)):
    #    print(contourVectsZipped[i])
    #contour=[x for y, x in sorted(contourVectsZipped, key=lambda pair: pair[1])]
    #contour=[]
    #for i in range(len(contourVectsZipped)):
    #    contour.append(contourVectsZipped[i][0])
    #print(contour)
    #plt_contour_points(contour,"contour",massCenter=massCenter)
    contour= [np.array(contour,dtype='int32')]
   # print(contour);    
    #print("contour")
   # print(line)
    return contour
    #return contours 

def adjust_contorur(contour, xMin, yMin):
    for i in range(len(contour)):
        #print("before adjusting = " + str(contour[i]))
        contour[i][0]=contour[i][0]-xMin
        contour[i][1]=contour[i][1]-yMin
        #print("after adjusting = " + str(contour[i]))
    return contour
def point_dist_2d(point1,point2):
    return math.sqrt(math.pow(point1[0]-point2[0],2)+math.pow(point1[1]-point2[1],2))
def create_mask(image,contours):
    mask = np.ones(image.shape[:2], dtype="uint8") * 255
    cv2.drawContours(mask, contours, -1, 0, -1)
    return mask

def remove_smal_contours(contours):
    avrage=0
    for i in range(len(contours)):
        avrage+=len(contours[i])
    avrage/=len(contours)
    avrage/=3
    for i in range(len(contours)):
        if(i>=len(contours)):
            break
        if len(contours[i])<avrage:
            contours.pop(i)
            i-=1
    return contours

def prepare_to_proces(imageToProces,imageMask,externalPolygon):
    
    boxCord=externalPolygon["box"]
    
    contours=externalPolygon["polygons"][0]["outer"]["path"]
    
    #print(contours)
    #contours=adjust_contorur(contours,boxCord[0],boxCord[1]) 
    imageMask=np.true_divide(imageMask,255)
    imageToProces=np.multiply(imageToProces,imageMask)
    #print(imageToProces)
    #imageToProces=imageToProces[boxCord[1]:boxCord[3],boxCord[0]:boxCord[2]]
    
    #histr = cv2.calcHist([imageToProces],[0],None,[256],[0,256]) 
    imageToProcesMin=imageToProces.min()
    imageToProcesMax=imageToProces.max()
    # show the plotting graph of an image 
    
    #print(imin,imax)
    externalPolygon.fill_mass_centers()
    massCenter = externalPolygon["polygons"][0]["outer"].get_mass_center()
    print("massCenter = "+str(massCenter))
    print("boxCord = " + str(boxCord))
    #massCenter[0]-=boxCord[0]
    #massCenter[1]-=boxCord[1]
    print(massCenter)
    im8_scal = imageToProces.copy()
    im8_scal = im8_scal*(255/imageToProcesMax)
    cv2Image8Bit = cv2.convertScaleAbs(im8_scal.copy())
    cv2Comtour = [np.array(contours, dtype=np.int32)]
    return cv2Image8Bit,cv2Comtour, [massCenter[0],massCenter[1]]
def proces_files(imageToProces,imageMask,externalPolygon):
    
    
    print("len(externalPolygon[polygons])"+str(len(externalPolygon["polygons"])))
    if(len(externalPolygon["polygons"])==0):
        return externalPolygon

    imageToProces,cv2Comtours,massCenter=prepare_to_proces(imageToProces,imageMask,externalPolygon)
    #cv2.imshow("oryginal 8 bit",imageToProces)
    for i in range(10,0,-1):
        cv2Comtours.append(scale_contour(cv2Comtours[0],i/10))
    
    mask=create_mask(imageToProces,[cv2Comtours[4]])
    print("mask shape: "+str(mask.shape))
    mask[round(massCenter[0]),round(massCenter[1])]=255
    imageWithoutMiddle = cv2.bitwise_and(imageToProces, imageToProces, mask=mask)
    ##cv2.imshow("oryginal middle removed",imageWithoutMiddle)
    #cv2.imshow("oryginal mask",mask)

    imageMedian = ndimage.median_filter(imageWithoutMiddle, size=7)
    tresh=(imageMedian.max()+imageMedian.min())/3
    #print("tresh= "+str(tresh))
    imageMedian = (imageMedian/imageMedian.max())*255
    imageMedian = cv2.convertScaleAbs(imageMedian.copy())
    #cv2.imshow("imageMedian",imageMedian)

    retMedian, threshMedian = cv2.threshold(imageMedian, tresh,255, 0)
    contoursMedian, hierarchyMedian = cv2.findContours(threshMedian, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ret, thresh = cv2.threshold(imageWithoutMiddle, 50,255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #cv2.drawContours(imageWithoutMiddle,contours,-1,(255),1)
    cv2.drawContours(imageMedian,contoursMedian,-1,(255),1)
    #
    #
    #print(hierarchyMedian)
    #cv2.imshow("imageMedian_cotours",imageMedian)
    remove_smal_contours(contoursMedian)
    
    mask= create_mask(imageWithoutMiddle,contoursMedian)
    
    imageDark = cv2.bitwise_and(imageWithoutMiddle, imageWithoutMiddle, mask=mask)
    mask=(255-mask)
    imageLight =cv2.bitwise_and(imageWithoutMiddle, imageWithoutMiddle, mask=mask)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(15,15))
    imageDark = clahe.apply(imageDark)
    imageLight = clahe.apply(imageLight)
    

    imageMedianLight = ndimage.median_filter(imageLight, size=7)
    tresh=(imageMedianLight.max()+imageMedianLight.min())/2
    #print("tresh= "+str(tresh))
    imageMedianLight = (imageMedianLight/imageMedianLight.max())*255
    imageMedianLight = cv2.convertScaleAbs(imageMedianLight.copy())
    
    #cv2.imshow("imageMedian",imageMedian)
    retMedianLight, threshMedianLight = cv2.threshold(imageMedianLight, tresh,255, 0)
    contoursMedianLight, hierarchyMedianLight = cv2.findContours(threshMedianLight, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("hierarchyMedianLight"+str(hierarchyMedianLight))
    contoursMedianLight=remove_smal_contours(contoursMedianLight)
    #print(len(contoursMedianLight))
    cv2.drawContours(imageLight,contoursMedianLight,-1,(255),1)
    #cv2.imshow("imageLight after hist+contours",imageLight)
    mask= create_mask(imageLight,contoursMedianLight)
    
    imageDarkSt2 = cv2.bitwise_and(imageLight, imageLight, mask=mask)
    mask=(255-mask)
    imageLightSt2 =cv2.bitwise_and(imageLight, imageLight, mask=mask)
    #contoursMedianLight=remove_smal_contours(contoursMedianLight)
    cv2.drawContours(imageDark,contoursMedianLight,-1,255-1)
    #cv2.imshow("contour 0",imageDark)
    comtourSt2=create_contour(cv2Comtours[0],contoursMedianLight,imageLightSt2,massCenter)
    comtourSt2.append(cv2Comtours[0])
    cv2.drawContours(imageDarkSt2,comtourSt2, -1, 255, 1)
    # #cv2.imshow("imageDark after hist",imageDarkSt2)
    #cv2.imshow("imageDark+light contours after hist",imageDarkSt2)
    maskk= create_mask(imageLight,comtourSt2)
    maskk=(255-maskk)
    #cv2.imshow("mask st2",maskk)
    tissue_polygons_out=v_polygons.from_ndarray(maskk)
    
    return tissue_polygons_out

#00000010_..._bones_-05_250135_algV<id>_labels.png
#00000010_..._bones_-05_250135_algV<id>_polygons.json


#                    img_Image = tissue_polygons_out.as_image(fill = True, w=out_w,h=out_h, force_labelRGB = True)
#                    img_Image.save(fn)

#                    org_point = [box[0], box[1]]
#                    tissue_polygons.move2point(org_point)
#                
#                
def make_file_name(filePath):
    fileName=filePath.split("_")
    fileName= fileName[len(fileName)-2]
    fileName=fileName.split("/")
    fileName=fileName[len(fileName)-1]
    fileName+= "_fat_algV0_"
    return fileName

def process_dir(dpDir,outDir):


    inputImageFiles = glob.glob(dpDir + '/*_lsi.png')
    inputImageFiles.sort()
    numOfFilleMin =0
    numOfFilleMax=len(inputImageFiles)
    #inputMaskFiles = glob.glob(contourDir + '/*_mask.png')
    #inputMaskFiles.sort()
    #print(inputMaskFiles[numOfFille])
    inputBoxFiles = glob.glob(dpDir + '/*_roi_polygons.json')
    inputBoxFiles.sort()
    for numOfFille in range(numOfFilleMin,numOfFilleMax):
        print("##########################################")
        print(numOfFille)
        imageToProces=Image.open(inputImageFiles[numOfFille])
        print(inputImageFiles[numOfFille])
        #fileName=inputImageFiles[numOfFille].split("_")
        fileName=make_file_name(inputImageFiles[numOfFille])
        print("fileName = "+fileName)
        print("proces_file_name = "+ inputImageFiles[numOfFille])
        #print(len(inputBoxFiles))
   #     with open(inputBoxFiles[numOfFille]) as f:
   #         contours_dict_data= json.load(f)
   #     my_read_contours = as_contours(contours_dict_data)
        #print(my_read_contours)

        #print(inputBoxFiles[numOfFille])
        with open(inputBoxFiles[numOfFille]) as f:
            polygonsDictData= json.load(f)
        #print("polygons_dict_data")
        #print(polygons_dict_data)
        # kdjkjskj    #my_read_polygons= v_polygons()
        myReadPolygons=v_polygons.from_dict(polygonsDictData)
   #     print("my_read_polygons")
   #     print(my_read_polygons.to_indendent_str())
        #print(my_read_polygons)
        print("imageToproces.shape = "+str(np.shape(imageToProces)))
        print("box: "+str(myReadPolygons["box"]))
        imageToProcesShape=np.shape(imageToProces)
        imageMask=myReadPolygons.as_numpy_mask(fill=True,w=imageToProcesShape[1],h=imageToProcesShape[0])
        #imageMask=imageMask[myReadPolygons["box"][1]:myReadPolygons["box"][3],myReadPolygons["box"][0]:myReadPolygons["box"][2]]
        #imageMask = imageio.imread(inputMaskFiles[numOfFille])
        #imageMask
        print("imageMask.shape = "+str(np.shape(imageMask)))
        tissue_polygons_out=proces_files(imageToProces,imageMask,myReadPolygons)
        img_Image = tissue_polygons_out.as_image(fill = True, w=imageToProcesShape[1],h=imageToProcesShape[0], force_labelRGB = True)
        img_Image.save( outDir+"/"+fileName+"_labels.png")
        tissue_polygons_dict = tissue_polygons_out.as_dict()
        jsonDumpSafe(outDir+"/"+fileName+"_polygons.json", tissue_polygons_dict)
        print("##########################################")
    
    #print(contours)
    
    #print(contours)
    #print(inputMaskFiles[0])
    
    

    
def TODO():
    



    
    cv2.drawContours(mask, contours, -1, 0, -1)
    mask=(255-mask)
    image2 = cv2.bitwise_and(image2, image2, mask=mask)
    #cv2.imshow("image2", image2)
    equ = cv2.equalizeHist(image2)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(15,15))
    cl1 = clahe.apply(image2)
    im_med = ndimage.median_filter(image2, size=7)
    im_med = (im_med/im_med.max())*255
    im_med = cv2.convertScaleAbs(im_med.copy())
    equ_imed = cv2.equalizeHist(im_med)
    res = np.hstack((image2,equ,im_med,equ_imed,cl1))
    #cv2.imshow('ar',image)
    #cv2.imshow("hist_equalization",res)
    
    ret, thresh = cv2.threshold(cl1, 100,255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    th2 = cv2.adaptiveThreshold(cl1,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
    contoursTh2, hierarchyTh2 = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    th3 = cv2.adaptiveThreshold(cl1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
    contoursTh3, hierarchyTh3 = cv2.findContours(th3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.ones(cl1.shape[:2], dtype="uint8") * 255
    #
    #print(mask)
# loop over the contours
    
    

    thresh2=cl1.copy()
    thresh3=cl1.copy()
    cv2.drawContours(cl1,contours,-1,(255),1)
    cv2.drawContours(thresh2,contoursTh2,-1,(255),1)
    cv2.drawContours(thresh3,contoursTh3,-1,(255),1)
    thresh=np.hstack((cl1,thresh2,thresh3))
    
    #cv2.imshow("thresh_im_comptour_v2", thresh)
    cv2.drawContours(mask, contours, -1, 0, -1)
    mask=(255-mask)
    cl1 = cv2.bitwise_and(cl1, cl1, mask=mask)
    equ = cv2.equalizeHist(cl1)
    res_fin = np.hstack((cl1,equ))
    #cv2.imshow("res_fin",res_fin)
    #plt.hist(image.flatten(),256,[1,256], color = 'r')
    #plt.hist(equ.flatten(),256,[1,256], color = 'b')
    #plt.hist(cl1.flatten(),256,[1,256], color = 'r')
    #plt.hist(im_med.flatten(),256,[1,256], color = 'g')
    #plt.hist(equ_imed.flatten(),256,[1,256])
    #plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()

####
#    im16  = np.load(img_path)
#
#    imin = im16.min()
#    imax = im16.max()
#
#    im = im16.copy()
#    #print(imin,imax)
#
#    im8_scal = im16.copy()
#    im8_scal = im8_scal*(255/gl_pixmax)
#
#    im[im<(imax/7)] = 0
#    im = (im/imax)*255
#
#    im_med = ndimage.median_filter(im16, size=5)
#    im_med = (im_med/im_med.max())*255
#
#    lmax = im_med.max()
#    #print(imax)
#    ret,thresh1 = cv2.threshold(im_med,28,255,cv2.THRESH_BINARY)
#
#    im8 = cv2.convertScaleAbs(im.copy())
#    im8_md = cv2.convertScaleAbs(im_med.copy())
#    im8_th = cv2.convertScaleAbs(thresh1.copy())
#
#    cont = []
#
#    contours, hierarchy = cv2.findContours(im8_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#
#    im_con             = im8.copy()
#    
#    sum_x = 0
#    sum_y = 0
#    n     = 0
#
#    #find largest contour
#    
#    L_cont = [] 
#    for L in contours:
#        #print(len(L))
#        if len(L)>len(L_cont):
#            L_cont = L
#
#    for T in L_cont:
#        #print(T[0][0],T[0][1])
#        x = T[0][0]
#        y = T[0][1]
#            
#        sum_x += x
#        sum_y += y
#        n     += 1
#
#    avg_x = int(sum_x/n)
#    avg_y = int(sum_y/n)
#
#    #print(len(L_cont))
#    
#    for T in L_cont:
#        #print(T[0][0],T[0][1])
#        x = T[0][0]
#        y = T[0][1]
#
#        cont.append([int(x),int(y)])
#
#    im8_mask = im8_th.copy()
#    im8_mask[:,:] = 0
#
#    cv2.fillPoly(im8_mask, pts =[L_cont], color=(255))
#
#    cv2.drawContours(im_con, L_cont, -1, 1024, 1)
#
#    max_x = max(cont, key=lambda x: x[0])[0]
#    max_y = max(cont, key=lambda x: x[1])[1]
#    min_x = min(cont, key=lambda x: x[0])[0]
#    min_y = min(cont, key=lambda x: x[1])[1]
#
#    #print((min_x,min_y),(max_x,max_y))
#
#    cv2.rectangle(im_med,(min_x,min_y),(max_x,max_y),(255,255,255),1)
#
#    #ax1 = plt.subplot(141)
#    #ax1.imshow(im8,cmap='gray')
#
#    #ax2 = plt.subplot(142,sharex=ax1,sharey=ax1)
#    #ax2.imshow(im_med,cmap='gray')
#
#    #ax3 = plt.subplot(143,sharex=ax1,sharey=ax1)
#    #ax3.imshow(im8_mask,cmap='gray')
#
#    #ax4 = plt.subplot(144,sharex=ax1,sharey=ax1)
#    #ax4.imshow(im_con,cmap='gray')
#
#    #plt.show() 
#
#    return(cont,im8,im8_mask,im8_scal,[min_x,min_y,max_x,max_y,avg_x,avg_y])


#-----------------------------------------------------------------------------------------
#MAIN
#-----------------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("-dpDir",  "--dp_dir",      dest="dp_dir",      help="input directory",         metavar="PATH", required=True)
    parser.add_argument("-osDir","--os_dir",        dest="os_dir",      help="output fat directory",              metavar="PATH", required=True)
    parser.add_argument("-v",      "--verbose",     dest="verbose",     help="verbose level",                                       required=False)

    args = parser.parse_args()

    verbose = 'off'                 if args.verbose is None else args.verbose
    dpDir  	= args.dp_dir
    outDir   = args.os_dir

    dpDir = os.path.normpath(dpDir)
    outDir = os.path.normpath(outDir)

    if not os.path.isdir(dpDir):
        print('Error : Input directory (%s) with numpy image files not found !',dpDir)
        exit(-1)

    try:
        if not os.path.isdir(outDir):
            pathlib.Path(outDir).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        print("##########dupa############")
        print("Creating shapem pattern dir (%) IO error: {}"%outDir,format(err))
        print("######################")
        sys.exit(1)
    
    
    print("Dir with numpy image input: "+dpDir)
    print("Dir with output: "+outDir)

    process_dir(dpDir,outDir)

    
    #plt.hist(im8_scal.ravel(),255,[1,255])
#
#    tc          = multiprocessing.cpu_count()
#
#    print('SETUP     > -----------------------------------------------------')
#    print('INFO 0000 > Detected threads      :%d '% tc)
#
#    at          = int((3*tc)/4)
#    cv2.setNumThreads(at)
#
#    print('INFO 0001 > CV2 threads           :%d ( min(%d, 8))'%(at,tc))
#    print('DEBUG     > -----------------------------------------------------')
#    print('INFO 0002 > Verbose level         :%s '% verbose)
#    print('BEGIN     > -----------------------------------------------------')
#
#    gname       = dpDir + '/*_dicom.json'
#    gname       = os.path.normpath(gname)
#
#    images      = glob.glob(gname)
#    imid        = 0
#
#    if images == []:
#        print('ERR  0002 > invalid file name or path (%s)'%gname)
#
#    images.sort()
#
#    points = []
#
#
#    print('FIRST LOOP  ---- finding of the maximum pixel value')
#
#    maxpixv 	= 0
#    imid        = 0
#    for iname in images:
#
#        xname           = os.path.basename(iname)
#        fname, fext     = os.path.splitext(xname)
#        fname, fsuf     = fname.split('_')
#
#        if imid!= 0:
#            print('NEXT      > -----------------------------------------------------')
#
#        print('INFO 0090 > file name     : ',fname)
#
#        npy_path 	= os.path.normpath(dpDir+'/'+fname+'_16bit.npy')
#        im16  		= np.load(npy_path)
#        locmax      = im16.max()
#
#        print('INFO 0091 > local maximum pixel value : ',locmax)
#
#        if maxpixv < locmax:
#            maxpixv = locmax
#    
#    print('INFO 0099 > global maximum pixel value : ',maxpixv)
#
#    print('SECOND LOOP  ---- generating shape')
#
#    imid        = 0
#    for iname in images:
#
#        xname           = os.path.basename(iname)
#        fname, fext     = os.path.splitext(xname)
#        fname, fsuf     = fname.split('_')
#
#        if imid!= 0:
#            print('NEXT      > -----------------------------------------------------')
#
#        print('INFO 0100 > file name     : ',fname)
#
#        npy_path = os.path.normpath(dpDir+'/'+fname+'_16bit.npy')
#
#        imid += 1
#        tcont,timg8, tmask,tscal, tmeta = process_dir(npy_path, maxpixv, verbose)
#
#        print('INFO 0101 > shape box     : ',[int(tmeta[0]),int(tmeta[1]),int(tmeta[2]),int(tmeta[3])])
#        print('INFO 0102 > contour nodes : ',len(tcont))
#
#        # creating file with metadata (box & contour)
#        meta = {}
#    
#        meta['box'] 	= [int(tmeta[0]),int(tmeta[1]),int(tmeta[2]),int(tmeta[3])]
#        meta['contour'] = tcont
#    
#        meta_path		= os.path.normpath(shDir + '/' + fname + '_vains.json')
#        fjson 			= open(meta_path,'w')
#
#        if fjson != None:
#             json.dump(meta,fjson,indent=4)
#
#        img_path = os.path.normpath(shDir + '/' + fname + '_veins_mask.bmp')
#        cv2.imwrite(img_path,tmask)
#
#        #writing image   
#
#        img_path = os.path.normpath(shDir + '/' + fname + '_veins_lsi.bmp')
#        cv2.imwrite(img_path,timg8)
#
#        img_path = os.path.normpath(shDir + '/' + fname + '_veins_gsi.bmp')
#        cv2.imwrite(img_path,tscal)
#
#    #-----------------------------------------------------------------------------------------
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()