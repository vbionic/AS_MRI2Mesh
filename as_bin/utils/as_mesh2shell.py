#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import copy
import trimesh
import shapely, shapely.ops
from multiprocessing import Pool, TimeoutError
import math
from argparse import ArgumentParser
import os, sys
import pathlib
import glob
import time
from time import gmtime, strftime
import pyglet
import json
import logging


def IsLineStringValidPolygon(LineStringObject, addStartPointToRings, minimalLenghtOfLine = 0):
    if(LineStringObject==None or len(LineStringObject.coords)==0):
        return False
    startPoint = LineStringObject.coords[0]
    endPoint = LineStringObject.coords[-1]
    validPolygonMinPointsNum = 3
    if(startPoint == endPoint):
        validPolygonMinPointsNum+=1
        #logging.info("Start point " + str(startPoint) + ", end point " + str(endPoint))
    if(addStartPointToRings):
        validPolygonMinPointsNum+=1
        
    stringLine_valid = (len(LineStringObject.coords) >= validPolygonMinPointsNum)
    stringLine_valid = stringLine_valid and (LineStringObject.length >= minimalLenghtOfLine)
    return stringLine_valid
    
def getOnlyValidLines(lines, addStartPointToRings, minimalLenghtOfLine=0):
    valid_lines = []
    if(lines == None):
        return []
    elif(type(lines) is list):
        for single_line in lines:
            if(IsLineStringValidPolygon(single_line, addStartPointToRings, minimalLenghtOfLine)):
                valid_lines.append(single_line)
    elif((lines.geom_type == 'MultiLineString') or (lines.geom_type == 'MultiLinearRing')):
        for single_line in list(lines):
            if(IsLineStringValidPolygon(single_line, addStartPointToRings, minimalLenghtOfLine)):
                valid_lines.append(single_line)
    elif((lines.geom_type == 'LineString') or (lines.geom_type == 'LinearRing')):
        if(IsLineStringValidPolygon(lines, addStartPointToRings, minimalLenghtOfLine)):
            valid_lines.append(lines)
    return valid_lines

def moveRingStartToStraight(_LinearRing):
    ExtPoints = list(_LinearRing.coords)
    firstP = ExtPoints[0]
    secondP = ExtPoints[1]
    # srednia nie jest dobra bo tworza sie jakies dodatkowe odcinki. Najlepiej wprowadzic nieistotne przesuniecie np. 0.001mm
    intP = ((secondP[0] + firstP[0])/2 + 0.001, (secondP[1] + firstP[1])/2)
    if(firstP == ExtPoints[-1]):
        ExtPoints[0]=intP
        ExtPoints.append(intP)
    else:
        ExtPoints[0]=intP
        ExtPoints.append(firstP)
    _LinearRingNew = shapely.geometry.LinearRing(ExtPoints)
    return _LinearRingNew

def ListAndValidatePolygons(_polygons):
    res_list = []
    needs_to_be_validate = False
    if(_polygons.geom_type == 'Polygon'):
        if _polygons.is_valid:
            res_list.append(_polygons)
        else:
            res_list.append(_polygons.buffer(0))
            needs_to_be_validate = True
    elif(_polygons.geom_type == 'MultiPolygon'):
        for mp in list(_polygons):
            if mp.is_valid:
                res_list.append(mp)
            else:
                res_list.append(mp.buffer(0))
                needs_to_be_validate = True
        
    # do the same once more - it was observed that after buffer(0) operation some Polygons turns into MultiPolygons
    while(needs_to_be_validate):
        needs_to_be_validate = False
        for poly in res_list:
            if(poly.geom_type == 'MultiPolygon'):
                needs_to_be_validate = True
                res_list += [(ps if ps.is_valid else ps.buffer(0)) for ps in list(poly)]
                res_list.remove(poly)
        
    return res_list 

def ListAndValidatePolygons_v2(x_polygons):
    res_list = []
    if(x_polygons.geom_type == 'Polygon'):
        if x_polygons.is_valid:
            res_list = [x_polygons]
        else:
            validated = ListAndValidatePolygons_v2(x_polygons.buffer(0))
            res_list += validated
    elif(x_polygons.geom_type == 'MultiPolygon'):
        for mp in list(x_polygons):
            if mp.is_valid:
                res_list.append(mp)
            else:
                validated = ListAndValidatePolygons_v2(mp.buffer(0))
                res_list += validated
        
    return res_list     

def check_for_a_new_hole(ofsetted_line):
    #przypadek gdy po rozszerzeniu poligonu bez dziur powstaje wiecej niz jedna linia odpowiada zamknieciu obszaru, czyli powstaniu dziury
    # nalezy przeniesc jedna z linii do zbioru dziur
    if(((type(ofsetted_line) is list) or (ofsetted_line.geom_type == 'MultiLineString')) and len(ofsetted_line) > 1):
        found_external_line = []
        found_interiors = []
        if (type(ofsetted_line) is list):
            ofseted_lines = ofsetted_line
        elif(ofsetted_line.geom_type == 'MultiLineString'):
            ofseted_lines = list(ofsetted_line)
        for line_examined_as_external in ofseted_lines:
            Found = True
            for line_examined_as_internal in ofseted_lines :
                if(line_examined_as_internal == line_examined_as_external):
                    continue
                external_linePolygon = shapely.geometry.Polygon(line_examined_as_external)
                internal_linePolygon = shapely.geometry.Polygon(line_examined_as_internal)
                if not external_linePolygon.contains(internal_linePolygon):
                    Found = False
                    break
            if Found:
                found_external_line = [line_examined_as_external]
                break
        if(len(found_external_line)!=0):
            for line_examined_as_internal in ofseted_lines :
                if(line_examined_as_internal != found_external_line[0]):
                    found_interiors.append(line_examined_as_internal)
            return found_external_line, found_interiors
        else:
            return ofsetted_line, None
    else:
        return ofsetted_line, None


def slice_mesh(wID, mesh, start_layer, last_layer, close_not_closed_holes_at_top, mesh_minz, mesh_maxz, printer_dx, printer_dz, wall_thickness, addStartPointToRings, resolution, joint_style, simplification_limit, delete_lines_shorter_than, PrintStatusInSubFunctions):
    
    if(True):
        logging.info("  #WID{}, layers {}-{})".format(wID, start_layer, last_layer))
    # get a single cross section of the mesh
    PoligonsHollow = []
    PoligonssFull = []
    PoligonssHoles = []
    plane_id = -1
    ploterlyPoligonsWithoutPlugs = False
    #for planeX_id in range(6, planes_num):
    for planeX_id in range(start_layer, last_layer+1):
    #for plane_id in [1,24,78,330]:
        plane_id+=1
        if(PrintStatusInSubFunctions):
            logging.info("  #{}: Plane {}(th_plain{})".format(wID, planeX_id, plane_id))
        input_z = mesh_minz + 0.001 + planeX_id * printer_dz
        output_z = planeX_id * printer_dz
        slice = mesh.section(plane_origin=[0,0,input_z], plane_normal=[0,0,1])
        PoligonsHollow.append([])
        PoligonssFull.append([])
        PoligonssHoles.append([])
        # the section will be in the original mesh frame
        if not(slice is None):
            #slice.show()
            to_2DMatrix = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
            slice_2D, to_3D = slice.to_planar(to_2D = to_2DMatrix)
            #logging.info("slice   bounds: \n" + str(slice.bounds[0][0:2]) + "\n" + str(slice.bounds[1][0:2]))
            #logging.info("slice 2Dbounds: \n" + str(slice_2D.bounds[0]) + "\n" + str(slice_2D.bounds[1]))
            #extruded = slice_2D.extrude(height=1)
            #extruded.export('test_extruded.stl')
            poligons = []
            if(not slice_2D.is_closed):
                if(close_not_closed_holes_at_top):
                    vertices = slice_2D.vertices
                    line = shapely.geometry.LineString(vertices)
                    poligon = line.convex_hull
                    poligons = ListAndValidatePolygons(poligon)
                    
                else:
                    logging.info("   #" + str(wID) + ": slice not closed at layer " + str(planeX_id) + ", at height" + str(input_z) + "mm. Stop slicing")
                    if(len(PoligonsHollow) > 0):
                        del PoligonsHollow[-1]
                        del PoligonssFull[-1]
                        del PoligonssHoles[-1]
                    break
            else:
                poligons = slice_2D.polygons_full
            
            # if we want to intersect a line with this 2D polygon, we can use shapely methods
            plane_offseted_lines_out = []
            plane_offseted_lines_in = []
            for poligon_id in range(0, len(poligons)):
                polygon = poligons[poligon_id]

                #
                # rozszerzenie poligonów
                #

                if addStartPointToRings:
                    polygonExtLinearRingNew = moveRingStartToStraight(polygon.exterior)
                else:
                    polygonExtLinearRingNew = polygon.exterior
                
                if simplification_limit != 0:        
                        polygonExtLinearRingNew  = polygonExtLinearRingNew.simplify(simplification_limit, preserve_topology=False)
                #Returns a LineString or MultiLineString geometry at a distance from the object on its right or its left side.
                #Distance must be a positive float value. The side parameter may be ‘left’ or ‘right’. The resolution of the offset around each vertex of the object is parameterized as in the buffer method.
                #The join style is for outside corners between line segments. Accepted integer values are 1 (round), 2 (mitre), and 3 (bevel). See also shapely.geometry.JOIN_STYLE.
                #Severely mitered corners can be controlled by the mitre_limit parameter (spelled in British English, en-gb). The ratio of the distance from the corner to the end of the mitred offset corner is the miter ratio. Corners with a ratio which exceed the limit will be beveled.
                #Note
                #
                #This method is only available for LinearRing and LineString objects.
                side = 'right'
                if(not polygonExtLinearRingNew.is_ccw):
                    side = 'left'
                ofsetted_lines_in  = polygonExtLinearRingNew#.parallel_offset(distance=printer_dx*(       0 + 0.5), side=side, resolution=resolution, join_style=joint_style, mitre_limit=5.0)
                               
                try:
                    ofsetted_lines_out = polygonExtLinearRingNew.parallel_offset(distance=printer_dx*(wall_thickness*1.1 ), side=side, resolution=resolution, join_style=joint_style, mitre_limit=1.0)
                except:
                    ## wywalil sie na tym, ale nie wiem co z tym zrobic. Zwyczajnie wykonam operacje jeszcze raz z innymi parametrami
                    #ofsetted_lines_out = polygonExtLinearRingNew.parallel_offset(distance=printer_dx*(wall_thickness*1.01 ), side=side, resolution=resolution, join_style=joint_style, mitre_limit=6.0)
                    #chyba chodzi o to ze obiekt nie jest zamkniety. Jeżeli tak to przerwij podzialy na tej warstwie
                    logging.error("!Error przy dzieleniu obiektu na warstwy. Prawdopodobnie obiekt nie jest zamkniety. Przerywam podzialy na tej warstwie (" + str(planeX_id) + ").")
                    break
                    #polygonExtLinearRingNew = polygonExtLinearRingNew.convex_hull.exterior
                    #ofsetted_lines_out = polygonExtLinearRingNew.parallel_offset(distance=printer_dx*(wall_thickness*1.1 ), side=side, resolution=resolution, join_style=joint_style, mitre_limit=5.0)
                
                ofsetted_lines_in = getOnlyValidLines(ofsetted_lines_in, addStartPointToRings, delete_lines_shorter_than)
                ofsetted_lines_out  = getOnlyValidLines(ofsetted_lines_out,  addStartPointToRings, delete_lines_shorter_than)
                ofsetted_lines_out, new_internal_lines = check_for_a_new_hole(ofsetted_lines_out)
                if( not(new_internal_lines is None) and (len(new_internal_lines) > 0) ):
                    ofsetted_lines_in+=new_internal_lines
                    ofsetted_lines_in = getOnlyValidLines(ofsetted_lines_in, addStartPointToRings, delete_lines_shorter_than)

                if simplification_limit != 0:
                    ofsetted_lines_in  = [xline.simplify(simplification_limit, preserve_topology=False) for xline in ofsetted_lines_in ]
                    ofsetted_lines_out = [xline.simplify(simplification_limit, preserve_topology=False) for xline in ofsetted_lines_out]
                    
                ofsetted_lines_out_validated = getOnlyValidLines(ofsetted_lines_out, addStartPointToRings, delete_lines_shorter_than)
                ofsetted_lines_in_validated  = getOnlyValidLines(ofsetted_lines_in,  addStartPointToRings, delete_lines_shorter_than)
                    
                plane_offseted_lines_out += ofsetted_lines_out_validated
                plane_offseted_lines_in  += ofsetted_lines_in_validated
                    
            #
            # stworz poligony
            #
                    
            # bez dziur
            for ofsetted_line_out in plane_offseted_lines_out:
                PoligonFull = shapely.geometry.Polygon(ofsetted_line_out)
                PoligonssFull[-1] += ListAndValidatePolygons_v2(PoligonFull)

            # same dziury
            for offset_line_for_holes in plane_offseted_lines_in:
                PoligonHole = shapely.geometry.Polygon(offset_line_for_holes)
                PoligonssHoles[-1] += ListAndValidatePolygons_v2(PoligonHole)
                    
            #
            # Łączenie poligonów
            #   

            if(len(PoligonssFull[plane_id]) > 1):
                mergedPolygons = shapely.ops.unary_union(PoligonssFull[plane_id])
                PoligonssFull[plane_id] = ListAndValidatePolygons_v2(mergedPolygons)
            if(len(PoligonssHoles[plane_id]) > 1):
                mergedPolygons = shapely.ops.unary_union(PoligonssHoles[plane_id])
                PoligonssHoles[plane_id] = ListAndValidatePolygons_v2(mergedPolygons)
            # ostateczny poligon jako ograniczony zewnerzem PoligonFull i z wycietymi dziurami z zewnetrza PoligonssHoles
            for poligonFull in PoligonssFull[plane_id]:
                poligonHollow = shapely.geometry.Polygon(poligonFull.exterior, [ph.exterior for ph in PoligonssHoles[plane_id]])
                PoligonsHollow[plane_id] += ListAndValidatePolygons_v2(poligonHollow)
                    
    return (PoligonsHollow, PoligonssFull, PoligonssHoles)

def create_plugs(wID, PoligonsHollow, PoligonssFull, PoligonssHoles, layer_offset, start_layer, last_layer, mesh_minz, mesh_maxz, printer_dx, printer_dz, wall_thickness, addStartPointToRings, resolution, joint_style, simplification_limit, delete_lines_shorter_than, postprocessMergedPoligons, show_layer, plug_buffer_size, PrintStatusInSubFunctions):

    if(True):
        logging.info("  #WID{}, layers {}-{})".format(wID, start_layer, last_layer))
    plugsPolygonsLayersList = []
    extrudedLayersList = []
    for plane_id in range(start_layer, last_layer+1):
        if(plane_id > len(PoligonssFull)-1):
            continue
        plugsPolygonsLayersList.append([])
        if( ((plane_id + layer_offset) > (len(PoligonssFull)-1)) or ((plane_id + layer_offset) < 0 ) or (plane_id == 0) ):
            continue
        if(PrintStatusInSubFunctions):
            logging.info("  #" + str(wID) + ": Plane " + str(plane_id) + " ")
    
        #check if all holes from previous layer are covered
        if((plane_id>0) and not(PoligonssHoles[plane_id+layer_offset] is None) and len(PoligonssHoles[plane_id+layer_offset])!= 0):
            CoveringPolygons = PoligonssFull[plane_id]

            #petla po wszystkich dziurach z poprzedniej warstwy
            polygonHoleId = 0
            holesToCover = [shapely.geometry.Polygon(htc) for htc in PoligonssHoles[plane_id+layer_offset]]
            while polygonHoleId < len(holesToCover):
                uncoveredPartOfHole = holesToCover[polygonHoleId]
                polygonHoleId+=1
                #dla danej dziury sprawdz czy cos jej nie przykrywa w tej warstwi
                for CoveringPolygonId in range(0, len(CoveringPolygons)):
                    CoveringPolygon = CoveringPolygons[CoveringPolygonId]
                    uncoveredPartOfHole = uncoveredPartOfHole.difference(CoveringPolygon)
                    # mozliwe ze powstanie wiecej poligonow niz jeden
                    # wtedy zajmij sie tylko pierwszym a dalsze nowe dodaj na koniec listy 
                    if(uncoveredPartOfHole.geom_type == 'MultiPolygon'):
                        uncoveredPartsOfHole = list(uncoveredPartOfHole)
                        for mpId in range (1, len(uncoveredPartsOfHole)):
                            if(not uncoveredPartsOfHole[mpId].is_valid):
                                uncoveredPartsOfHole[mpId] = uncoveredPartsOfHole[mpId].buffer(0)
                            holesToCover.append(uncoveredPartsOfHole[mpId])
                        uncoveredPartOfHole = uncoveredPartsOfHole[0]
                        
                    if(not uncoveredPartOfHole.is_valid):
                        uncoveredPartOfHole = uncoveredPartOfHole.buffer(0)

                # jezeli cos zostalo z dziury to stworz zatyczkę
                if(not uncoveredPartOfHole.is_empty):
                    if(PrintStatusInSubFunctions):
                        logging.info("   #" + str(wID) + ": uncovered hole!")
                    interiorsOfThisPlug = list(uncoveredPartOfHole.interiors)

                    # małe dziury zamień na standardowy, okragły otwór
                    if(uncoveredPartOfHole.area < 0.1):
                        uncoveredPartOfHole = uncoveredPartOfHole.centroid.buffer(0.1)

                    # zatyczka powinna byc troche wieksza niz dziura, wiec ja rozszerz
                    if addStartPointToRings:
                        polygonExtLinearRingNew = moveRingStartToStraight(uncoveredPartOfHole.exterior)
                    else:
                        polygonExtLinearRingNew = uncoveredPartOfHole.exterior
                    side = 'right'
                    if(not polygonExtLinearRingNew.is_ccw):
                        side = 'left'
                    if(polygonExtLinearRingNew.geom_type == 'MultiLinearRing'):
                        notImplemented = 1 # tego sie nie spodziewam
                        
                    if simplification_limit != 0:        
                        polygonExtLinearRingNew  = polygonExtLinearRingNew.simplify(simplification_limit, preserve_topology=False)
                    
                    valids = getOnlyValidLines(polygonExtLinearRingNew, addStartPointToRings, delete_lines_shorter_than)
                    if(len(valids) > 0):
                        polygonExtLinearRingNew = valids[0]
                        try:
                            hole_ofsetted_lines  = polygonExtLinearRingNew.parallel_offset(distance=plug_buffer_size, side=side, resolution=resolution, join_style=joint_style, mitre_limit=5.0)
                        except:
                            logging.error("   !Error na warstwie {}. Nie tworze zatyczki".format(plane_id))
                            hole_ofsetted_lines = None
                            
                        hole_ofsetted_lines = getOnlyValidLines(hole_ofsetted_lines, addStartPointToRings, delete_lines_shorter_than)
                        #przypadek gdy po rozszerzeniu poligonu bez dziur powstaje wiecej niz jedna linia odpowiada zamknieciu obszaru, czyli powstaniu dziury
                        # nalezy przeniesc jedna z linii do zbioru dziur
                        hole_ofsetted_lines, new_internal_lines = check_for_a_new_hole(hole_ofsetted_lines)
                        if( not(new_internal_lines is None) and (len(new_internal_lines) > 0) ):
                            interiorsOfThisPlug+=new_internal_lines
                        
                        if simplification_limit != 0:
                            for hole_ofsetted_line in hole_ofsetted_lines:
                                hole_ofsetted_line  = hole_ofsetted_line.simplify(simplification_limit, preserve_topology=False)
                        hole_ofsetted_lines = getOnlyValidLines(hole_ofsetted_lines, addStartPointToRings, delete_lines_shorter_than)

                        #jezeli zatyczka ma miec jakies wyciecia (list(uncoveredPartOfHole.interiors))
                        # to je troche przesun bo jak beda sie nakladac granice zatyczki i podstawowych elementow to bedzie kiszka
                        interiorsOfThisPlug = getOnlyValidLines(interiorsOfThisPlug, addStartPointToRings, delete_lines_shorter_than)
                        interiorsOfThisPlugOffsetted = []
                        for interiorOfThisPlug in interiorsOfThisPlug:
                            if addStartPointToRings:
                                polygonIntLinearRingNew = moveRingStartToStraight(interiorOfThisPlug)
                            else:
                                polygonIntLinearRingNew = interiorOfThisPlug
                            side = 'right'
                            if(not polygonIntLinearRingNew.is_ccw):
                                side = 'left'
                                
                            if simplification_limit != 0:
                                polygonIntLinearRingNew  = polygonIntLinearRingNew.simplify(simplification_limit, preserve_topology=False)
                            valids = getOnlyValidLines(polygonIntLinearRingNew, addStartPointToRings, delete_lines_shorter_than)
                            if(len(valids) > 0):
                                polygonIntLinearRingNew = valids[0]
                                try:
                                    polygonIntLinearRingNew  = polygonIntLinearRingNew.parallel_offset(distance=-0.09, side=side, resolution=resolution, join_style=joint_style, mitre_limit=5.0)
                                except:
                                    logging.error("   !Error na warstwie {}. Nie tworze wyciecia w zatyczce".format(plane_id))
                                    polygonIntLinearRingNew = None
                                if not (polygonIntLinearRingNew is None):
                                    if simplification_limit != 0:
                                        polygonIntLinearRingNew  = polygonIntLinearRingNew.simplify(simplification_limit, preserve_topology=False)
                                    interiorsOfThisPlugOffsetted += getOnlyValidLines(polygonIntLinearRingNew, addStartPointToRings, delete_lines_shorter_than)
                                                

                        # z rozszerzenia moze powstac wiecej niz jeden poligon
                        for hole_ofsetted_line in hole_ofsetted_lines:
                            #PoligonHolePlug = shapely.geometry.Polygon(hole_ofsetted_line,list(uncoveredPartOfHole.interiors))
                            PoligonHolePlug = shapely.geometry.Polygon(hole_ofsetted_line,interiorsOfThisPlugOffsetted)
                            if(not PoligonHolePlug.is_valid):
                                PoligonHolePlug=PoligonHolePlug.buffer(0)
                            if(PoligonHolePlug.geom_type != 'Polygon'):
                                not_implemented_case=1
                            plugsPolygonsLayersList[-1].append(PoligonHolePlug)
                        
    #
    # merge plugs with basic polygons
    #
    
    if(PrintStatusInSubFunctions):
        logging.info("  #" + str(wID) + ": Merge plugs with basic polygons")
    LayerMergedPolygonListList = []
    for dst_plane_id in range(0, len(plugsPolygonsLayersList)):
        src_plane_id = dst_plane_id+start_layer
        if(src_plane_id > len(PoligonssFull)-1):
            break
        if(PrintStatusInSubFunctions):
            logging.info("  #" + str(wID) + ": Plane " + str(src_plane_id) + " ")
        z = printer_dz * src_plane_id
        
        layerMergedPolygonList = []
        if(len(PoligonsHollow[src_plane_id]) + len(plugsPolygonsLayersList[dst_plane_id]) > 1):
            layerMergedPolygon = shapely.ops.unary_union(PoligonsHollow[src_plane_id] + plugsPolygonsLayersList[dst_plane_id])
            if(layerMergedPolygon.geom_type == 'Polygon'):
                layerMergedPolygonList = [layerMergedPolygon if layerMergedPolygon.is_valid else layerMergedPolygon.buffer(0)]
            else:
                layerMergedPolygonList =  [(mp if mp.is_valid else mp.buffer(0)) for mp in list(layerMergedPolygon)]
        elif(len(PoligonsHollow[src_plane_id]) >0):
            layerMergedPolygonList = [PoligonsHollow[src_plane_id][0]]
        elif(len(plugsPolygonsLayersList[dst_plane_id]) >0):
            layerMergedPolygonList = [plugsPolygonsLayersList[dst_plane_id][0]]

        postprocessdMergedPolygonsList = []
        if(postprocessMergedPoligons):
            for poly in layerMergedPolygonList:
                polyoutline = poly.exterior
                polyinlines = poly.interiors
                if simplification_limit != 0:
                    polyoutline = polyoutline.simplify(simplification_limit, preserve_topology=False)
                    polyoutline = getOnlyValidLines(polyoutline, addStartPointToRings, delete_lines_shorter_than)

                    lineList = []
                    for line in polyinlines:
                        line  = line.simplify (simplification_limit, preserve_topology=False)
                        lineList  += getOnlyValidLines(line,  addStartPointToRings, delete_lines_shorter_than)
                    polyinlines = lineList
                    for polyout in polyoutline:
                        postrocessedPoly = shapely.geometry.Polygon(polyout,polyinlines)
                        if(postrocessedPoly.geom_type == 'Polygon'):
                            postprocessdMergedPolygonsList.append(postrocessedPoly if postrocessedPoly.is_valid else postrocessedPoly.buffer(0))
                        else:
                            postprocessdMergedPolygonsList += [(mp if mp.is_valid else mp.buffer(0)) for mp in list(postrocessedPoly)]
                    
        if(postprocessMergedPoligons):
            LayerMergedPolygonListList.append(postprocessdMergedPolygonsList)
        else:
            LayerMergedPolygonListList.append(layerMergedPolygonList)

    return LayerMergedPolygonListList
    
def calc_support_params(min_support_radius_at_hight, min_support_radius_at_bed, sizing_start_height, support_start_height, height):

    my_sizing_start_height = sizing_start_height if (sizing_start_height < support_start_height) else support_start_height
    min_support_radius = min_support_radius_at_hight
    if (height < my_sizing_start_height):
        coeff = (my_sizing_start_height - height) / my_sizing_start_height
        coeff *= coeff
        min_support_radius = min_support_radius_at_hight + (min_support_radius_at_bed - min_support_radius_at_hight) * coeff
    min_support_area = min_support_radius * min_support_radius * 3.14
    return (min_support_radius, min_support_area)
    
def create_support(wID, PoligonsHollow, PoligonssFull, PoligonssHoles, min_support_diameter_at_hight, min_support_diameter_at_bed, shrinking_angle, mesh_minz, printer_dx, printer_dz, wall_thickness, addStartPointToRings, resolution, joint_style, simplification_limit, delete_lines_shorter_than, postprocessMergedPoligons, show_layer, plug_buffer_size, PrintStatusInSubFunctions):
    show_cups = False
    
    discontinuedPolygonsPoints = []
    buffering_start_height = min_support_diameter_at_bed
    min_support_radius_at_hight = min_support_diameter_at_hight/2 
    min_support_radius_at_bed = min_support_diameter_at_bed/2 
    prev_externaPolygons = []
    externalPolygonLayerList = [ [] for i in range(len(PoligonssFull)) ]

    for plane_id in range(len(PoligonssFull)-1, -1, -1):

        if(PrintStatusInSubFunctions):
            logging.info("  #" + str(wID) + ": Plane " + str(plane_id) + " ")
            
        shrinking_dx = (shrinking_angle/45) * printer_dz
        #shrink previous external poligons
        prev_externaPolygonsShrinked = []
        if(len(prev_externaPolygons) > 0):
            tmpList = shapely.ops.unary_union(prev_externaPolygons)
            tmpList = tmpList.buffer(-shrinking_dx)
            #prev_externaPolygonsShrinked += ListAndValidatePolygons_v2(tmpList)
            if(tmpList.geom_type == 'Polygon'):
                if(not tmpList.is_valid):
                    tmpList=tmpList.buffer(0)
                if(tmpList.area > 1):
                    prev_externaPolygonsShrinked.append(tmpList)
            else:
                for tmpPoly in tmpList:
                    if(tmpPoly.geom_type == 'MultiPolygon'):
                        for poly in list(tmpPoly):
                            if(not poly.is_valid):
                                poly=poly.buffer(0)
                            if(poly.area > 1):
                                prev_externaPolygonsShrinked.append(poly)
                    else:
                            if(not tmpPoly.is_valid):
                                tmpPoly=tmpPoly.buffer(0)
                            if(tmpPoly.area > 1):
                                prev_externaPolygonsShrinked.append(tmpPoly)
                        
            
        #merge external polygon from upper layer with object from this layer
        externalPolygons = []
        CoveringPolygons = [shapely.geometry.Polygon(poly.exterior) for poly in PoligonsHollow[plane_id]]
        
        if(len(prev_externaPolygonsShrinked) > 0):
            tmpList = shapely.ops.unary_union(CoveringPolygons + prev_externaPolygonsShrinked)
            if(tmpList.geom_type == 'Polygon'):
                if(not tmpList.is_valid):
                    tmpList=tmpList.buffer(0)
                if(tmpList.area > 1):
                    externalPolygons.append(tmpList)
            else:
                for tmpPoly in list(tmpList):
                    if(not tmpPoly.is_valid):
                        tmpPoly=tmpPoly.buffer(0)
                    if(tmpPoly.area > 1):
                        externalPolygons.append(tmpPoly)
        else:
            externalPolygons = CoveringPolygons
    

        #add support for discontinued external polygon 
        continued_extPolys1 = []
        continued_extPolys2 = []
        height = plane_id * printer_dz
        
        
        #dodaj wczesniej zidentyfikowane podpory
        for prev_discontinuedCentroids in discontinuedPolygonsPoints:
            support_start_height, centroidPoint = prev_discontinuedCentroids
            min_support_radius, min_support_area = calc_support_params(min_support_radius_at_hight, min_support_radius_at_bed, buffering_start_height, support_start_height, height)
            continued_extPolys1.append(shapely.geometry.Polygon(centroidPoint.buffer(min_support_radius)))
            
            
        #dodaj nowe podpory (ktore powinny sie zaczac na tej warstwie)
        for prev_extPoly in prev_externaPolygons:
            continued = False
            min_support_radius, min_support_area = calc_support_params(min_support_radius_at_hight, min_support_radius_at_bed, buffering_start_height, buffering_start_height, height)  
            for curr_extPoly in externalPolygons:
                if prev_extPoly.intersects(curr_extPoly) and (curr_extPoly.area > min_support_area-0.01):
                    continued = True
                    break
            if(not continued):
                min_support_radius, min_support_area = calc_support_params(min_support_radius_at_hight, min_support_radius_at_bed, buffering_start_height, height, height)  
                continued_extPolys2.append(shapely.geometry.Polygon(prev_extPoly.centroid.buffer(min_support_radius)))
                found = False
                for cpoint in discontinuedPolygonsPoints:
                    support_start_height, centroidPoint = cpoint
                    if(centroidPoint.buffer(1).intersects(prev_extPoly.centroid) ):
                        found = True
                        break
                if(not found):
                    discontinuedPolygonsPoints.append((height, prev_extPoly.centroid))

        prev_externaPolygons = externalPolygons
        
        if(len(continued_extPolys1) > 0):
            externalPolygons += continued_extPolys1
            tmpList = shapely.ops.unary_union(externalPolygons)
            externalPolygons = ListAndValidatePolygons(tmpList)
            
        if(len(continued_extPolys2) > 0):
            externalPolygons += continued_extPolys2
            tmpList = shapely.ops.unary_union(externalPolygons)
            externalPolygons = ListAndValidatePolygons(tmpList)

        externalPolygonLayerList[plane_id] += externalPolygons
    #
    # merge support with basic polygons
    #
    
    LayerMergedPolygonListList = []
    if(PrintStatusInSubFunctions):
        logging.info("  #" + str(wID) + ": Merge plugs with basic polygons")
    for plane_id in range(0, len(externalPolygonLayerList)):
        if(PrintStatusInSubFunctions):
            logging.info("  #" + str(wID) + ": Plane " + str(plane_id) + " ")
        z = printer_dz * plane_id
        
        LayerMergedPolygonList = []
        # ostateczny poligon jako ograniczony zewnerzem supportem i z wycietymi dziurami (zewnetrze PoligonssHoles)
        for externalPolygon in externalPolygonLayerList[plane_id]:
            poligonSupported = shapely.geometry.Polygon(externalPolygon.exterior, [ph.exterior for ph in PoligonssHoles[plane_id]])
            if(poligonSupported.geom_type == 'Polygon'):
                poligonSupported = poligonSupported if poligonSupported.is_valid else poligonSupported.buffer(0)
                if(poligonSupported.geom_type == 'Polygon'):
                    if(poligonSupported.area > 0):
                        LayerMergedPolygonList.append(poligonSupported)
                elif(poligonSupported.geom_type == 'MultiPolygon'): # zdarzylo sie ze po buffer z poligonu zrobił sie multipoligon
                    tmpList = [( ph if ph.is_valid else ph.buffer(0)) for ph in list(poligonSupported)]
                    for ph in tmpList:
                        if(ph.area>0):
                            if(ph.geom_type != 'Polygon'):
                                not_implemented = 1
                            LayerMergedPolygonList.append(ph)
            else:
                tmpList = [( ph if ph.is_valid else ph.buffer(0)) for ph in list(poligonSupported)]
                for ph in tmpList:
                    if(ph.area>0):
                        if(ph.geom_type != 'Polygon'):
                            not_implemented = 1
                        LayerMergedPolygonList.append(ph)


        postprocessdMergedPolygonsList = []
        if(postprocessMergedPoligons):
            for poly in LayerMergedPolygonList:
                polyoutline = poly.exterior
                polyinlines = poly.interiors
                if simplification_limit != 0:
                    polyoutline = polyoutline.simplify(simplification_limit, preserve_topology=False)
                    polyoutline = getOnlyValidLines(polyoutline, addStartPointToRings, delete_lines_shorter_than)

                    lineList = []
                    for line in polyinlines:
                        line  = line.simplify (simplification_limit, preserve_topology=False)
                        lineList  += getOnlyValidLines(line,  addStartPointToRings, delete_lines_shorter_than)
                    polyinlines = lineList
                    for polyout in polyoutline:
                        postrocessedPoly = shapely.geometry.Polygon(polyout,polyinlines)
                        postprocessdMergedPolygonsList += ListAndValidatePolygons_v2(postrocessedPoly)
                
        if(postprocessMergedPoligons):
            LayerMergedPolygonListList.append(postprocessdMergedPolygonsList)
        else:
            LayerMergedPolygonListList.append(LayerMergedPolygonList)

    return LayerMergedPolygonListList

def extrude_planes(wID, LayerMergedPolygonListList, start_layer, last_layer, mesh_minz, mesh_maxz, printer_dx, printer_dz, wall_thickness, addStartPointToRings, resolution, joint_style, simplification_limit, delete_lines_shorter_than, postprocessMergedPoligons, show_layer, plug_buffer_size, show_cups, do_repairs_pack, PrintStatusInSubFunctions):

    if(True):
        logging.info("  #WID{}, layers {}-{})".format(wID, start_layer, last_layer))
    extrudedLayersList = []
    #tmScene = trimesh.scene.scene.Scene()
    max_layer_id = last_layer if len(LayerMergedPolygonListList) > last_layer else len(LayerMergedPolygonListList)-1
    for plane_id in range(start_layer, max_layer_id+1):
        if(PrintStatusInSubFunctions):
            logging.info("  #" + str(wID) + ": Plane " + str(plane_id) + " ")
        z = printer_dz * (plane_id)
        extrudedLayersList.append([])
        
        Polygons = LayerMergedPolygonListList[plane_id]
        if(len(Polygons) > 0):
            #create extruded layer
            for polygonId in range(0, len(Polygons)):
                myExtrudedPolygon = trimesh.primitives.Extrusion(polygon = Polygons[polygonId],height= printer_dz+0.05)
                myExtrudedPolygon.slide(z)
                extrudedLayersList[-1].append(myExtrudedPolygon)
    #tmScene.show()
    return extrudedLayersList

    
def merge_extrude_planes(wID, extrudedLayersList, start_layer, last_layer, mesh_minz, printer_dx, printer_dz, wall_thickness, addStartPointToRings, resolution, joint_style, simplification_limit, delete_lines_shorter_than, postprocessMergedPoligons, show_layer, plug_buffer_size, show_cups, do_repairs_pack, PrintStatusInSubFunctions):

    last_layer = last_layer if (last_layer < len(extrudedLayersList)) else (len(extrudedLayersList)-1)
    unioning_layer_by_layer = False
    if (unioning_layer_by_layer):
        unionedLayers = []
        for plane_id in range(start_layer, last_layer+1):
            if(PrintStatusInSubFunctions):
                logging.info("  #" + str(wID) + ": Plane " + str(plane_id) + " with " + str(len(extrudedLayersList[plane_id])) + " objects")
            if(len(extrudedLayersList[plane_id]) == 1):
                unionedLayers.append(extrudedLayersList[plane_id][0])
            elif(len(extrudedLayersList[plane_id]) > 1):
                unionedLayer = extrudedLayersList[plane_id][0].union(extrudedLayersList[plane_id][1:], engine="blender")
                unionedLayers.append(unionedLayer)
        my_union = unionedLayers[0]
        for plane_id in range(1, len(unionedLayers)):
            if(PrintStatusInSubFunctions):
                logging.info("  Plane " + str(plane_id))
            #unionedLayers[plane_id].show()
            my_union = my_union.union([unionedLayers[plane_id],], engine="blender")
    else :
        if(PrintStatusInSubFunctions):
            logging.info("  #" + str(wID) + ": Merge " + str(last_layer - start_layer + 1) + " layers")
        list_all = [item for sublist in extrudedLayersList[start_layer: last_layer+1] for item in sublist]
        if(len(list_all) > 1):
            my_union = list_all[0].union(list_all[1:], engine="scad")
        elif(len(list_all) == 1):
            my_union = list_all[0]
        else:
            return None
        if(do_repairs_pack):
            trimesh.repair.fix_inversion(my_union)
            trimesh.repair.fix_normals(my_union)
            trimesh.repair.broken_faces(my_union)
            trimesh.repair.fix_winding(my_union)
    return my_union

def add_meshes(i, list_of_meshes, start_layer, last_layer ):
    if(True):
        logging.info("  #WID{}, layers {}-{})".format(i, start_layer, last_layer))
    
    last_layer_checked = last_layer if (last_layer < len(list_of_meshes)) else len(list_of_meshes)-1
    work_list = copy.deepcopy(list_of_meshes[start_layer : last_layer_checked+1])
    more_to_do = len(work_list) > 1
    while (more_to_do):
        tmp_list = []
        while (more_to_do):
            if(len(work_list) == 1):
                tmp_list.append(work_list.pop(0))
                more_to_do = False
            else:
                tmp_list.append(work_list.pop(0) + work_list.pop(0))
                more_to_do = len(work_list) >= 1
        work_list = tmp_list 
        more_to_do = len(work_list) > 1
        
    return work_list[0]

def mesh_to_png(mesh,fileName):
    figure = plt.figure()
    ax = mplot3d.Axes3D(figure)
    ax.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh.vectors))
    ax.set_xlim(mesh.min_[0],mesh.max_[0])
    ax.set_ylim(mesh.min_[1],mesh.max_[1])
    ax.set_zlim(mesh.min_[2],mesh.max_[2])
    plt.savefig(fileName)

##############################################################################
# MAIN
##############################################################################
def main():
    parser = ArgumentParser()
    parser.add_argument("-osDir",  "--os_dir",      dest="os_dir",      help="output stl directory",    required=True ,  metavar="PATH")
    parser.add_argument("-ssDir",  "--ss_dir",      dest="ss_dir",      help="shell  stl directory",    required=True ,  metavar="PATH")
    parser.add_argument("-cfgDir", "--cfg_dir",     dest="cfg_dir",     help="configs directory",       required=False,  metavar="PATH")
    parser.add_argument("-v",      "--verbose",     dest="verbose",     help="verbose level",           required=False,  )
    parser.add_argument("-fn",     "--file_name",   dest="file_name",   help="output files name",       required=False,  )
    
    args = parser.parse_args()
    
    verbose = False if args.verbose is None else args.verbose
   
    input_dir           = args.os_dir
    output_dir          = args.ss_dir
    config_dir          = args.cfg_dir
    output_file         = strftime("%d%b%Y%H%M", gmtime())  if args.file_name is None else args.file_name
    output_file         += "_shell"
    output_png_file     = output_file.split()[0] + ".png"
    output_file         += ".stl"
    output_file_full    = os.path.join(output_dir, output_file)
    log_file_full       = os.path.join(output_dir, "log.txt")
    output_png_file_full= os.path.join(output_dir, output_png_file)
    
    input_dir   = os.path.abspath(input_dir)
    output_dir  = os.path.abspath(output_dir)
    config_dir  = os.path.abspath(config_dir) if config_dir is not None else output_dir
    
    if not os.path.isdir(output_dir):
        try:
            pathlib.Path(output_dir).mkdir(mode=0o775, parents=True, exist_ok=True)
        except Exception as err:
            logging.error("Output dir IO error: {}".format(err))
            sys.exit(1)
            
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(log_file_full,mode='w'),logging.StreamHandler(sys.stdout)])
        
    if not(verbose is None):
        logging.info("===============================================")
    
    if not os.path.isdir(input_dir):
        logging.error('Input directory (%s) with 3D model files not found !',input_dir)
        exit(-1)
        
    
    if not os.path.isdir(config_dir):
        try:
            pathlib.Path(config_dir).mkdir(mode=0o775, parents=True, exist_ok=True)
        except Exception as err:
            logging.error("Config dir IO error: {}".format(err))
            sys.exit(1)
    #-----------------------------------------------------------------------------------------
    input_files_list = glob.glob(input_dir + '/*.stl')
    input_files_list.sort();  
    file_list = []
    for filename in input_files_list:
        file_list.append(os.path.basename(filename))
    file_list.sort()
    if not(verbose is None):
        logging.info("List of detected 3D STL files: {}".format(file_list))
    if(len(file_list)==0):
        logging.error("No STL file in the intput dir.")
        sys.exit(1)
    input_file_full = os.path.join(input_dir, file_list[0])
    input_file = os.path.split(input_file_full)[1]
    
    #-----------------------------------------------------------------------------------------
    config_files_list = glob.glob(config_dir + '/*.json')
    config_files_list.sort();  
    file_list = []
    for filename in config_files_list:
        file_list.append(os.path.basename(filename))
    file_list.sort()
    logging.info("List of detected config files: {}".format(file_list))
    if(len(file_list)==0):
        logging.warning("No config file in the intput dir. I will use default configuration and save it to the config_dir.")
        config_file_full    = None
        config_file         = None
    else:
        config_file_full    = os.path.join(config_dir, file_list[0])
        config_file         = os.path.split(config_file_full)[1]
    
    #-----------------------------------------------------------------------------------------
    logging.info("Dir with 3D model             : {}".format(input_dir      ))
    logging.info("         3D model file        : {}".format(input_file     ))
    logging.info("Dir with configuration        : {}".format(config_dir     ))
    logging.info("         configuration file   : {}".format(config_file    ))
    logging.info("Dir with output 3D shell      : {}".format(output_dir     ))
    logging.info("         output 3D shell file : {}".format(output_file    ))
    logging.info("         output 3D png file   : {}".format(output_png_file))
    
    threadsNum                    = os.cpu_count()
    maxNumOfThreads               = threadsNum-2
    logging.info("Detected {} avilable threads, use {} threads".format(threadsNum, maxNumOfThreads))
    #-----------------------------------------------------------------------------------------
    #default configuration:
    config = {}
        
    config["use_threads"]                   = True # przykładowe przetwarzanie na 4 watkach trwalo 38s, a na jednym 107s
                
    config["printer_dz"]                    = 0.3
    config["printer_dx"]                    = 0.5
    config["wall_thickness"]                = 4; # num of printer_dx

    config["do_support_box"]                = False#True # dodaj support schodzacy sie pod katem shrinking_angle
    config["shrinking_angle"]               = 30
    config["min_support_diameter_at_hight"] = 5 # minimalna średnica podpory w mm pzy obiekcie
    config["min_support_diameter_at_bed"]   = 25 # minimalna średnica podpory w mm na stole
    
    config["do_plugs"]                      = True # zatkaj dziury z poprzedniej warstwy odsloniete przez kolejna warstwe (wynik skonczonej szerokosci warstwy i mozliwie płaskej cientej powierzchni 
    config["do_plugs_at_top"]               = False # Dodaj warstwy nad obiektem na potrzeby zatkania górnego otworu
    config["plug_buffer_size"]              = config["printer_dx"]*2 # rozmiar zakladki dla zatyczek (tylko gdy do_plugs = True)
    config["add_number_of_plug_planes"]     = 4; # dodaje tyle warstw nad dziurą
    config["show_cups"]                     = False # debug, zadziala tylko dla use_threads=False
        
    config["joint_style"]                   = 1 # 1-round najlepszy, 2, 3 - bardziej kanciaste
    config["resolution"]                    = 16 # default
    config["simplification_limit"]          = 0.5 # upraszcza łamane obrysow dopuszczajac maksymalny blad odchylenia lamanej o simplification_limit milimetrow od oryginalu
    config["addStartPointToRings"]          = True # przenowi poczatek lamanej na prosty odcinek, dzieki czemu unika scietych lamanych po operacji offset (bugfix)
    config["delete_lines_shorter_than"]     = 1 # olej linie obrysów ktore sa krotsze niz parametr delete_lines_shorter_than. Czesto sa to linie powstale z przeciecia poligonow.
    config["postprocessMergedPoligons"]     = False # po polaczeniu poligonu warstwy z poligonem zatyczek, jeszcze raz uprasza otrzymany poligon 
    config["do_repairs_pack"]               = True # w trimesh sa dostepne narzedzia do naprawy mesh'ow. Dla spokoju uzyj ich po wygenerowaniu mesh dla kazdej warstwy
        
    config["show_layer"]                    = False # debug, zadziala tylko dla use_threads=False
    config["MergeExtrudedLayers"]           = False # bardzo czasochlonne laczenie warstw za pomoca polecenia union (wykonywane w scad lub blender). Jezeli False to tworzy STL z kazda warstwa jako osobny obiekt - sli3er radzi sobie z takim plikiem
    config["PrintStatusInSubFunctions"]     = not(verbose is None) and (verbose == "all")

    #-----------------------------------------------------------------------------------------
    # configuration from file:
    if not(config_file_full is None):
        logging.info("Load configuration from file")
        with open (config_file_full) as f:
            config_data= json.load(f)
        config.update(config_data)
            
    else:
        logging.info("Use default configuration")
        
        if not(config_dir is None):
            logging.info("Dump configuration to file")
            config_file_full = os.path.normpath(config_dir + '/' + 'default_printer_config.json')
            fjson = open(os.open(config_file_full, os.O_CREAT | os.O_WRONLY, 0o664), 'w')

            if not(fjson is None):
                 json.dump(config, fjson, indent=4)
                 
            fjson.flush()
            fjson.close()
    
    logging.info(" Configuration: {}".format(config))
        
    use_threads                     = config["use_threads"                    ]
    printer_dz                      = config["printer_dz"                     ]
    printer_dx                      = config["printer_dx"                     ]
    wall_thickness                  = config["wall_thickness"                 ]
    do_support_box                  = config["do_support_box"                 ]
    shrinking_angle                 = config["shrinking_angle"                ]
    min_support_diameter_at_hight   = config["min_support_diameter_at_hight"  ]
    min_support_diameter_at_bed     = config["min_support_diameter_at_bed"    ]
    do_plugs                        = config["do_plugs"                       ]
    do_plugs_at_top                 = config["do_plugs_at_top"                ]
    plug_buffer_size                = config["plug_buffer_size"               ]
    add_number_of_plug_planes       = config["add_number_of_plug_planes"      ]
    show_cups                       = config["show_cups"                      ]
    joint_style                     = config["joint_style"                    ]
    resolution                      = config["resolution"                     ]
    simplification_limit            = config["simplification_limit"           ]
    addStartPointToRings            = config["addStartPointToRings"           ]
    delete_lines_shorter_than       = config["delete_lines_shorter_than"      ]
    postprocessMergedPoligons       = config["postprocessMergedPoligons"      ]
    do_repairs_pack                 = config["do_repairs_pack"                ]
    show_layer                      = config["show_layer"                     ]
    MergeExtrudedLayers             = config["MergeExtrudedLayers"            ]
    PrintStatusInSubFunctions       = config["PrintStatusInSubFunctions"      ]

    ##############################################################################
    # load a file by name or from a buffer
    logging.info("-----------------------------------------------")
    logging.info("Loading mesh file \"" + input_file + "\"")
    
    start_time = time.time()
    start_time_total = start_time
    
    mesh = trimesh.load(input_file_full)
    mesh_bounds = mesh.bounds
    mesh_minz = mesh_bounds[0][2]
    mesh_maxz = mesh_bounds[1][2]
    planes_num = int((mesh_maxz - mesh_minz) / printer_dz)

    if use_threads:
        pool = Pool(processes = maxNumOfThreads)

    elapsed_time = time.time() - start_time
    logging.info(" File loaded in " + str(round(elapsed_time, 2)) + "s")
    
    ##############################################################################
    logging.info("-----------------------------------------------")
    logging.info("Slicing object into "+ str(planes_num) + " planes ...")
    start_time = time.time()
    
    layers_per_thread = math.floor((planes_num+(maxNumOfThreads-1)) / maxNumOfThreads)
    
    if(use_threads):    
        # launching multiple evaluations asynchronously *may* use more processes
        multiple_results = [pool.apply_async(slice_mesh, (i, mesh, i*layers_per_thread, min((i+1)*layers_per_thread-1, planes_num-1), do_plugs_at_top, mesh_minz, mesh_maxz, printer_dx, printer_dz, wall_thickness, addStartPointToRings, resolution, joint_style, simplification_limit, delete_lines_shorter_than, PrintStatusInSubFunctions)) for i in range(maxNumOfThreads)]
        results = [res.get() for res in multiple_results]
    else:
        results = []
        # launching multiple evaluations asynchronously *may* use more processes
        for i in range(maxNumOfThreads):
            start_layer_id  = i     *layers_per_thread
            stop_layer_id   = (i+1) *layers_per_thread - 1
            #start_layer_id = stop_layer_id = 125
            results.append(slice_mesh(i, mesh, start_layer_id, stop_layer_id, do_plugs_at_top, mesh_minz, mesh_maxz, printer_dx, printer_dz, wall_thickness, addStartPointToRings, resolution, joint_style, simplification_limit, delete_lines_shorter_than, PrintStatusInSubFunctions))


    PoligonsHollow = []
    PoligonssFull = []
    PoligonssHoles = []
    for i,j,k in results:
        PoligonsHollow += i
        PoligonssFull += j
        PoligonssHoles += k
    
    elapsed_time = time.time() - start_time
    logging.info(" Slicing object took " + str(round(elapsed_time, 2)) + "s")
    
    LayerMergedPolygonListList = PoligonsHollow
    ##############################################################################
    #
    # stworz zatyczki
    #
    
    logging.info("-----------------------------------------------")
    logging.info("Create plugs...")
    start_time = time.time()
    
    if(do_plugs and do_plugs_at_top):
        planes_num += add_number_of_plug_planes
        PoligonsHollow += [[] for i in range(add_number_of_plug_planes)]
        PoligonssFull += [[] for i in range(add_number_of_plug_planes)]
        PoligonssHoles += [[] for i in range(add_number_of_plug_planes)]
        
    if(do_plugs):
        layers_per_thread = math.floor((planes_num+(maxNumOfThreads-1)) / maxNumOfThreads)

        hole_layer_dz_set = [i for i in range(-add_number_of_plug_planes, add_number_of_plug_planes+1)] 
        (hole_layer_dz_set).remove(0) # filter out zero
        plug_margin_set = [round( (1 - 0.6 * (abs(dz)/add_number_of_plug_planes)) * plug_buffer_size, 3) for dz in hole_layer_dz_set]
        plugs_params_set = [(hole_layer_dz_set[i], plug_margin_set[i]) for i in range(0, len(hole_layer_dz_set))]
    
        for layer_offset, plug_buffer_size_for_layer in plugs_params_set:
            logging.info("  dz = " + str(layer_offset) + ", margin = " + str(plug_buffer_size_for_layer) + "...")
            if(use_threads):
                multiple_results = [pool.apply_async(create_plugs,(i, LayerMergedPolygonListList, PoligonssFull, PoligonssHoles, layer_offset, i*layers_per_thread, min((i+1)*layers_per_thread-1, planes_num-1), mesh_minz, mesh_maxz, printer_dx, printer_dz, wall_thickness, addStartPointToRings, resolution, joint_style, simplification_limit, delete_lines_shorter_than, postprocessMergedPoligons, show_layer, plug_buffer_size_for_layer, PrintStatusInSubFunctions)) for i in range(maxNumOfThreads)]
                results = [res.get() for res in multiple_results]
            else:
                results = []
                # launching multiple evaluations asynchronously *may* use more processes
                for i in range(maxNumOfThreads):
                    results.append(create_plugs(i, LayerMergedPolygonListList, PoligonssFull, PoligonssHoles, layer_offset, i*layers_per_thread, min((i+1)*layers_per_thread-1, planes_num-1), mesh_minz, mesh_maxz, printer_dx, printer_dz, wall_thickness, addStartPointToRings, resolution, joint_style, simplification_limit, delete_lines_shorter_than, postprocessMergedPoligons, show_layer, plug_buffer_size_for_layer, PrintStatusInSubFunctions))

            LayerMergedPolygonListList = []
            for l in results:
                LayerMergedPolygonListList += l
    
        elapsed_time = time.time() - start_time
        logging.info(" Creating plugs took " + str(round(elapsed_time, 2)) + "s")
    
    ##############################################################################
    if(do_support_box):
        logging.info("-----------------------------------------------")
        logging.info("Create support (always on single thread)...")
        start_time = time.time()

        LayerMergedPolygonListList = create_support(i, LayerMergedPolygonListList, PoligonssFull, PoligonssHoles, min_support_diameter_at_hight, min_support_diameter_at_bed, shrinking_angle, mesh_minz, printer_dx, printer_dz, wall_thickness, addStartPointToRings, resolution, joint_style, simplification_limit, delete_lines_shorter_than, postprocessMergedPoligons, show_layer, plug_buffer_size_for_layer, PrintStatusInSubFunctions)

        elapsed_time = time.time() - start_time
        logging.info(" Creating support took " + str(round(elapsed_time, 2)) + "s")
    ##############################################################################
    #
    # rozciagnij poligony  wzdłóż osi Z (2D -> 3D)
    #
    logging.info("-----------------------------------------------")
    logging.info("Extruding planes...")
    start_time = time.time() 
    
    if(use_threads):
        multiple_results = [pool.apply_async(extrude_planes,(i, LayerMergedPolygonListList, i*layers_per_thread, min((i+1)*layers_per_thread-1, planes_num-1), mesh_minz, mesh_maxz, printer_dx, printer_dz, wall_thickness, addStartPointToRings, resolution, joint_style, simplification_limit, delete_lines_shorter_than, postprocessMergedPoligons, show_layer, plug_buffer_size, show_cups, do_repairs_pack, PrintStatusInSubFunctions)) for i in range(maxNumOfThreads)]
        results = [res.get() for res in multiple_results]
    else:
        results = []
        # launching multiple evaluations asynchronously *may* use more processes
        for i in range(maxNumOfThreads):
            results.append(extrude_planes(i, LayerMergedPolygonListList, i*layers_per_thread, min((i+1)*layers_per_thread-1, planes_num-1), mesh_minz, mesh_maxz, printer_dx, printer_dz, wall_thickness, addStartPointToRings, resolution, joint_style, simplification_limit, delete_lines_shorter_than, postprocessMergedPoligons, show_layer, plug_buffer_size, show_cups, do_repairs_pack, PrintStatusInSubFunctions))

    extrudedLayersList = []
    for l in results:
        extrudedLayersList += l
        
    elapsed_time = time.time() - start_time
    logging.info(" Extruding took " + str(round(elapsed_time, 2)) + "s")
    
    ##############################################################################
    if(not MergeExtrudedLayers):
    ##############################################################################
        logging.info("-----------------------------------------------")
        logging.info("Adding all layers...")
        start_time = time.time()
        list_all = [item for sublist in extrudedLayersList for item in sublist]

        
        if(use_threads):
            logging.info(" in separate threads...")
            layers_per_thread = math.floor((len(list_all) + maxNumOfThreads - 1) / maxNumOfThreads)
            multiple_results = [pool.apply_async(add_meshes,(i, list_all[0:], i*layers_per_thread, min((i+1)*layers_per_thread-1, planes_num-1), )) for i in range(maxNumOfThreads)]
            results = [res.get() for res in multiple_results]
            logging.info(" main thread...")
            sum_of_unions = add_meshes(0, results[0:], 0, len(results)+1)
            if(use_threads):
                    pool.close()
                    pool.join()
        else:
            sum_of_unions = list_all[0]
            layer_id = 0
            for next_layer in list_all[1:]:
                layer_id += 1
                try:
                    sum_of_unions += next_layer
                except Exception as e:
                    logging.error( "Eception when adding layer number " + str(layer_id) + ":") 
                    logging.error(e)
        elapsed_time = time.time() - start_time
        logging.info(" took " + str(round(elapsed_time, 2)) + "s")
    ##############################################################################
    else:
    ##############################################################################
        logging.info("-----------------------------------------------")
        logging.info("Merging extruded planes stage 1 (scad)...")
        start_time = time.time()
    
        if(use_threads):
            multiple_results = [pool.apply_async(merge_extrude_planes,(i, extrudedLayersList, i*layers_per_thread, min((i+1)*layers_per_thread-1, planes_num-1), mesh_minz, mesh_maxz, printer_dx, printer_dz, wall_thickness, addStartPointToRings, resolution, joint_style, simplification_limit, delete_lines_shorter_than, postprocessMergedPoligons, show_layer, plug_buffer_size, show_cups, do_repairs_pack, PrintStatusInSubFunctions)) for i in range(maxNumOfThreads)]
            results = [res.get() for res in multiple_results]
        else:
            results = []
            # launching multiple evaluations asynchronously *may* use more processes
            for i in range(maxNumOfThreads):
                results.append(merge_extrude_planes(i, extrudedLayersList, i*layers_per_thread, min((i+1)*layers_per_thread-1, planes_num-1), mesh_minz, mesh_maxz, printer_dx, printer_dz, wall_thickness, addStartPointToRings, resolution, joint_style, simplification_limit, delete_lines_shorter_than, postprocessMergedPoligons, show_layer, plug_buffer_size, show_cups, do_repairs_pack, PrintStatusInSubFunctions))
    
        if(use_threads):
                pool.close()
                pool.join()

        unions = list(filter(None, results))

        elapsed_time = time.time() - start_time
        logging.info(" Scad took " + str(round(elapsed_time, 2)) + "s")

        ##############################################################################
        logging.info("Saving intermediate stls...")
        start_time = time.time()
        unionId = 0
        for my_union in unions:
            file_name = "Out_simpLimit" + str(simplification_limit)
            file_name += "_dz" + str(printer_dz)
            file_name += "_jointStyle" + str(joint_style)
            file_name += "_resol" + str(resolution)
            file_name += "_wallTh" + str(wall_thickness)
            file_name += "_#" + str(unionId)
            file_name +=  ".stl"
            logging.info("  " + file_name + "...")
            my_union.export(file_name)
            unionId +=1
        elapsed_time = time.time() - start_time
        logging.info(" Saving took " + str(round(elapsed_time, 2)) + "s")
            
        ##############################################################################
        logging.info("Merging extruded planes stage 2 (final) (scad)...")
        #start_time = time.time()
    
        if(MergeExtrudedLayers):
            unions_as_list_of_lists = [[i] for i in unions ]
            sum_of_unions = merge_extrude_planes(-1, unions_as_list_of_lists, 0, len(unions_as_list_of_lists)-1, mesh_minz, mesh_maxz, printer_dx, printer_dz, wall_thickness, addStartPointToRings, resolution, joint_style, simplification_limit, delete_lines_shorter_than, postprocessMergedPoligons, show_layer, plug_buffer_size, show_cups, do_repairs_pack, PrintStatusInSubFunctions)
        else:
            sum_of_unions = unions[0]
            for next_union in unions[1:]:
                sum_of_unions += next_union

        elapsed_time = time.time() - start_time
        logging.info(" Scad took " + str(round(elapsed_time, 2)) + "s")

    ##############################################################################
    logging.info("-----------------------------------------------")
    logging.info("Saving PNG of final stl {}".format(output_png_file))
    #logging.info("sum_of_unions: {}".format(sum_of_unions))
    scene = sum_of_unions.scene()
    
    # a 45 degree homogenous rotation matrix around
    # the Y axis at the scene centroid
    rotate = trimesh.transformations.rotation_matrix(
        angle=np.radians(0.0),
        direction=[0, 1, 0],
        #point=scene.centroid)
        point = sum_of_unions.center_mass)
        
    #for i in range(4):
    # rotate the camera view transform
    camera_old, _geometry = scene.graph[scene.camera.name]
    camera_new = np.dot(camera_old, rotate)

    # apply the new transform
    scene.graph[scene.camera.name] = camera_new
    
    # saving an image requires an opengl context, so if -nw
    # is passed don't save the image
    try:
        # save a render of the object as a png
        png = scene.save_image(resolution=[256, 256], visible=True)
        with open(output_png_file_full, 'wb') as f:
            f.write(png)
            f.close()
    except BaseException as E:
        logging.error("unable to save image", str(E))
    #mesh_to_png(sum_of_unions, output_png_file_full)
    ##############################################################################
    logging.info("-----------------------------------------------")
    logging.info("Saving final stl {}".format(output_file))
    start_time = time.time()
    sum_of_unions.export(output_file_full)
        
    
    elapsed_time = time.time() - start_time
    logging.info(" Saving took " + str(round(elapsed_time, 2)) + "s")
    ##############################################################################
    elapsed_time = time.time() - start_time_total
    logging.info("Total time " + str(round(elapsed_time, 2)) + "s")
    logging.info("===============================================")
    

if __name__ == '__main__':
    main()
