import open3d as o3d
import numpy as np
from argparse import ArgumentParser
import os
import logging
import sys, getopt

def Run(iDir,verbose):
    try:
        #xyz_fn = os.path.normpath(iDir+"/skinlines/straight_point_cloud.xyz")
        xyz_fn = os.path.normpath(iDir+"/skinlines/point_cloud.xyz")
        pcd = o3d.io.read_point_cloud(xyz_fn)
    except Exception as err:
        logging.error('opening file "%s" failed, error "%s"'%(xyz_fn,err))
        exit(1)
    #o3d.visualization.draw_geometries([pcd])

    number_of_inliers = len(pcd.points)
    number_of_points = len(pcd.points)

    while number_of_inliers > number_of_points/15:
        plane_model, inliers = pcd.segment_plane(distance_threshold=2.0,
                                                 ransac_n=3,
                                                 num_iterations=1000)
        [a, b, c, d] = plane_model
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        number_of_inliers = len(inliers)
        inlier_cloud = pcd.select_by_index(inliers)
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        outlier_cloud.paint_uniform_color([0, 1.0, 0])

        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
        pcd = outlier_cloud


parser = ArgumentParser()
parser.add_argument("-iDir",      "--input_dir"      ,     dest="idir"   ,    help="input directory" ,    metavar="PATH", required=True)
parser.add_argument("-v"   ,      "--verbose"        ,     dest="verbose",    help="verbose level"   ,                    required=False)

args = parser.parse_args()

verbose = 'off'                 if args.verbose is None else args.verbose
iDir 	= args.idir

logging.basicConfig(level=logging.DEBUG,format="%(asctime)s [%(levelname)s] %(message)s",handlers=[logging.FileHandler(iDir+"/showStraight.log",mode='w'),logging.StreamHandler(sys.stdout)])


if not os.path.isdir(iDir):
    logging.error('Error : Input directory (%s) not found !',iDir)
    exit(1)



logging.info("\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")
logging.info("START:     as_showStraight.py")
logging.info("in:       "    +   iDir    )
logging.info("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n")


if verbose == 'off':
    verbose = False
else:
    verbose = True

    
Run(iDir, verbose)
