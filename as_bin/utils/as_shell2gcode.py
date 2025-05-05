import os
import pathlib
from argparse import ArgumentParser
import glob
from time import gmtime, strftime
from PIL import Image, ImageDraw, ImageFont
#-----------------------------------------------------------------------------------------
#MAIN
#-----------------------------------------------------------------------------------------
if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument("-shellDir",  "--shell_dir",    dest="sh_dir",       help="input STL directory",                     metavar="PATH", required=True)
    parser.add_argument("-iniFile",   "--ini_file",     dest="ini_file",     help="input printer config file",     required=True)
    parser.add_argument("-gcodeDir",  "--gcode_dir",    dest="gc_dir",       help="output gco directory",                    metavar="PATH", required=True)
    parser.add_argument("-v",         "--verbose",      dest="verbose",      help="verbose level",                                           required=False)
    parser.add_argument("-fn",        "--file_name",    dest="file_name",    help="output files name",                                       required=False)
    
    args = parser.parse_args()
    
    verbose = 'off'                 if args.verbose is None else args.verbose
   
    idDirStl      = args.sh_dir
    idFileConf    = args.ini_file
    outDir  	  = args.gc_dir
    filesName     =    "3DModel"+strftime("%d%b%Y%H%M", gmtime())                 if args.file_name is None else args.file_name+"_surface"
    
    idDirStl = os.path.normpath(idDirStl)
    idFileConf = os.path.normpath(idFileConf)
    outDir = os.path.normpath(outDir)
    
    if not os.path.isdir(idDirStl):
        print('Error : Input directory (%s) with STL files not found !',idDirStl)
        print("dupa")
        exit(-1)
    #if not os.path.isdir(idDirConf):
    #    print('Error : Input directory (%s) with config files not found !',idDirSpacing)
    #    exit(-1)
    try:
        if not os.path.isdir(outDir):
            pathlib.Path(outDir).mkdir(mode=0o775, parents=True, exist_ok=True)
    except Exception as err:
        print("Output dir IO error: {}".format(err))
        exit(1)
    
    print("Dir with stl file    : "+idDirStl)
    print("Dir with output file : "+outDir)
    print("ini file             : "+idFileConf)
    try:
        inputFile = glob.glob(idDirStl + '/*shell.stl')[0]
    except IndexError:
        print("STL file to slice not found")
        exit(1)
    if not os.path.isfile(idFileConf):
        print("Config file for PrusaSlicer not found")
        exit(1)

    xname           = os.path.basename(inputFile)
    fname, fext     = os.path.splitext(xname)

    #try:
    #    inputConfigFile = glob.glob(idFileConf + '/*.ini')[0]
    #except IndexError:
    #    print("Config file for PrusaSlicer not found")
    #    sys.exit(1)
    #on MacOS PrusaSlicer
    #on Tesla slic3r-prusa3d
    print(inputFile)
    
    cmd="slic3r -g "+inputFile +" --load "+idFileConf+" --output "+outDir

    print(cmd)
    #ret=0
    ret = os.system(cmd)
    if ret == 0:
        try:
            outputFileName = glob.glob(outDir + '/*.gcode')[0]
        except IndexError:
            print("STL file to slice not found")
            exit(1)
        print(outputFileName)
        outputFileName=outputFileName.split("/")
        outputFileName=outputFileName[len(outputFileName)-1]
        print(outputFileName)
        outputFileName=outputFileName.split("_")
        outputFileName=outputFileName[len(outputFileName)-1]
        print(outputFileName)
        outputFileName=outputFileName.split(".")
        outputFileName=outputFileName[0]
        print(outputFileName)
        img = Image.new('RGB', (256, 256), color = (0, 0, 0))
        fnt = ImageFont.truetype('/usr/local/share/fonts/OpenImageIO/DroidSans.ttf', 56)
        d = ImageDraw.Draw(img)
        d.text((0,72), outputFileName+"\n *.gcode", font=fnt, fill=(255,55,0))
 
        img.save(outDir +"/gcode.png")
    exit(ret)
    
#-----------------------------------------------------------------------------------------
