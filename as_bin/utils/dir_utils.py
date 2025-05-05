import os, sys
import pathlib
import glob
import logging
#----------------------------------------------------------------------------
def createDirIfNeeded(dir, err_list=None):
    try:
        if not os.path.isdir(dir):
            pathlib.Path(dir).mkdir(mode=0o775, parents=True, exist_ok=True)
            logging.info("  Created dir {}".format(dir))
    except Exception as err:
        err = { "dir": dir,
                "msg":"  Creating dir ({}) IO error: {}".format(dir, err)
                }
        logging.error(err["msg"])
        if not err_list is None:
            err_list.append(err)
            return False
        else:
            sys.exit(1)
    return True

        
def move_between_dirs(src_dir, dst_dir, glob_mask):
    src_dir = os.path.normpath(src_dir)
    dst_dir = os.path.normpath(dst_dir)
    fns = glob.glob(os.path.join(src_dir, glob_mask))
    for fn in fns:
        dst_fn = fn.replace(src_dir, dst_dir)
        os.replace(fn, dst_fn)
   
#----------------------------------------------------------------------------     
        
def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)