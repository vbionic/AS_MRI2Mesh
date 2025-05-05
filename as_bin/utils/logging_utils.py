import os, sys
import pathlib
import logging
#----------------------------------------------------------------------------
def redirect_log(rdir, fn1, fn2, copy_and_remove_prev = True):
    lDir = os.path.join(rdir, "_log")
    try:
        if not os.path.isdir(lDir):
            pathlib.Path(lDir).mkdir(mode=0o775, parents=True, exist_ok=True)
            logging.info("Created log dir {}".format(lDir))
    except Exception as err:
        logging.error("Creating dir ({}) IO error: {}".format(lDir, err))
        sys.exit(1)

    log = logging.getLogger()  # root logger
    prev_log_format = log.handlers[0].formatter._fmt
    filec_fn = lDir+f"/{fn1}"
    filehc = logging.FileHandler(filec_fn, 'w')
    filehc.setFormatter(logging.Formatter(prev_log_format))
    filel_fn = lDir+f"/{fn2}"
    filehl = logging.FileHandler(filel_fn, 'w')
    filehl.setFormatter(logging.Formatter(prev_log_format))

    #logging_level = tr_args.logging_level
    #log.setLevel(logging_level)
    for hdlr in log.handlers[:]:  # remove all old handlers
        if(type(hdlr) is logging.FileHandler):
            old_log_fn = hdlr.baseFilename 
            hdlr.close()
            log.removeHandler(hdlr)
            if copy_and_remove_prev:
                with open(old_log_fn, 'r') as f:
                    lines = f.read()
                os.remove(old_log_fn)
                filehc.stream.writelines(lines)
                filehl.stream.writelines(lines)
                
    for fn in [filec_fn, filel_fn]:
        try:
            os.chmod(fn, 0o666)
        except:
            logging.warning(f"Could not change log file permitions {fn}. I'm not the owner of the file?")
        
    log.addHandler(filehc)      # set the new handler
    log.addHandler(filehl)      # set the new handler
    
    # start new logging
    logging.info("change log file to {} and {}".format(filehc.baseFilename, filehl.baseFilename))
    return lDir