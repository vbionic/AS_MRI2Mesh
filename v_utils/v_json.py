import json
import os
import logging

def jsonUpdate(jsonPath, dictToAdd, do_delete_before_dump = True):
    if not(type(dictToAdd) is dict):
        logging.debug(" object to add to the json {} is not a dictionary ({}). Try it anyway but it may fail.".format(jsonPath, type(dictToAdd)))
        
    logging.debug("Updating json file {}".format(jsonPath))
        
    fjson_path = os.path.normpath(jsonPath)
    if(os.path.isfile(fjson_path) and (os.stat(fjson_path).st_size != 0)):
        f= open (fjson_path)
        try:
            old_dict_data = json.load(f)
        except json.JSONDecodeError as err:
            old_dict_data = {}
            f.seek(0)
            fdata = f.read()
            if(err.pos > 0):
                logging.warning(" Json file {} with error at pos {}. Try loading from the file trimmed up to this position".format(jsonPath, err.pos))
                logging.warning(" original data: {}".format(fdata))
                trimmed = fdata[:err.pos]
                logging.warning(" trimmed data: {}".format(trimmed))
                old_dict_data = json.loads(trimmed)
                logging.warning(" parsed dict: {}".format(old_dict_data))
            else:
                old_dict_data = {}
                logging.warning(" Json file {} with errors. I skip reading.".format(jsonPath))
        except:
            old_dict_data = {}
            logging.warning(" Json file {} with errors. I skip reading.".format(jsonPath))
        finally:
            f.close()

    else:
        logging.debug(" json file {} not found or empty".format(jsonPath))
        old_dict_data = {}
    
    if not(type(dictToAdd) is dict):
        try:
            old_dict_data.update(dictToAdd)
        except:
            logging.error(" object to add to the json {} was not a dictionary ({}). Updating of the json file failed.".format(jsonPath, type(dictToAdd)))
    else:
        old_dict_data.update(dictToAdd)

    jsonDumpSafe(jsonPath, old_dict_data, do_delete_before_dump)
def convert_ndarray_to_list(obj):
    if type(obj) is dict:
        ks_vs = obj.items()
    elif type(obj) is list:
        ks_vs = list(enumerate(obj))
    else:
        return obj
    for k,v in ks_vs:
        if v.__class__.__name__ == 'ndarray':
            logging.warning(f" @jsonDumpSafe: Try convert entry of type {v.__class__.__name__} to json serializable list.")
            obj[k] = list(v)
        elif type(obj) is dict or type(obj) is list:
            obj[k] = convert_ndarray_to_list(v)
    return obj

def jsonDumpSafe(jsonPath, dictToDump, do_delete_before_dump = True, do_convert_ndarray_to_list = True):

    fjson_path = os.path.normpath(jsonPath)
    if(do_delete_before_dump):
        if(os.path.isfile(fjson_path)):
            os.remove(fjson_path)

    fjson = open(os.open(fjson_path, os.O_CREAT | os.O_WRONLY, 0o664), 'w');   

    if do_convert_ndarray_to_list:
        dictToDump = convert_ndarray_to_list(dictToDump)

    if not(fjson is None):
        json.dump(dictToDump, fjson, indent=4)
    else:
        logging.error(" The json {} does not exist.".format(jsonPath))

    fjson.flush()
    fjson.close()
        
##############################################################################
# MAIN
##############################################################################
def main():
    import time
    logging.info("cast to dictionary and save to json")
    meta_path = os.path.normpath('tmp_test_json.json');    
    jsonUpdate(meta_path, {"current_time":time.ctime()})
    
if __name__ == '__main__':
    main()