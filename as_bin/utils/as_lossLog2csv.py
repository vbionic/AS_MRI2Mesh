import collections.abc
import json
import os

##############################################################################
# MAIN
##############################################################################
def main():
    dirs=[
        #"my_tr_1comp_scale0811",
        #"my_tr_1comp_scale1010",
        #"my_tr_3comp_scale0811",
        #"my_tr_3comp_scale1010",
        #"my_tr_1comp_scale0811_wybrane",
        #"my_tr_1comp_scale1010_wybrane",
        #"my_tr_3comp_scale0811_wybrane",
        #"my_tr_3comp_scale1010_wybrane",
        "as_data_0.50mm/st05_shape_processing/my_tr_scale0811_bones_1ctim_as_datasetV1",
        "as_data_0.50mm/st05_shape_processing/my_tr_scale0811_bones_3ctim_as_datasetV1",
        "as_data_0.50mm/st05_shape_processing/my_tr_scale0811_bones_3ctim_as_datasetV1_LRc",
        "as_data_0.50mm/st05_shape_processing/my_tr_scale0811_bones_3ctim_as_dataset",
        ]
    csv_fn = "loss_log.csv"
    keys = ["loss_train", "loss_val"]

    with open(csv_fn, "w") as fcsv:
        fns_dicts = []
        fcsv.write("eid;")
        for dir in dirs:
            loss_log_fn = dir+"/loss_log.json"
            print("read from json file {}".format(loss_log_fn))
            with open (loss_log_fn) as f:
                loss_dict = json.load(f)
                fns_dicts.append(loss_dict)
            #fcsv.write("{};;".format(dir))
            
        for key in keys:
            for dir in dirs:
                fcsv.write("{}_{};".format(key,dir))

        eid = 0
        foundEpochResults = True
        while foundEpochResults:
            fcsv.write("\n{};".format(eid))
            foundEpochResults = False
            for key in keys:
                for loss_dict_data in fns_dicts:
                    eidx = "e{}".format(eid)
                    if(eidx in loss_dict_data.keys()):
                        edict = loss_dict_data[eidx]
                        val = edict[key]
                        csv_l = "{};".format(val)
                        foundEpochResults = True
                    else:
                        csv_l = ";"
                    fcsv.write(csv_l)
            eid+=1    

if __name__ == '__main__':
    main()