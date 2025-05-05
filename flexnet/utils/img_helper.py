import matplotlib.pyplot as plt
import numpy as np
import copy 
import math

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def plot_img_array_save(img_array, filename, state_dict = None):
    consider_last_N = 40
    skip_pN_highest = 0.2
    
    nrow = len(img_array[0])
    ncol = len(img_array)
    dy, dx = 0, 0
    for img in img_array[0]:
        dy, dx = [max(x) for x in list(zip([dy, dx],img.shape[0:2]))]

    #nrow = int((nimg + ncol-1) / ncol)
    if (not (state_dict is None)):#type(state_dict) is np.ndarray:
        nrow = nrow+2
    fig_dx = 1 + dx/512 * ncol*3
    fig_dy = 1 + dy/512 * nrow*3
    f = plt.figure(figsize=(fig_dx, fig_dy))#, frameon = False)#, gridspec_kw={'hspace': 0})
    #top_ax = f.add_axes([.1, .1, .8, .2])
    #f, plots = plt.subplots(nrow, ncol, sharex='all', sharey='all', figsize=(fig_dx, fig_dy), squeeze=True, gridspec_kw={'hspace': 0})
    myt = 0.05
    myb = 0.1
    fdy = 1 - myt - myb
    mxl = 0.1
    mxr = 0.05
    fdx = 1 - mxl - mxr
    fy = myb
    fx = mxl
    dy = fdy/nrow
    dx = fdx/ncol
    
    for cidx, imgs_row in enumerate(img_array):
        for ridx, img in enumerate(imgs_row):
            y = fy + fdy - (ridx+1) * dy
            x = fx       + (cidx  ) * dx
            curr_ax = f.add_axes([x, y, dx*0.95, dy*0.95])
            if(cidx != 0):
                curr_ax.set_yticks([])
            if(ridx != 0):
                curr_ax.set_xticks([])
            else:
                curr_ax.xaxis.tick_top()
            try:
                curr_ax.imshow(img)
            except:
                _ = 0

    
    y = fy  
    x = fx 
    bottom_ax = f.add_axes([x, y, fdx, dy*1.8], facecolor='w')
    max_loss_ploted = 1.0
    if(type(state_dict) is dict):
        loss_eval = []
        loss_train = []
        epochs = []
        found = True
        keys = list(state_dict.keys())
        for epoch_id in range(len(state_dict["epochs"])):
            if epoch_id==0:
                mes_keys = list(state_dict["epochs"][epoch_id]['measures'].keys())
                mes_keys_cls_loss = [key for key in mes_keys if (key.find("loss_") == 0) and (key.find("_val") != -1) and not (key.find("loss_val") == 0)]
                loss_cls_eval = [[] for _ in (mes_keys_cls_loss)]
            epochs.append(epoch_id)
            loss_train.append(state_dict["epochs"][epoch_id]['measures']['loss_train'])
            loss_eval.append(state_dict["epochs"][epoch_id]['measures']['loss_val'])
            for tid, cls_key in enumerate(mes_keys_cls_loss):
                loss_cls_eval[tid].append(state_dict["epochs"][epoch_id]['measures'][cls_key])

        min_v, max_v = 1.0, 0
        for losses in (loss_train, loss_eval):
            tail = copy.copy(losses)
            tail = [x for x in tail if not math.isnan(x)]
            tail.sort()
            if(len(tail) > 0):
                hist_len = min(len(tail), consider_last_N)
                tail = tail[0:hist_len]
                len_considered = int(hist_len*(1-skip_pN_highest))
                min_v = min(min_v, tail[0]               )
                max_v = max(max_v, tail[len_considered-1])
        range_v = max_v - min_v
        max_loss_ploted = max_v + range_v * 0.05
        min_loss_ploted = min_v - range_v * 0.05

        eval_line  = bottom_ax.plot(epochs, loss_eval,  'b-',  label="eval")
        train_line = bottom_ax.plot(epochs, loss_train, 'r--', label="train")
        loss_eval_min_val = min(loss_eval)
        loss_eval_min_epoch = loss_eval.index(loss_eval_min_val)
        min_loss = bottom_ax.plot(loss_eval_min_epoch, loss_eval_min_val, 'gx', label="{:.04}@e{}".format(loss_eval_min_val, loss_eval_min_epoch))

        for tid, cls_key in enumerate(mes_keys_cls_loss):
            bottom_ax.plot(epochs, loss_cls_eval[tid],  ':',  linewidth = 1.0, label=cls_key[5:])
        #bottom_ax.set_xticks(epochs)
        bottom_ax.set_title('Loss')
        bottom_ax.axis([ -1, epochs[-1]+1, min_loss_ploted, max_loss_ploted])
        bottom_ax.legend()
        bottom_ax.grid()
        #bottom_ax.set_xticks([])
        #bottom_ax.set_yticks([])
        #for plotr in plots:
        #    for plotc in plotr:
        #        plotc.label_outer()

    #plt.tight_layout()
    plt.savefig(filename)
    #plt.imsave(filename, fig2data(f))
    plt.close(f)


from functools import reduce
def plot_side_by_side(img_arrays, filename = None, state_dict = None):
    
    if(filename == None):
        filename = "out.png"
    plot_img_array_save(img_arrays, filename, state_dict = state_dict)

import itertools
def plot_errors(results_dict, title):
    markers = itertools.cycle(('+', 'x', 'o'))

    plt.title('{}'.format(title))

    for label, result in sorted(results_dict.items()):
        plt.plot(result, marker=next(markers), label=label)
        plt.ylabel('dice_coef')
        plt.xlabel('epoch')
        plt.legend(loc=3, bbox_to_anchor=(1, 0))

    plt.show()

def masks_to_colorimg(masks):
    colors = np.asarray([(201, 58, 64), (242, 207, 1), (0, 152, 75), (101, 172, 228),(56, 34, 132), (160, 194, 56)], dtype=np.uint8)
    channels, height, width = masks.shape

    if(masks.shape[0] == 1):
        colorimg = np.ones((masks.shape[1], masks.shape[2], 3), dtype=np.uint8) * 255
        colorimg[masks[0,:,:] > 0.5] = colors[0]
    
    else:
        colorimg_tmp = np.zeros((masks.shape[0], masks.shape[1], masks.shape[2], 3), dtype=np.uint8)
        for mid in range(masks.shape[0]):
            colorimg_tmp[mid][masks[mid,:,:] > 0.5] = colors[mid]
        colorimg_tmp_masked = np.ma.masked_equal(colorimg_tmp, 0)
        colorimg_masked = np.mean(colorimg_tmp_masked, axis=0)
        colorimg_masked = colorimg_masked.astype(np.uint8)
        colorimg = np.ma.filled(colorimg_masked, fill_value=255)
    return colorimg

def stack_masks_as_rgb(r,g,b):
    layers = [r,g,b]
    for id in range(3):
        #if rgb / rgba -> bitwise or between points in all the layers
        if(len(layers[id].shape) > 2):
            #comps = layers[id].shape[2]
            layers[id] = np.bitwise_or.reduce(layers[id], axis=2)
            #layers[id] = layers[id][:, :, 0]
        #if float encoded
        if((layers[id].dtype is np.dtype('float32')) or (layers[id].dtype is np.dtype('float16'))):
            layers[id] = (layers[id] * 255.999).astype(np.uint8)
    rgb = np.dstack(layers)  # stacks 3 h x w arrays -> h x w x 3
    return rgb