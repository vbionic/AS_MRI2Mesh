import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np

# https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
# https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d
def f1_loss(pred, target, cls_valid_ref = None, out_weight_mask = None, cls_num_loss_weights = None, do_dbg_nans = False):
    pred = pred.contiguous()
    target = target.contiguous()   
    
    # tak nie moge bo jest problem z propagacja wsteczna gradientu
    #if not(cls_valid_ref is None):
    #    for iid in range(cls_valid_ref.shape[0]):
    #        for cid in range(cls_valid_ref.shape[1]):
    #            if(cls_valid_ref[iid,cid] == 0):
    #                pred  [iid,cid,:,:]  = pred  [iid,cid,:,:] * 0
    #                target[iid,cid,:,:]  = target[iid,cid,:,:] * 0
       
    # musze to zrobic tak:
    if not(cls_valid_ref is None) and (out_weight_mask is None):
        mask = torch.ones(target.shape, dtype=torch.float16, device=pred.device)
        for img_id in range(cls_valid_ref.shape[0]):
            for cls_id in range(cls_valid_ref.shape[1]):
                if cls_valid_ref[img_id, cls_id] == 0:
                    mask[img_id, cls_id, :, :] = 0 
        
        pred    = pred      * mask
        target  = target    * mask

        del mask

    if not(out_weight_mask is None):
        pred    = pred      * out_weight_mask
        target  = target    * out_weight_mask

    tp = (   target *   pred ).sum(dim=[0,2,3])
   #tn = ((1-target)*(1-pred)).sum(dim=[0,2,3])
    fp = ((1-target)*   pred ).sum(dim=[0,2,3])
    fn = (   target *(1-pred)).sum(dim=[0,2,3])

    eps = torch.finfo(target.dtype).eps

    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)

    f1 = (2 * p * r) / (p + r + eps)
    #f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    
    loss_cls = 1 - f1 
    
    # scale loss for each class according to number of the class's samples in a dataset
    if not(cls_num_loss_weights is None):
        for cls_id in range(cls_num_loss_weights.shape[0]):
            loss_cls[cls_id].mul_(cls_num_loss_weights[cls_id])

    # zero loss for a class if there was no ref for this class in this batch
    if not(cls_valid_ref is None):
        cls_valid_any = torch.where(cls_valid_ref.sum(dim=[0], dtype = torch.bool), 1, 0)
        for cls_id in range(cls_valid_any.shape[0]):
            if(cls_valid_any[cls_id] == 0):
                loss_cls[cls_id] = loss_cls[cls_id]*0
        loss_mean = loss_cls.sum()/ cls_valid_any.sum()
    else:
        loss_mean = loss_cls.mean()

            
    if(do_dbg_nans and np.any(np.isnan(loss_cls.data.cpu().numpy()))):
        print("NaN!")
        print(" f1_loss_clss[0].data.cpu().numpy(): {}".format(loss.data.cpu().numpy()))
        print(" f1_loss_clss: {}".format(loss))
        print(" out_weight_mask: {}".format(out_weight_mask))
        print(" cls_num_loss_weights: {}".format(cls_num_loss_weights))
        print(" pred has any NaN: {}".format(torch.isnan(pred).any()))
        print("  pred.shape: {}".format(pred.shape))
        print("  pred: {}".format(pred))
        print(" target has any NaN: {}".format(torch.isnan(target).any()))
        print("  target.shape: {}".format(target.shape))
        print("  target: {}".format(target))

    return loss_cls, loss_mean

def dice_loss(pred, target, smooth = 1., cls_valid_ref = None, out_weight_mask = None, cls_num_loss_weights = None):
    pred = pred.contiguous()
    target = target.contiguous()    

    if not(out_weight_mask is None):
        pred    = pred      * out_weight_mask
        target  = target    * out_weight_mask

    intersection = (pred * target)

    intersection = intersection.sum(dim=2)
    intersection = intersection.sum(dim=2)

    pred = pred.sum(dim=2)
    pred = pred.sum(dim=2)

    target = target.sum(dim=2)
    target = target.sum(dim=2)
    
    m = (2. * intersection + smooth)
    d = (pred + target + smooth)
    v = (m / d)
    loss = (1 - v)
    
    loss = loss.mean(dim=0)

    if not(cls_num_loss_weights is None):
        for cls_id in range(pred.shape[1]):
            loss[cls_id].mul_(cls_num_loss_weights[cls_id])

    return loss#.mean()

def bce_loss(pred, target, smooth = 1., cls_valid_ref = None, out_weight_mask = None, cls_num_loss_weights = None, cls_pos_loss_weights = None, do_dbg_nans = False):
    #binary_cross_entropy_with_logits and BCEWithLogits are safe to autocast.
    bce_table = F.binary_cross_entropy_with_logits(pred, target, reduction='none', pos_weight=cls_pos_loss_weights)
    if not out_weight_mask is None:
        bce_table = bce_table*out_weight_mask
    if not cls_num_loss_weights is None:
        for img_id in range(pred.shape[0]):
            for cls_id in range(pred.shape[1]):
                if(cls_num_loss_weights[cls_id].data!=1):
                    bce_table[img_id, cls_id, :, :].mul_(cls_num_loss_weights[cls_id])
    #bce = bce_table.mean()
    bce_clss = bce_table.mean(dim=[0,2,3])
    
    if(do_dbg_nans and np.any(np.isnan(bce_clss.data.cpu().numpy()))):
        print("NaN!")
        print(" bce_clss[0].data.cpu().numpy(): {}".format(bce_clss[0].data.cpu().numpy()))
        print(" bce_clss: {}".format(bce_clss))
        print(" out_weight_mask: {}".format(out_weight_mask))
        print(" cls_num_loss_weights: {}".format(cls_num_loss_weights))
        print(" pred has any NaN: {}".format(torch.isnan(pred).any()))
        print("  pred.shape: {}".format(pred.shape))
        print("  pred: {}".format(pred))
        print(" target has any NaN: {}".format(torch.isnan(target).any()))
        print("  target.shape: {}".format(target.shape))
        print("  target: {}".format(target))
        print(" bce_table has any NaN: {}".format(torch.isnan(bce_table).any()))
        print("  bce_table.shape: {}".format(bce_table.shape))
        print("  bce_table: {}".format(bce_table))

    return bce_clss#.mean()
        

def calc_loss(pred, target, metrics, cls_valid_ref = None, out_weight_mask = None, cls_num_loss_weights = None, cls_pos_loss_weights = None, clss_names = None, do_dbg_nans = False):
    num_cls  = target.shape[1]
    num_inps = target.shape[0]

    pred_sig = torch.sigmoid(pred) 
    loss_clss, loss_mean  = f1_loss(pred_sig, target, cls_valid_ref = cls_valid_ref, out_weight_mask = out_weight_mask, cls_num_loss_weights = cls_num_loss_weights, do_dbg_nans = do_dbg_nans)
        
    # accumulate the metrics
    for cid in range(num_cls):
        cls_name = clss_names[cid] if (not clss_names is None) else cid
        mulf = num_inps
        if not(cls_valid_ref is None):
            mulf = torch.sum(cls_valid_ref[:,cid])
        metrics['loss_{}'.format(cls_name)] += (loss_clss[cid] * mulf).data.cpu().numpy()

    metrics['loss'] += loss_mean.data.cpu().numpy() * num_inps
    
    #if(bce_weight != 0):
    #    bce_clss = bce_loss(pred, target, cls_valid_ref = cls_valid_ref, out_weight_mask = out_weight_mask, cls_num_loss_weights = cls_num_loss_weights, cls_pos_loss_weights = cls_pos_loss_weights, do_dbg_nans = do_dbg_nans)
    #if(bce_weight != 1):
    #    pred = torch.sigmoid(pred)
    #    dice_clss = dice_loss(pred, target, cls_valid_ref = cls_valid_ref, out_weight_mask = out_weight_mask, cls_num_loss_weights = cls_num_loss_weights)
    #        
    #if(bce_weight != 1) and (bce_weight != 0):
    #    loss_clss = bce_clss * bce_weight + dice_clss * (1-bce_weight)
    #elif(bce_weight != 1):
    #    loss_clss = dice_clss
    #else:
    #    loss_clss = bce_clss
        
    return loss_mean