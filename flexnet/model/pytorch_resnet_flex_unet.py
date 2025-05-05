import os, sys
import torch
import torch.nn as nn
from torchvision import models
from math import log2, pow
import logging
from argparse import ArgumentParser

#-----------------------------------------------------------------------------------------
curr_script_path = os.path.dirname(os.path.abspath(__file__))
flexnet_path = os.path.normpath(os.path.join(curr_script_path, ".."))
flexnet_host_path = os.path.normpath(os.path.join(flexnet_path, ".."))
sys.path.append(flexnet_host_path)
#-----------------------------------------------------------------------------------------
from flexnet.utils.gen_unet_utils import load_model
from v_utils.v_arg import print_cfg_dict, arg2boolAct
#-----------------------------------------------------------------------------------------

def convrelu(in_channels, out_channels, kernel, padding, norm_layer = None):
    if not norm_layer is None:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
            norm_layer(out_channels),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
            nn.ReLU(inplace=True))
    
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv2x2(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """2x2 convolution without padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=2, stride=stride,
                     padding=0, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResNetFlexUNet(nn.Module):
    
    @staticmethod
    def parse_arguments(argv):
        
        # possible choices:
        _resnet_types = [
            None, '',
            'resnet18', 'resnet34',
            'resnet50', 'resnet101', 'resnet152',
            #'resnext50_32x4d', 'resnext101_32x8d',
            #'wide_resnet50_2', 'wide_resnet101_2',
            ]
        _conv_dn_types= ['ResnetBasicBlock', 'ResnetBottelneck']
        _conv_up_types= ['convrelu', 'ResnetBasicBlock', 'ResnetBottelneck']
        _upsampling_modes = ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear']
        _tf = [True, False]

        parser = ArgumentParser(prog = "ResNetFlexUNet")
        #sys.argv

        ioa = parser.add_argument_group('in out arguments')
        ioa.add_argument   ("--n_class"                                                               , type=int             , required=True,  metavar='I'    , help="Liczba segmentowanych klas. Odpowiada liczbie generowanych obrazów wyjsciowych.")
        ioa.add_argument   ("--in_channels"                                                           , type=int             , required=True,  metavar='I'    , help="Liczba kanalow wejsciowych obrazow")
        
        dna = parser.add_argument_group('downsampling branch arguments')
        dna.add_argument   ("--conv_dn_predefined_resnet_type"           , default=''                 , type=str             , required=False, choices=_resnet_types,  help="Dla galezi decymacyjnej mozna podac typ sieci ResNet")
        dna.add_argument   ("--conv_dn_type"                             , default='ResnetBottelneck' , type=str             , required=False, choices=_conv_dn_types, help="Typ blokow skladowych Resnet uzytych do budowy galezi decymacyjnej (poza warstwa przetwarzajaca dane w oryginalnej rozdzielczosci)")
        dna.add_argument   ("--conv_dn_low_layer_div"                    , default=32                 , type=int             , required=False, metavar='I'    , help="Decymacja rozdzielczosci na najnizszej warstwie. Domyslnie 32 - tak jak dla ResNet")
        dna.add_argument   ("--conv_dn_layers_depth"                     , default=[2,2,2,2]          , type=int,   nargs='*', required=False, metavar='I'    , help="Glebokosc warstw (liczba blokow skladowych Resnet) na kojenych warstwach galezi decymacyjnej (poza warstwa przetwarzajaca dane w oryginalnej rozdzielczosci)")
        dna.add_argument   ("--conv_dn_chN_exp_f"                        , default= 2.0               , type=float           , required=False, metavar='F'    , help="Wspolczynnik zwiekszania liczby kanalow przy kolejnych decymacjach w galezi decymacyjnej.")
        dna.add_argument   ("--dec_factor"                               , default= 2                 , type=int             , required=False, metavar='I'    , help="Wspolczynnik decymacji na każdym poziomie sieci. Wymusza rowniesz taki sam wspolczynnik interpolacji.")
        
        upa = parser.add_argument_group('upsampling branch arguments')
        upa.add_argument   ("--conv_up_type"                             , default='ResnetBottelneck' , type=str             , required=False, choices=_conv_up_types,   help="Typ blokow skladowych Resnet uzytych do budowy galezi inkrementacyjnej (poza warstwa przetwarzajaca dane w oryginalnej rozdzielczosci)")
        upa.add_argument   ("--conv_up_layers_depth"                     , default=[2,2,2,2]          , type=int,   nargs='*', required=False, metavar='I'    , help="Glebokosc warstw (liczba blokow skladowych Resnet) na kojenych warstwach galezi inkrementacyjnej (poza warstwa przetwarzajaca dane w oryginalnej rozdzielczosci)")
        upa.add_argument   ("--conv_up_chN_compatible_with_org_unet_impl", default=True               , action=arg2boolAct   , required=False, metavar='B'    , help="Wymusza ograniczona liczbe kanalow w galezi inkrementacyjnej (nie na kazdej warstwie jest ona mnozona przez 2)")
        upa.add_argument   ("--conv_up_final_type"                       , default='ResnetBottelneck' , type=str             , required=False, choices=_conv_up_types,   help="conv_up_type dla najnizszej warstwy (przetwarzajacej dane w oryginalnej rozdzielczosci)")
        upa.add_argument   ("--conv_up_final_layers_depth"               , default=2                  , type=int             , required=False, metavar='I'    , help="conv_up_layers_depth dla najnizszej warstwy (przetwarzajacej dane w oryginalnej rozdzielczosci)")
        upa.add_argument   ("--upsampling_mode"                          , default='bilinear'         , type=str             , required=False, choices=_upsampling_modes,  help="conv_up_layers_depth dla najnizszej warstwy (przetwarzajacej dane w oryginalnej rozdzielczosci)")
        
        upa = parser.add_argument_group('horizotal branch at org size arguments')
        upa.add_argument   ("--conv_org_0_chN"                           , default=64                 , type=int             , required=False, metavar='I'    , help="Parametr pierwszego ConvRelu na warstwie bez podpróbkowania - liczba kanalow")
        upa.add_argument   ("--conv_org_0_kernel"                        , default=3                  , type=int             , required=False, metavar='I'    , help="Parametr pierwszego ConvRelu na warstwie bez podpróbkowania - rozmiar maski")
        
        cra = parser.add_argument_group('convrelu arguments')
        cra.add_argument   ("--convrelu_use_batchnorm"                   , default=False              , action=arg2boolAct   , required=False, metavar='B'    , help="Wymusza wstawienie normalizacji batch pomiędzy warstwe splotowa a Relu w module convrelu (zawsze uzywany na sciezkach hor, mozna tez wybrac w sciezkach dn i up)")
        cra.add_argument   ("--fix_phase"                                , default=False              , action=arg2boolAct   , required=False, metavar='B'    , help="Zmiana conv decymujacych na takie ktore nie przesowaja fazy w obrazie")
        

        pta = parser.add_argument_group('pretrining arguments')
        pta.add_argument   ("--pretrained_model_state_dict_path"         , default=""                 , type=str             , required=False, metavar='PATH' , help="plik ze wartosciami wspolczynnikow ktorymi nalezy inicjowac Flexnet")
        pta.add_argument   ("--initialize_with_mean"                     , default=False              , action=arg2boolAct   , required=False, metavar='B'    , help="inicjuj wartości w warstwach splotowych tak zeby liczyly srednia")
        
        if not(("-h" in argv) or ("--help" in argv)): 
            # get training arguments
            fn_args, rem_args = parser.parse_known_args(argv)
            return fn_args, rem_args
        else: 
            # help
            return parser.format_help()

    def __init__(self, fn_args):
        
        super().__init__()

        self._norm_layer = nn.BatchNorm2d
        if fn_args.convrelu_use_batchnorm:
            self._convrelu_norm_layer = nn.BatchNorm2d
        else:
            self._convrelu_norm_layer = None

        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        chN = self.inplanes

        predefined_params = {
            'resnet18'        : [[2, 2,  2, 2], 2, 'ResnetBasicBlock'] ,
            'resnet34'        : [[3, 4,  6, 3], 2, 'ResnetBasicBlock'] ,
            'resnet50'        : [[3, 4,  6, 3], 2, 'ResnetBottelneck'] ,
            'resnet101'       : [[3, 4, 23, 3], 2, 'ResnetBottelneck'] ,
            'resnet152'       : [[3, 8, 36, 3], 2, 'ResnetBottelneck'] ,
            #'resnext50_32x4d' : [[3, 4,  6, 3], 2, 'ResnetBottelneck', {'width_per_group': 4, 'groups': 32}] ,
            #'resnext101_32x8d': [[3, 4, 23, 3], 2, 'ResnetBottelneck', {'width_per_group': 8, 'groups': 32}] ,
            #'wide_resnet50_2' : [[3, 4,  6, 3], 2, 'ResnetBottelneck', {'width_per_group': 64 * 2}] ,
            #'wide_resnet101_2': [[3, 4, 23, 3], 2, 'ResnetBottelneck', {'width_per_group': 64 * 2}] ,
            }
        org_impl_conv_up_out_chN = [128, 256, 256, 512]
        if not(fn_args.conv_dn_predefined_resnet_type is None) and fn_args.conv_dn_predefined_resnet_type != '':
            fn_args.conv_dn_layers_depth, fn_args.conv_dn_chN_exp_f, fn_args.conv_dn_type = predefined_params[fn_args.conv_dn_predefined_resnet_type]
            if fn_args.conv_dn_low_layer_div != 32:
                logging.warning("predefined resnet is used that has constant div=32, but div={} is given. Change it to 32".format(fn_args.conv_dn_low_layer_div))
            fn_args.conv_dn_low_layer_div = 32
                
        self.dec_factor = fn_args.dec_factor
        self.initialize_with_mean = fn_args.initialize_with_mean
        self.conv_dn_low_layer_div = fn_args.conv_dn_low_layer_div
        self.lids = int(log2(fn_args.conv_dn_low_layer_div) )  # 4 for 32
        # if there is more layers than in Resnet than extend layers table approprietly 
        for lid in range(5, self.lids):
            fn_args.onv_dn_layers_depth.append(2)
            fn_args.onv_up_layers_depth.append(fn_args.conv_up_layers_depth[-1])
            org_impl_conv_up_out_chN.append(org_impl_conv_up_out_chN[-1]*2)
        self.conv_dn            = nn.ModuleList([None] * self.lids     )
        self.conv_hr            = nn.ModuleList([None] * self.lids     )
        self.conv_up            = nn.ModuleList([None] *(self.lids - 1))
        self.conv_dn_out_chN    = [0] *  self.lids
        self.conv_hr_out_chN    = [0] *  self.lids
        self.conv_up_out_chN    = [0] * (self.lids -1)

        self.fix_phase                  = fn_args.fix_phase
        self.conv_up_layers_depth       = fn_args.conv_up_layers_depth
        self.conv_dn_layers_depth       = fn_args.conv_dn_layers_depth
        self.conv_up_final_layers_depth = fn_args.conv_up_final_layers_depth

        #create layers
        if(self.fix_phase):
            self.conv_dn     [0] = nn.Sequential(
                 nn.Conv2d(fn_args.in_channels, self.inplanes, kernel_size=(6 + self.dec_factor-2), stride=self.dec_factor, padding=self.dec_factor, bias=False),
                 self._norm_layer(self.inplanes),
                 nn.ReLU(inplace=True)
            )        
        else:
            self.conv_dn     [0] = nn.Sequential(
                 nn.Conv2d(fn_args.in_channels, self.inplanes, kernel_size=7, stride=self.dec_factor, padding=3, bias=False),
                 self._norm_layer(self.inplanes),
                 nn.ReLU(inplace=True)
            )        
        # pass 0
        self.conv_original_size0 = convrelu(fn_args.in_channels, fn_args.conv_org_0_chN, fn_args.conv_org_0_kernel, int((fn_args.conv_org_0_kernel-1)/2), norm_layer = self._convrelu_norm_layer)
        self.conv_original_size1 = convrelu(fn_args.conv_org_0_chN, chN, 3, 1, norm_layer = self._convrelu_norm_layer)   
        

        self.conv_dn_out_chN[0] = self.find_out_ch_num(self.conv_dn     [0])
        self.conv_hr    [0] = convrelu(self.conv_dn_out_chN [0], chN, 1, 0, norm_layer = self._convrelu_norm_layer)  
        self.conv_hr_out_chN    [0] = self.find_out_ch_num(self.conv_hr    [0])
 
    
        self.conv_dn     [1] = nn.Sequential(
            nn.MaxPool2d(kernel_size=self.dec_factor, stride=self.dec_factor, padding=0),
            self._make_resnet_layer(fn_args.conv_dn_type,  64, fn_args.conv_dn_layers_depth[0])
            )
        self.conv_dn_out_chN[1] = self.find_out_ch_num(self.conv_dn     [1])
        self.conv_hr    [1] = convrelu(self.conv_dn_out_chN [1], chN, 1, 0, norm_layer = self._convrelu_norm_layer) 
        self.conv_hr_out_chN    [1] = self.find_out_ch_num(self.conv_hr    [1])
        
        for lid in range(2, self.lids):
            chN = int(chN * fn_args.conv_dn_chN_exp_f) #limit output of each layer to this value

            resnet_layer_out_ch_limit_to_n = chN
            self.conv_dn     [lid] = self._make_resnet_layer(fn_args.conv_dn_type, resnet_layer_out_ch_limit_to_n, fn_args.conv_dn_layers_depth[lid-1], stride=self.dec_factor)
            # sprawdzam czy na pewno wyjscie ma taka l. kanalow jak sie spodziewam (chN):
            self.conv_dn_out_chN[lid] = self.find_out_ch_num(self.conv_dn     [lid]) # powinno byc rowne chN
            
            hor_conv_in_ch_n = self.conv_dn_out_chN [lid]
            hor_conv_out_ch_limit_to_n = hor_conv_in_ch_n
            self.conv_hr    [lid] = convrelu(hor_conv_in_ch_n, hor_conv_out_ch_limit_to_n, 1, 0, norm_layer = self._convrelu_norm_layer)
            self.conv_hr_out_chN    [lid] = self.find_out_ch_num(self.conv_hr    [lid]) # powinno byc rowne chN

        # conv up
        for lid in range(self.lids-2, -1, -1):
            in_hor_channels = self.conv_hr_out_chN [lid  ] #self.conv_hr   [lid]._modules['0'].out_channels
            in_ver_channels = self.conv_hr_out_chN [lid+1]  if (lid == (self.lids-2)) else self.conv_up_out_chN [lid+1]
            in_channels = in_hor_channels + in_ver_channels
            if fn_args.conv_up_chN_compatible_with_org_unet_impl:
                out_channels = org_impl_conv_up_out_chN[lid]
            else:
                out_channels = self.base_width * int(pow(2, lid+1)) 

            self.conv_up    [lid] = self._make_conv_up_layer(block_type = fn_args.conv_up_type, in_channels = in_channels, out_channels = out_channels, num_blocks = fn_args.conv_up_layers_depth[lid])

            self.conv_up_out_chN[lid] = self.find_out_ch_num(self.conv_up    [lid]) # powinno byc rowne chN

        # ostatnie przetwarzania:
        in_hor_channels = self.find_out_ch_num(self.conv_original_size1)
        in_ver_channels = self.conv_up_out_chN[0]
        in_channels = in_hor_channels + in_ver_channels
        out_channels = self.base_width #tak jak w org implementacji UNet
        self.conv_original_size2 = self._make_conv_up_layer(block_type = fn_args.conv_up_final_type, in_channels = in_channels, out_channels = out_channels, num_blocks = fn_args.conv_up_final_layers_depth)
        self.conv_last = nn.Conv2d(out_channels, fn_args.n_class, 1)

        align_corners = None if not (fn_args.upsampling_mode in ['linear', 'bilinear', 'bicubic', 'trilinear']) else False
        self.upsample = nn.Upsample(scale_factor=self.dec_factor, mode=fn_args.upsampling_mode, align_corners=align_corners)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if(self.initialize_with_mean):
                    nn.init.constant_(m.weight, 1.0/(m.kernel_size[0] * m.kernel_size[1]*m.in_channels))
                    if(not m.bias is None):
                        nn.init.constant_(m.bias, 0)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                if(self.initialize_with_mean):
                    m.track_running_stats = False
                    m.affine = False
                    m.training = False


        # # Zero-initialize the last BN in each residual branch,
        # # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck):
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)

        
        
        logging.info(" -" * 25)
        logging.info("Parametrs of UNet:")
        logging.info(" in_channels (num components)  : {}".format(fn_args.in_channels                       ))
        logging.info(" conv down backbone resnet name: {}".format(fn_args.conv_dn_predefined_resnet_type    ))
        logging.info(" conv down block type          : {}".format(fn_args.conv_dn_type                      ))
        logging.info(" conv_dn_low_layer_div         : {}".format(fn_args.conv_dn_low_layer_div             ))
        logging.info(" conv_dn_layers_depth          : {}".format(fn_args.conv_dn_layers_depth              ))
        logging.info(" conv_dn_chN_exp_f             : {}".format(fn_args.conv_dn_chN_exp_f                 ))
        logging.info(" conv up block type            : {}".format(fn_args.conv_up_type                      ))
        logging.info(" conv_up_layers_depth          : {}".format(fn_args.conv_up_layers_depth              ))
        logging.info(" conv up final block type      : {}".format(fn_args.conv_up_final_type                ))
        logging.info(" conv_up_final_layers_depth    : {}".format(fn_args.conv_up_final_layers_depth        ))
        logging.info(" upsampling_mode               : {}".format(fn_args.upsampling_mode                   ))
        logging.info(" convrelu_use_batchnorm        : {}".format(fn_args.convrelu_use_batchnorm            ))
        logging.info(" number of channels at block output:")
        logging.info(" -conv down                       : {}".format(self.conv_dn_out_chN))
        logging.info(" -conv horizontal                 : {}".format(self.conv_hr_out_chN))
        logging.info(" -conv up                         : {}".format(self.conv_up_out_chN))
        logging.info(" out_channels (num classes)    : {}".format(fn_args.n_class                           ))
        
        logging.info(" -" * 25)
        pretrained = fn_args.pretrained_model_state_dict_path != ""
        if pretrained:
            if(self.initialize_with_mean):
                logging.error("initialize_with_mean is set but also pretrained_model_state_dict_path is set to {}. Skip params initialization".format(fn_args.pretrained_model_state_dict_path))
            logging.info("Use pretrained model_state_dict file {} to warmstart training".format(fn_args.pretrained_model_state_dict_path))
            fn_args.pretrained_model_state_dict_path =  os.path.normpath(fn_args.pretrained_model_state_dict_path)
            if os.path.isfile(fn_args.pretrained_model_state_dict_path):
                load_model(fn_args.pretrained_model_state_dict_path, self, strict=False)
            else:
                logging.warning("Could not find {} file. Skip params initialization".format(fn_args.pretrained_model_state_dict_path))

    def find_out_ch_num(self, layer_module):
        if (type(layer_module) is nn.Sequential) or (type(layer_module) is models.resnet.Bottleneck) or (type(layer_module) is  models.resnet.BasicBlock):
            num_mod = len(layer_module._modules)
            layer_modules_list = list(layer_module._modules.values())
            mod_idx = num_mod-1
            while mod_idx >= 0:
                ret_val = self.find_out_ch_num(layer_modules_list[mod_idx])
                if not(ret_val is None):
                    return ret_val
                mod_idx -= 1
            return None
        elif type(layer_module) is nn.Conv2d:
            return layer_module.out_channels
        elif type(layer_module) is nn.BatchNorm2d:
            return layer_module.num_features
        else:
            return None

    #from torchvision.models.resnet.py
    def _make_resnet_layer(self, block_type, planes, num_blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if(block_type == 'ResnetBasicBlock'):
            block = models.resnet.BasicBlock 
        elif(block_type == 'ResnetBottelneck'):
            block = models.resnet.Bottleneck
        else:
            logging.error("block_type for decimation branch can one be from <'ResnetBasicBlock', 'ResnetBottelneck'>, but '{}' was specified".format(block_type))
            sys.exit(30)
        if dilate:
            self.dilation *= stride
            stride = 1

        blks = []
        first_blk = block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer)
        if (stride != 1) or (self.inplanes != (planes * block.expansion)):
            
            if(self.fix_phase and (stride != 1)):
                d_out_channels = first_blk.conv1.out_channels
                d_out_channels = first_blk.conv1.out_channels
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, d_out_channels, kernel_size = self.dec_factor, stride=stride, padding=0, bias=False),
                    norm_layer(d_out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(d_out_channels, planes * block.expansion, kernel_size = 1, stride=1, padding=0, bias=False),
                    norm_layer(planes * block.expansion),
                )  
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            first_blk.downsample = downsample

        if (self.fix_phase) and (stride != 1):
            if(block_type == 'ResnetBasicBlock'):
                c_in_channels = first_blk.conv1.in_channels
                c_out_channels = first_blk.conv1.out_channels
                c_stride = first_blk.conv1.stride
                c_groups = first_blk.conv1.groups
                c_dilation = first_blk.conv2.dilation
                #first_blk.conv1 = conv2x2(c_in_channels, c_out_channels, c_stride, c_groups, c_dilation)
                first_blk.conv1 = nn.Conv2d(c_in_channels, c_out_channels, kernel_size = self.dec_factor, stride=c_stride, padding=0, dilation=c_dilation, groups=c_groups, bias=False)
            elif(block_type == 'ResnetBottelneck'):
                c_in_channels = first_blk.conv2.in_channels
                c_out_channels = first_blk.conv2.out_channels
                c_stride = first_blk.conv2.stride
                c_groups = first_blk.conv2.groups
                c_dilation = first_blk.conv2.dilation
                #first_blk.conv2 = conv2x2(c_in_channels, c_out_channels, c_stride, c_groups, c_dilation)
                first_blk.conv2 = nn.Conv2d(c_in_channels, c_out_channels, kernel_size = self.dec_factor, stride=c_stride, padding=0, dilation=c_dilation, groups=c_groups, bias=False)
        blks.append(first_blk)
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            blks.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*blks)

    def _make_conv_up_layer(self, block_type, in_channels, out_channels, num_blocks):
        blks = []
        for id in range(num_blocks):
            ic = in_channels if id==0 else out_channels
            oc = out_channels
            if block_type == 'convrelu':
                blk = convrelu(ic, oc, 3, 1, norm_layer = self._convrelu_norm_layer)
            elif block_type == 'ResnetBasicBlock':
                # downsample for bypass path inside the block so it matches size for the processed data
                downsample = nn.Sequential(
                    conv1x1(ic, oc, 1),
                    self._norm_layer(oc),
                )
                blk = models.resnet.BasicBlock(ic, oc, downsample = downsample, groups=self.groups, base_width=self.base_width, norm_layer=self._norm_layer)
            elif block_type == 'ResnetBottelneck':
                blk = models.resnet.Bottleneck(ic, oc, groups=self.groups, base_width=self.base_width, norm_layer=self._norm_layer)
                # downsample for bypass path inside the block so it matches size for the processed data
                downsample = nn.Sequential(
                    conv1x1(ic, oc, 1),
                    self._norm_layer(oc),
                )
                #override expansion parameter and re-initialize
                blk.expansion = 1 
                blk.__init__(ic, oc, downsample = downsample, groups=self.groups, base_width=self.base_width, norm_layer=self._norm_layer)
                
            blks.append(blk)
        return nn.Sequential(*blks)
    
    def forward(self, input):
        # pass 0
        # jednak tutaj, a myslalem ze "jezeli na koniec licze x_original (a nie potrzebuje tego wczesniej), to ta zmienna nie bedzie zajmować pamięc podczas innych obliczen"
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)
        
        lX = [None] * self.lids
        # down 0
        lX[0] = self.conv_dn[0](input)  
        # down X  
        for  lid in range(1, self.lids):        
            lX[lid] = self.conv_dn[lid](lX[lid-1])
            if(self.initialize_with_mean):
                lX[lid] = lX[lid]/(pow(2, self.conv_dn_layers_depth[lid-1]))
        
        last_l_id =  self.lids-1 
        # hor last
        lX[last_l_id] = self.conv_hr[last_l_id](lX[last_l_id])
        # up last -> oneby last
        x = self.upsample(lX[last_l_id])
        
        
        for  lid in range(self.lids-2, -1, -1):   
            # hor 
            lX[lid] = self.conv_hr[lid](lX[lid])
            x = torch.cat([x, lX[lid]], dim=1)
            x = self.conv_up[lid](x)
            if(self.initialize_with_mean):
                x = x/(pow(2, self.conv_up_layers_depth[lid]))#self.conv_up[lid]))/(math.pow(2, self.conv_dn_layers_depth[lid-1]))
            # up 
            x = self.upsample(x)
 
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x) 
        if(self.initialize_with_mean):
            x = x/(pow(2, self.conv_up_final_layers_depth))       
        
        out = self.conv_last(x)        
        
        return out
    
    def check_anomaly_in_tensor(self, t):
        #t[0,0,0,0]=float("inf")
        #t[0,0,0,1]=float("-inf")
        #t[0,0,0,2]=float("nan")
        hasNaN_T = torch.isnan(t)
        hasInf_T = torch.isinf(t)
        hasNaN = hasNaN_T.any()
        hasInf = hasInf_T.any()
        if(hasNaN):
            nans_idxs = hasNaN_T.nonzero()
            logging.info(" found NaN at forward_stage_id {}! Num = {}. First at {}".format(self.forward_stage_id, nans_idxs.shape[0], nans_idxs[0]))
        if(hasInf):
            infs_idxs = hasInf_T.nonzero()
            logging.info(" found Inf at forward_stage_id {}! Num = {}. First at {}".format(self.forward_stage_id, infs_idxs.shape[0], infs_idxs[0]))
        if(hasNaN or hasInf):
            logging.info("  tensor.shape: {}".format(t.shape))
            logging.info("  tensor = {}".format(t))
        self.forward_stage_id += 1

    def print_forward_stage(self, st):
        logging.info("stage '{}', current forward_stage_id {}".format(st, self.forward_stage_id))

    def forward_anomaly_det(self, input):
        self.forward_stage_id = 0
        self.print_forward_stage("pass 0")
        self.check_anomaly_in_tensor(input)
        # jednak tutaj, a myslalem ze "jezeli na koniec licze x_original (a nie potrzebuje tego wczesniej), to ta zmienna nie bedzie zajmować pamięc podczas innych obliczen"
        x_original = self.conv_original_size0(input)
        self.check_anomaly_in_tensor(x_original)
        x_original = self.conv_original_size1(x_original)
        self.check_anomaly_in_tensor(x_original)
        
        lX = [None] * self.lids
        self.print_forward_stage("down 0")
        lX[0] = self.conv_dn[0](input)  
        self.check_anomaly_in_tensor(lX[0])
        self.print_forward_stage("down X")
        for  lid in range(1, self.lids):        
            lX[lid] = self.conv_dn[lid](lX[lid-1])
            self.check_anomaly_in_tensor(lX[lid])
        
        last_l_id =  self.lids-1 
        self.print_forward_stage("hor last")
        lX[last_l_id] = self.conv_hr[last_l_id](lX[last_l_id])
        self.check_anomaly_in_tensor(lX[last_l_id])
        self.print_forward_stage(" up last -> oneby last")
        x = self.upsample(lX[last_l_id])
        self.check_anomaly_in_tensor(x)
        
        self.print_forward_stage("conv_ups")
        for  lid in range(self.lids-2, -1, -1):   
            # hor 
            lX[lid] = self.conv_hr[lid](lX[lid])
            self.check_anomaly_in_tensor(lX[lid])
            x = torch.cat([x, lX[lid]], dim=1)
            x = self.conv_up[lid](x)
            self.check_anomaly_in_tensor(x)
            # up 
            x = self.upsample(x)
            self.check_anomaly_in_tensor(x)
        
        self.print_forward_stage("conv_original_size2")
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)  
        self.check_anomaly_in_tensor(x)
        
        self.print_forward_stage("conv_last")
        out = self.conv_last(x) 
        #out[0,0,0,0] = float('nan')
        self.check_anomaly_in_tensor(out)
        
        return out
    
    def get_in_channels_num(self):
        return self.conv_dn[0]._modules['0'].in_channels