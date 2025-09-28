

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os


from APQ.utils.converter import Converter
from APQ.utils.accuracy_predictor import AccuracyPredictor
import sys
import copy
import argparse

import json
import torch
from APQ.elastic_nn.modules.dynamic_op import DynamicSeparableConv2d, DynamicSeparableQConv2d
from APQ.elastic_nn.networks.dynamic_quantized_proxyless import DynamicQuantizedProxylessNASNets
from APQ.imagenet_codebase.run_manager import ImagenetRunConfig, RunManager
from torchsummary import summary
parser = argparse.ArgumentParser(description='Test')
#parser.add_argument('--exp_dir', type=str, default=None)
parser.add_argument('--exp_dir', type=str, default='APQ/exps/test/')
parser.add_argument('--acc_predictor_dir', type=str, default='./APQ/models')
args, _ = parser.parse_known_args()
import yaml
import shutil

def create_or_replace_directory(directory_path):
    # Check if the directory exists
    if os.path.exists(directory_path):
        # If it exists, delete it
        shutil.rmtree(directory_path)
    
    # Create a new directory
    os.makedirs(directory_path)

# Custom YAML dumper to avoid quotes around merge key
class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True

def write_problem_yaml_first_conv(file_path, C, M, P, Q, R, S, HStride, WStride, inputs_histogram, weights_histogram, outputs_histogram):
    # Define the YAML data structure
    #problem_base = {}
    data = {
        "problem": {
            #"<<<": "*problem_base",
            "<<<": "*problem_base",
            "instance": {"C": C, "M": M, "P": P, "Q": Q, "R": R, "S": S, "HStride": HStride, "WStride": WStride},
            "name": "Conv2d",
            "dnn_name": "mobilenet_v2",
            "notes": "Conv2d",
            "histograms": {
                "Inputs": inputs_histogram,
                "Weights": weights_histogram,
                "Outputs": outputs_histogram
            }
        }
    }
    
    # Write to YAML file with inline formatting
    with open(file_path, "w") as file:
        file.write("{{include_text('../problem_base.yaml')}}\n")
        
        yaml.dump(data, file, Dumper=NoAliasDumper, default_flow_style=None, sort_keys=False, explicit_start=False)
        
    # Post-process the file to remove quotes around the asterisk
    with open(file_path, "r") as file:
        content = file.read()

    # Replace instances of `'*'` with `*` directly
    content = content.replace("'*problem_base'", "*problem_base")

    # Write the modified content back to the file
    with open(file_path, "w") as file:
        file.write(content)
    
    
def write_problem_yaml_depth_conv(file_path, G, P, Q, R, S, HStride, WStride, inputs_histogram, weights_histogram, outputs_histogram):
    # Define the YAML data structure
    data = {
        "problem": {
            "<<<": "*problem_base",
            "instance": {"G": G, "P": P, "Q": Q, "R": R, "S": S, "HStride": HStride, "WStride": WStride},
            "name": "Conv2d",
            "dnn_name": "mobilenet_v2",
            "notes": "Conv2d",
            "histograms": {
                "Inputs": inputs_histogram,
                "Weights": weights_histogram,
                "Outputs": outputs_histogram
            }
        }
    }
    
    # Write to YAML file with inline formatting
    with open(file_path, "w") as file:
        file.write("{{include_text('../problem_base.yaml')}}\n")
        yaml.dump(data, file, Dumper=NoAliasDumper, default_flow_style=None, sort_keys=False)
    # Post-process the file to remove quotes around the asterisk
    with open(file_path, "r") as file:
        content = file.read()

    # Replace instances of `'*'` with `*` directly
    content = content.replace("'*problem_base'", "*problem_base")

    # Write the modified content back to the file
    with open(file_path, "w") as file:
        file.write(content)
    
        
def write_problem_yaml_point_conv(file_path, C, M, P, Q, inputs_histogram, weights_histogram, outputs_histogram):
    # Define the YAML data structure
    data = {
        "problem": {
            "<<<": "*problem_base",
            "instance": {"C": C, "M": M, "P": P, "Q": Q},
            "name": "Conv2d",
            "dnn_name": "mobilenet_v2",
            "notes": "Conv2d",
            "histograms": {
                "Inputs": inputs_histogram,
                "Weights": weights_histogram,
                "Outputs": outputs_histogram
            }
        }
    }
    
    # Write to YAML file with inline formatting
    with open(file_path, "w") as file:
        file.write("{{include_text('../problem_base.yaml')}}\n")
        yaml.dump(data, file, Dumper=NoAliasDumper, default_flow_style=None, sort_keys=False)
    # Post-process the file to remove quotes around the asterisk
    with open(file_path, "r") as file:
        content = file.read()

    # Replace instances of `'*'` with `*` directly
    content = content.replace("'*problem_base'", "*problem_base")

    # Write the modified content back to the file
    with open(file_path, "w") as file:
        file.write(content)
        
        
def write_problem_yaml_linear(file_path, C, M, inputs_histogram, weights_histogram, outputs_histogram):
    # Define the YAML data structure
    problem_base='*problem_base'
    data = {
        "problem": {
            "<<<": problem_base,
            "instance": {"C": C, "M": M},
            "name": "Linear",
            "dnn_name": "mobilenet_v2",
            "notes": "Linear",
            "histograms": {
                "Inputs": inputs_histogram,
                "Weights": weights_histogram,
                "Outputs": outputs_histogram
            }
        }
    }
    # Write to YAML file with inline formatting
    with open(file_path, "w") as file:
        file.write("{{include_text('../problem_base.yaml')}}\n")
        yaml.dump(data, file, Dumper=NoAliasDumper, default_flow_style=None, sort_keys=False)
    # Post-process the file to remove quotes around the asterisk
    with open(file_path, "r") as file:
        content = file.read()

    # Replace instances of `'*'` with `*` directly
    content = content.replace("'*problem_base'", "*problem_base")

    # Write the modified content back to the file
    with open(file_path, "w") as file:
        file.write(content)

def CIMNAS_software(d_sam, ks_sam, e_sam, q_pw_w_sam, q_pw_a_sam, q_dw_w_sam, q_dw_a_sam):

    converter = Converter()
    specsnew = converter.random_spec_setSampled(d_sam=d_sam, ks_sam=ks_sam, e_sam=e_sam, q_pw_w_sam=q_pw_w_sam, q_pw_a_sam=q_pw_a_sam, q_dw_w_sam=q_dw_w_sam, q_dw_a_sam=q_dw_a_sam)
    keys_for_dict1 = {'wid', 'ks', 'e', 'd'}
    keys_for_dict2 = {'pw_w_bits_setting', 'pw_a_bits_setting', 'dw_w_bits_setting', 'dw_a_bits_setting'}
    info = {key: specsnew[key] for key in keys_for_dict1 if key in specsnew}
    q_info = {key: specsnew[key] for key in keys_for_dict2 if key in specsnew}

    accuracy_predictor = AccuracyPredictor(args,quantize=True)
    acc = accuracy_predictor.predict_accuracy([specsnew])
    acc=acc.item()
    
    ckpt_path = '{}/checkpoint/model_best.pth.tar'.format(args.exp_dir)
    if os.path.exists(ckpt_path):
        DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = 1
        DynamicSeparableQConv2d.KERNEL_TRANSFORM_MODE = 1

        dynamic_proxyless = DynamicQuantizedProxylessNASNets(
            ks_list=[3, 5, 7], expand_ratio_list=[4, 6], depth_list=[2, 3, 4], base_stage_width='proxyless',
            width_mult_list=1.0, dropout_rate=0, n_classes=1000
        )

        proxylessnas_init = torch.load(
            './APQ/models/imagenet-OFA',
            map_location='cpu'
        )['state_dict']
        dynamic_proxyless.load_weights_from_proxylessnas(proxylessnas_init)
        init_lr = 1e-3
        run_config = ImagenetRunConfig(
            test_batch_size=1, image_size=224, n_worker=16, valid_size=100, dataset='imagenet', train_batch_size=256,
            init_lr=init_lr, n_epochs=30,
        )

        run_manager = RunManager('~/tmp', dynamic_proxyless, run_config, init=False)

        proxylessnas_init = torch.load(
            ckpt_path,
            map_location='cpu'
        )['state_dict']
        dynamic_proxyless.load_weights_from_proxylessnas(proxylessnas_init)

        dynamic_proxyless.set_active_subnet(**info)
        dynamic_proxyless.set_quantization_policy(**q_info)
        
        xx=dynamic_proxyless.get_active_subnet(dynamic_proxyless)
        state_dict=xx.state_dict()
        layer_names=[]
        
        def compute_symmetric_zero_centered_histogram(data, N, quant_levels=8):
            # Flatten the 4D input tensor to a 1D tensor
            data = data.flatten()
            
            # Define the quantization range
            min_val, max_val = data.min(), data.max()
            data_scaled = (data - min_val) / (max_val - min_val)  # Normalize to [0, 1]
            
            # Quantize the data to integer levels (e.g., 256 levels)
            data_quantized = torch.round(data_scaled * (quant_levels - 1))
            
            # Define histogram bins
            num_bins = 2**N - 1
            bins = torch.linspace(0, quant_levels - 1, num_bins + 1)  # Range to fit quantized values

            # Compute histogram
            hist = torch.histc(data_quantized, bins=num_bins, min=0, max=quant_levels - 1)
            
            # Normalize the histogram to sum to 1.0
            hist = hist / hist.sum()
            
            # Optionally make the histogram zero-centered (centermost bin to zero)
            center_idx = num_bins // 2
            hist[center_idx] = 0.0  # Set the centermost bin to zero

            # Re-normalize the histogram after zero-centering
            hist = hist / hist.sum()
            
            return hist
        
        histsweights={}
        kernel_sizes = {}
        stride_values = {}
        bits={}
        for layer_name, param_tensor in state_dict.items():
            if len(param_tensor.shape) > 1:
                layername=layer_name.rsplit('.weight', 1)[0]
                layer_names.append(layername)
                histweight=compute_symmetric_zero_centered_histogram(param_tensor, 5)
                histsweights[layername]=histweight.tolist()
                
                if "conv" in layer_name and "weight" in layer_name:
                    # Extract layer module from the model using the name (removing ".weight" suffix)
                    layer_name2 = layer_name.rsplit('.weight', 1)[0]
                    layer = dict(xx.named_modules()).get(layer_name2, None)
                    
                    # Check if it's a Conv2d layer and retrieve stride
                    #if isinstance(layer, torch.nn.Conv2d):
                    stride_values[layername] = layer.stride
                    bits[layername]=[layer.w_bit, 8 if layer.a_bit < 0 else layer.a_bit]
                else:
                    stride_values[layername] = None
                    layer_name2 = layer_name.rsplit('.weight', 1)[0]
                    layer = dict(xx.named_modules()).get(layer_name2, None)
                    bits[layername]=[layer.w_bit, 8 if layer.a_bit < -2 else layer.a_bit]
                
                if "conv" in layer_name and param_tensor.ndimension() == 4:  # 4D tensor -> Conv2D weights
                    kernel_size = param_tensor.shape[2:]  # Last two dimensions are the kernel size
                    newLayername=layer_name.rsplit('.weight', 1)[0]+'.conv'
                    kernel_sizes[layername] = list(kernel_size)
                else:
                    kernel_sizes[layername] = None
                    
        run_manager = RunManager('~/tmp', xx, run_config, init=False)

        histsinp, histsout, inputshape, outshape = run_manager.val_get_act(is_test=True, layer_names=layer_names)
        count=0
        bits_dict={}

        folder='workloads/mobilenet_v2/'
        create_or_replace_directory(folder)
        bits_file='workloads/mobilenet_v2/bits.json'
        
        create_or_replace_directory(folder)
        
        for layer_n in layer_names:
            if count<10:
                yamlfile='0'+str(count)+'.yaml'
                yamlfiledict='0'+str(count)
            elif count>=10:
                yamlfile=str(count)+'.yaml'
                yamlfiledict=str(count)
            bits_dict[yamlfiledict]=bits[layer_n]
            
            file_path=folder+yamlfile
            inputs_histogram=histsinp[layer_n]
            outputs_histogram=histsout[layer_n]
            weights_histogram=histsweights[layer_n]
            if layer_n=='classifier.linear':
                C=inputshape[layer_n][1]
                M=outshape[layer_n][1]
                write_problem_yaml_linear(file_path, C, M, inputs_histogram, weights_histogram, outputs_histogram)
            elif (layer_n=='first_conv.conv') or (layer_n=='feature_mix_layer.conv'):
                C=inputshape[layer_n][1]
                M=outshape[layer_n][1]
                P=outshape[layer_n][2]
                Q=outshape[layer_n][3]
                R=kernel_sizes[layer_n][0]
                S=kernel_sizes[layer_n][1]
                HStride=stride_values[layer_n][0]
                WStride=stride_values[layer_n][1]
                write_problem_yaml_first_conv(file_path, C, M, P, Q, R, S, HStride, WStride, inputs_histogram, weights_histogram, outputs_histogram)
            elif ("inverted_bottleneck" in layer_n) or ("point_linear" in layer_n):
                C=inputshape[layer_n][1]
                M=outshape[layer_n][1]
                P=outshape[layer_n][2]
                Q=outshape[layer_n][3]
                write_problem_yaml_point_conv(file_path, C, M, P, Q, inputs_histogram, weights_histogram, outputs_histogram)
            elif "depth_conv" in layer_n:
                G=inputshape[layer_n][1]
                P=outshape[layer_n][2]
                Q=outshape[layer_n][3]
                R=kernel_sizes[layer_n][0]
                S=kernel_sizes[layer_n][1]
                HStride=stride_values[layer_n][0]
                WStride=stride_values[layer_n][1]
                write_problem_yaml_depth_conv(file_path, G, P, Q, R, S, HStride, WStride, inputs_histogram, weights_histogram, outputs_histogram)
            
            count=count+1
        
        bits_dict['acc']=acc
        with open(bits_file, "w") as file:
            json.dump(bits_dict, file, indent=4)

    else:
        print('NO SUCH DIRECTORY --> check model path!!!')

