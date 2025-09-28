
from imagenet_codebase.run_manager import ImagenetRunConfig, RunManager
import os
import copy
import torch
from elastic_nn.modules.dynamic_op import DynamicSeparableConv2d, DynamicSeparableQConv2d
from elastic_nn.networks.dynamic_quantized_proxyless import DynamicQuantizedProxylessNASNets
import json
import argparse
from torchsummary import summary

parser = argparse.ArgumentParser(description='Quantization-aware Finetuning')
parser.add_argument('--exp_name', type=str, default='test')
parser.add_argument('--id', type=int, default=-1)
args, _ = parser.parse_known_args()
print(args)

if __name__ == '__main__':
    exp_dir = 'exps/{}'.format(args.exp_name)

    arch_path = '{}/arch'.format(exp_dir)
    print(arch_path)
    tmp_lst = json.load(open(arch_path, 'r'))
    info, q_info = tmp_lst
    print('Info')
    print(info)
    print('QInfo')
    print(q_info)

    DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = 1
    DynamicSeparableQConv2d.KERNEL_TRANSFORM_MODE = 1
    
    #DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = None
    #DynamicSeparableQConv2d.KERNEL_TRANSFORM_MODE = None

    dynamic_proxyless = DynamicQuantizedProxylessNASNets(
        ks_list=[3, 5, 7], expand_ratio_list=[4, 6], depth_list=[2, 3, 4], base_stage_width='proxyless',
        width_mult_list=1.0, dropout_rate=0, n_classes=1000
    )
    
    dynamic_proxyless2 = DynamicQuantizedProxylessNASNets(
        ks_list=[3, 5, 7], expand_ratio_list=[4, 6], depth_list=[2, 3, 4], base_stage_width='proxyless',
        width_mult_list=1.0, dropout_rate=0, n_classes=1000
    ).cuda()
    
    #print('model is:')
    print(dynamic_proxyless)
    
    #print('attributes')
    #attrs=vars(dynamic_proxyless)   
    #print(', '.join("%s: %s" % item for item in attrs.items()))
    
    print('layers check')
    #print(dynamic_proxyless.first_conv)
    
    #attrs=vars(dynamic_proxyless.first_conv)  
    #print(', '.join("%s: %s" % item for item in attrs.items()))
    
    #print("type",type(dynamic_proxyless))
    
    dynamic_proxyless.sample_active_subnet()
    
    
    # STOPPED HERE --> how to get a subnet!
    
    random_subnet = dynamic_proxyless.get_active_subnet(preserve_weight=True)
    #print('HERE is a RANDONLY sampled network')
    #print(random_subnet)
    
    #print('len random subnet',len(random_subnet)) #to stop here
    
    torch.save(random_subnet, './exps/test/archnew')
    
    #to print all attributes of the model
    #print(dir(dynamic_proxyless))
    
    #print(dynamic_proxyless.children)
    
    #print(run_config[1231231]) # to stop
    
    #print(xxx)
    ##IF YOU WANT SUMMARY OF THE MODEL
    #print('printing summary:')
    
    #summary(dynamic_proxyless2, (3, 224, 224))

    proxylessnas_init = torch.load(
        './models/imagenet-OFA',
        map_location='cpu'
    )['state_dict']
    
    dynamic_proxyless.load_weights_from_proxylessnas(proxylessnas_init)
    
    #attrs=vars(dynamic_proxyless.first_conv.conv)  
    #print(', '.join("%s: %s" % item for item in attrs.items()))
    
    #print('len random subnet',len(random_subnet)) #to stop here
    
    init_lr = 1e-3
    
    run_config = ImagenetRunConfig(
        test_batch_size=1000, image_size=224, n_worker=16, valid_size=5000, dataset='imagenet', train_batch_size=256,
        init_lr=init_lr, n_epochs=5,
    )
    
    
    print('Run config:')
    for k, v in run_config.config.items():
        print('\t%s: %s' % (k, v))

    #print(run_config[1231231]) # to stop
    
    tmp_dynamic_proxyless = copy.deepcopy(dynamic_proxyless)

    run_manager = RunManager(exp_dir, tmp_dynamic_proxyless, run_config, init=False)
    
    #print(run_manager)

    tmp_dynamic_proxyless.set_active_subnet(**info)
    tmp_dynamic_proxyless.set_quantization_policy(**q_info)

    run_manager.reset_running_statistics()
    
    print('WE ARE HERE')
    
    #losses, top1, top5=run_manager.validate()
    
    #print('losses',losses) 
    #print('top1',top1)
    #print('top5',top5) 
    
    #torch.set_printoptions(precision=100)
    
    losses, top1, top5  = run_manager.validate()
    
    #print(run_config[1231231]) # to stop
    
    acc = run_manager.finetune()

    acc_list = []
    acc_list.append((json.dumps(info), json.dumps(q_info), acc))
    output_dir = '{}/acc'.format(exp_dir)
    json.dump(acc_list, open(output_dir, 'w'))
    print('[Finished] Acc: {}'.format(acc))
