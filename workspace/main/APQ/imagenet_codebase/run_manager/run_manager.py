# Code for "APQ: Joint Search for Network Architecture, Pruning and Quantization Policy"
# CVPR 2020
# Tianzhe Wang, Kuan Wang, Han Cai, Ji Lin, Zhijian Liu, Song Han
# {usedtobe, kuanwang, hancai, jilin, zhijian, songhan}@mit.edu

import json
from datetime import timedelta
import random

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision

from APQ.imagenet_codebase.utils import *
from APQ.elastic_nn.utils import set_running_statistics
#from surgeon_pytorch import Extract, get_nodes

class RunConfig:

    def __init__(self, n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
                 dataset, train_batch_size, test_batch_size, valid_size,
                 opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
                 model_init, validation_frequency, print_frequency):
        self.n_epochs = n_epochs
        self.init_lr = init_lr
        self.lr_schedule_type = lr_schedule_type
        self.lr_schedule_param = lr_schedule_param

        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.valid_size = valid_size

        self.opt_type = opt_type
        self.opt_param = opt_param
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.no_decay_keys = no_decay_keys

        self.model_init = model_init
        self.validation_frequency = validation_frequency
        self.print_frequency = print_frequency

    @property
    def config(self):
        config = {}
        for key in self.__dict__:
            if not key.startswith('_'):
                config[key] = self.__dict__[key]
        return config

    def copy(self):
        return RunConfig(**self.config)

    """ learning rate """

    def calc_learning_rate(self, epoch, batch=0, nBatch=None):
        if self.lr_schedule_type == 'cosine':
            T_total = self.n_epochs * nBatch
            T_cur = epoch * nBatch + batch
            lr = 0.5 * self.init_lr * (1 + math.cos(math.pi * T_cur / T_total))
        elif self.lr_schedule_type is None:
            lr = self.init_lr
        else:
            raise ValueError('do not support: %s' % self.lr_schedule_type)
        return lr

    def adjust_learning_rate(self, optimizer, epoch, batch=0, nBatch=None):
        """ adjust learning of a given optimizer and return the new learning rate """
        new_lr = self.calc_learning_rate(epoch, batch, nBatch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def warmup_adjust_learning_rate(self, optimizer, T_total, nBatch, epoch, batch=0, warmup_lr=0):
        T_cur = epoch * nBatch + batch + 1
        new_lr = T_cur / T_total * (self.init_lr - warmup_lr) + warmup_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    """ data provider """

    @property
    def data_provider(self):
        raise NotImplementedError

    @property
    def train_loader(self):
        return self.data_provider.train

    @property
    def valid_loader(self):
        return self.data_provider.valid

    @property
    def test_loader(self):
        return self.data_provider.test

    def random_sub_train_loader(self, n_images, batch_size, num_worker=None):
        return self.data_provider.build_sub_train_loader(n_images, batch_size, num_worker)

    def random_sub_val_loader(self, n_images, batch_size, num_worker=None):
        return self.data_provider.build_sub_val_loader(n_images, batch_size, num_worker)

    """ optimizer """

    # noinspection PyTypeChecker
    def build_optimizer(self, net_params):
        if self.no_decay_keys is not None:
            assert isinstance(net_params, list) and len(net_params) == 2
            net_params = [
                {'params': net_params[0], 'weight_decay': self.weight_decay},
                {'params': net_params[1], 'weight_decay': 0},
            ]
        else:
            net_params = [{'params': net_params, 'weight_decay': self.weight_decay}]

        if self.opt_type == 'sgd':
            opt_param = {} if self.opt_param is None else self.opt_param
            momentum, nesterov = opt_param.get('momentum', 0.9), opt_param.get('nesterov', True)
            optimizer = torch.optim.SGD(net_params, self.init_lr, momentum=momentum, nesterov=nesterov)
        elif self.opt_type == 'adam':
            optimizer = torch.optim.Adam(net_params, self.init_lr)
        else:
            raise NotImplementedError
        return optimizer


class RunManager:

    def __init__(self, path, net, run_config: RunConfig, init=True, no_gpu=False, print_info=False):
        self.path = path
        self.net = net
        self.run_config = run_config

        self.best_acc = 0
        self.start_epoch = 0

        os.makedirs(self.path, exist_ok=True)

        # move network to GPU if available
        if torch.cuda.is_available() and (not no_gpu):
            self.device = torch.device('cuda:0')
            self.net = self.net.to(self.device)
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        # net info
        net_info = get_net_info(self.net, self.run_config.data_provider.data_shape, print_info=print_info)
        with open('%s/net_info' % self.path, 'w') as fout:
            fout.write(json.dumps(net_info, indent=4) + '\n')

        # criterion
        if self.run_config.label_smoothing > 0:
            self.criterion = lambda pred, target: \
                cross_entropy_with_label_smoothing(pred, target, self.run_config.label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # optimizer
        if self.run_config.no_decay_keys:
            keys = self.run_config.no_decay_keys.split('#')
            net_params = [
                self.network.get_parameters(keys, mode='exclude'),  # parameters with weight decay
                self.network.get_parameters(keys, mode='include'),  # parameters without weight decay
            ]
        else:
            try:
                net_params = self.network.weight_parameters()
            except Exception:
                net_params = self.network.parameters()
        self.optimizer = self.run_config.build_optimizer(net_params)

        # initialize model (default)
        if init:
            self.network.init_model(run_config.model_init)

    """ save path and log path """

    @property
    def save_path(self):
        if self.__dict__.get('_save_path', None) is None:
            save_path = os.path.join(self.path, 'checkpoint')
            os.makedirs(save_path, exist_ok=True)
            self.__dict__['_save_path'] = save_path
        return self.__dict__['_save_path']

    @property
    def logs_path(self):
        if self.__dict__.get('_logs_path', None) is None:
            logs_path = os.path.join(self.path, 'logs')
            os.makedirs(logs_path, exist_ok=True)
            self.__dict__['_logs_path'] = logs_path
        return self.__dict__['_logs_path']

    @property
    def network(self):
        if isinstance(self.net, nn.DataParallel):
            return self.net.module
        else:
            return self.net

    """ save and load models """

    def save_model(self, checkpoint=None, is_best=False, model_name=None):
        if checkpoint is None:
            checkpoint = {'state_dict': self.network.state_dict()}

        if model_name is None:
            model_name = 'checkpoint.pth.tar'

        checkpoint['dataset'] = self.run_config.dataset  # add `dataset` info to the checkpoint
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        model_path = os.path.join(self.save_path, model_name)
        with open(latest_fname, 'w') as fout:
            fout.write(model_path + '\n')
        torch.save(checkpoint, model_path)

        if is_best:
            best_path = os.path.join(self.save_path, 'model_best.pth.tar')
            torch.save({'state_dict': checkpoint['state_dict']}, best_path)

    def load_model(self, model_fname=None):
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        if model_fname is None and os.path.exists(latest_fname):
            with open(latest_fname, 'r') as fin:
                model_fname = fin.readline()
                if model_fname[-1] == '\n':
                    model_fname = model_fname[:-1]
        # noinspection PyBroadException
        try:
            if model_fname is None or not os.path.exists(model_fname):
                model_fname = '%s/checkpoint.pth.tar' % self.save_path
                with open(latest_fname, 'w') as fout:
                    fout.write(model_fname + '\n')
            print("=> loading checkpoint '{}'".format(model_fname))

            if torch.cuda.is_available():
                checkpoint = torch.load(model_fname)
            else:
                checkpoint = torch.load(model_fname, map_location='cpu')

            self.network.load_state_dict(checkpoint['state_dict'])
            # set new manual seed
            new_manual_seed = int(time.time())
            torch.manual_seed(new_manual_seed)
            torch.cuda.manual_seed_all(new_manual_seed)
            np.random.seed(new_manual_seed)
            random.seed(new_manual_seed)

            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1
            if 'best_acc' in checkpoint:
                self.best_acc = checkpoint['best_acc']
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            print("=> loaded checkpoint '{}'".format(model_fname))
        except Exception:
            print('fail to load checkpoint from %s' % self.save_path)

    def save_config(self):
        """ dump run_config and net_config to the model_folder """
        net_save_path = os.path.join(self.path, 'net.config')
        json.dump(self.network.config, open(net_save_path, 'w'), indent=4)
        print('Network configs dump to %s' % net_save_path)

        run_save_path = os.path.join(self.path, 'run.config')
        json.dump(self.run_config.config, open(run_save_path, 'w'), indent=4)
        print('Run configs dump to %s' % run_save_path)

    def write_log(self, log_str, prefix, should_print=True):
        """ prefix: valid, train, test """
        if prefix in ['valid', 'test']:
            with open(os.path.join(self.logs_path, 'valid_console.txt'), 'a') as fout:
                fout.write(log_str + '\n')
                fout.flush()
        if prefix in ['valid', 'test', 'train']:
            with open(os.path.join(self.logs_path, 'train_console.txt'), 'a') as fout:
                if prefix in ['valid', 'test']:
                    fout.write('=' * 10)
                fout.write(log_str + '\n')
                fout.flush()
        else:
            with open(os.path.join(self.logs_path, '%s.txt' % prefix), 'a') as fout:
                fout.write(log_str + '\n')
                fout.flush()
        if should_print:
            print(log_str)

    """ train and test """

    def validate(self, epoch=0, is_test=True, net=None, use_train_mode=False, no_logs=False, run_str=None, sub=False,
                 in_mem=None):

        if is_test:
            data_loader = self.run_config.test_loader
        else:
            data_loader = self.run_config.valid_loader

        if sub:
            data_loader = self.run_config.random_sub_val_loader(2000, 250)

        if in_mem is not None:
            data_loader = in_mem

        if net is None:
            net = self.net
        if not isinstance(net, nn.DataParallel):
            net = nn.DataParallel(net)

        #to display the intermediate output
        activation = {}
        def get_activation(name):
            def hook(model, input, output): 
                activation[name] = output.detach()
            return hook
        
        
        #layerNeeded='blocks.0.mobile_inverted_conv.depth_conv.conv'
        
        #layerNeeded='classifier.linear'
        #'feature_mix_layer.conv'
        #'classifier.linear'
        #net.register_forward_hook(get_activation(layerNeeded))
        
        net.eval()
        
        if use_train_mode:
            net = copy.deepcopy(net)
            # set bn layers to train mode
            for m in net.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    m.train()

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        end = time.time()
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                # compute output
                output = net(images)
                #added here a line to check intermediate outputs
                
                #this is to display the hook for activations
                #if i==1:
                    #print(activation[layerNeeded])
                    
                    #print(dir(net))
                    
                    #print(net)
                    #print(output)
                    #print(output.dtype)
                loss = self.criterion(output, labels)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self.run_config.print_frequency == 0 or i + 1 == len(data_loader):
                    if is_test:
                        prefix = 'Test [%d]' % (epoch + 1)
                    else:
                        prefix = 'Valid [%d]' % (epoch + 1)

                    test_log = prefix + ': [{0}/{1}]\t' \
                                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                                        'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})'. \
                        format(i, len(data_loader) - 1, batch_time=batch_time, loss=losses, top1=top1)
                    test_log += '\tTop-5 acc {top5.val:.3f} ({top5.avg:.3f})'.format(top5=top5)
                    test_log += '\t' + 'img_size: %d' % images.size(2)
                    if run_str is not None:
                        test_log += '\t' + run_str
                    if not no_logs:
                        print(test_log)

        return losses.avg, top1.avg, top5.avg

    def val_get_act(self, layer_names, epoch=0, is_test=True, net=None, use_train_mode=False, no_logs=False, run_str=None, sub=False,
                 in_mem=None):

        if is_test:
            data_loader = self.run_config.test_loader
        else:
            data_loader = self.run_config.valid_loader

        if sub:
            data_loader = self.run_config.random_sub_val_loader(2000, 250)

        if in_mem is not None:
            data_loader = in_mem

        if net is None:
            net = self.net
        if not isinstance(net, nn.DataParallel):
            net = nn.DataParallel(net)

        #to display the intermediate output
        activation = {}
        #inputn = {}
        histsinp={}
        histsout={}
        layer_name_map = {}
        inputshape={}
        outshape={}
        # Function to print input/output shapes for each layer with its name
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
        
        def compute_symmetric_zero_centered_histogramNoquant(data, N):
            # Flatten the 4D input tensor to a 1D tensor
            data = data.flatten()
            #print(data.shape)
            # Number of bins based on N
            num_bins = 2**N - 1
            # Define bin edges symmetrically around zero
            min_val, max_val = -1.0, 1.0  # Customize range as needed
            bins = torch.linspace(min_val, max_val, num_bins + 1)

            # Compute histogram
            hist = torch.histc(data, bins=num_bins, min=min_val, max=max_val)
            
            # Normalize the histogram to sum to 1.0
            hist = hist / hist.sum()
            
            # Optionally make the histogram zero-centered (centermost bin to zero)
            # Find the index of the center bin
            center_idx = num_bins // 2
            hist[center_idx] = 0.0  # Set the centermost bin to zero

            # Re-normalize the histogram after zero-centering
            hist = hist / hist.sum()
            
            return hist  # Exclude the last bin edge
        
        
        def get_inputs(name):
            def hook(module, input, output):
                # Get the layer name from the dictionary using the module as the key
                #layer_name = layer_name_map[module]
                input_shape = input[0].shape if isinstance(input, tuple) else input.shape
                output_shape = output.shape if output is not None else None
                inputshape[name]=list(input_shape)
                outshape[name]=list(output_shape)
                # Check if the module has a weight attribute and print its shape
                #if hasattr(module, 'weight') and module.weight is not None:
                #    weight_shape = module.weight.shape
                #else:
                #    weight_shape = None
                
                #print(f"module: {module} | Weight shape: {weight_shape}")
                #print(f"module: {module} | Weight shape: {input_shape}")
                
                #min_val = torch.min(input[0])
                #max_val = torch.max(input[0])
                #normalized_tensor = (input[0] - min_val) / (max_val - min_val)
                #meanval=torch.mean(normalized_tensor)
                #print('')
                #print(module)
                #print(meanval)
                #if name=='blocks.12.mobile_inverted_conv.depth_conv.conv':
                    #print(name)
                    #print(output.detach())
                    #print(output.detach().shape)
                    #print(compute_symmetric_zero_centered_histogram(output.detach(), N=5))
                histsinp[name]=compute_symmetric_zero_centered_histogram(input[0], N=5).tolist()
                
                histsout[name]=compute_symmetric_zero_centered_histogram(output.detach(), N=5).tolist()
                #inputn[name] = meanval
            return hook

            #print(f"Layer: {layer_name} | Input shape: {input_shape} | Output shape: {output_shape} | Weight shape: {weight_shape}")


        # Register hooks on all layers including inverted bottleneck and other layers
        # Recursive function to register hooks on all layers and sub-layers (including sub-layers of sub-layers)
        
        def register_hooks_using_state_dict(model):
            hooks = []
            hook = model.module.first_conv.conv.register_forward_hook(get_inputs('first_conv.conv'))
            hooks.append(hook)
            #print('ok')
            #hook = model.module.feature_mix_layer.conv.register_forward_hook(hook_fn)
            hook = model.module.feature_mix_layer.conv.register_forward_hook(get_inputs('feature_mix_layer.conv'))
            hooks.append(hook)
            #hook = model.module.classifier.linear.register_forward_hook(hook_fn)
            hook = model.module.classifier.linear.register_forward_hook(get_inputs('classifier.linear'))
            hooks.append(hook)

            
            for idx, module in enumerate(model.module.blocks):
                #print(f"Registering hook for module {idx} - {module.__class__.__name__}")
                name1='blocks.'+str(idx)+'.'
                for idx2, module2 in module.named_modules():
                    #print(f"Registering hook for module {idx2} - {module2.__class__.__name__}")
                    #if module2.__class__.__name__=='Sequential':
                    #    name2=str(idx2)+'.conv'
                    #    fullname=name1+name2
                    #    hook = module2.register_forward_hook(get_inputs(fullname))
                    #    hooks.append(hook)
                        
                    if module2.__class__.__name__=='QConv2d':
                        #name2=str(idx2)+'.conv'
                        name2=str(idx2)
                        #print(f"Registering hook for module {idx2} - {module2.__class__.__name__}")
                        #print(dir(module))
                        fullname=name1+name2
                        #print(fullname)
                        hook = module2.register_forward_hook(get_inputs(fullname))
                        hooks.append(hook)
            #print(len(hooks))
            return hooks
        
        # Register hooks for all submodules
        def register_hooks_using_state_dict3(model):
            hooks = []
            # Loop through all named submodules (including sublayers)
            for name, module in model.named_modules():
                # Skip the top-level model itself
                if name == "":
                    continue
                
                if isinstance(module, nn.Conv2d) or isinstance(module, QConv2d) or isinstance(module, QLinear):
                    #print(f"Registering hook for: {name} - {module.__class__.__name__}")
                    hook = module.register_forward_hook(hook_fn)
                    hooks.append(hook)
                    #layer_name_map[module] = name
            return hooks
        
        

        
        #for layer in net.children():
        #    hook = layer.register_forward_hook(hook_fn)
        #    hooks.append(hook)
        
        def get_activation(name):
            def hook(model, input, output):
                inputn[name] = input[0]
                activation[name] = output.detach()
                print(f"Layer: {model.__class__.__name__} | Input shape: {input[0].shape}")
            return hook
        
        #hooks = register_hooks(net)
        hooks = register_hooks_using_state_dict(net)
        # Store inputs for each layer
        
        #layerNeeded='blocks.0.mobile_inverted_conv.depth_conv.conv'
        
        #layerNeeded1='classifier.linear'
       # layerNeeded2='feature_mix_layer.conv'
        #hooks = []
        #for layerNeeded in layer_names:
            #print(layerNeeded)
        #    hook = net.register_forward_hook(get_activation(layerNeeded))
        #    hooks.append(hook)
        #'feature_mix_layer.conv'
        #'classifier.linear'
        
        #net.register_forward_hook(get_activation(layerNeeded1))
        #net.register_forward_hook(get_activation(layerNeeded2))
        net.eval()
        
        if use_train_mode:
            net = copy.deepcopy(net)
            # set bn layers to train mode
            for m in net.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    m.train()

        #batch_time = AverageMeter()
        #losses = AverageMeter()
        #top1 = AverageMeter()
        #top5 = AverageMeter()

        end = time.time()
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader):
                if i>=1:
                    break
                #print('i')
                #print(i)
                images, labels = images.to(self.device), labels.to(self.device)
                # compute output
                output = net(images)
                #added here a line to check intermediate outputs
                #print(torch.mean(activation[layerNeeded1]))
                #print(torch.mean(activation[layerNeeded2]))
                #exit()
                #print(inputn)
                
                #for key, value in inputn.items():
                #    print(f"{key}: {value}")
                #exit()
                #for key, value in hists.items():
                #    print(f"{key}: {value}")
                
        for hook in hooks:
            hook.remove()
        return histsinp, histsout, inputshape, outshape
    
    
    def train_one_epoch(self, adjust_lr_func, train_log_func):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to train mode
        self.net.train()

        end = time.time()
        for i, (images, labels) in enumerate(self.run_config.train_loader):
            data_time.update(time.time() - end)
            new_lr = adjust_lr_func(i)
            images, labels = images.to(self.device), labels.to(self.device)
            #print('images', images.shape)
            #print('labels', labels.shape)

            # compute output
            if isinstance(self.network, torchvision.models.Inception3):
                output, aux_outputs = self.net(images)
                loss1 = self.criterion(output, labels)
                loss2 = self.criterion(aux_outputs, labels)
                loss = loss1 + 0.4 * loss2
            else:
                output = self.net(images)
                loss = self.criterion(output, labels)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))

            # compute gradient and do SGD step
            self.net.zero_grad()  # or self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.run_config.print_frequency == 0 or i + 1 == len(self.run_config.train_loader):
                batch_log = train_log_func(i, batch_time, data_time, losses, top1, top5, new_lr)
                batch_log += '\t' + 'img_size: %d' % images.size(2)
                # print(batch_log)
                self.write_log(batch_log, 'train')
        return top1, top5

    def train(self, warmup_epoch=0, warmup_lr=0):
        nBatch = len(self.run_config.train_loader)
        total_epochs = self.run_config.n_epochs + warmup_epoch

        def train_log_func(epoch_, i, batch_time, data_time, losses, top1, top5, lr):
            batch_log = 'Train [{0}][{1}/{2}]\t' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                        'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                        'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})'. \
                format(epoch_ + 1, i, nBatch - 1,
                       batch_time=batch_time, data_time=data_time, losses=losses, top1=top1)
            batch_log += '\tTop-5 acc {top5.val:.3f} ({top5.avg:.3f})'.format(top5=top5)
            batch_log += '\tlr {lr:.5f}'.format(lr=lr)
            return batch_log

        for epoch in range(self.start_epoch, total_epochs):
            print('\n', '-' * 30, 'Train epoch: %d' % (epoch + 1), '-' * 30, '\n')

            end = time.time()
            if epoch < warmup_epoch:
                adjust_lr_func = lambda i: self.run_config.warmup_adjust_learning_rate(
                    self.optimizer, warmup_epoch * nBatch, nBatch, epoch, i, warmup_lr
                )
            else:
                adjust_lr_func = lambda i: self.run_config.adjust_learning_rate(
                    self.optimizer, epoch - warmup_epoch, i, nBatch
                )

            train_top1, train_top5 = self.train_one_epoch(
                adjust_lr_func,
                lambda i, batch_time, data_time, losses, top1, top5, new_lr:
                train_log_func(epoch, i, batch_time, data_time, losses, top1, top5, new_lr),
            )
            time_per_epoch = time.time() - end
            seconds_left = int((total_epochs - epoch - 1) * time_per_epoch)
            print('Time per epoch: %s, Est. complete in: %s' % (
                str(timedelta(seconds=time_per_epoch)),
                str(timedelta(seconds=seconds_left))))

            if (epoch + 1) % self.run_config.validation_frequency == 0:
                val_loss, val_acc, val_acc5 = self.validate(epoch=epoch, is_test=False)

                is_best = np.mean(val_acc) > self.best_acc
                self.best_acc = max(self.best_acc, np.mean(val_acc))
                val_log = 'Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f} ({4:.3f})'. \
                    format(epoch + 1 - warmup_epoch, self.run_config.n_epochs,
                           val_loss, val_acc, self.best_acc)
                val_log += '\ttop-5 acc {0:.3f}\tTrain top-1 {top1.avg:.3f}\ttop-5 {top5.avg:.3f}\t'. \
                    format(np.mean(val_acc5), top1=train_top1, top5=train_top5)
                self.write_log(val_log, 'valid')
            else:
                is_best = False

            self.save_model({
                'epoch': epoch,
                'best_acc': self.best_acc,
                'optimizer': self.optimizer.state_dict(),
                'state_dict': self.network.state_dict(),
            }, is_best=is_best)

    def finetune(self, warmup_epoch=0, warmup_lr=0, in_mem=None):
        nBatch = len(self.run_config.train_loader)
        print('nBatch: ', nBatch)
        total_epochs = self.run_config.n_epochs + warmup_epoch
        print('total_epochs: ', total_epochs)

        def finetune_log_func(epoch_, i, batch_time, data_time, losses, top1, top5, lr):
            batch_log = 'Finetune [{0}][{1}/{2}]\t' \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                        'Loss {losses.val:.4f} ({losses.avg:.4f})\t' \
                        'Top-1 acc {top1.val:.3f} ({top1.avg:.3f})'. \
                format(epoch_ + 1, i, nBatch - 1,
                       batch_time=batch_time, data_time=data_time, losses=losses, top1=top1)
            batch_log += '\tTop-5 acc {top5.val:.3f} ({top5.avg:.3f})'.format(top5=top5)
            batch_log += '\tlr {lr:.5f}'.format(lr=lr)
            return batch_log

        for epoch in range(self.start_epoch, total_epochs):
            print('\n', '-' * 30, 'Finetune epoch: %d' % (epoch + 1), '-' * 30, '\n')

            end = time.time()
            if epoch < warmup_epoch:
                print('case: epoch < warmup_epoch')
                adjust_lr_func = lambda i: self.run_config.warmup_adjust_learning_rate(
                    self.optimizer, warmup_epoch * nBatch, nBatch, epoch, i, warmup_lr
                )
            else:
                print('case: epoch >= warmup_epoch')
                adjust_lr_func = lambda i: self.run_config.adjust_learning_rate(
                    self.optimizer, epoch - warmup_epoch, i, nBatch
                )
            print('adjust_lr_func', adjust_lr_func)
            
            print('train one epoch')
            train_top1, train_top5 = self.train_one_epoch(
                adjust_lr_func,
                lambda i, batch_time, data_time, losses, top1, top5, new_lr:
                finetune_log_func(epoch, i, batch_time, data_time, losses, top1, top5, new_lr),
            )
            time_per_epoch = time.time() - end
            seconds_left = int((total_epochs - epoch - 1) * time_per_epoch)
            print('Time per epoch: %s, Est. complete in: %s' % (
                str(timedelta(seconds=time_per_epoch)),
                str(timedelta(seconds=seconds_left))))

            if (epoch + 1) % self.run_config.validation_frequency == 0:
                val_loss, val_acc, val_acc5 = self.validate(epoch=epoch, is_test=True, in_mem=in_mem)

                is_best = np.mean(val_acc) > self.best_acc
                self.best_acc = max(self.best_acc, np.mean(val_acc))
                val_log = 'Valid [{0}/{1}]\tloss {2:.3f}\ttop-1 acc {3:.3f} ({4:.3f})'. \
                    format(epoch + 1 - warmup_epoch, self.run_config.n_epochs,
                           val_loss, val_acc, self.best_acc)
                val_log += '\ttop-5 acc {0:.3f}\tFinetune top-1 {top1.avg:.3f}\ttop-5 {top5.avg:.3f}\t'. \
                    format(np.mean(val_acc5), top1=train_top1, top5=train_top5)
                self.write_log(val_log, 'valid')
            else:
                is_best = False

            self.save_model({
                'epoch': epoch,
                'best_acc': self.best_acc,
                'optimizer': self.optimizer.state_dict(),
                'state_dict': self.network.state_dict(),
            }, is_best=is_best)

        return self.best_acc

    def reset_running_statistics(self, net=None, val_mode=None):
        if net is None:
            net = self.network
        if not val_mode:
            print('-' * 30, 'Reset Running Statistics', '-' * 30)
        sub_train_loader = self.run_config.random_sub_train_loader(2000, 250)
        set_running_statistics(net, sub_train_loader)
        calibrate(net, sub_train_loader)

    def after_set_bit(self, net=None, val_mode=None):
        if net is None:
            net = self.network
        if not val_mode:
            print('-' * 30, 'After Set Bit', '-' * 30)
        sub_train_loader = self.run_config.random_sub_train_loader(2000, 250)
        # set_running_statistics(net, sub_train_loader)
        calibrate(net, sub_train_loader)
