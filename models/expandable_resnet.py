"""
Expandable ResNet architecture for continual learning.
Integrated with lightweight Adapters instead of full parallel branches.
"""

import torch
import torch.nn as nn


class AdapterLayer(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        hidden_dim = max(1, in_channels // reduction)
        self.down = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.up = nn.Conv2d(hidden_dim, in_channels, kernel_size=1, bias=False)
        
        nn.init.kaiming_normal_(self.down.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.relu(self.down(x)))


class ExpandableBottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.adapters = nn.ModuleList()
        self.frozen = False
        
    def add_adapter(self, reduction=16):
        input_dim = self.conv1.in_channels
        output_dim = self.conv3.out_channels
        
        if input_dim == output_dim and self.stride == 1:
            adapter = AdapterLayer(input_dim, reduction=reduction)
            if hasattr(self.conv1, 'weight'):
                device = self.conv1.weight.device
                adapter.to(device)
            self.adapters.append(adapter)
            return True
        return False
        
    def freeze(self):
        self.frozen = True
        for param in [self.conv1, self.bn1, self.conv2, self.bn2, 
                     self.conv3, self.bn3]:
            for p in param.parameters():
                p.requires_grad = False
        if self.downsample is not None:
            for p in self.downsample.parameters():
                p.requires_grad = False
                
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        
        for adapter in self.adapters:
            out = out + adapter(x)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out


class ExpandableResNet(nn.Module):
    def __init__(self, backbone_name='resnet50d', num_classes=100):
        super().__init__()
        
        if backbone_name == 'resnet50d':
            try:
                import timm
                base_model = timm.create_model('resnet50d', pretrained=True)
            except ImportError:
                raise RuntimeError("timm not available. Please install: pip install timm>=0.9.0")
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}. Only 'resnet50d' is supported.")
        
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        
        if hasattr(base_model, 'relu'):
            self.relu = base_model.relu
        elif hasattr(base_model, 'act1'):
            self.relu = base_model.act1
        else:
            self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = base_model.maxpool
        
        def wrap_layer(layer):
            new_layer = nn.Sequential()
            for i, block in enumerate(layer):
                if hasattr(block, 'conv1') and hasattr(block, 'conv2') and hasattr(block, 'conv3'):
                    in_channels = block.conv1.in_channels
                    planes = block.conv1.out_channels
                    
                    stride = 1
                    if hasattr(block, 'stride'):
                        stride = block.stride
                    elif hasattr(block.conv2, 'stride'):
                        s = block.conv2.stride
                        stride = s[0] if isinstance(s, tuple) else s
                    
                    downsample = block.downsample
                    
                    new_block = ExpandableBottleneck(in_channels, planes, stride, downsample)
                    
                    new_block.conv1.load_state_dict(block.conv1.state_dict())
                    new_block.bn1.load_state_dict(block.bn1.state_dict())
                    new_block.conv2.load_state_dict(block.conv2.state_dict())
                    new_block.bn2.load_state_dict(block.bn2.state_dict())
                    new_block.conv3.load_state_dict(block.conv3.state_dict())
                    new_block.bn3.load_state_dict(block.bn3.state_dict())
                    
                    new_layer.add_module(str(i), new_block)
                else:
                    new_layer.add_module(str(i), block)
            return new_layer

        self.layer1 = wrap_layer(base_model.layer1)
        self.layer2 = wrap_layer(base_model.layer2)
        self.layer3 = wrap_layer(base_model.layer3)
        self.layer4 = wrap_layer(base_model.layer4)
        
        if hasattr(base_model, 'avgpool'):
            self.avgpool = base_model.avgpool
        elif hasattr(base_model, 'global_pool'):
            self.avgpool = base_model.global_pool
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            
        if hasattr(base_model.fc, 'in_features'):
            self.feature_dim = base_model.fc.in_features
        elif hasattr(base_model.fc, 'weight'):
            self.feature_dim = base_model.fc.weight.shape[1]
        else:
            self.feature_dim = 2048
            
        self.fc = None
        self.num_classes = num_classes
        self.frozen_modules = set()
        self.expansion_history = []
        
    def add_task_head(self, num_classes: int):
        if self.fc is None:
            self.fc = nn.Linear(self.feature_dim, num_classes)
            self.num_classes = num_classes
            if next(self.parameters()).is_cuda:
                self.fc = self.fc.cuda()
                
    def expand_layer(self, layer_name: str):
        if layer_name not in ['layer1', 'layer2', 'layer3', 'layer4']:
            raise ValueError(f"Invalid layer name: {layer_name}")
        
        layer = getattr(self, layer_name)
        expanded_count = 0
        
        for block in layer:
            if isinstance(block, ExpandableBottleneck):
                if block.add_adapter(reduction=16):
                    expanded_count += 1
        
        if expanded_count > 0:
            print(f"Added adapters to {expanded_count} blocks in {layer_name}")
            self.expansion_history.append({'layer': layer_name, 'type': 'adapter'})
        else:
            print(f"No blocks in {layer_name} were eligible for adapters (stride/dim mismatch)")
            
    def freeze_module(self, module_name: str):
        if module_name == 'fc' and self.fc is not None:
            for param in self.fc.parameters():
                param.requires_grad = False
            self.frozen_modules.add(module_name)
        elif hasattr(self, module_name):
            module = getattr(self, module_name)
            for param in module.parameters():
                param.requires_grad = False
            self.frozen_modules.add(module_name)
        else:
            raise ValueError(f"Module {module_name} not found")
            
    def unfreeze_module(self, module_name: str):
        if module_name in self.frozen_modules:
            if module_name == 'fc' and self.fc is not None:
                for param in self.fc.parameters():
                    param.requires_grad = True
            elif hasattr(self, module_name):
                module = getattr(self, module_name)
                for param in module.parameters():
                    param.requires_grad = True
            self.frozen_modules.discard(module_name)
    
    def get_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        
        if len(x.shape) > 2:
            x = torch.flatten(x, 1)
        
        return x
    
    def forward(self, x):
        # Force float32 for interpolation to avoid AMP errors
        if x.shape[-1] < 224:
            x = x.float() 
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            
        features = self.get_features(x)
        if self.fc is None:
            raise ValueError("Classifier not initialized. Call add_task_head first.")
        return self.fc(features)
        
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def get_effective_num_parameters(self):
        """
        Count parameters that remain active after pruning.

        If torch.nn.utils.prune is applied, parameters are represented as
        <name>_orig with a corresponding <name>_mask buffer. We count only
        mask entries equal to 1 for those tensors. For regular (unmasked)
        parameters we count full tensor size, so naturally-zero values are not
        misinterpreted as pruned capacity.
        """
        total = 0

        for module in self.modules():
            for name, param in module.named_parameters(recurse=False):
                if name.endswith('_orig'):
                    base_name = name[:-5]
                    mask_name = f"{base_name}_mask"
                    if hasattr(module, mask_name):
                        mask = getattr(module, mask_name)
                        total += int(mask.sum().item())
                    else:
                        total += param.numel()
                    continue

                if name.endswith('_mask'):
                    continue

                total += param.numel()

        return int(total)


def create_expandable_resnet(backbone='resnet50d', num_initial_classes=10):
    model = ExpandableResNet(backbone, num_initial_classes)
    model.add_task_head(num_initial_classes)
    return model
