"""
Expandable ResNet architecture for continual learning.
Integrated with lightweight Adapters instead of full parallel branches.
"""

import torch
import torch.nn as nn


class AdapterLayer(nn.Module):
    def __init__(self, in_channels, out_channels=None, stride=1, reduction=16):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        hidden_dim = max(1, in_channels // reduction)
        self.down = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=stride, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.up = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        
        nn.init.kaiming_normal_(self.down.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.relu(self.down(x)))


class ExpandableBottleneck(nn.Module):
    expansion = 4
    
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        downsample=None,
        adapt_downsample_blocks=True
    ):
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
        self.adapt_downsample_blocks = bool(adapt_downsample_blocks)
        
    def add_adapter(self, reduction=16):
        input_dim = self.conv1.in_channels
        output_dim = self.conv3.out_channels
        stride = self.stride[0] if isinstance(self.stride, tuple) else self.stride
        if not self.adapt_downsample_blocks and (input_dim != output_dim or stride != 1):
            return False

        adapter = AdapterLayer(
            input_dim,
            out_channels=output_dim,
            stride=self.stride,
            reduction=reduction
        )
        if hasattr(self.conv1, 'weight'):
            device = self.conv1.weight.device
            adapter.to(device)
        self.adapters.append(adapter)
        return True
        
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
    def __init__(
        self,
        backbone_name='resnet50d',
        num_classes=100,
        adapt_downsample_blocks=True
    ):
        super().__init__()
        self.adapt_downsample_blocks = bool(adapt_downsample_blocks)
        
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
                    
                    new_block = ExpandableBottleneck(
                        in_channels,
                        planes,
                        stride,
                        downsample,
                        adapt_downsample_blocks=self.adapt_downsample_blocks
                    )
                    
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
        self.expansion_history = []
        
    def add_task_head(self, num_classes: int):
        num_classes = int(num_classes)
        if self.fc is None:
            self.fc = nn.Linear(self.feature_dim, num_classes)
            self.num_classes = num_classes
            if next(self.parameters()).is_cuda:
                self.fc = self.fc.cuda()
            return

        current_classes = int(self.fc.out_features)
        if current_classes == num_classes:
            self.num_classes = num_classes
            return

        device = self.fc.weight.device
        new_fc = nn.Linear(self.feature_dim, num_classes).to(device)
        with torch.no_grad():
            copy_classes = min(current_classes, num_classes)
            new_fc.weight[:copy_classes].copy_(self.fc.weight[:copy_classes])
            if self.fc.bias is not None and new_fc.bias is not None:
                new_fc.bias[:copy_classes].copy_(self.fc.bias[:copy_classes])

        self.fc = new_fc
        self.num_classes = num_classes
                
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
        features = self.get_features(x)
        if self.fc is None:
            raise ValueError("Classifier not initialized. Call add_task_head first.")
        return self.fc(features)
        
    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def get_effective_num_parameters(self):
        return self.get_num_parameters()


def create_expandable_resnet(
    backbone='resnet50d',
    num_initial_classes=10,
    adapt_downsample_blocks=True
):
    model = ExpandableResNet(
        backbone,
        num_initial_classes,
        adapt_downsample_blocks=adapt_downsample_blocks
    )
    model.add_task_head(num_initial_classes)
    return model
