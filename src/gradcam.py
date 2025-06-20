import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# from attention import SpatialAttention, ChannelAttention


class ProbBase(object):
    def __init__(self, model, target_layer, relu, device, attention=None):
        self.model = model
        self.device = device
        self.relu = relu
        # self.attention = (
        #     SpatialAttention().to(self.device)
        #     if attention == "spatial"
        #     else (ChannelAttention().to(self.device) if attention == "channel" else None)
        # )
        self.model.to(self.device)
        self.model.eval()
        self.target_layer = target_layer
        self.outputs_backward = OrderedDict()
        self.outputs_forward = OrderedDict()
        self.set_hook_func()

    def set_hook_func(self):
        raise NotImplementedError

    def forward(self, x):
        self.image_size = x.size(-1)
        self.embed = self.model(x)
        return self.embed


class GradCAM(ProbBase):
    def set_hook_func(self):
        def func_f(module, input, f_output):
            self.outputs_forward[id(module)] = f_output

        # print(f'outputs_forward:{self.outputs_forward.values()}\n')

        for module in self.model.named_modules():
            if module[0] == self.target_layer:
                module[1].register_forward_hook(func_f)  # saves output of forward pass

    def get_captured_activations(self):
        """Retrieve the captured activations from the forward pass."""
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                return self.outputs_forward[id(module)]
        raise ValueError(f"Could not find target layer: {self.target_layer}")

    def normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm.item()

    def compute_gradient_weights(self):
        self.grads = self.normalize(self.grads.squeeze())
        self.map_size = self.grads.size()[2:]
        self.alpha = nn.AvgPool2d(self.map_size)(self.grads)

    def generate_from_tensors(self, activations, gradients):
        """
        Generates Grad-CAM heatmaps from provided activation and gradient tensors.
        This method is now stateless and works on a per-batch basis.
        """
        # Normalize gradients and compute weights (alpha)
        # The gradient is for the whole batch
        grads_norm = self.normalize(gradients.squeeze())
        map_size = grads_norm.size()[2:]
        alpha = nn.AvgPool2d(map_size)(grads_norm)  # Shape: [batch, channels, 1, 1]

        # The key is that `activations` and `alpha` (from gradients) are for the same batch
        # Perform weighted combination
        # Using einsum for a clear, batch-wise weighted sum
        # 'bchw,bc->bhw' -> for each item in batch (b), sum channels (c) of (activations * alpha)
        gcam = torch.einsum("bchw,bc->bhw", activations, alpha.squeeze(-1).squeeze(-1))
        gcam = gcam.unsqueeze(1)  # Add channel dim back: [batch, 1, H, W]

        # Upsample and apply ReLU/abs
        gcam = F.interpolate(gcam, (self.image_size, self.image_size), mode="bilinear", align_corners=False)

        if self.relu:
            gcam = F.relu(gcam)
        else:
            gcam = torch.abs(gcam)

        return gcam
