## Modified from Torch-Pruning Package (https://github.com/VainF/Torch-Pruning/tree/master)

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))))
import torch
import torch.nn.functional as F
import torch_pruning as tp
import timm
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from tqdm import tqdm
from torchvision.transforms.functional import InterpolationMode
from torch.utils.mobile_optimizer import optimize_for_mobile

import presets

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Timm ViT Pruning')
    parser.add_argument('--model_name', default='deit_base_patch16_224', type=str, help='model name')
    parser.add_argument('--data_path', default='/p/vast1/MLdata/james-imagenet/', type=str, help='model name')
    parser.add_argument('--test_accuracy', default=False, action='store_true', help='test accuracy')
    parser.add_argument('--val_batch_size', default=128, type=int, help='val batch size')
    parser.add_argument('--load_from', default='pruned/model_taylor_0.25.pth', type=str, help='load the pruned model')
    args = parser.parse_args()
    return args

def forward(self, x):
    """https://github.com/huggingface/pytorch-image-models/blob/054c763fcaa7d241564439ae05fbe919ed85e614/timm/models/vision_transformer.py#L79"""
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)
    self.fused_attn = hasattr(torch.nn.functional, 'scaled_dot_product_attention')  # FIXME

    if self.fused_attn:
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p,
        )
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, -1) # original implementation: x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x

def prepare_imagenet(imagenet_root, train_batch_size=64, val_batch_size=128, num_workers=4, use_imagenet_mean_std=False):
    """The imagenet_root should contain train and val folders.
    """

    print('Parsing dataset...')
    val_dst = ImageFolder(os.path.join(imagenet_root, 'val'), 
                          transform=presets.ClassificationPresetEval(
                                mean=[0.485, 0.456, 0.406] if use_imagenet_mean_std else [0.5, 0.5, 0.5],
                                std=[0.229, 0.224, 0.225] if use_imagenet_mean_std else [0.5, 0.5, 0.5],
                                crop_size=224,
                                resize_size=256,
                                interpolation=InterpolationMode.BILINEAR,
                            )
    )
    val_loader = torch.utils.data.DataLoader(val_dst, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)
    return val_loader

def validate_model(model, val_loader, device, flag=0):
    model.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        for k, (images, labels) in enumerate(tqdm(val_loader)):
            if (flag == 0):
              images, labels = images.cuda(), labels.cuda()
            else:
              images, labels = images.cpu(), labels.cpu()
            outputs = model(images)
            loss += torch.nn.functional.cross_entropy(outputs, labels, reduction='sum').item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
    return correct / len(val_loader.dataset), loss / len(val_loader.dataset)

def main():
    args = parse_args()
    device = 'cpu'

    if args.test_accuracy:
        val_loader = prepare_imagenet(args.data_path, train_batch_size=args.train_batch_size, val_batch_size=args.val_batch_size, use_imagenet_mean_std=args.use_imagenet_mean_std)

    # Load the original model
    model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True).to(device)
    model = model.cuda()

    model.eval()
    model = model.cpu()
    img = torch.randn(1,3,224,224)

    scripted_model = torch.jit.script(model)                                                                       
    scripted_model.save("vit_unpruned_scripted.pt")                                                                       
    optimized_scripted_model = torch.utils.mobile_optimizer.optimize_for_mobile(scripted_model)                                                 
    optimized_scripted_model.save("vit_unpruned_optimized_scripted.pt")                                                   
    optimized_scripted_model._save_for_lite_interpreter("vit_unpruned_optimized_scripted_lite.ptl")                       
    ptl_unquant = torch.jit.load("vit_unpruned_optimized_scripted_lite.ptl")

    with torch.autograd.profiler.profile(use_cuda=False) as prof1:
       out = ptl_unquant(img)                                                                                     
    print("Unpruned non-quantized lite model: {:.2f}ms".format(prof1.self_cpu_time_total/1000))


    scripted_model = torch.jit.script(model)
    scripted_model.save("vit_unpruned_scripted.pt")

    backend = "x86" # replaced with ``qnnpack`` causing much worse inference speed for quantized model on this notebook
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend

    quantized_model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
    scripted_quantized_model = torch.jit.script(quantized_model)
    scripted_quantized_model.save("vit_unpruned_scripted_quantized.pt")
    
    from torch.utils.mobile_optimizer import optimize_for_mobile
    optimized_scripted_quantized_model = optimize_for_mobile(scripted_quantized_model)
    optimized_scripted_quantized_model.save("vit_unpruned_optimized_scripted_quantized.pt")

    optimized_scripted_quantized_model._save_for_lite_interpreter("vit_unpruned_optimized_scripted_quantized_lite.ptl")
    ptl = torch.jit.load("vit_unpruned_optimized_scripted_quantized_lite.ptl")

    with torch.autograd.profiler.profile(use_cuda=False) as prof2:
      out = ptl(img)

    print("Unpruned quantized lite model: {:.2f}ms".format(prof2.self_cpu_time_total/1000))

    #Load pruned model
    model = torch.load(args.load_from, map_location=torch.device('cpu'))

    if args.test_accuracy:
        print("Testing accuracy of the pruned model...")
        flag = 1
        acc_pruned, loss_pruned = validate_model(model, val_loader, device, flag)
        print("Accuracy: %.4f, Loss: %.4f"%(acc_pruned, loss_pruned))

    model.eval()
    model = model.cpu()
    scripted_model = torch.jit.script(model)
    scripted_model.save("vit_pruned_scripted.pt")

    backend = "x86" # replaced with ``qnnpack`` causing much worse inference speed for quantized model on this notebook
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend

    quantized_model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
    scripted_quantized_model = torch.jit.script(quantized_model)
    scripted_quantized_model.save("vit_pruned_scripted_quantized.pt")

    if args.test_accuracy:
        print("Testing accuracy of the pruned and quantized model...")
        flag = 1
        acc_pruned, loss_pruned = validate_model(scripted_quantized_model, val_loader, device, flag)
        print("Accuracy: %.4f, Loss: %.4f"%(acc_pruned, loss_pruned))

    optimized_scripted_quantized_model = optimize_for_mobile(scripted_quantized_model)
    optimized_scripted_quantized_model.save("vit_pruned_optimized_scripted_quantized.pt")

    optimized_scripted_quantized_model._save_for_lite_interpreter("vit_pruned_optimized_scripted_quantized_lite.ptl")
    ptl = torch.jit.load("vit_pruned_optimized_scripted_quantized_lite.ptl")

    img = torch.randn(1,3,224,224)

    with torch.autograd.profiler.profile(use_cuda=False) as prof3:
      out = ptl(img)

    print("Pruned quantized lite model: {:.2f}ms".format(prof3.self_cpu_time_total/1000))

if __name__=='__main__':
    main()
