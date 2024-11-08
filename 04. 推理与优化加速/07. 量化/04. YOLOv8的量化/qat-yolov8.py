import torch
import os, argparse
from torch.cuda import amp
import torch.optim as optim
from pytorch_quantization import nn as quant_nn
import tqdm, json
from typing import Callable
from copy import deepcopy
from ultralytics.utils.torch_utils import init_seeds

# from ultralytics.utils import DEFAULT_CFG as cfg
# DEFAULT_CFG -> 默认的配置文件路径: /home/anaconda3/envs/quant/lib/python3.8/site-packages/ultralytics/cfg/default.yaml
import quantize

from ultralytics.cfg import get_cfg
cfg = get_cfg("ultralytics/ultralytics/cfg/default.yaml")

print("torch.__version__:  ", torch.__version__)

class SummaryTool:
    def __init__(self, file):
        self.file = file
        self.data = []

    def append(self, item):
        self.data.append(item)
        json.dump(self.data, open(self.file, "w"), indent=4)


from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.downloads import attempt_download_asset as attempt_download
from ultralytics.nn.modules import Conv
import torch.nn as nn
from ultralytics.utils import yaml_load, IterableSimpleNamespace

# from ultralytics.engine.model import Model
def load_yolov8_model(weight, device) -> DetectionModel:
    attempt_download(weight)
    model = torch.load(weight, map_location=device)["model"]
    for m in model.modules():
        if type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    model.args = cfg  #  model.args : 字典类型  改为  cfg: IterableSimpleNamespace类型
    model.float()
    model.eval()

    with torch.no_grad():
        model.fuse()
    return model



from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.utils import clean_url, emojis
def build_dataset(cfg, img_path, mode='train', batch=None, gs=32):
    """Build YOLO Dataset."""
    cfg.data = "ultralytics/ultralytics/cfg/datasets/coco128.yaml"
    try:
        if cfg.task == 'classify':
            data = check_cls_dataset(cfg.data)
        elif cfg.data.split('.')[-1] in ('yaml', 'yml') or cfg.task in ('detect', 'segment', 'pose'):
            data = check_det_dataset(cfg.data)
            if 'yaml_file' in data:
                cfg.data = data['yaml_file']  # for validating 'yolo train data=url.zip' usage
    except Exception as e:
        raise RuntimeError(emojis(f"Dataset '{clean_url(cfg.data)}' error ❌ {e}")) from e
    return build_yolo_dataset(cfg, img_path, batch, data, mode=mode, rect=mode == 'val', stride=gs)

from ultralytics.utils import LOGGER
def get_dataloader(cfg, dataset_path, batch_size=16, mode='train', gs=32):
    assert mode in ['train', 'val']
    # with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
    dataset = build_dataset(cfg, dataset_path, mode, batch=batch_size, gs=gs)
    shuffle = mode == 'train'
    if getattr(dataset, 'rect', False) and shuffle:
        LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
        shuffle = False
    workers = cfg.workers if mode == 'train' else cfg.workers * 2
    return build_dataloader(dataset, batch_size, workers, shuffle)  # return dataloader



def run_qat(weight, cocodir, device, ignore_policy, save_ptq, save_qat, supervision_stride, iters, eval_origin, eval_ptq):
    quantize.initialize()

    if save_ptq and os.path.dirname(save_ptq) != "":
        os.makedirs(os.path.dirname(save_ptq), exist_ok=True)

    if save_qat and os.path.dirname(save_qat) != "":
        os.makedirs(os.path.dirname(save_qat), exist_ok=True)
    
    device  = torch.device(device)
    print("Load model ....")
    model   = load_yolov8_model(weight, device)
    print("Load dataset ....")
    train_dataloader = get_dataloader(cfg, cocodir + "images/train2017", batch_size=cfg.batch, mode='train')
    val_dataloader   = get_dataloader(cfg, cocodir + "images/train2017", batch_size=cfg.batch, mode='val')
    print("Insert QDQ ....")
    quantize.replace_bottleneck_forward(model)
    quantize.replace_to_quantization_module(model, ignore_policy=ignore_policy)
    print("Apply custom_rules ....")
    quantize.apply_custom_rules_to_quantizer(model, export_onnx)
    print("Calibrate model ....")
    quantize.calibrate_model(model, train_dataloader, device)

    json_save_dir = "." if os.path.dirname(save_ptq) == "" else os.path.dirname(save_ptq)
    summary_file = os.path.join(json_save_dir, "summary.json")
    summary = SummaryTool(summary_file)

    if eval_origin:
        print("Evaluate Origin...")
        with quantize.disable_quantization(model):
            ap = evaluate_coco(model, val_dataloader, True, json_save_dir)
            summary.append(["Origin", ap])

    if eval_ptq:
        print("Evaluate PTQ...")
        ap = evaluate_coco(model, val_dataloader, True, json_save_dir)
        summary.append(["PTQ", ap])

    if save_ptq:
        print(f"Save ptq model to {save_ptq}")
        torch.save({"model": model}, save_ptq)

    if save_qat is None:
        print("Done as save_qat is None.")
        return

    best_ap = 0
    def per_epoch(model, epoch, lr):

        nonlocal best_ap
        ap = evaluate_coco(model, val_dataloader)
        summary.append([f"QAT{epoch}", ap])

        if ap > best_ap:
            print(f"Save qat model to {save_qat} @ {ap:.5f}")
            best_ap = ap
            torch.save({"model": model}, save_qat)

    # def preprocess(datas):
    #     return datas[0].to(device).float() / 255.0

    def preprocess_batch(batch, device):
        batch['img'] = batch['img'].to(device, non_blocking=True).float() / 255
        return batch

    def supervision_policy():
        supervision_list = []
        for item in model.model:
            supervision_list.append(id(item))

        keep_idx = list(range(0, len(model.model) - 1, supervision_stride))
        keep_idx.append(len(model.model) - 2)
        def impl(name, module):
            if id(module) not in supervision_list: return False
            idx = supervision_list.index(id(module))
            if idx in keep_idx:
                print(f"Supervision: {name} will compute loss with origin model during QAT training")
            else:
                print(f"Supervision: {name} no compute loss during QAT training, that is unsupervised only and doesn't mean don't learn")
            return idx in keep_idx
        return impl

    quantize.finetune(
        model, train_dataloader, per_epoch, early_exit_batchs_per_epoch=iters, 
        preprocess=preprocess_batch, supervision_policy=supervision_policy())
    


import torch, onnx
import onnxsim
from ultralytics.utils.checks import check_imgsz, check_requirements
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.torch_utils import get_latest_opset
def export_onnx(model : DetectionModel, save_file, size=640, dynamic_batch=False, noanchor=False, prefix=colorstr('ONNX:')):
    """YOLOv8 ONNX export."""
    """
    model: DetectionModel class

    """
    requirements = ['onnx>=1.12.0']
    check_requirements(requirements)

    output_names = ['output0']
    dynamic = cfg.dynamic
    if dynamic:
        dynamic['output0'] = {0: 'batch', 2: 'anchors'}  # shape(1, 84, 8400)

    device  = next(model.parameters()).device

    imgsz = check_imgsz(cfg.imgsz, stride=model.stride, min_dim=2)
    im = torch.zeros(cfg.batch, 3, *imgsz).to(device)

    quantize.export_onnx(model.cpu() if dynamic else model,  # dynamic=True only compatible with cpu
            im.cpu() if dynamic else im,
            save_file,
            verbose=False,
            opset_version=13,
            do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
            input_names=['images'],
            output_names=output_names,
            dynamic_axes=dynamic or None
            )
    
    # Simplify
    model_onnx = onnx.load(save_file)  # load onnx model
    if cfg.simplify:
        try:
            LOGGER.info(f'{prefix} simplifying with onnxsim {onnxsim.__version__}...')
            # subprocess.run(f'onnxsim "{f}" "{f}"', shell=True)
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, 'Simplified ONNX model could not be validated'
        except Exception as e:
            LOGGER.info(f'{prefix} simplifier failure: {e}')
    onnx.save(model_onnx, save_file)


# def evaluate(torch_model, model_name='yolov8n.pt'):
#     args = dict(model=model_name, data='ultralytics/ultralytics/cfg/datasets/coco128.yaml')
#     validator = DetectionValidator(args=args)

#     # result = validator(model=torch_model)
#     # mAP = result["metrics/mAP50-95(B)"]
#     return validator(model=torch_model)["metrics/mAP50-95(B)"]

def evaluate_coco(model, val_dataloader):
    validator = yolo.detect.DetectionValidator(dataloader=val_dataloader, args=cfg)
    val_model = deepcopy(model) # deepcopy
    mAP = validator(model=val_model)["metrics/mAP50-95(B)"]
    return mAP

def run_export(weight, save, size, dynamic, noanchor, noqadd):
    
    quantize.initialize()
    if save is None:
        name = os.path.basename(weight)
        name = name[:name.rfind('.')]
        save = os.path.join(os.path.dirname(weight), name + ".onnx")
        
    model = torch.load(weight, map_location="cpu")["model"]
    model.float()
    if not noqadd:
        quantize.replace_bottleneck_forward(model)

    export_onnx(model, save, size, dynamic_batch=dynamic, noanchor=noanchor)
    print(f"Save onnx to {save}")



def run_sensitive_analysis(weight, device, cocodir, summary_save):

    quantize.initialize()
    device  = torch.device(device)
    model   = load_yolov8_model(weight, device)
    train_dataloader = get_dataloader(cfg, cocodir + "images/train2017", batch_size=cfg.batch, mode='train')
    val_dataloader   = get_dataloader(cfg, cocodir + "images/train2017", batch_size=cfg.batch, mode='val')
    quantize.replace_to_quantization_module(model)
    quantize.calibrate_model(model, train_dataloader, device)

    summary = SummaryTool(summary_save)
    print("Evaluate PTQ...")
    ap = evaluate_coco(model, val_dataloader)
    summary.append([ap, "PTQ"])

    print("Sensitive analysis by each layer...")
    for i in range(0, len(model.model)):
        layer = model.model[i]
        if quantize.have_quantizer(layer):
            print(f"Quantization disable model.{i}")
            quantize.disable_quantization(layer).apply()
            ap = evaluate_coco(model, val_dataloader)
            summary.append([ap, f"model.{i}"])
            quantize.enable_quantization(layer).apply()
        else:
            print(f"ignore model.{i} because it is {type(layer)}")
    
    summary = sorted(summary.data, key=lambda x:x[0], reverse=True)
    print("Sensitive summary:")
    for n, (ap, name) in enumerate(summary[:10]):
        print(f"Top{n}: Using fp16 {name}, ap = {ap:.5f}")


from ultralytics.models import yolo
def run_test(weight, device, cocodir):

    device  = torch.device(device)
    model   = load_yolov8_model(weight, device)
    val_dataloader = get_dataloader(cfg, cocodir + "images/train2017", batch_size=cfg.batch, mode='val')
    evaluate_coco(model, val_dataloader)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--weight', type=str, default='yolov8n.pt', help='initial weight patsh')
    parser.add_argument('--cocodir', type=str,  default="/data-nbd/yolov8/datasets/coco128/", help="coco directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")

    parser.add_argument('--save', type=str,  required=False, help="coco directory")
    parser.add_argument('--size', type=int,  default=640, help="input size for export onnx ...")
    parser.add_argument('--dynamic', type=bool,  default=True, help="dynamic batch for export onnx ...")
    parser.add_argument('--noanchor', type=bool,  default=True, help="export no anchor nodes ...")
    parser.add_argument("--noqadd", type=bool,  default=True, help="export do not add QuantAdd")

    parser.add_argument("--ptq", type=str, default="ptq.pt", help="file")
    parser.add_argument("--qat", type=str, default="qat.pt", help="file")
    parser.add_argument("--eval-origin", action="store_true", help="do eval for origin model")
    parser.add_argument("--eval-ptq", action="store_true", help="do eval for ptq model")
    
    parser.add_argument("--ignore-policy", type=str, default="model\.24\.m\.(.*)", help="regx")
    parser.add_argument("--supervision-stride", type=int, default=1, help="supervision stride")
    parser.add_argument("--iters", type=int, default=200, help="iters per epoch")
    parser.add_argument("--summary", type=str, default="sensitive-summary.json", help="summary save file")

    parser.add_argument("--confidence", type=float, default=0.001, help="confidence threshold")
    parser.add_argument("--nmsthres", type=float, default=0.65, help="nms threshold")

    parser.add_argument('--export', type=bool, default=False, help='Do Export weight to onnx file ...')
    parser.add_argument('--finetune', type=bool, default=True, help='Do PTQ/QAT finetune ...')
    parser.add_argument('--sensitive', type=bool, default=False, help='Do Sensitive layer analysis ...')
    parser.add_argument('--test', type=bool, default=False, help='Do evaluate ...')


    args = parser.parse_args()
    init_seeds(2023)

    if args.export:
        run_export(args.weight, args.save, args.size, args.dynamic, args.noanchor, args.noqadd)
    elif args.finetune:
        print(args)
        run_qat(
            args.weight, args.cocodir, args.device, args.ignore_policy, 
            args.ptq, args.qat, args.supervision_stride, args.iters,
            args.eval_origin, args.eval_ptq
        )
    elif args.sensitive:
        run_sensitive_analysis(args.weight, args.device, args.cocodir, args.summary)
    
    elif args.test:
        run_test(args.weight, args.device, args.cocodir)

    else:
        parser.print_help()

