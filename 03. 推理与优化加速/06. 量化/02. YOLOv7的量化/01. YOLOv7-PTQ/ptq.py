import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.append(ROOT / 'yolov7')  # to run '$ python *.py' files in subdirectories

import warnings
warnings.filterwarnings("ignore")

import torch
import quantize

import argparse


def run_SensitiveAnalysis(weight, cocodir, device='cpu'):
    # prepare model
    print("Prepare Model ....")
    model = quantize.prepare_model(weight, device)
    quantize.replace_to_quantization_model(model)
    # prepare dataset
    print("Prepare Dataset ....")
    train_dataloader = quantize.prepare_train_dataset(cocodir)
    val_dataloader = quantize.prepare_val_dataset(cocodir)
    # calibration model
    print("Begining Calibration ....")
    quantize.calibrate_model(model, train_dataloader, device)
    # sensitive analysis
    print("Begining Sensitive Analysis ....")
    quantize.sensitive_analysis(model, val_dataloader, args.sensitive_summary)


def run_PTQ(args, device='cpu'):
    # prepare model
    print("Prepare Model ....")
    model = quantize.prepare_model(args.weights, device)
    quantize.replace_to_quantization_model(model, args.ignore_layers)
    # prepare dataset
    print("Prepare Dataset ....")
    val_dataloader = quantize.prepare_val_dataset(args.cocodir, batch_size=args.batch_size)
    train_dataloader = quantize.prepare_train_dataset(args.cocodir, batch_size=args.batch_size)
    # calibration model
    print("Begining Calibration ....")
    quantize.calibrate_model(model, train_dataloader, device)
    
    summary = quantize.SummaryTool(args.ptq_summary)
    
    if args.eval_origin:
        print("Evaluate Origin...")
        with quantize.disable_quantization(model):
            ap = quantize.evaluate_coco(model, val_dataloader, conf_thres=args.conf_thres, iou_thres=args.iou_thres)
            summary.append(["Origin", ap])
    if args.eval_ptq:
        print("Evaluate PTQ...")
        ap = quantize.evaluate_coco(model, val_dataloader, conf_thres=args.conf_thres, iou_thres=args.iou_thres)
        summary.append(["PTQ", ap])

    if args.save_ptq:
        print("Export PTQ...")
        quantize.export_ptq(model, args.ptq, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--weights', type=str, default='weights/yolov7.pt', help='initial weights path')
    parser.add_argument('--cocodir', type=str,  default="../../datasets/coco2017val", help="coco directory")
    parser.add_argument('--batch_size', type=int,  default=4, help="batch size for data loader")
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    
    parser.add_argument('--sensitive', type=bool, default=True, help="use sensitive analysis or not before ptq")
    parser.add_argument("--sensitive_summary", type=str, default="sensitive-summary.json", help="summary save file")
    parser.add_argument("--ignore_layers", type=str, default="model\.105\.m\.(.*)", help="regx")
    
    parser.add_argument("--save_ptq", type=bool, default=False, help="file")
    parser.add_argument("--ptq", type=str, default="yolov7-ptq.onnx", help="file")
    
    parser.add_argument("--confidence", type=float, default=0.4, help="confidence threshold")
    parser.add_argument("--nmsthres", type=float, default=0.6, help="nms threshold")
    
    parser.add_argument("--eval_origin", action="store_true", help="do eval for origin model")
    parser.add_argument("--eval_ptq", action="store_true", help="do eval for ptq model")
    
    parser.add_argument("--ptq_summary", type=str, default="ptq_summary.json", help="summary save file")
    
    args = parser.parse_args()
    
    is_cuda = (args.device != 'cpu') and torch.cuda.is_available()
    device = torch.device("cuda:0" if is_cuda else "cpu")
    
    
    # 敏感层分析
    if args.sensitive:
        print("Sensitive Analysis....")
        run_SensitiveAnalysis(args.weights, args.cocodir, device)
    
    
    # PTQ 量化
    # ignore_layers= ["model\.105\.m\.(.*)", model\.99\.m\.(.*)]
    # args.ignore_layer = ignore_layers
    print("Begining PTQ.....")
    run_PTQ(args, device)
    
    print("PTQ Quantization Has Finished....")
    
    
    
