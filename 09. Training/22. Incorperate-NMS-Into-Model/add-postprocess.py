import cv2
import numpy as np
import torch
from torch import nn
import onnx
from onnx import helper
import torchvision


# 使用onnx将预处理和后处理代码变成网络结构添加到网络中
# step0:构建预处理和后处理网络并转成onnx
class Postprocess(nn.Module):
    # max_det means max_det per image, it must correspont to the output_tensor shape of openvino C++ inference code 
    # !注意:max-det是单张图片最多允许检测的目标数, 需要和openvino-C++中设置output_tensor保持一致
    def __init__(self, conf_thres=0.5, iou_thres=0.4, multi_label=True, max_det=100, max_nms=30000, max_wh=1024):
        super().__init__()
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.multi_label = multi_label
        self.max_det = max_det
        self.max_nms = max_nms
        self.max_wh = max_wh
    
    def forward(self, prediction):
        # batch-size, grid-size, pred-size
        bs, gs, ps = prediction.shape
        max_cls_idx = ps - 5
        # convert prediction from [bs, 15120, 5+cls] to [bs * 15120, 5+cls+img_idx]
        prediction = torch.cat([prediction.view(-1, ps), torch.arange(bs).reshape(-1, 1).repeat(1, gs).reshape(-1)[..., None]], dim=1)
        # get mask
        xc = prediction[..., 4] > self.conf_thres  # candidates
        # filter
        x = prediction[xc]
        # Compute conf
        x[:, 5:-1] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = self.xywh2xyxy(x[:, :4])
        
        # multi-label
        # i for box_idx, j for cls_idx
        i, j = (x[:, 5:-1] > self.conf_thres).nonzero(as_tuple=False).T
        # x: box(xywh) + cls_conf + cls_idx + image_idx      
        x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float(), x[i, -1:].float()), 1)
        
        n = int(x.shape[0])
        if n == 0:
            output = torch.zeros([0, 7], dtype=torch.float32)
            return output
        
        # Batched NMS
        # offset images
        c1 = x[:, -1:] * max_cls_idx * self.max_wh
        # offset classes
        c2 = x[:, 5:6] * self.max_wh
        boxes, scores = x[:, :4] + c1 + c2, x[:, 4]  # boxes (offset by class image), scores
        i = torchvision.ops.nms(boxes, scores, self.iou_thres)  # NMS
        reserved = x[i]  

        # limit the max-det for every image
        output = []
        for i in range(bs):
            imask = reserved[..., -1] == i
            ireserved = reserved[imask]
            if int(ireserved.shape[0]) > self.max_det:
                 ireserved = ireserved[:self.max_det]
            output.append(ireserved)
        output = torch.cat(output, dim=0)
        
        return output     # output: box(xywh) + cls_conf + cls_idx + image_idx
    
    @staticmethod
    def xywh2xyxy(x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y
    

def modify_onnx(model_weight, post_weight, save_weight):
    model = onnx.load(model_weight)
    """给onnx模型增加预处理和后处理部分"""
    # =====================Part1:给原模型增加后处理部分=====================
    # step1:加载后处理网络，将模型中的以output为输出的节点，修改为post_onnx的输入节点
    post_onnx = onnx.load(post_weight)
    # 给所有的post-onnx网络的节点名字前面加上'post/'
    for item in post_onnx.graph.node:
        # 修改当前节点的名字
        item.name = f"post/{item.name}"
        # 修改当前节点的输入的名字
        for index in range(len(item.input)):
            item.input[index] = f"post/{item.input[index]}"
        # 修改当前节点的输出的名字
        for index in range(len(item.output)):
            item.output[index] = f"post/{item.output[index]}" 
    # 修改原模型的最后一层的输出节点名字改为post-onnx的输入节点的名字
    for item in model.graph.node:
        if item.name == "/model.24/Concat_3":      # 这个需要到netron上查看到底叫啥名
            item.output[0] = "post/" + post_onnx.graph.input[0].name
            print("Change original model output to post model input successfully!")

    # setp2: 把post-onnx的node全部放到原模型的node中
    for item in post_onnx.graph.node:
        model.graph.node.append(item)    # 这里我看了model.graph.node这个转成列表后append不是在网络末尾追加吗，但是这个是将预处理加入到网络首部中，不应该是insert吗
    # 答：其实model.graph.node这个列表里的元素可以使完全乱序的，因为这个列表里的每个元素都标记好了他的输入是叫啥名，输出时叫啥名，所以无论在列表中顺序怎么乱，最终都能按照名字一一对应上

    # step3: 把post-onnx的输出名称作为原模型的输出名称
    output_name = "post/" + post_onnx.graph.output[0].name
    model.graph.output[0].CopyFrom(post_onnx.graph.output[0])
    model.graph.output[0].name = output_name

    onnx.save(model, save_weight)
    print("Done!")
    
def export_mypost_model(post_weight, dynamic=True):
    post = Postprocess().eval()
    # 15120: 640*384 ; 9: xyxy + conf + 4cls
    dummy = torch.randn(10, 15120, 9) if dynamic else torch.randn(1, 15200, 9)
    dynamic_axes = {
        "postinput": {0: "batch"},
        "postoutput": {0: "batch"},
    }
    torch.onnx.export(
        post, 
        (dummy), 
        post_weight,
        input_names=["postinput"], 
        output_names=["postoutput"],
        opset_version=18,
        dynamic_axes = dynamic_axes if dynamic else None
        )
    
def verify_onnxmodel_result(onnxmodel, imgs, multi_batch=True):
    import onnxruntime as ort
    imgs = [imgs] if not isinstance(imgs, list) else imgs
    ort_session = ort.InferenceSession(onnxmodel)
    imgs = [np.load(img) for img in imgs]
    imgs = np.concatenate(imgs, axis=0)
    input_name = ort_session.get_inputs()[0].name
    out = ort_session.run(None, {input_name: imgs})[0]
    print()
    
def verify_mypost_result_pt(right_before, right_after):
    import numpy as np
    right_before = np.load(right_before)
    post = Postprocess()
    res = post(torch.from_numpy(right_before))
    right_after = np.load(right_after)
    print()     # 对比res和right_after是否相等

def verify_mypost_result_onnx(weight, right_before, right_after):
    import numpy as np
    import onnxruntime as ort
    ort_session = ort.InferenceSession(weight)
    input_name = ort_session.get_inputs()[0].name
    x = np.load(right_before)
    out = ort_session.run(None, {input_name: x})[0]
    right_after = np.load(right_after)
    print()     # 对比res和right_after是否相等
    
def verify_new_onnx(weight, imgs):
    import onnxruntime as ort
    ort_session = ort.InferenceSession(weight)
    imgs = [imgs] if not isinstance(imgs, list) else imgs
    imgs = [np.load(img) for img in imgs]
    imgs = np.concatenate(imgs, axis=0)
    # dummy = torch.randn([5, 3, 384, 640], dtype=torch.float32)
    input_name = ort_session.get_inputs()[0].name
    out = ort_session.run(None, {input_name: imgs})[0]
    print(out.shape)
        

if __name__ == '__main__':
    model_weight = "weights/lookout-helmets-persons-detection/multi-batch/last.onnx"
    post_weight = "weights/lookout-helmets-persons-detection/multi-batch/post.onnx"
    save_weight = "weights/lookout-helmets-persons-detection/multi-batch/new.onnx"
    ov_weight = "weights/lookout-helmets-persons-detection/multi-batch/ov.xml"
    right_before = "official-before-nms2.npy"   # two imgs batched bofore nms after detect
    right_after = "official-after-nms2.npy"     # two imgs batched after nms after detect
    imgs = ["img1.npy", "img2.npy"]             # two imgs after letterbox before detect
    # verify_onnxmodel_result(model_weight, imgs, multi_batch=True)         # verify the result of original model transformed to onnx
    # verify_mypost_result_pt(right_before, right_after)                    # verify the result of my post-process model in pytorch
    export_mypost_model(post_weight)                                        # export my post-process model to onnx
    # verify_mypost_result_onnx(post_weight, right_before, right_after)     # verify the result of my post-process model in onnx
    modify_onnx(model_weight, post_weight, save_weight)                   # add post-process model to original model
    verify_new_onnx(save_weight, imgs)                                      # verify the result of new model with post-process added
    # export to ov model command:
    # mo --input_model input-model-path --input_shape [-1,3,384,640] --output_dir output-model-dir --compress_to_fp16=True
    # mo --input_model weights/lookout-helmets-persons-detection/multi-batch/new.onnx --input_shape [-1,3,384,640] --output_dir weights/lookout-helmets-persons-detection/multi-batch/ --compress_to_fp16=True