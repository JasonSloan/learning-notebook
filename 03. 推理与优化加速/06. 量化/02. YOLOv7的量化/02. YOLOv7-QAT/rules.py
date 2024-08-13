import os
import onnx


# 在onnx模型中查找 特定输入节点的所有节点，以列表的形式进行返回
def find_all_with_input_node(model, name):
    all = []
    for node in model.graph.node:
        if len(node.input) > 0 and name in node.input:
            all.append(node)
    return all


# 在onnx模型中查找指定输入的节点，找到了，立刻返回该节点
def find_with_input_node(model, name):
    for node in model.graph.node:
        if len(node.input) > 0 and name in node.input:
            return node


# 在onnx模型中查找给定的QuantizeLinear节点相关联的Conv
def find_quantizelinear_conv(model, qnode):
    # 找到q节点相连的dq节点(qnode为以Concat的输出作为输入的节点), 返回值dq为 以Concat层的输出作为输入的QuantizeLinear节点对应的DequantizeLinear节点)
    dq = find_with_input_node(model, qnode.output[0]) 
    # conv为以DequantizeLinear节点为输入的层(conv层)
    conv = find_with_input_node(model, dq.output[0])
    return conv


# 在onnx模型中查找特定的输出名称的节点,找到了就返回该节点
def find_with_output_node(model, name):
    for node in model.graph.node:
        if len(node.output) > 0 and name in node.output:
            return node



# 在onnx模型中查找指定量化节点的相关卷积模块的名称
def find_quantize_conv_name(model, weight_qname):

    dq = find_with_output_node(model, weight_qname)

    q = find_with_output_node(model, dq.input[0])

    return ".".join(q.input[0].split(".")[:-1])
    # model.63.conv.weight   ===>  model.63.conv



def find_quantizer_pairs(onnx_file):
    model = onnx.load(onnx_file)  # 加载 ONNX 模型
  
    match_pairs = []
    for node in model.graph.node:  # 遍历onnx模型中的每个节点
        if node.op_type == "Concat":  # 如果节点类型为"Concat"
            #  找到那些将node节点(Concat)的输出(node.output[0])作为其输入的所有节点
            qnodes = find_all_with_input_node(model, node.output[0])
            major = None
            for qnode in qnodes:
                if qnode.op_type != "QuantizeLinear": # 如果节点的类型不是"QuantizeLinear"
                    continue
                
                conv = find_quantizelinear_conv(model, qnode)   # qnode为以Concat的输出作为输入的节点
                if major is None:
                    major = find_quantize_conv_name(model, conv.input[1]) # 找出weight_DequantizeLinear所对应的conv_name
                else:
                    match_pairs.append([major, find_quantize_conv_name(model, conv.input[1])])

                for subnode in model.graph.node:
                    if len(subnode.input) > 0 and subnode.op_type == "QuantizeLinear" and subnode.input[0] in node.input:
                        subconv = find_quantizelinear_conv(model, subnode)
                        match_pairs.append([major, find_quantize_conv_name(model, subconv.input[1])])

        elif node.op_type == "MaxPool":  # 如果节点类型为"MaxPool"
            qnode = find_with_input_node(model, node.output[0])
            if not (qnode and qnode.op_type == "QuantizeLinear"):
                continue

            major = find_quantizelinear_conv(model, qnode)
            major = find_quantize_conv_name(model, major.input[1])
            same_input_nodes = find_all_with_input_node(model, node.input[0])

            for same_input_node in same_input_nodes:
                if same_input_node.op_type == "QuantizeLinear":
                    subconv = find_quantizelinear_conv(model, same_input_node)
                    match_pairs.append([major, find_quantize_conv_name(model, subconv.input[1])])
    return match_pairs


# 用于获取模块module中给定路径path的属性值
# 首先将路径拆分为属性名称的列表, 即 path.split(".")
# 然后逐级访问模块的属性, sub_attr(value, names[1:])
# 直到达到路径的末尾，并返回该属性的值, if len(names) == 1: return value
def get_attr_with_path(module, path):

    def sub_attr(module, names):
        name = names[0]
        # 使用内置的getattr函数, 获取模块 m 中名为 name 的属性的值，
        # 并将该值存储在变量 value 中
        value = getattr(module, name)

        if len(names) == 1:
            return value
        
        return sub_attr(value, names[1:])
    return sub_attr(module, path.split("."))



# 使用match_pairs，在model的基础上，进行scale的替换
import quantize
def apply_custom_rules_to_quantizer(qdq_model, device="cpu"):
    quantize.export_ptq(qdq_model, "custom_rules_temp.onnx", device)        # 未标定前的模型

    match_pairs = find_quantizer_pairs("custom_rules_temp.onnx")

    for major, sub in match_pairs:
        print(f"Rules: {sub} match to {major}")


        # 获取major的输入量化器，将其替换到sub子模块的输入量化器
        get_attr_with_path(qdq_model, sub)._input_quantizer = get_attr_with_path(qdq_model, major)._input_quantizer

        # # 获取主模块和子模块的输入量化器
        # input_quantizer_major = get_attr_with_path(model, major)._input_quantizer
        # input_quantizer_sub = get_attr_with_path(model, sub)._input_quantizer

        # # 将子模块的输入量化器设置为主模块的输入量化器
        # input_quantizer_sub = input_quantizer_major
        

    # 移除temp.onnx
    os.remove("custom_rules_temp.onnx")
