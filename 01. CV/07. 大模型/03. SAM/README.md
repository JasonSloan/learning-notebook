[参考链接](https://mp.weixin.qq.com/s/gUOaxlIC2LjMlkTm0NvcVA)

![](assets/model.jpg)

todo: 尚未想明白的问题

1. 相对位置特征relative postional embedding是什么意思，为什么加这个，它和正常的正余弦叠加的positional_embedding比有什么好处？[参考链接](https://www.youtube.com/watch?v=7XHucAvHNKs)

# 0. 微调方法

[Adapter/LoRA微调方式微调SAM到其他语义分割数据](https://github.com/KidsWithTokens/Medical-SAM-Adapter)

# 一. 模型结构

![](assets/arch.jpg)

## 1. Encoder

### 1.1 image encoder

![](assets/encoder.jpg)

image encoder就是ViT, 只不过没有了多余的[cls]的embedding(上图中"0*"的位置),  然后图像在做运算的时候没有展平成[N, T, E]的维度, 而是一直保持[N, C, H, W]的维度。

image encoder输入图像维度[bs, 3, 1024, 1024] (把原图做了个letterbox), 输出维度为[bs, 256, 64, 64] (假设为ViT_b模型)

![](assets/vit.jpg)

### 1.2 prompt encoder

SAM在做分割的时候, 可以接受以下几种类型的prompt:

(1) 点坐标以及点的label值(0或者1, 0代表背景, 1代表前景)

(2) 边框的四个点坐标

(3) 掩膜

![](assets/prompt.jpg)

prompt encoder输入为np.ndarray(维度不一定), 输出为sparse_embeddings[bs, ?, 256]、dense_embeddings[bs, 256, 64, 64]

todo: prompt encoder原理未搞明白

## 2. Decoder

### 2.1 mask decoder 

[参考链接](https://towardsdatascience.com/how-does-the-segment-anything-models-sam-s-decoder-work-0e4ab4732c37)

mask decoder接受4个输入:

**Image embedding**: 来自于image encoder, 即使用ViT将图像转成embedding, 维度[bs, 256, 64, 64]

**Image positional embedding** : 来自于prompt encoder, 维度[bs, 256, 64, 64]

**Dense prompt embedding** : 来自于prompt encoder, 维度[bs, 256, 64, 64]

**Sparse prompt embedding** : 来自于prompt encoder, 维度[bs,  T, 256], T代表不定长

mask decoder返回两个输出:

**masks** : 4种类型的掩膜, 维度[4, 256, 256]

**confidence scores** : 4种掩膜的得分

为什么是4种掩膜, 如下图, 每一个prompt点可能代表了使用者三种意图(分割出一块车玻璃、整块车玻璃、整个车), 再加上一个最有可能(得分最高)的掩膜, 一共是4种掩膜

![](assets/masks.jpg)

分成两部分理解:

![](assets/predict_masks.jpg)

第一部分:

![](assets/part1.jpg)

**iou_token** : 固定维度1×256, 上图中蓝色的iou_token是为了得到绿色的iou_token, 绿色的iou_token最终目的是得出置信度得分(类似于ViT中的cls_token, 通过注意力机制可以将prompt embeddings, image embedding, image positional embedding这三种信息融合进去, 从而可以预测得分)

**mask_tokens** : 固定维度4×256,  上图中淡蓝色的mask_tokens是为了得到深绿色的mask_tokens, 绿色的mask_tokens最终目的是为了得到4个level的mask

**sparse_prompt_embeddings** : 不固定维度T×256, 上图中的sparse_prompt_embeddings只是为了将信息融合进去, 得到的绿色的sparse_prompt_embeddings弃掉不用(为什么弃掉不用还要输出呢? 因为transformer结构注定了输入输出维度相同, 输入这三种, 输出也有这三种)

**image_embeddings ** : 来自于image encoder，提取到的图像特征向量

**dense_prompt_embeddings** : 来自于prompt encoder

**image_embeddings ** 和 **dense_prompt_embeddings** 会进行element-wise相加, 相当于把两种信息融合在一起了

**image positional embedding** : 即image positional encoding, 来自于prompt encoder
**The transformer call** : 接受这三类输入, 并得到两类输出attended tokens embedding(在图中是hs)以及attended image embedding(在图中是src2)，具体见本页2.2节

**attended tokens embedding** : (5+T)×256

**attended image embedding** :  bs×256×64×64, 融合了各种信息后的图像特征向量



第二部分:

![](assets/part2.jpg)

第二部分part1：产生4张分割掩膜

做法是将mask_tokens通过一个mlp从维度4X256映射到4X32, 同时将attended image embedding通过上采样将分辨率由256X64X64提高到32X256X256， 然后拉平成32X65536，与4X32的iou_tokens做矩阵运算，得到4X65536，再reshape成4X256X256，此时得到的是低分辨率的掩膜，最后插值成原图大小

第二部分part2：产生4张分割掩膜的置信度得分

将iou_tokens做一个mlp从1X256维度映射到1X4维度，得到4张特征图级别的置信度得分

## 2.2 mask decoder中的注意力机制（The transformer call）

### 2.2.1 TwoWayTransformer.forward

![](assets/twoway.jpg)

![](assets/twoway2.jpg)

TwoWayTransformer.forward主要做了3次transformer的操作（对应于图中的layer[0]、layer[1]和final_attn_token_to_image)，其中layer[0]、layer[1]调用了TwoWayAttentionBlock

### 2.2.2 TwoWayAttentionBlock.forward

![](assets/twowayattn.jpg)

TwoWayAttentionBlock分为三种注意力，一种prompt token自注意力，第二种是token to image 注意力，第三种是image to token注意力

第一种：prompt token自注意力

![](assets/attenforward.jpg)

之所以可以出来动态长度的（5+T）的自注意力，是因为将（5+T）放在了batch维度上，而正常的q与k之间的矩阵乘法对于第0个维度是没有要求的

第二种：token to image 注意力

![](assets/cross1.jpg)

第三种：image to token 注意力

![](assets/cross2.jpg)









