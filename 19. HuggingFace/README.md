# 一. huggingface快速测试一个模型的方法

![](assets/download.jpg)

下载Files and Versions中所有的文件放入一个新建的文件夹models中

pip install 所有的依赖,运行模型并指定刚才下载的文件所在文件夹即可,例如:

```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("./models", trust_remote_code=True)
model = AutoModel.from_pretrained("./models", trust_remote_code=True, device='cuda')
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)
```

可以使下载加速的国内镜像设置方法：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```





