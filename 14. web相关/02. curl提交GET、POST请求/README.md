```python
语法格式：
curl -X POST [options] [URL]
 
# 使用该-F选项时，curl使用的默认Content-Type是“multipart/form-data”，以key=value配对形式
curl -X POST -F 'name=Jason' -F 'email=jason@example.com' http://127.0.0.1:8000/login
 
# 使用-d选项，可以使用&符号对发送数据进行合并
curl -X POST -d 'name=Jason&email=jason@example.com' http://127.0.0.1:8000/login
 
# 使用-H选项，指定Content-Type为application/json发送JSON对象
curl -X POST -H "Content-Type:application/json"  -d '{"user": "admin", "passwd":"12345678"}' http://127.0.0.1:8000/login  # 通过-d指定json data内容
 
# 文件上传，需在文件位置之前添加@符号
curl -X POST -F 'image=@/home/user/Pictures/wallpaper.jpg' http://example.com/upload
```

