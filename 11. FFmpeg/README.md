

```bash
# 指定一个时间第3分19秒, 获取该时间所在整个视频的第几帧(大约)
ffprobe -read_intervals "%+03:19" -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 input.mp4

# 使用ffplay播放视频流
ffplay -rtsp_transport tcp rtsp://192.168.103.241:1935/live/724-2
```

