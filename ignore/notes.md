# ReCoNet Notes

### Setup from the Paper
We implement our style transfer pipeline on PyTorch 0.3 [26] with cuDNN 7 [7].
All tensor calculations are performed on a single GTX 1080 Ti GPU. Further
details of the training process can be found in our supplementary materials.

```bash
docker run -p 8888:8888 -p 6006:6006 -it --rm -v $PWD/runs:/opt/summaries tensorflow/tensorflow tensorboard --logdir /opt/summaries 
```

### duration
32700 steps