# GroupNorm-caffe
caffe implementation of Group Normalization https://arxiv.org/abs/1803.08494
# This code only provide CUDA version
# add to caffe.proto
message GroupNormParameter {

  optional float eps = 1 [default = 1e-5];
  
  optional int32 group_num = 2 [default = 32];
  
}

optional GroupNormParameter group_norm_param = 10000;
