#include <algorithm>
#include <vector>

#include "caffe/layers/group_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GroupNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  
  if (bottom[0] != top[0]) {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }

  // compute mean
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * group_num_, chip_num_ * spatial_dim_,
        1. / (chip_num_ * spatial_dim_), bottom_data,cube_sum_multiplier_.gpu_data(), 0.,
        mean_.mutable_gpu_data());

  // subtract mean
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,num_ * group_num_,
      chip_num_ * spatial_dim_, 1, -1, mean_.gpu_data(),
      cube_sum_multiplier_.gpu_data(), 1., top_data);
 
  // compute variance using var(X) = E((X-EX)^2)
  caffe_gpu_mul(top[0]->count(), top[0]->gpu_data(), top[0]->gpu_data(),temp_.mutable_gpu_data());  // (X-EX)^2
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * group_num_, chip_num_ * spatial_dim_,
        1. / (chip_num_ * spatial_dim_), temp_.gpu_data(),
        cube_sum_multiplier_.gpu_data(), 0.,variance_.mutable_gpu_data());
  
  // normalize variance
  caffe_gpu_add_scalar(variance_.count(), eps_, variance_.mutable_gpu_data());
  caffe_gpu_sqrt(variance_.count(), variance_.gpu_data(),variance_.mutable_gpu_data());

  // div variance    
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * group_num_,
      chip_num_ * spatial_dim_, 1, 1., variance_.gpu_data(),
      cube_sum_multiplier_.gpu_data(), 0., temp_.mutable_gpu_data());

  caffe_gpu_div(top[0]->count(), top_data, temp_.gpu_data(), top_data);
  
  // TODO(cdoersch): The caching is only needed because later in-place layers
  //                 might clobber the data.  Can we skip this if they won't?
  
  caffe_copy(x_norm_.count(), top_data, x_norm_.mutable_gpu_data());
}

template <typename Dtype>
void GroupNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  
  const Dtype* top_data = x_norm_.gpu_data();
  const Dtype* top_diff;
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  if (bottom[0] != top[0]) {
    top_diff = top[0]->gpu_diff();
  } else {
    caffe_copy(x_norm_.count(), top[0]->gpu_diff(), x_norm_.mutable_gpu_diff());
    top_diff = x_norm_.gpu_diff();
  }
 
  // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
  //
  // dE(Y)/dX =
  //   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
  //     ./ sqrt(var(X) + eps)
  //
  // where \cdot and ./ are hadamard product and elementwise division,
  // respectively, dE/dY is the top diff, and mean/var/sum are all computed
  // along all dimensions except the channels dimension.  In the above
  // equation, the operations allow for expansion (i.e. broadcast) along all
  // dimensions except the channels dimension where required.
 
  // sum(dE/dY \cdot Y)
  caffe_gpu_mul(temp_.count(), top_data, top_diff, bottom_diff);
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * group_num_, chip_num_ * spatial_dim_, 1.,
      bottom_diff, cube_sum_multiplier_.gpu_data(), 0.,
      mean_.mutable_gpu_data());
  
  // reshape (broadcast) the above
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * group_num_,
      chip_num_ * spatial_dim_, 1, 1., mean_.gpu_data(),
      cube_sum_multiplier_.gpu_data(), 0., bottom_diff);

  // sum(dE/dY \cdot Y) \cdot Y
  caffe_gpu_mul(temp_.count(), top_data, bottom_diff, bottom_diff);

  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * group_num_, chip_num_ * spatial_dim_, 1.,
      top_diff, cube_sum_multiplier_.gpu_data(), 0.,
      mean_.mutable_gpu_data());
 
  // reshape (broadcast) the above to make
  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * group_num_,
      chip_num_ * spatial_dim_, 1, 1., mean_.gpu_data(),
      cube_sum_multiplier_.gpu_data(), 1., bottom_diff);

  // dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
  caffe_gpu_axpby(temp_.count(), Dtype(1), top_diff,
      Dtype(-1. / (chip_num_ * spatial_dim_)), bottom_diff);

  // note: temp_ still contains sqrt(var(X)+eps), computed during the forward
  // pass.
  caffe_gpu_div(temp_.count(), bottom_diff, temp_.gpu_data(), bottom_diff);
}

 

INSTANTIATE_LAYER_GPU_FUNCS(GroupNormLayer);


}  // namespace caffe




