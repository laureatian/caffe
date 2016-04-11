#include <vector>

#include "caffe/layers/conv_layer.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_im2col.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif

#include "caffe/util/benchmark.hpp"
namespace caffe {
#define HYBRID
template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
#ifndef HYBRID
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int_tp i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    // Multi queue execution, all previous work needs to be done first
    this->device_->FinishQueues();
    for (int_tp n = 0; n < this->num_; ++n) {
      // Multi queue execution, go through work queues
      this->device_->SwitchQueue(n);
      this->forward_gpu_gemm(bottom_data, n * this->bottom_dim_, weight,
          top_data, n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data, n * this->top_dim_, bias);
      }
    }
    // Multi queue execution, finish all queues
    this->device_->FinishQueues();
  }
#else
    const Dtype* weight_gpu = this->blobs_[0]->gpu_data();
    const Dtype* weight_cpu = this->blobs_[0]->cpu_data();
    const auto weight_shape = this->blobs_[0]->shape();
    assert(weight_shape.size() == 4);
    const uint output_channels = weight_shape[0];
    const uint hybrid_offset = (output_channels+1)*80/100;

    const uint weight_dim = weight_shape[1]*weight_shape[2]*weight_shape[3];
    caffe::Timer total_timer;
    total_timer.Start();
    if(hybrid_offset != output_channels) {
        const auto top_shape = top[0]->shape();
        uint top_image_dim = top_shape[2]*top_shape[3];
        uint cpu_channels = output_channels - hybrid_offset;
        Dtype* top_cpu_data = new Dtype[top_image_dim * cpu_channels];// = top[i]->mutable_cpu_data();
        for (int_tp i = 0; i < bottom.size(); ++i) {
          const Dtype* bottom_gpu_data = bottom[i]->gpu_data();
          Dtype* top_gpu_data = top[i]->mutable_gpu_data();
          //const Dtype
          weight_cpu += hybrid_offset*weight_dim;
          // Multi queue execution, all previous work needs to be done first
          this->device_->FinishQueues();
          for (int_tp n = 0; n < this->num_; ++n) {
            // Multi queue execution, go through work queues
            this->device_->SwitchQueue(n);
            this->forward_hybrid_gemm(bottom_gpu_data, n * this->bottom_dim_,
                                      weight_gpu, weight_cpu, hybrid_offset, output_channels,
                top_gpu_data, top_cpu_data, n * this->top_dim_);
            if (this->bias_term_) {
              const Dtype* bias = this->blobs_[1]->gpu_data();
              this->forward_gpu_bias(top_gpu_data, n * this->top_dim_, bias);
            }
          }
          // Multi queue execution, finish all queues
          this->device_->FinishQueues();
        }
        delete [] top_cpu_data;
    } else {
        for (int_tp i = 0; i < bottom.size(); ++i) {
          const Dtype* bottom_data = bottom[i]->gpu_data();
          Dtype* top_data = top[i]->mutable_gpu_data();
          // Multi queue execution, all previous work needs to be done first
          this->device_->FinishQueues();
          for (int_tp n = 0; n < this->num_; ++n) {
            // Multi queue execution, go through work queues
            this->device_->SwitchQueue(n);
            this->forward_gpu_gemm(bottom_data, n * this->bottom_dim_, weight_gpu,
                top_data, n * this->top_dim_);
            if (this->bias_term_) {
              const Dtype* bias = this->blobs_[1]->gpu_data();
              this->forward_gpu_bias(top_data, n * this->top_dim_, bias);
            }
          }
          // Multi queue execution, finish all queues
          this->device_->FinishQueues();
        }
    }
    total_timer.Stop();
    std::cout << "Total Time(Hybrid): " << total_timer.MilliSeconds() << " ms." << std::endl;

#endif
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int_tp i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int_tp n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff, n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int_tp n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data, n * this->bottom_dim_,
              top_diff, n * this->top_dim_, weight_diff);
        }
      }
      // gradient w.r.t. bottom data, if necessary.
      if (propagate_down[i]) {
        // Multi queue execution, all previous work needs to be done first
        this->device_->FinishQueues();
        for (int_tp n = 0; n < this->num_; ++n) {
          // Multi queue execution, go through work queues
          this->device_->SwitchQueue(n);
          this->backward_gpu_gemm(top_diff, n * this->top_dim_, weight,
                                  bottom_diff, n * this->bottom_dim_);
        }
        // Multi queue execution, finish all queues
        this->device_->FinishQueues();
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
