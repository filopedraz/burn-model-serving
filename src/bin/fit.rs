use burn::nn::transformer::TransformerEncoderConfig;
use burn::optim::{decay::WeightDecayConfig, AdamConfig};
use burn::tensor::backend::ADBackend;

use burn_model_serving::training::ExperimentConfig;
use burn_model_serving::AgNewsDataset;

#[cfg(not(feature = "f16"))]
#[allow(dead_code)]
type ElemType = f32;
#[cfg(feature = "f16")]
type ElemType = burn::tensor::f16;

pub fn launch<B: ADBackend>(device: B::Device) {
    let config = ExperimentConfig::new(
        TransformerEncoderConfig::new(256, 1024, 8, 4).with_norm_first(true),
        AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5))),
    );

    burn_model_serving::training::train::<B, AgNewsDataset>(
        device,
        AgNewsDataset::train(),
        AgNewsDataset::test(),
        config,
        "./models/text-classification",
    );
}

#[cfg(any(
    feature = "ndarray",
    feature = "ndarray-blas-netlib",
    feature = "ndarray-blas-openblas",
    feature = "ndarray-blas-accelerate",
))]
mod ndarray {
    use burn::autodiff::ADBackendDecorator;
    use burn::backend::ndarray::{NdArrayBackend, NdArrayDevice};

    use crate::{launch, ElemType};

    pub fn run() {
        launch::<ADBackendDecorator<NdArrayBackend<ElemType>>>(NdArrayDevice::Cpu);
    }
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use burn::autodiff::ADBackendDecorator;
    use burn::backend::tch::{TchBackend, TchDevice};

    use crate::{launch, ElemType};

    pub fn run() {
        #[cfg(not(target_os = "macos"))]
        let device = TchDevice::Cuda(0);
        #[cfg(target_os = "macos")]
        let device = TchDevice::Mps;

        launch::<ADBackendDecorator<TchBackend<ElemType>>>(device);
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use burn::autodiff::ADBackendDecorator;
    use burn::backend::tch::{TchBackend, TchDevice};

    use crate::{launch, ElemType};

    pub fn run() {
        launch::<ADBackendDecorator<TchBackend<ElemType>>>(TchDevice::Cpu);
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use burn::autodiff::ADBackendDecorator;
    use burn::backend::wgpu::{AutoGraphicsApi, WgpuBackend, WgpuDevice};

    use crate::{launch, ElemType};

    pub fn run() {
        launch::<ADBackendDecorator<WgpuBackend<AutoGraphicsApi, ElemType, i32>>>(
            WgpuDevice::default(),
        );
    }
}

fn main() {
    #[cfg(any(
        feature = "ndarray",
        feature = "ndarray-blas-netlib",
        feature = "ndarray-blas-openblas",
        feature = "ndarray-blas-accelerate",
    ))]
    ndarray::run();
    #[cfg(feature = "tch-gpu")]
    tch_gpu::run();
    #[cfg(feature = "tch-cpu")]
    tch_cpu::run();
    #[cfg(feature = "wgpu")]
    wgpu::run();
}
