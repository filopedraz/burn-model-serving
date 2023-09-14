use burn::nn::transformer::TransformerEncoderConfig;
use burn::optim::{decay::WeightDecayConfig, AdamConfig};
use burn::tensor::backend::ADBackend;

use burn::autodiff::ADBackendDecorator;
use burn::backend::wgpu::{AutoGraphicsApi, WgpuBackend, WgpuDevice};

use burn_model_serving::training::ExperimentConfig;
use burn_model_serving::AgNewsDataset;

#[allow(dead_code)]
type ElemType = f32;

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

fn main() {
    launch::<ADBackendDecorator<WgpuBackend<AutoGraphicsApi, ElemType, i32>>>(WgpuDevice::default());
}
