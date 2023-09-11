use std::env;

use burn::optim::decay::WeightDecayConfig;
use burn_model_serving::{training::{ExperimentConfig, train}, DbPediaDataset};

pub fn fit() {
    #[cfg(feature = "f16")]
    type Elem = burn::tensor::f16;
    #[cfg(not(feature = "f16"))]
    type Elem = f32;

    type Backend = burn::autodiff::ADBackendDecorator<burn::backend::tch::TchBackend<Elem>>;
    
    let config = ExperimentConfig::new(
        burn::nn::transformer::TransformerEncoderConfig::new(384, 1536, 12, 6)
            .with_norm_first(true),
        burn::optim::AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1.0e-6))),
    );

    train::<Backend, DbPediaDataset>(
        if cfg!(target_os = "macos") {
            println!("Using MPS");
            burn::backend::tch::TchDevice::Mps
        } else {
            burn::backend::tch::TchDevice::Cuda(0)
        },
        DbPediaDataset::train(),
        DbPediaDataset::test(),
        config,
        "./models/text-generation",
    );
}

fn predict() {
    println!("Predicting");
}


fn main() {

    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        eprintln!("Please provide an argument. Options: fit, predict");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "fit" => fit(),
        "predict" => predict(),
        _ => {
            eprintln!("Invalid argument. Options: fit, predict");
            std::process::exit(1);
        }
    }

    
}
