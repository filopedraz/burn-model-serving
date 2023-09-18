use crate::{
    data::{num_classes, BertCasedTokenizer, TextClassificationBatcher, Tokenizer},
    model::{TextClassificationModel, TextClassificationModelConfig},
    training::ExperimentConfig,
};

use burn::{
    backend::WgpuBackend,
    config::Config,
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::backend::Backend,
};

use std::sync::Arc;

#[allow(dead_code)]
type ElemType = f32;
pub type ServingBackend = WgpuBackend;

pub fn load_model_and_tokenizer<B: Backend>(
    artifact_dir: &str,
    device: B::Device,
) -> (TextClassificationBatcher<B>, TextClassificationModel<B>) {
    // Load experiment configuration
    let config = ExperimentConfig::load(format!("{artifact_dir}/config.json").as_str())
        .expect("Config file present");

    // Initialize tokenizer
    let tokenizer = Arc::new(BertCasedTokenizer::default());

    // Get number of classes from dataset
    let n_classes = num_classes();

    // Initialize batcher for batching samples
    let batcher = TextClassificationBatcher::<B>::new(
        tokenizer.clone(),
        device.clone(),
        config.max_seq_length,
    );

    // Load pre-trained model weights
    println!("Loading weights ...");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into())
        .expect("Trained model weights");

    // Create model using loaded weights
    println!("Creating model ...");
    let model = TextClassificationModelConfig::new(
        config.transformer,
        n_classes,
        tokenizer.vocab_size(),
        config.max_seq_length,
    )
    .init_with::<B>(record) // Initialize model with loaded weights
    .to_device(&device); // Move model to computation device

    (batcher, model)
}
