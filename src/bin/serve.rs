use burn::{
    backend::WgpuBackend,
    config::Config,
    data::dataloader::batcher::Batcher,
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::backend::Backend,
};
use burn_model_serving::{
    data::{class_name, num_classes, BertCasedTokenizer, TextClassificationBatcher, Tokenizer},
    model::{TextClassificationModel, TextClassificationModelConfig},
    training::ExperimentConfig,
};
use burn_wgpu::WgpuDevice;
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, sync::Arc};
use warp::{body::json, Filter};

#[allow(dead_code)]
type ElemType = f32;
type ServingBackend = WgpuBackend;

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

async fn handle_prediction<B: Backend>(
    body: ClassifierRequest,
    batcher: Arc<TextClassificationBatcher<B>>,
    model: Arc<TextClassificationModel<B>>,
) -> Result<impl warp::Reply, Infallible> {
    // Run inference on the given text samples
    println!("Running inference ...");
    let item = batcher.batch(body.texts.clone()); // Batch samples using the batcher
    let predictions = model.infer(item); // Get model predictions

    // Print out predictions for each sample
    let mut classes = Vec::new();
    for (i, text) in body.texts.into_iter().enumerate() {
        let prediction = predictions.clone().slice([i..i + 1]); // Get prediction for current sample
        let logits = prediction.to_data(); // Convert prediction tensor to data
        let class_index = prediction.argmax(1).into_data().convert::<i32>().value[0]; // Get class index with the highest value
        let class = class_name(class_index as usize); // Get class name

        // Print sample text, predicted logits and predicted class
        println!("\n=== Item {i} ===\n- Text: {text}\n- Logits: {logits}\n- Prediction: {class}\n================");
        classes.push(class);
    }

    let result = ClassifierResponse { classes };

    Ok(warp::reply::json(&result))
}

#[derive(Deserialize)]
struct ClassifierRequest {
    texts: Vec<String>,
}

#[derive(Serialize)]
struct ClassifierResponse {
    classes: Vec<String>,
}

#[tokio::main]
async fn main() {
    let (batcher, model) = load_model_and_tokenizer::<ServingBackend>(
        "./models/text-classification",
        WgpuDevice::default(),
    );

    let batcher = Arc::new(batcher);
    let model = Arc::new(model);

    println!("Server started at http://localhost:3030");
    let classify = warp::post()
        .and(warp::path("classify"))
        .and(json::<ClassifierRequest>())
        .and(with_batcher(batcher.clone()))
        .and(with_model(model.clone()))
        .and_then(handle_prediction);

    warp::serve(classify).run(([127, 0, 0, 1], 3030)).await;
}

fn with_batcher(
    batcher: Arc<TextClassificationBatcher<ServingBackend>>,
) -> impl Filter<Extract = (Arc<TextClassificationBatcher<ServingBackend>>,), Error = Infallible> + Clone
{
    warp::any().map(move || batcher.clone())
}

fn with_model(
    model: Arc<TextClassificationModel<ServingBackend>>,
) -> impl Filter<Extract = (Arc<TextClassificationModel<WgpuBackend>>,), Error = Infallible> + Clone
{
    warp::any().map(move || model.clone())
}
