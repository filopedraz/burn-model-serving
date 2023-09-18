#![allow(clippy::new_without_default)]

use crate::{
    data::{class_name, TextClassificationBatcher},
    model::TextClassificationModel,
    state::{load_model_and_tokenizer, ServingBackend},
};

use burn::data::dataloader::batcher::Batcher;
use burn_wgpu::WgpuDevice;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WebModel {
    batcher: TextClassificationBatcher<ServingBackend>,
    model: TextClassificationModel<ServingBackend>,
}

#[wasm_bindgen]
impl WebModel {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let (batcher, model) = load_model_and_tokenizer::<ServingBackend>(
            "./models/text-classification",
            WgpuDevice::default(),
        );
        Self { batcher, model }
    }

    pub fn inference(&self, input: &str) -> String {
        let item = self.batcher.batch(vec![input.to_string()]); // Batch samples using the batcher
        let predictions = self.model.infer(item); // Get model predictions
        let prediction = predictions.clone().slice([0..1]); // Get prediction for current sample
        let class_index = prediction.argmax(1).into_data().convert::<i32>().value[0]; // Get class index with the highest value
        let class = class_name(class_index as usize);
        println!("class: {:?}", class);
        class
    }
}

#[cfg(test)]
mod tests {

    use super::WebModel;

    #[test]
    fn inference_manual_from_test_data() {
        let mdoel = WebModel::new();
        let input = "This is the first text.";

        let output = mdoel.inference(&input);

        println!("output: {:?}", output);
    }
}
