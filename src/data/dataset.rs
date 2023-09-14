use burn::data::dataset::{source::huggingface::HuggingfaceDatasetLoader, Dataset, SqliteDataset};

// Define a struct for text classification items
#[derive(new, Clone, Debug)]
pub struct TextClassificationItem {
    pub text: String, // The text for classification
    pub label: usize, // The label of the text (classification category)
}

// Trait for text classification datasets
pub trait TextClassificationDataset: Dataset<TextClassificationItem> {
    fn num_classes() -> usize; // Returns the number of unique classes in the dataset
    fn class_name(label: usize) -> String; // Returns the name of the class given its label
}

// Struct for items in the AG News dataset
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AgNewsItem {
    pub text: String, // The text for classification
    pub label: usize, // The label of the text (classification category)
}

// Struct for the AG News dataset
pub struct AgNewsDataset {
    dataset: SqliteDataset<AgNewsItem>, // Underlying SQLite dataset
}

// Implement the Dataset trait for the AG News dataset
impl Dataset<TextClassificationItem> for AgNewsDataset {
    /// Returns a specific item from the dataset
    fn get(&self, index: usize) -> Option<TextClassificationItem> {
        self.dataset
            .get(index)
            .map(|item| TextClassificationItem::new(item.text, item.label)) // Map AgNewsItems to TextClassificationItems
    }

    /// Returns the length of the dataset
    fn len(&self) -> usize {
        self.dataset.len()
    }
}

// Implement methods for constructing the AG News dataset
impl AgNewsDataset {
    /// Returns the training portion of the dataset
    pub fn train() -> Self {
        Self::new("train")
    }

    /// Returns the testing portion of the dataset
    pub fn test() -> Self {
        Self::new("test")
    }

    /// Constructs the dataset from a split (either "train" or "test")
    pub fn new(split: &str) -> Self {
        let dataset: SqliteDataset<AgNewsItem> = HuggingfaceDatasetLoader::new("ag_news")
            .dataset(split)
            .unwrap();
        Self { dataset }
    }
}

/// Implements the TextClassificationDataset trait for the AG News dataset
impl TextClassificationDataset for AgNewsDataset {
    /// Returns the number of unique classes in the dataset
    fn num_classes() -> usize {
        4
    }

    /// Returns the name of a class given its label
    fn class_name(label: usize) -> String {
        match label {
            0 => "World",
            1 => "Sports",
            2 => "Business",
            3 => "Technology",
            _ => panic!("invalid class"),
        }
        .to_string()
    }
}
