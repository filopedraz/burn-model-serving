pub fn pre_process() {
    println!("pre_process the data");
}

pub fn post_process() {
    println!("post_process the data");
}

pub fn num_classes() -> usize {
    4
}

pub fn class_name(label: usize) -> String {
    match label {
        0 => "World",
        1 => "Sports",
        2 => "Business",
        3 => "Technology",
        _ => panic!("invalid class"),
    }
    .to_string()
}
