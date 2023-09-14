# Serving a Model with Burn

## Training the Model

You can decide which backend to use by setting the `--features` flag [`tch-gpu`, `tch-cpu`, `wgpu`, `ndarray`]. For example, to use the Torch GPU Backend, run:

```bash
cargo run --bin fit --release
```

## Serving the Model

To serve the model, run:

```bash
cargo run --bin serve --release
```

## Query the Model

To query the model, run:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"texts":["This is the first text", "This is the second text"]}' http://127.0.0.1:3030/classify
```

## Improvements

- [ ] Model Storage: decide where to store the models weights and configuration. HF?
- [ ] Model Lifecycle: decide how to handle model loading and unloading. HF?
