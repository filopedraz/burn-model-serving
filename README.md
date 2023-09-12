# Serving a Model with Burn

## Training the Model

You can decide which backend to use by setting the `--features` flag [`tch-gpu`, `tch-cpu`, `wgpu`, `ndarray`]. For example, to use the Torch GPU Backend, run:

```bash
cargo run --bin fit --release --features tch-gpu
```