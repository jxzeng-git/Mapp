# SGI: Secure Two-Party Graph Neural Network Inference

___SGI___ is a Python, C++, and Rust library for **Secure Two-Party Graph Neural Network Inference**

## Overview

This library implements a cryptographic system for efficient inference on general graph convolutional networks.

# Directory structure

This repository contains several folders that implement the different building blocks of Delphi. The high-level structure of the repository is as follows.
* [`server-SGI++/experiments`](server-SGI++/experiments): Rust crate for running latency, bandwidth, throughput, accuracy, and memory usage experiments in server.
* [`client-SGI++/experiments`](client-SGI++/experiments): Rust crate for running latency, bandwidth, throughput, accuracy, and memory usage experiments in server.
* [`client-SGI++/Dataset`](client-SGI++/Dataset): MUTAG dataset.
* [`client-SGI++/weights`](client-SGI++/weights): The weights of a trained GCN model on MUTAG dataset.

## Build and perfrom inference

The library compiles on the `nightly` toolchain of the Rust compiler. To install the latest version of Rust, first install `rustup` by following the instructions [here](https://rustup.rs/), or via your platform's package manager. Once `rustup` is installed, install the Rust toolchain by invoking:
```bash
rustup install nightly
```

After that, both server and client use `cargo`, the standard Rust build tool, to build the library:
```bash
cd server-SGI++
cargo +nightly run --release --bin minionn-accuracy -- --images DATA_PATH --weights MODEL_WEIGHTS_PATH 
```

```bash
cd client-SGI++
cargo +nightly run --release --bin minionn-accuracy -- --images DATA_PATH --weights MODEL_WEIGHTS_PATH 
```
