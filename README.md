# Mapping: Model-Agnostic Privacy-Preserving Two-Party Inference Framework for Graph Neural Networks

___Mapping___ is a Python, C++, and Rust library for **Mapping: Model-Agnostic Privacy-Preserving Two-Party Inference Framework for Graph Neural Networks**

## Overview

This library implements a Privacy-Preserving Two-Party Inference Framework for Graph Neural Networks.

# Directory structure

This repository contains several folders that implement the different building blocks of Delphi. The high-level structure of the repository is as follows.
* [`server-SGI++/experiments`](server-SGI++/experiments): Rust crate for running latency, bandwidth, throughput, accuracy, and memory usage experiments using secure protocol SGI++ in server-side.
* [`client-SGI++/experiments`](client-SGI++/experiments): Rust crate for running latency, bandwidth, throughput, accuracy, and memory usage experiments using secure protocol SGI++ in client-side.
* [`client-SGI++/Dataset`](client-SGI++/Dataset): MUTAG dataset.
* [`client-SGI++/weights`](client-SGI++/weights): The weights of a trained GIN model on MUTAG dataset.
* [`proxy-generator`](proxy-generator): The code of Mapping's proxy generator.

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
