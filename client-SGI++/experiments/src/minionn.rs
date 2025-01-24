use ::neural_network as nn;
use nn::{
    layers::{convolution::Padding, Layer},
    NeuralNetwork,
};
use rand::{CryptoRng, RngCore};

use super::*;

pub fn construct_minionn<R: RngCore + CryptoRng>(
    vs: Option<&tch::nn::Path>,
    batch_size: usize,
    num_poly: usize,
    rng: &mut R,
) -> NeuralNetwork<TenBitAS, TenBitExpFP> {
    let relu_layers = vec![1,3,5,7,9,11,13,15];
    // let relu_layers = vec![3,5,7,9,11,13,15];
    // let relu_layers = vec![1,3,5,7,9,11,15];
    // let relu_layers = match num_poly {
    //     0 => vec![1, 3, 5, 7, 9, 11, 13, 15],
    //     1 => vec![3, 5, 7, 9, 11, 13, 15],
    //     3 => vec![1, 5, 7, 9, 11, 13, 15],
    //     5 => vec![1, 3, 7, 9, 11, 13, 15],
    //     7 => vec![1, 3, 5, 9, 11, 13, 15],
    //     9 => vec![1, 3, 5, 7, 11, 13, 15],
    //     11 => vec![1, 3, 5, 7, 9, 13, 15],
    //     13 => vec![1, 3, 5, 7, 9, 11, 15],
    //     99 => vec![3,7,11,15],
    //     15 => vec![1, 3, 5, 7, 9, 11, 13],
    //     _ => unreachable!(),
    // };

    let mut network = match &vs {
        Some(vs) => NeuralNetwork {
            layers: vec![],
            eval_method: ::neural_network::EvalMethod::TorchDevice(vs.device()),
        },
        None => NeuralNetwork {
            layers: vec![],
            ..Default::default()
        },
    };

    // 第1层是全连接层
    // 这里需要替换，将7替换成数据集对应维度，input_dim
    let input_dims = (batch_size, 7, 1, 1);
    // 这里需要替换，将64替换成数据集对应的hidden_dim
    let fc = sample_fc_layer(vs, input_dims, 64, rng).0;
    network.layers.push(Layer::LL(fc));
    // 第2层是ReLU层
    add_activation_layer(&mut network, &relu_layers);
    // 第3层是全连接层
    // 这里需要替换，将64替换成数据集对应的hidden_dim
    let input_dims = network.layers.last().unwrap().output_dimensions();
    let fc = sample_fc_layer(vs, input_dims, 64, rng).0;
    network.layers.push(Layer::LL(fc));
    // 第4层是ReLU层
    add_activation_layer(&mut network, &relu_layers);

    // ******* 5、6、7、8是GIN的第*二*大层 ******

    // 第5层是全连接层
    let input_dims = network.layers.last().unwrap().output_dimensions();
    // 这里需要替换，将64替换成数据集对应的hidden_dim
    let fc = sample_fc_layer(vs, input_dims, 64, rng).0;
    network.layers.push(Layer::LL(fc));
    // 第6层是ReLU层
    add_activation_layer(&mut network, &relu_layers);
    // 第7层是全连接层
    let input_dims = network.layers.last().unwrap().output_dimensions();
    // 这里需要替换，将64替换成数据集对应的hidden_dim
    let fc = sample_fc_layer(vs, input_dims, 64, rng).0;
    network.layers.push(Layer::LL(fc));
    // 第8层是ReLU层
    add_activation_layer(&mut network, &relu_layers);

    // ******* 9、10、11、12是GIN的第*三*大层 ******

    // 第9层是全连接层
    let input_dims = network.layers.last().unwrap().output_dimensions();
    // 这里需要替换，将64替换成数据集对应的hidden_dim
    let fc = sample_fc_layer(vs, input_dims, 64, rng).0;
    network.layers.push(Layer::LL(fc));
    // 第10层是ReLU层
    add_activation_layer(&mut network, &relu_layers);
    // 第11层是全连接层
    let input_dims = network.layers.last().unwrap().output_dimensions();
    // 这里需要替换，将64替换成数据集对应的hidden_dim
    let fc = sample_fc_layer(vs, input_dims, 64, rng).0;
    network.layers.push(Layer::LL(fc));
    // 第12层是ReLU层
    add_activation_layer(&mut network, &relu_layers);

    // ******* 13、14、15、16是GIN的第*三*大层 ******

    // 第13层是全连接层
    let input_dims = network.layers.last().unwrap().output_dimensions();
    // 这里需要替换，将64替换成数据集对应的hidden_dim
    let fc = sample_fc_layer(vs, input_dims, 64, rng).0;
    network.layers.push(Layer::LL(fc));
    // 第14层是ReLU层
    add_activation_layer(&mut network, &relu_layers);
    // 第15层是全连接层
    let input_dims = network.layers.last().unwrap().output_dimensions();
    // 这里需要替换，将64替换成数据集对应的hidden_dim
    let fc = sample_fc_layer(vs, input_dims, 64, rng).0;
    network.layers.push(Layer::LL(fc));
    // 第16层是ReLU层
    add_activation_layer(&mut network, &relu_layers);

    // 池化层0
    // 这里需要替换，将7替换成数据集对应维度，input_dim
    let input_dims = (1, 7, 1, 1);
    let fc = sample_fc_layer(vs, input_dims, 2, rng).0;
    network.layers.push(Layer::LL(fc));
    // 池化层1
    // 这里需要替换，将64替换成数据集对应的hidden_dim
    let input_dims = (1, 64, 1, 1);
    let fc = sample_fc_layer(vs, input_dims, 2, rng).0;
    network.layers.push(Layer::LL(fc));
    // 池化层2
    // 这里需要替换，将64替换成数据集对应的hidden_dim
    let input_dims = (1, 64, 1, 1);
    let fc = sample_fc_layer(vs, input_dims, 2, rng).0;
    network.layers.push(Layer::LL(fc));
    // 池化层3
    // 这里需要替换，将64替换成数据集对应的hidden_dim
    let input_dims = (1, 64, 1, 1);
    let fc = sample_fc_layer(vs, input_dims, 2, rng).0;
    network.layers.push(Layer::LL(fc));
    // 池化层4
    // 这里需要替换，将64替换成数据集对应的hidden_dim
    let input_dims = (1, 64, 1, 1);
    let fc = sample_fc_layer(vs, input_dims, 2, rng).0;
    network.layers.push(Layer::LL(fc));

    network
}
