use crate::AdditiveShare;
use bench_utils::{timer_end, timer_start};
use neural_network::{
    layers::{Layer, LayerInfo, NonLinearLayer, NonLinearLayerInfo, fully_connected::FullyConnectedParams},
    NeuralArchitecture, NeuralNetwork,tensors::Kernel, Evaluate,
};
use rand::{CryptoRng, RngCore};
use std::{
    io::{Read, Write},
    marker::PhantomData,
    os::raw::c_char, ops::Add,
};
use ndarray::s;

use algebra::{
    fixed_point::{FixedPoint, FixedPointParameters},
    fp_64::Fp64Parameters,
    FpParameters, PrimeField,  UniformRandom,
};

use rand::SeedableRng;
use rand_chacha::ChaChaRng;

use io_utils::{counting::CountingIO, imux::IMuxSync};

use neural_network::{
    layers::*,
    tensors::{Input, Output},
};

use crypto_primitives::{
    beavers_mul::{FPBeaversMul, Triple},
    gc::fancy_garbling::{Encoder, GarbledCircuit, Wire}, Share,
};

use crate::{gc::ReluProtocol, linear_layer::LinearProtocol, quad_approx::QuadApproxProtocol};
use protocols_sys::*;
use std::collections::BTreeMap;

pub struct NNProtocol<P: FixedPointParameters> {
    _share: PhantomData<P>,
}

pub const CLIENT: usize = 1;
pub const SERVER: usize = 2;

pub struct ServerState<P: FixedPointParameters> {
    pub linear_state: BTreeMap<usize, Output<P::Field>>,
    pub relu_encoders: Vec<Encoder>,
    pub relu_output_randomizers: Vec<P::Field>,
    pub approx_state: Vec<Triple<P::Field>>,
    pub sfhe: ServerFHE,
    pub sfhe2: ClientFHE,
}
// This is a hack since Send + Sync aren't implemented for the raw pointer types
// Not sure if there's a cleaner way to guarantee this
unsafe impl<P: FixedPointParameters> Send for ServerState<P> {}
unsafe impl<P: FixedPointParameters> Sync for ServerState<P> {}

pub struct ClientState<P: FixedPointParameters> {
    pub relu_circuits: Vec<GarbledCircuit>,
    pub relu_server_labels: Vec<Vec<Wire>>,
    pub relu_client_labels: Vec<Vec<Wire>>,
    pub relu_next_layer_randomizers: Vec<P::Field>,
    pub approx_state: Vec<Triple<P::Field>>,
    /// Randomizers for the input of a linear layer.
    pub linear_randomizer: BTreeMap<usize, Input<P::Field>>,
    /// Shares of the output of a linear layer
    pub linear_post_application_share: BTreeMap<usize, Output<AdditiveShare<P>>>,
    pub cfhe: ClientFHE,
    pub cfhe2: ServerFHE,
}

pub struct NNProtocolType;
// The final message from the server to the client, contains a share of the
// output.
pub type MsgSend<'a, P> = crate::OutMessage<'a, Output<AdditiveShare<P>>, NNProtocolType>;
pub type MsgRcv<P> = crate::InMessage<Output<AdditiveShare<P>>, NNProtocolType>;

// aggregation阶段传输的密文

pub type OnlineServerMsgSend<'a> = crate::OutMessage<'a, Vec<c_char>, NNProtocolType>;
pub type OnlineServerMsgRcv = crate::InMessage<Vec<c_char>, NNProtocolType>;

pub type OnlineClientMsgSend<'a> = crate::OutMessage<'a, Vec<c_char>, NNProtocolType>;
pub type OnlineClientMsgRcv = crate::InMessage<Vec<c_char>, NNProtocolType>;

/// ```markdown
///                   Client                     Server
/// --------------------------------------------------------------------------
/// --------------------------------------------------------------------------
/// Offline:
/// 1. Linear:
///                 1. Sample randomizers r
///                 for each layer.
///
///                       ------- Enc(r) ------>
///                                              1. Sample randomness s_1.
///                                              2. Compute Enc(Mr + s_1)
///                       <--- Enc(Mr + s_1) ---
///                 2. Store -(Mr + s1)
///
/// 2. ReLU:
///                                              1. Sample online output randomizers s_2
///                                              2. Garble ReLU circuit with s_2 as input.
///                       <-------- GC ---------
///                 1. OT input:
///                     Mr_i + s_(1, i),
///                     r_{i + 1}
///                       <-------- OT -------->
///
/// 3. Quadratic approx:
///                       <- Beaver's Triples ->
///
/// --------------------------------------------------------------------------
///
/// Online:
///
/// 1. Linear:
///                       -- x_i + r_i + s_{2, i} ->
///
///
///                                               1. Derandomize the input
///                                               1. Compute y_i = M(x_i + r_i) + s_{1, i}
///
/// 2. ReLU:
///                                               2. Compute garbled labels for y_i
///                       <- garbled labels -----
///                 1. Evaluate garbled circuit,
///                 2. Set next layer input to
///                 be output of GC.
///
/// 3. Quad Approx
///                   ---- (multiplication protocol) ----
///                  |                                  |
///                  ▼                                  ▼
///                y_i + a                              a
///
///                       ------ y_i + a + r_i -->
/// ```
impl<P: FixedPointParameters> NNProtocol<P>
where
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
{
    pub fn offline_server_protocol<R: Read + Send, W: Write + Send, RNG: CryptoRng + RngCore>(
        reader: &mut IMuxSync<CountingIO<R>>,
        writer: &mut IMuxSync<CountingIO<W>>,
        neural_network: &NeuralNetwork<AdditiveShare<P>, FixedPoint<P>>,
        rng: &mut RNG,
    ) -> Result<ServerState<P>, bincode::Error> {
        let mut num_relu = 0;
        let mut num_approx = 0;
        let mut linear_state = BTreeMap::new();
        // 获取client端的密钥
        let sfhe: ServerFHE = crate::server_keygen(reader)?;
        // server端生成一套同态加密的密钥，解密在server端
        let sfhe2: ClientFHE = crate::client_keygen(writer)?;

        let start_time = timer_start!(|| "Server offline phase");
        let linear_time = timer_start!(|| "Linear layers offline phase");
        for (i, layer) in neural_network.layers.iter().enumerate() {
            match layer {
                Layer::NLL(NonLinearLayer::ReLU(dims)) => {
                    let (b, c, h, w) = dims.input_dimensions();
                    num_relu += 1 * c * h * w;
                }
                Layer::NLL(NonLinearLayer::PolyApprox { dims, .. }) => {
                    let (b, c, h, w) = dims.input_dimensions();
                    num_approx += 1 * c * h * w;
                }
                Layer::LL(layer) => {
                    let randomizer = match &layer {
                        LinearLayer::Conv2d { .. } | LinearLayer::FullyConnected { .. } => {
                            let mut cg_handler = match &layer {
                                LinearLayer::Conv2d { .. } => SealServerCG::Conv2D(
                                    server_cg::Conv2D::new(&sfhe, layer, &layer.kernel_to_repr()),
                                ),
                                LinearLayer::FullyConnected { .. } => {
                                    SealServerCG::FullyConnected(server_cg::FullyConnected::new(
                                        &sfhe,
                                        layer,
                                        &layer.kernel_to_repr(),
                                    ))
                                }
                                _ => unreachable!(),
                            };
                            LinearProtocol::<P>::offline_server_protocol(
                                reader,
                                writer,
                                layer.input_dimensions(),
                                layer.output_dimensions(),
                                &mut cg_handler,
                                rng,
                            )?
                        }
                        // AvgPool and Identity don't require an offline phase
                        LinearLayer::AvgPool { dims, .. } => {
                            Output::zeros(dims.output_dimensions())
                        }
                        LinearLayer::Identity { dims } => Output::zeros(dims.output_dimensions()),
                    };
                    linear_state.insert(i, randomizer);
                }
            }
        }
        timer_end!(linear_time);

        let relu_time =
            timer_start!(|| format!("ReLU layers offline phase, with {:?} activations", num_relu));
        let crate::gc::ServerState {
            encoders: relu_encoders,
            output_randomizers: relu_output_randomizers,
        } = ReluProtocol::<P>::offline_server_protocol(reader, writer, num_relu, rng)?;
        timer_end!(relu_time);

        let approx_time = timer_start!(|| format!(
            "Approx layers offline phase, with {:?} activations",
            num_approx
        ));
        let approx_state = QuadApproxProtocol::offline_server_protocol::<FPBeaversMul<P>, _, _, _>(
            reader, writer, &sfhe, num_approx, rng,
        )?;
        timer_end!(approx_time);
        timer_end!(start_time);
        Ok(ServerState {
            linear_state,
            relu_encoders,
            relu_output_randomizers,
            approx_state,
            sfhe,
            sfhe2,
        })
    }

    pub fn offline_client_protocol<R: Read + Send, W: Write + Send, RNG: RngCore + CryptoRng>(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        neural_network_architecture: &NeuralArchitecture<AdditiveShare<P>, FixedPoint<P>>,
        rng: &mut RNG,
    ) -> Result<ClientState<P>, bincode::Error> {
        let mut num_relu = 0;
        let mut num_approx = 0;
        let mut in_shares = BTreeMap::new();
        let mut out_shares = BTreeMap::new();
        let mut relu_layers = Vec::new();
        let mut approx_layers = Vec::new();
        // 生成client端的密钥，解密在client端
        let cfhe: ClientFHE = crate::client_keygen(writer)?;
        // 获取server端的密钥
        let cfhe2: ServerFHE = crate::server_keygen(reader)?;

        let start_time = timer_start!(|| "Client offline phase");
        let linear_time = timer_start!(|| "Linear layers offline phase");
        for (i, layer) in neural_network_architecture.layers.iter().enumerate() {
            // GIN的纵向结构
            if i < 16{
                match layer {
                    LayerInfo::NLL(dims, NonLinearLayerInfo::ReLU) => {
                        relu_layers.push(i);
                        let (b, c, h, w) = dims.input_dimensions();
                        num_relu += 1 * c * h * w;
                    }
                    LayerInfo::NLL(dims, NonLinearLayerInfo::PolyApprox { .. }) => {
                        approx_layers.push(i);
                        let (b, c, h, w) = dims.input_dimensions();
                        num_approx += 1 * c * h * w;
                    }
                    LayerInfo::LL(dims, linear_layer_info) => {
                        let input_dims = dims.input_dimensions();
                        let output_dims = dims.output_dimensions();
                        let (in_share, mut out_share) = match &linear_layer_info {
                            LinearLayerInfo::Conv2d { .. } | LinearLayerInfo::FullyConnected => {
                                let mut cg_handler = match &linear_layer_info {
                                    LinearLayerInfo::Conv2d { .. } => {
                                        SealClientCG::Conv2D(client_cg::Conv2D::new(
                                            &cfhe,
                                            linear_layer_info,
                                            input_dims,
                                            output_dims,
                                        ))
                                    }
                                    LinearLayerInfo::FullyConnected => {
                                        SealClientCG::FullyConnected(client_cg::FullyConnected::new(
                                            &cfhe,
                                            linear_layer_info,
                                            input_dims,
                                            output_dims,
                                        ))
                                    }
                                    _ => unreachable!(),
                                };
                                LinearProtocol::<P>::offline_client_protocol(
                                    reader,
                                    writer,
                                    layer.input_dimensions(),
                                    layer.output_dimensions(),
                                    &mut cg_handler,
                                    rng,
                                )?
                            }
                            _ => {
                                // AvgPool and Identity don't require an offline communication
                                if out_shares.keys().any(|k| k == &(i - 1)) {
                                    // If the layer comes after a linear layer, apply the function to
                                    // the last layer's output share
                                    let prev_output_share = out_shares.get(&(i - 1)).unwrap();
                                    let mut output_share = Output::zeros(dims.output_dimensions());
                                    linear_layer_info
                                        .evaluate_naive(prev_output_share, &mut output_share);
                                    (Input::zeros(dims.input_dimensions()), output_share)
                                } else {
                                    // Otherwise, just return randomizers of 0
                                    (
                                        Input::zeros(dims.input_dimensions()),
                                        Output::zeros(dims.output_dimensions()),
                                    )
                                }
                            }
                        };
                        // We reduce here becase the input to future layers requires
                        // shares to already be reduced correctly; for example,
                        // `online_server_protocol` reduces at the end of each layer.
                        for share in &mut out_share {
                            share.inner.signed_reduce_in_place();
                        }
                        // r
                        in_shares.insert(i, in_share);
                        // -(Lr + s)
                        out_shares.insert(i, out_share);
                    }
                }
            }
            // GIN的横向结构
            // 先处理后四层，第0层随后再添加
            // 把后面需要用到的中间结果存放到out_shares中
            else {
                match layer {
                    LayerInfo::LL(dims, linear_layer_info) => {
                        let input_dims = dims.input_dimensions();
                        let output_dims = dims.output_dimensions();
                        match &linear_layer_info {
                            LinearLayerInfo::FullyConnected => {
                                let mut cg_handler = 
                                    SealClientCG::FullyConnected(client_cg::FullyConnected::new(
                                        &cfhe,
                                        linear_layer_info,
                                        input_dims,
                                        output_dims,
                                    ));
                                let mut k = (i-16)*4;
                                if k == 16 {
                                    k = 14;
                                }
                                let c_randomness = in_shares.get(&k).expect("");
                                let mut c:Input<AdditiveShare<P>> = Input::zeros(c_randomness.dim());
                                c.randomize_local_share(c_randomness);
                                let mut c_sum:Input<FixedPoint<P>> = Input::zeros(input_dims);
                                for row in 0..c.dim().0{
                                    for col in 0..c.dim().1{
                                        c_sum[[0,col,0,0]] -= c[[row,col,0,0]].inner;
                                    }
                                }
                                let mut out_share = LinearProtocol::<P>::offline_client_protocol_2(
                                    reader,
                                    writer,
                                    &mut c_sum,
                                    output_dims,
                                    &mut cg_handler,
                                    rng,
                                )?;
                                for share in &mut out_share {
                                    share.inner.signed_reduce_in_place();
                                }
                                out_shares.insert(i, out_share);
                            }
                            _ => { }
                        };
                    }
                    _ => { }
                }
            }

        }
        timer_end!(linear_time);
        // Preprocessing for next step with ReLUs; if a ReLU is layer i,
        // we want to take output shares for the (linear) layer i - 1,
        // and input shares for the (linear) layer i + 1.
        let mut current_layer_shares = Vec::new();
        let mut relu_next_layer_randomizers = Vec::new();
        let relu_time =
            timer_start!(|| format!("ReLU layers offline phase with {} ReLUs", num_relu));
        for &i in &relu_layers {
            let current_layer_output_shares = out_shares
                .get(&(i - 1))
                .expect("should exist because every ReLU should be preceeded by a linear layer");
            current_layer_shares.extend_from_slice(current_layer_output_shares.as_slice().unwrap());
            // 这个if判断用来解决最后一层不能是ReLU层的问题
            if i < 15 {
                let next_layer_randomizers = in_shares
                .get(&(i + 1))
                .expect("should exist because every ReLU should be succeeded by a linear layer");
                relu_next_layer_randomizers
                .extend_from_slice(next_layer_randomizers.as_slice().unwrap());
            } else {
                let next_layer_randomizers = in_shares
                .get(&(i - 1))
                .expect("should exist because every ReLU should be succeeded by a linear layer");
                relu_next_layer_randomizers
                .extend_from_slice(next_layer_randomizers.as_slice().unwrap());
            }
            
        }

        let crate::gc::ClientState {
            gc_s: relu_circuits,
            server_randomizer_labels: randomizer_labels,
            client_input_labels: relu_labels,
        } = ReluProtocol::<P>::offline_client_protocol(
            reader,
            writer,
            num_relu,
            current_layer_shares.as_slice(),
            rng,
        )?;
        let (relu_client_labels, relu_server_labels) = if num_relu != 0 {
            let size_of_client_input = relu_labels.len() / num_relu;
            let size_of_server_input = randomizer_labels.len() / num_relu;

            assert_eq!(
                size_of_client_input,
                ReluProtocol::<P>::size_of_client_inputs(),
                "number of inputs unequal"
            );

            let client_labels = relu_labels
                .chunks(size_of_client_input)
                .map(|chunk| chunk.to_vec())
                .collect();
            let server_labels = randomizer_labels
                .chunks(size_of_server_input)
                .map(|chunk| chunk.to_vec())
                .collect();

            (client_labels, server_labels)
        } else {
            (vec![], vec![])
        };
        timer_end!(relu_time);

        let approx_time = timer_start!(|| format!(
            "Approx layers offline phase with {} approximations",
            num_approx
        ));
        let approx_state = QuadApproxProtocol::offline_client_protocol::<FPBeaversMul<P>, _, _, _>(
            reader, writer, &cfhe, num_approx, rng,
        )?;
        timer_end!(approx_time);
        timer_end!(start_time);
        Ok(ClientState {
            relu_circuits,
            relu_server_labels,
            relu_client_labels,
            relu_next_layer_randomizers,
            approx_state,
            linear_randomizer: in_shares,
            linear_post_application_share: out_shares,
            cfhe,
            cfhe2,
        })
    }

    pub fn online_server_protocol<R: Read + Send, W: Write + Send + Send>(
        reader: &mut IMuxSync<CountingIO<R>>,
        writer: &mut IMuxSync<CountingIO<W>>,
        neural_network: &NeuralNetwork<AdditiveShare<P>, FixedPoint<P>>,
        state: &ServerState<P>,
    ) -> Result<(), bincode::Error> {
        let (first_layer_in_dims, first_layer_out_dims) = {
            let layer = neural_network.layers.first().unwrap();
            assert!(
                layer.is_linear(),
                "first layer of the network should always be linear."
            );
            (layer.input_dimensions(), layer.output_dimensions())
        };

        let initial_read = reader.count();
        let initial_write = writer.count();
        let mut lt_read = reader.count();
        let mut lt_write = writer.count();
        let mut agg_read = reader.count();
        let mut agg_write = writer.count();
        let mut relu_read = reader.count();
        let mut relu_write = writer.count();
        let mut other_read = reader.count();
        let mut other_write = writer.count();


        let mut num_consumed_relus = 0;
        let mut num_consumed_triples = 0;
        let mut next_layer_input = Output::zeros(first_layer_out_dims);
        let mut next_layer_derandomizer = Input::zeros(first_layer_in_dims);
        // A*s后添加的随机向量
        let mut server_randomness: Output<P::Field> = Output::zeros(first_layer_out_dims);
        let start_time = timer_start!(|| "Server online phase");
        let mut x_add_c = BTreeMap::new();
        let mut prob_bar_i = BTreeMap::new();
        for (i, layer) in neural_network.layers.iter().enumerate() {
            let pre_read = reader.count();
            let pre_write = writer.count();

            match layer {
                Layer::NLL(NonLinearLayer::ReLU(dims)) => {
                    let start_time = timer_start!(|| "ReLU layer");
                    // Have the server encode the current input, via the garbled circuit,
                    // and then send the labels over to the other party.
                    // 在这里搞个循环，经过n次循环完成gc层的处理
                    let mut next_layer_input_row = Input::zeros((1,next_layer_input.dim().1,1,1));
                    let layer_size = next_layer_input_row.len();
                    assert_eq!(dims.input_dimensions(), next_layer_input_row.dim());
                    let layer_encoders =
                        &state.relu_encoders[num_consumed_relus..(num_consumed_relus + layer_size)];
                    for row in 0..next_layer_input.dim().0 {
                        for col in 0..next_layer_input.dim().1 {
                            next_layer_input_row[[0,col,0,0]] = next_layer_input[[row,col,0,0]];
                        }
                        ReluProtocol::online_server_protocol(
                            writer,
                            &next_layer_input_row.as_slice().unwrap(),
                            layer_encoders,
                        )?;
                    }
                    
                    let relu_output_randomizers = state.relu_output_randomizers
                        [num_consumed_relus..(num_consumed_relus + layer_size)]
                        .to_vec();
                    num_consumed_relus += layer_size;
                    let next_layer_derandomizer_row: Input<<P as FixedPointParameters>::Field> = ndarray::Array1::from_iter(relu_output_randomizers)
                        .into_shape(dims.output_dimensions())
                        .expect("shape should be correct")
                        .into();
                    next_layer_derandomizer = Input::zeros((next_layer_input.dim().0,next_layer_input.dim().1,1,1));
                    for row in 0..next_layer_input.dim().0 {
                        for col in 0..next_layer_input.dim().1 {
                            next_layer_derandomizer[[row,col,0,0]] = next_layer_derandomizer_row[[0,col,0,0]];
                        }
                    }
                    let cur_relu_read = reader.count();
                    let cur_relu_write = writer.count();
                    relu_read +=cur_relu_read - pre_read;
                    relu_write += cur_relu_write - pre_write;

                    timer_end!(start_time);
                }
                Layer::NLL(NonLinearLayer::PolyApprox { dims, poly, .. }) => {
                    let start_time = timer_start!(|| "Approx layer");
                    let mut next_layer_input_row = Input::zeros((1,next_layer_input.dim().1,1,1));
                    let layer_size = next_layer_input_row.len();
                    assert_eq!(dims.input_dimensions(), next_layer_input_row.dim());
                    let triples = &state.approx_state
                        [num_consumed_triples..(num_consumed_triples + layer_size)];
                    num_consumed_triples += layer_size;
                    next_layer_derandomizer = Input::zeros((next_layer_input.dim().0,next_layer_input.dim().1,1,1));
                    for row in 0..next_layer_input.dim().0 {
                    // for row in 0..1 {
                        for col in 0..next_layer_input.dim().1 {
                            next_layer_input_row[[0,col,0,0]] = next_layer_input[[row,col,0,0]];
                        }
                        let shares_of_eval =
                        QuadApproxProtocol::online_server_protocol::<FPBeaversMul<P>, _, _>(
                            SERVER, // party_index: 2
                            reader,
                            writer,
                            &poly,
                            next_layer_input_row.as_slice().unwrap(),
                            triples,
                        )?;
                        let shares_of_eval: Vec<_> =
                            shares_of_eval.into_iter().map(|s| s.inner.inner).collect();
                        let next_layer_derandomizer_row: Input<<P as FixedPointParameters>::Field> = ndarray::Array1::from_iter(shares_of_eval)
                            .into_shape(dims.output_dimensions())
                            .expect("shape should be correct")
                            .into();
                        for col in 0..next_layer_input.dim().1 {
                            next_layer_derandomizer[[row,col,0,0]] = next_layer_derandomizer_row[[0,col,0,0]];
                        }
                    }

                    // let shares_of_eval =
                    //     QuadApproxProtocol::online_server_protocol::<FPBeaversMul<P>, _, _>(
                    //         SERVER, // party_index: 2
                    //         reader,
                    //         writer,
                    //         &poly,
                    //         next_layer_input.as_slice().unwrap(),
                    //         triples,
                    //     )?;
                    // let shares_of_eval: Vec<_> =
                    //     shares_of_eval.into_iter().map(|s| s.inner.inner).collect();
                    // next_layer_derandomizer = ndarray::Array1::from_iter(shares_of_eval)
                    //     .into_shape(dims.output_dimensions())
                    //     .expect("shape should be correct")
                    //     .into();
                    timer_end!(start_time);
                }
                Layer::LL(layer) => {
                    let start_time = timer_start!(|| "Linear layer");
                    // server端代码
                    if i % 4 == 0
                        && i != 0 && i < 16
                    {
                        // 这里只能获取到前3个X+C，最后一个要通过最后一个ReLU层获取
                        let mut input = {
                            let recv:MsgRcv<P> = crate::bytes::deserialize(reader).unwrap();
                            recv.msg()
                        };
                        input.randomize_local_share(&next_layer_derandomizer);
                        // 把这一句放到了后面
                        // x_add_c.insert((i/4), input);

                        // 这里需要将input加密发送给client端，所以server端需要生成一个同态加密的密钥，应该在offline阶段生成
                        // offline阶段已经将密钥生成，现在研究如何把用它去进行加密
                        // 现在已经完成了对一行向量的加密，然后再将密文序列化
                        let mut kernel_vec: Vec<*const u64> = vec![std::ptr::null(); input.dim().0 as usize];
                        // let mut matrix_ct_vec = Vec::new();
                        // 这一步非常重要，如果不这么做的话指针会乱
                        let input_u64 = input.to_repr();
                        for row in 0..input.dim().0 {
                            kernel_vec[row] = input_u64
                                .slice(s![row, .., .., ..])
                                .into_slice()
                                .expect("Error converting kernel")
                                .as_ptr();
                        }
                        let ct = unsafe { server_aggregation(&state.sfhe2, input.dim().0 as u64,input.dim().1 as u64, kernel_vec.as_ptr()) };
                        // // 将密文序列化
                        let ct_vec = unsafe {
                            std::slice::from_raw_parts(ct.inner, ct.size as usize)
                                .to_vec()
                        };
                        // matrix_ct_vec.push(ct_vec);
                        // 将X-r密文发送给client
                        let sent_message = OnlineClientMsgSend::new(&ct_vec);
                        crate::bytes::serialize(writer, &sent_message)?;

                        // 从client端接收加密的AX - r
                        let x_ct: OnlineServerMsgRcv = crate::bytes::deserialize(reader)?;
                        let mut x_ct_i = x_ct.msg();
                        // 将 x_ct_i变为SerialCT
                        let ct = SerialCT {
                            inner: x_ct_i.as_mut_ptr(),
                            size: x_ct_i.len() as u64,
                        };

                        // 将AX - r 解密,用于后续计算
                        let ax_r_ptr = unsafe{ server_decrypt_ax(&state.sfhe2, input.dim().0 as u64, input.dim().1 as u64, ct) };
                        // unsafe{ server_free_ax(ax_r_ptr, input.dim().0 as u64)};
                        let mut ax_r: Input<crypto_primitives::AdditiveShare<FixedPoint<P>>> = Input::zeros(input.dim());
                        for row in 0..ax_r.dim().0{
                            for col in 0..ax_r.dim().1{
                                let ax_r_val = unsafe { *(*(ax_r_ptr.offset(row as isize))).offset(col as isize) };
                                ax_r[[row, col, 0, 0]] = AdditiveShare::new(FixedPoint::with_num_muls(
                                    P::Field::from_repr(ax_r_val.into()),
                                    0,
                                ));
                            }
                        }
                        // 用于最后的池化，从上面挪下来的
                        x_add_c.insert((i/4), input);
                        
                        // 计算完成后添加的随机向量
                        let layer_randomizer = state.linear_state.get(&i).unwrap();

                        // 只求一次A*s
                        // if i == 4{
                            
                        //     let mut s:Input<AdditiveShare<P>> = Input::zeros(next_layer_input.dim());
                        //     s.randomize_local_share(&next_layer_derandomizer);

                        //     let mut s_row = Input::zeros((1,s.dim().1,1,1));
                        //     for col in 0..s_row.dim().1 {
                        //         s_row[[0,col,0,0]] = s[[0,col,0,0]];
                        //     }

                        //     let RANDOMNESS: [u8; 32] = [
                        //         0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4, 0x76,
                        //         0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a, 0x52, 0xd2,
                        //     ];

                        //     let mut tmp_rng = ChaChaRng::from_seed(RANDOMNESS);

                        //     // A*s后添加的随机向量
                        //     server_randomness = Output::zeros((next_layer_input.dim().0,layer.output_dimensions().1,1,1));
                        //     for r in &mut server_randomness {
                        //         *r = P::Field::uniform(&mut tmp_rng);
                        //     }

                        //     // 使用与i对应的随机向量s构建一个layer
                        //     let input_dims = (next_layer_input.dim().0,1,1,1);
                        //     let output_dims = (next_layer_input.dim().0,layer.output_dimensions().1,1,1);
                        //     let layer_dims = LayerDims {
                        //         input_dims,
                        //         output_dims,
                        //     };
                        //     // 把client_relu_output的值转置得到kernel需要的形式
                        //     let mut kernel = Kernel::zeros((layer.output_dimensions().1,1,1,1));
                        //     for p in 0..layer.output_dimensions().1{
                        //         kernel[[p,0,0,0]] = -s_row[[0,p,0,0]].inner;
                        //     }
                        //     let bias:Kernel<FixedPoint<P>> = Kernel::zeros((layer.output_dimensions().1, 1, 1, 1));
                        //     let layer_params = FullyConnectedParams::<AdditiveShare<P>,_>::new(kernel.clone(), bias.clone());
                        //     let inter_layer = LinearLayer::FullyConnected {
                        //         dims: layer_dims,
                        //         params: layer_params,
                        //     };
                        //     let mut cg_handler = 
                        //         SealServerCG::FullyConnected(server_cg::FullyConnected::new(
                        //             &state.sfhe,
                        //             &inter_layer,
                        //             &inter_layer.kernel_to_repr(),
                        //         ));
                        //     LinearProtocol::<P>::online_server_protocol_2(
                        //         reader,
                        //         writer,
                        //         &mut server_randomness,
                        //         input_dims,
                        //         output_dims,
                        //         &mut cg_handler,
                        //     )?;
                        // }
                        
                        LinearProtocol::online_server_protocol_5(
                            reader,
                            ax_r,
                            layer,
                            layer_randomizer,
                            &server_randomness,
                            &mut next_layer_input,
                        )?;
                        let cur_agg_read = reader.count();
                        let cur_agg_write = writer.count();
                        agg_read +=cur_agg_read - pre_read;
                        agg_write += cur_agg_write - pre_write;

                    } else if i < 16 {
                        let layer_randomizer = state.linear_state.get(&i).unwrap();
                        LinearProtocol::online_server_protocol(
                            reader,
                            layer,
                            layer_randomizer,
                            &next_layer_derandomizer,
                            &mut next_layer_input,
                        )?;

                        let cur_fc_read = reader.count();
                        let cur_fc_write = writer.count();
                        lt_read += cur_fc_read - pre_read;
                        lt_write += cur_fc_write - pre_write;
                    } else {
                        if i == 16 {
                            let mut input: Input<AdditiveShare<P>> = Input::zeros((0,0,0,0));
                            input = {
                                let recv:MsgRcv<P> = crate::bytes::deserialize(reader).unwrap();
                                recv.msg()
                            };
                            
                            x_add_c.insert(0, input);
                            // 最后一个ReLU层的X+C需要进行一次性解密
                            input = {
                                let recv:MsgRcv<P> = crate::bytes::deserialize(reader).unwrap();
                                recv.msg()
                            };
                            
                            input.randomize_local_share(&next_layer_derandomizer);
                            x_add_c.insert((i/4), input);
                        }
                        let rows = next_layer_derandomizer.dim().0 as f64;
                        let layer_randomizer = state.linear_state.get(&i).unwrap();
                        let k = i-16;
                        let input = x_add_c.get(&k).unwrap();
                        let mut input_sum:Input<AdditiveShare<P>> = Input::zeros(layer.input_dimensions());
                        for row in 0..input.dim().0{
                            for col in 0..input.dim().1{
                                input_sum[[0,col,0,0]] += input[[row,col,0,0]];
                            }
                        }
                        LinearProtocol::online_server_protocol_4(
                            reader,
                            layer,
                            layer_randomizer,
                            &input_sum,
                            &mut next_layer_input,
                            rows
                        )?;
                        prob_bar_i.insert(k, next_layer_input.clone());
                        let cur_other_read = reader.count();
                        let cur_other_write = writer.count();
                        other_read +=cur_other_read - pre_read;
                        other_write += cur_other_write - pre_write;
                    }
                    // next_layer_input = Output::zeros(layer.output_dimensions());
                    // LinearProtocol::online_server_protocol(
                    //     reader,
                    //     layer,
                    //     layer_randomizer,
                    //     &next_layer_derandomizer,
                    //     &mut next_layer_input,
                    // )?;
                    // next_layer_derandomizer = Output::zeros(layer.output_dimensions());
                    // Since linear operations involve multiplications
                    // by fixed-point constants, we want to truncate here to
                    // ensure that we don't overflow.
                    for share in next_layer_input.iter_mut() {
                        share.inner.signed_reduce_in_place();
                    }
                    timer_end!(start_time);
                }
            }
        }
        let cur_chihua_read = reader.count();
        let cur_chihua_write = writer.count();

        let pool_time = timer_start!(|| "Pooling time");
        let prob_bar_0 = prob_bar_i.get(&0).unwrap();
        let prob_bar_1 = prob_bar_i.get(&1).unwrap();
        let prob_bar_2 = prob_bar_i.get(&2).unwrap();
        let prob_bar_3 = prob_bar_i.get(&3).unwrap();
        let prob_bar_4 = prob_bar_i.get(&4).unwrap();
        let mut prob_bar:Input<AdditiveShare<P>> = Input::zeros((1,prob_bar_0.dim().1,1,1));
        for i in 0..prob_bar_0.dim().1{
            prob_bar[[0,i,0,0]] += prob_bar_0[[0,i,0,0]] + prob_bar_1[[0,i,0,0]] + prob_bar_2[[0,i,0,0]] + prob_bar_3[[0,i,0,0]] + prob_bar_4[[0,i,0,0]];
        }
        // for i in 0..2{
        //     prob_bar[[0,i,0,0]] += prob_bar_0[[0,i,0,0]];
        // }
        // 这里最后结合server应该给client端发送的结果进行修改
        let sent_message = MsgSend::new(&prob_bar);
        crate::bytes::serialize(writer, &sent_message)?;
        
        // let sent_message = MsgSend::new(&next_layer_input);
        // crate::bytes::serialize(writer, &sent_message)?;
        let last_other_read = reader.count();
        let last_other_write = writer.count();
        other_read +=last_other_read - cur_chihua_read;
        other_write += last_other_write - cur_chihua_write;
        
        println!("server端特征变换: 读取{}字节, 写入{}字节", lt_read, lt_write);
        println!("server端ReLU: 读取{}字节, 写入{}字节", relu_read, relu_write);
        println!("server端聚合: 读取{}字节, 写入{}字节", agg_read, agg_write);
        println!("server端池化等: 读取{}字节, 写入{}字节", other_read, other_write);

        timer_end!(pool_time);
        timer_end!(start_time);
        Ok(())
    }

    /// Outputs shares for the next round's input.
    pub fn online_client_protocol<R: Read + Send, W: Write + Send + Send>(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        input: &Input<FixedPoint<P>>,
        A: &Input<FixedPoint<P>>,
        X: &Input<FixedPoint<P>>,
        architecture: &NeuralArchitecture<AdditiveShare<P>, FixedPoint<P>>,
        state: &ClientState<P>,
    ) -> Result<Output<FixedPoint<P>>, bincode::Error> {
        // let first_layer_in_dims = {
        //     let layer = architecture.layers.first().unwrap();
        //     assert!(
        //         layer.is_linear(),
        //         "first layer of the network should always be linear."
        //     );
        //     assert_eq!(layer.input_dimensions(), input.dim());
        //     layer.input_dimensions()
        // };
        // assert_eq!(first_layer_in_dims, input.dim());

        let mut num_consumed_relus = 0;
        let mut num_consumed_triples = 0;

        let start_time = timer_start!(|| "Client online phase");
        let mut lr = Input::zeros(input.dim());
        for row in 0..lr.dim().0 {
            for col in 0..lr.dim().1 {
                lr[[row,col,0,0]] = state.linear_randomizer[&0][[0,col,0,0]];
            }
        }

        let (mut next_layer_input, _) = input.share_with_randomness(&lr);

        // let mut a_s = Input::zeros((input.dim().0,64,1,1));
        
        for (i, layer) in architecture.layers.iter().enumerate() {
            match layer {
                LayerInfo::NLL(dims, nll_info) => {
                    match nll_info {
                        NonLinearLayerInfo::ReLU => {
                            let start_time = timer_start!(|| "ReLU layer");
                            // The client receives the garbled circuits from the server,
                            // uses its already encoded inputs to get the next linear
                            // layer's input.
                            let mut next_layer_input_row = Input::zeros((1,next_layer_input.dim().1,1,1));
                            let layer_size = next_layer_input_row.len();
                            assert_eq!(dims.input_dimensions(), next_layer_input_row.dim());

                            let layer_client_labels = &state.relu_client_labels
                                [num_consumed_relus..(num_consumed_relus + layer_size)];
                            let layer_server_labels = &state.relu_server_labels
                                [num_consumed_relus..(num_consumed_relus + layer_size)];
                            let next_layer_randomizers = &state.relu_next_layer_randomizers
                                [num_consumed_relus..(num_consumed_relus + layer_size)];

                            let layer_circuits = &state.relu_circuits
                                [num_consumed_relus..(num_consumed_relus + layer_size)];

                            let layer_client_labels = layer_client_labels
                                .into_iter()
                                .flat_map(|l| l.clone())
                                .collect::<Vec<_>>();
                            let layer_server_labels = layer_server_labels
                                .into_iter()
                                .flat_map(|l| l.clone())
                                .collect::<Vec<_>>();
                            for row in 0..next_layer_input.dim().0 {
                                let output = ReluProtocol::online_client_protocol(
                                    reader,
                                    layer_size,              // num_relus
                                    &layer_server_labels,    // Labels for layer
                                    &layer_client_labels,    // Labels for layer
                                    &layer_circuits,         // circuits for layer.
                                    &next_layer_randomizers, // circuits for layer.
                                )?;
                                next_layer_input_row = ndarray::Array1::from_iter(output)
                                    .into_shape(dims.output_dimensions())
                                    .expect("shape should be correct")
                                    .into();
                                for col in 0..next_layer_input.dim().1{
                                    next_layer_input[[row,col,0,0]] = next_layer_input_row[[0,col,0,0]];
                                }
                            }
                            num_consumed_relus += layer_size;
                            timer_end!(start_time);
                        }
                        NonLinearLayerInfo::PolyApprox { poly, .. } => {
                            let start_time = timer_start!(|| "Approx layer");
                            let mut next_layer_input_row = Input::zeros((1,next_layer_input.dim().1,1,1));
                            let layer_size = next_layer_input_row.len();
                            assert_eq!(dims.input_dimensions(), next_layer_input_row.dim());
                            let triples = &state.approx_state
                                [num_consumed_triples..(num_consumed_triples + layer_size)];
                            num_consumed_triples += layer_size;
                            for row in 0..next_layer_input.dim().0 {
                            // for row in 0..1 {
                                for col in 0..next_layer_input.dim().1 {
                                    next_layer_input_row[[0,col,0,0]] = next_layer_input[[row,col,0,0]];
                                }
                                
                                let output_row = QuadApproxProtocol::online_client_protocol::<
                                    FPBeaversMul<P>,
                                    _,
                                    _,
                                >(
                                    CLIENT, // party_index: 1
                                    reader,
                                    writer,
                                    &poly,
                                    next_layer_input_row.as_slice().unwrap(),
                                    triples,
                                )?;
                                next_layer_input_row = ndarray::Array1::from_iter(output_row)
                                    .into_shape(dims.output_dimensions())
                                    .expect("shape should be correct")
                                    .into();
                                next_layer_input_row
                                    .randomize_local_share(&state.linear_randomizer[&(i + 1)]);
                                for col in 0..next_layer_input.dim().1 {
                                    next_layer_input[[row,col,0,0]] = next_layer_input_row[[0,col,0,0]];
                                }
                            }
                            // let output = QuadApproxProtocol::online_client_protocol::<
                            //     FPBeaversMul<P>,
                            //     _,
                            //     _,
                            // >(
                            //     CLIENT, // party_index: 1
                            //     reader,
                            //     writer,
                            //     &poly,
                            //     next_layer_input.as_slice().unwrap(),
                            //     triples,
                            // )?;
                            // next_layer_input = ndarray::Array1::from_iter(output)
                            //     .into_shape(dims.output_dimensions())
                            //     .expect("shape should be correct")
                            //     .into();
                            // next_layer_input
                            //     .randomize_local_share(&state.linear_randomizer[&(i + 1)]);
                            timer_end!(start_time);
                        }
                    }
                }
                LayerInfo::LL(_, layer_info) => {
                    let start_time = timer_start!(|| "Linear layer");
                    // Send server secret share if required by the layer
                    // 这个next_layer_input是被server一次性加密的
                    let mut input = next_layer_input.clone();
                    // next_layer_input = state.linear_post_application_share[&i].clone();
                    // client端代码
                    if i % 4 == 0
                        && i != 0 && i < 16
                    {
                        let mul_a_time = timer_start!(|| "乘A操作");
                        let sent_message = MsgSend::new(&input);
                        crate::bytes::serialize(writer, &sent_message)?;

                        let linear_post_application_share = state.linear_post_application_share[&i].clone();
                        next_layer_input = Input::zeros((input.dim().0,linear_post_application_share.dim().1,1,1));
                        for row in 0..input.dim().0 {
                            for col in 0..linear_post_application_share.dim().1 {
                                next_layer_input[[row,col,0,0]] = linear_post_application_share[[0,col,0,0]];
                            }
                        }
                        let layer_size = input.dim().1;
                        // 需要根据GIN的结构推导一下上一个ReLU层对应的随机向量的位置
                        let relu_layer_randomizers = &state.relu_next_layer_randomizers
                            [(i/2-1)*layer_size..(i/2)*layer_size];
                        // let relu_layer_randomizers = &state.relu_next_layer_randomizers
                        //     [(i/2-2)*layer_size..(i/2-1)*layer_size];
                        // let relu_layer_randomizers = &state.relu_next_layer_randomizers
                        //     [(i/4-1)*layer_size..(i/4)*layer_size];
                        let mut vec_relu_layer_randomizers:Vec<<P as FixedPointParameters>::Field> = Vec::new();
                        vec_relu_layer_randomizers.extend_from_slice(relu_layer_randomizers);
                        let mut last_layer_randomizers:Input<<P as FixedPointParameters>::Field> = Input::zeros((1,input.dim().1,1,1));
                        last_layer_randomizers = ndarray::Array1::from_iter(vec_relu_layer_randomizers)
                            .into_shape((1,input.dim().1,1,1))
                            .expect("shape should be correct")
                            .into();
                        // input的值减去这个随机向量,最后还要再加上这个随机向量
                        // let mut c:Input<AdditiveShare<P>> = Input::zeros(input.dim());
                        // c.zip_mut_with(&last_layer_randomizers, |out,s|{
                        //     *out = FixedPoint::randomize_local_share(out, s)
                        // });

                        // 从server端接收加密的X - r
                        let x_ct: OnlineClientMsgRcv = crate::bytes::deserialize(reader)?;
                        let mut x_ct_i = x_ct.msg();
                        // 将 x_ct_i变为SerialCT
                        let ct = SerialCT {
                            inner: x_ct_i.as_mut_ptr(),
                            size: x_ct_i.len() as u64,
                        };
                        // 接着要通过加r获取到X的密文
                        // Convert the vector from P::Field -> u64
                        let mut last_layer_randomizers_u64 = Output::zeros(last_layer_randomizers.dim());
                        last_layer_randomizers_u64
                            .iter_mut()
                            .zip(&last_layer_randomizers)
                            .for_each(|(e1, e2)| *e1 = e2.into_repr().0);
                        let last_layer_randomizers_vec: *const u64 = last_layer_randomizers_u64
                            .slice(s![0, .., .., ..])
                            .into_slice()
                            .expect("Error converting kernel")
                            .as_ptr();
                        // println!("&&&&&&{:?}",last_layer_randomizers_u64.dim());
                        let mut a_vec: Vec<*const u64> = vec![std::ptr::null(); A.dim().0 as usize];
                        // let mut matrix_ct_vec = Vec::new();
                        let A_u64 = A.to_repr();
                        for row in 0..A.dim().0 {
                            a_vec[row] = A_u64
                                .slice(s![row, .., .., ..])
                                .into_slice()
                                .expect("Error converting kernel")
                                .as_ptr();
                        }
                        let ct = unsafe { client_aggregation(&state.cfhe2, A.dim().0 as u64,input.dim().1 as u64, last_layer_randomizers_vec, ct, a_vec.as_ptr()) };
                        // 将处理好的密文AX-r发送给server
                        let ct_vec = unsafe {
                            std::slice::from_raw_parts(ct.inner, ct.size as usize)
                                .to_vec()
                        };
                        let sent_message = OnlineClientMsgSend::new(&ct_vec);
                        crate::bytes::serialize(writer, &sent_message)?;



                        // for row in 0..input.dim().0{
                        //     for col in 0..input.dim().1{
                        //         input[[row,col,0,0]] -=  c[[row,col,0,0]];
                        //     }
                        // }
                        // A乘以input
                        // let mut input_tmp:Input<AdditiveShare<P>> = Input::zeros(input.dim());
                        // for row in 0..input_tmp.dim().0{
                        //     for col in 0..input_tmp.dim().1{
                        //         for k in 0..A.dim().1{
                        //             input_tmp[[row,col,0,0]].inner += A[[row,k,0,0]] * input[[k,col,0,0]].inner;
                        //         }
                        //     }
                        // }
                        
                        // input = input_tmp;
                        
                        // let mut a_row_sum = Input::zeros((A.dim().0,1,1,1));
                        // for row in 0..A.dim().0 {
                        //     for col in 0..A.dim().1 {
                        //         a_row_sum[[row,0,0,0]] += A[[row,col,0,0]];
                        //     }
                        // }
                        
                        // 把A发送到server与server端的密文进行基于同态加密的矩阵运算
                        // let mut a_s = Input::zeros(input.dim());
                        // if i==4{
                        //     match &layer_info {
                        //         LinearLayerInfo::FullyConnected => {
                        //             let mut cg_handler = 
                        //                 SealClientCG::FullyConnected(client_cg::FullyConnected::new(
                        //                     &state.cfhe,  //这个cfhe可以通过offline的state获取，这样就不用重新生成了
                        //                     layer_info,
                        //                     a_row_sum.dim(),
                        //                     input.dim(),
                        //                 ));
                        //             a_s = LinearProtocol::<P>::online_client_protocol_2(
                        //                 reader,
                        //                 writer,
                        //                 a_row_sum.clone(),
                        //                 a_s.dim(),
                        //                 &mut cg_handler,
                        //             )?;
                        //         }
                        //         _ => { }
                        //     }
                        // }
                        
                        // input减去a*s加上之前client端的随机向量
                        // for row in 0..input.dim().0{
                        //     for col in 0..input.dim().1{
                        //         input[[row,col,0,0]] = input[[row,col,0,0]] - a_s[[row,col,0,0]];
                        //     }
                        // }
                        // for i in &mut input.iter_mut() {
                        //     i.inner.signed_reduce_in_place();
                        // }
                        // for row in 0..input.dim().0{
                        //     for col in 0..input.dim().1{
                        //         input[[row,col,0,0]] = input[[row,col,0,0]] + c[[row,col,0,0]];
                        //     }
                        // }
                        timer_end!(mul_a_time);
                        // LinearProtocol::online_client_protocol(
                        //     writer,
                        //     &input,
                        //     &layer_info,
                        //     &mut next_layer_input,
                        // )?;
                    } else if i < 16 {
                        // 这里应该是聚合
                        let linear_post_application_share = state.linear_post_application_share[&i].clone();
                        next_layer_input = Input::zeros((input.dim().0,linear_post_application_share.dim().1,1,1));
                        for row in 0..input.dim().0 {
                            for col in 0..linear_post_application_share.dim().1 {
                                next_layer_input[[row,col,0,0]] = linear_post_application_share[[0,col,0,0]];
                            }
                        }
                        LinearProtocol::online_client_protocol(
                            writer,
                            &input,
                            &layer_info,
                            &mut next_layer_input,
                        )?;
                    } else {
                        // 池化层的在线阶段先不跑
                        // 这里需要把最后一层的X+C发送给server
                        if i == 16{
                            // 传输X+C
                            let (mut x_add_c, _) = X.share_with_randomness(&lr);
                            let sent_message = MsgSend::new(&x_add_c);
                            crate::bytes::serialize(writer, &sent_message)?;
                            // 传输X_4+C
                            let sent_message = MsgSend::new(&input);
                            crate::bytes::serialize(writer, &sent_message)?;
                        }                       
                    }
                    // LinearProtocol::online_client_protocol(
                    //     writer,
                    //     &input,
                    //     &layer_info,
                    //     &mut next_layer_input,
                    // )?;
                    // // If this is not the last layer, and if the next layer
                    // // is also linear, randomize the output correctly.
                    // if i != (architecture.layers.len() - 1)
                    //     && architecture.layers[i + 1].is_linear()
                    // {
                    //     next_layer_input.randomize_local_share(&state.linear_randomizer[&(i + 1)]);
                    // }
                    // linear layer
                    timer_end!(start_time);
                }
            }
        }
        let c_lp_0 = state.linear_post_application_share.get(&16).unwrap();
        let c_lp_1 = state.linear_post_application_share.get(&17).unwrap();
        let c_lp_2 = state.linear_post_application_share.get(&18).unwrap();
        let c_lp_3 = state.linear_post_application_share.get(&19).unwrap();
        let c_lp_4 = state.linear_post_application_share.get(&20).unwrap();
        let mut c_lp:Input<AdditiveShare<P>> = Output::zeros((1,c_lp_0.dim().1,1,1));
        for i in 0..c_lp_0.dim().1{
            c_lp[[0,i,0,0]] += c_lp_0[[0,i,0,0]] + c_lp_1[[0,i,0,0]] + c_lp_2[[0,i,0,0]] + c_lp_3[[0,i,0,0]] + c_lp_4[[0,i,0,0]];
        }
        // for i in 0..2{
        //     c_lp[[0,i,0,0]] += c_lp_0[[0,i,0,0]];
        // }
        let mut prob_bar:Input<AdditiveShare<P>> = Output::zeros((1,c_lp_0.dim().1,1,1));
        prob_bar = {
            let recv:MsgRcv<P> = crate::bytes::deserialize(reader).unwrap();
            recv.msg()
        };

        let mut prob:Input<FixedPoint<P>> = Output::zeros((1,c_lp_0.dim().1,1,1));
        let rows:f64 = X.dim().0 as f64;
        for i in 0..c_lp_0.dim().1{
            prob[[0,i,0,0]] = prob_bar[[0,i,0,0]].inner + c_lp[[0,i,0,0]].inner * rows.into();
        }
        // client online阶段
        timer_end!(start_time);
        Ok(prob)
        
        // let result = crate::bytes::deserialize(reader).map(|output: MsgRcv<P>| {
        //     let server_output_share = output.msg();
        //     server_output_share.combine(&next_layer_input)
        // })?;
        // timer_end!(start_time);
        // Ok(result)
    }
}
