use crate::{AdditiveShare, InMessage, OutMessage};
use algebra::{
    fixed_point::{FixedPoint, FixedPointParameters},
    fp_64::Fp64Parameters,
    FpParameters, PrimeField, UniformRandom,
};
use crypto_primitives::additive_share::Share;
use io_utils::imux::IMuxSync;
use neural_network::{
    layers::*,
    tensors::{Input, Output},
    Evaluate,
};
use protocols_sys::{SealClientCG, SealServerCG, *};
use rand::{CryptoRng, RngCore};
use std::{
    io::{Read, Write},
    marker::PhantomData,
    os::raw::c_char, ops::Add,
};
use ndarray::s;
pub struct LinearProtocol<P: FixedPointParameters> {
    _share: PhantomData<P>,
}

pub struct LinearProtocolType;

pub type OfflineServerMsgSend<'a> = OutMessage<'a, Vec<c_char>, LinearProtocolType>;
pub type OfflineServerMsgRcv = InMessage<Vec<c_char>, LinearProtocolType>;
pub type OfflineServerKeyRcv = InMessage<Vec<c_char>, LinearProtocolType>;

pub type OfflineClientMsgSend<'a> = OutMessage<'a, Vec<c_char>, LinearProtocolType>;
pub type OfflineClientMsgRcv = InMessage<Vec<c_char>, LinearProtocolType>;
pub type OfflineClientKeySend<'a> = OutMessage<'a, Vec<c_char>, LinearProtocolType>;

pub type MsgSend<'a, P> = crate::OutMessage<'a, Input<AdditiveShare<P>>, LinearProtocolType>;
pub type MsgRcv<P> = crate::InMessage<Input<AdditiveShare<P>>, LinearProtocolType>;

impl<P: FixedPointParameters> LinearProtocol<P>
where
    P: FixedPointParameters,
    <P::Field as PrimeField>::Params: Fp64Parameters,
    P::Field: PrimeField<BigInt = <<P::Field as PrimeField>::Params as FpParameters>::BigInt>,
{
    pub fn offline_server_protocol<R: Read + Send, W: Write + Send, RNG: RngCore + CryptoRng>(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        input_dims: (usize, usize, usize, usize),
        output_dims: (usize, usize, usize, usize),
        server_cg: &mut SealServerCG,
        rng: &mut RNG,
    ) -> Result<Output<P::Field>, bincode::Error> {
        // TODO: Add batch size
        let start_time = timer_start!(|| "Server linear offline protocol");
        // let preprocess_time = timer_start!(|| "Preprocessing");

        // Sample server's randomness `s` for randomizing the i+1-th layer's share.
        let mut server_randomness: Output<P::Field> = Output::zeros(output_dims);
        // TODO
        for r in &mut server_randomness {
            *r = P::Field::uniform(rng);
        }
        // Convert the secret share from P::Field -> u64
        let mut server_randomness_c = Output::zeros(output_dims);
        server_randomness_c
            .iter_mut()
            .zip(&server_randomness)
            .for_each(|(e1, e2)| *e1 = e2.into_repr().0);
        let mut server_randomness_c_row = Output::zeros((1,output_dims.1,output_dims.2,output_dims.3));
        for row in 0..input_dims.0{
            for out_col in 0..output_dims.1{
                server_randomness_c_row[[0,out_col,0,0]] = server_randomness_c[[row,out_col,0,0]];
            }
            // Preprocess filter rotations and noise masks
            server_cg.preprocess(&server_randomness_c_row);
            // Receive client Enc(r_i)
            // let rcv_time = timer_start!(|| "Receiving Input");
            let client_share: OfflineServerMsgRcv = crate::bytes::deserialize(reader)?;
            let client_share_i = client_share.msg();
            // Compute client's share for layer `i + 1`.
            // That is, compute -Lr + s
            let enc_result_vec = server_cg.process(client_share_i);

            let sent_message = OfflineServerMsgSend::new(&enc_result_vec);
            crate::bytes::serialize(writer, &sent_message)?;
        }
        
        timer_end!(start_time);
        Ok(server_randomness)
    }

    // Output randomness to share the input in the online phase, and an additive
    // share of the output of after the linear function has been applied.
    // Basically, r and -(Lr + s).
    pub fn offline_client_protocol<
        'a,
        R: Read + Send,
        W: Write + Send,
        RNG: RngCore + CryptoRng,
    >(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        input_dims: (usize, usize, usize, usize),
        output_dims: (usize, usize, usize, usize),
        client_cg: &mut SealClientCG,
        rng: &mut RNG,
    ) -> Result<(Input<P::Field>, Output<AdditiveShare<P>>), bincode::Error> {
        // TODO: Add batch size

        // let preprocess_time = timer_start!(|| "Client preprocessing");

        // Generate random share -> r2 = -r1 (because the secret being shared is zero).
        let client_share: Input<FixedPoint<P>> = Input::zeros(input_dims);
        let (r1, r2) = client_share.share(rng);
        let mut r2_row = Input::zeros((1,input_dims.1,input_dims.2,input_dims.3));
        let mut client_share_next = Input::zeros(output_dims);
        let mut client_share_next_row = Input::zeros((1,output_dims.1,output_dims.2,output_dims.3));
        for row in 0..input_dims.0{
            for col in 0..input_dims.1{
                r2_row[[0,col,0,0]] = r2[[row,col,0,0]];
            }
            let ct_vec = client_cg.preprocess(&r2_row.to_repr());
            // timer_end!(preprocess_time);

            // Send layer_i randomness for processing by server.
            // let send_time = timer_start!(|| "Sending input");
            let sent_message = OfflineClientMsgSend::new(&ct_vec);
            crate::bytes::serialize(writer, &sent_message)?;
            // timer_end!(send_time);

            // let rcv_time = timer_start!(|| "Receiving Result");
            let enc_result: OfflineClientMsgRcv = crate::bytes::deserialize(reader)?;
            // timer_end!(rcv_time);

            // let post_time = timer_start!(|| "Post-processing");
            
            // Decrypt + reshape resulting ciphertext and free C++ allocations
            client_cg.decrypt(enc_result.msg());
            client_cg.postprocess(&mut client_share_next_row);
            for out_col in 0..output_dims.1{
                client_share_next[[row,out_col,0,0]] = client_share_next_row[[0,out_col,0,0]];
            }
        }
        // // Preprocess and encrypt client secret share for sending
        // let ct_vec = client_cg.preprocess(&r2.to_repr());
        // timer_end!(preprocess_time);
        // // Send layer_i randomness for processing by server.
        // let send_time = timer_start!(|| "Sending input");
        // let sent_message = OfflineClientMsgSend::new(&ct_vec);
        // crate::bytes::serialize(writer, &sent_message)?;
        // timer_end!(send_time);
        // let rcv_time = timer_start!(|| "Receiving Result");
        // let enc_result: OfflineClientMsgRcv = crate::bytes::deserialize(reader)?;
        // timer_end!(rcv_time);
        // let post_time = timer_start!(|| "Post-processing");
        // let mut client_share_next = Input::zeros(output_dims);
        // // Decrypt + reshape resulting ciphertext and free C++ allocations
        // client_cg.decrypt(enc_result.msg());
        // client_cg.postprocess(&mut client_share_next);

        // Should be equal to -(L*r1 - s)
        assert_eq!(client_share_next.dim(), output_dims);
        // Extract the inner field element.
        let layer_randomness = r1
            .iter()
            .map(|r: &AdditiveShare<P>| r.inner.inner)
            .collect::<Vec<_>>();
        let layer_randomness = ndarray::Array1::from_vec(layer_randomness)
            .into_shape(input_dims)
            .unwrap();
        // timer_end!(post_time);

        Ok((layer_randomness.into(), client_share_next))
    }

    // client用来进行offline阶段的线性预测
    pub fn offline_client_protocol_2<
        'a,
        R: Read + Send,
        W: Write + Send,
        RNG: RngCore + CryptoRng,
    >(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        input: &mut Input<FixedPoint<P>>,
        output_dims: (usize, usize, usize, usize),
        client_cg: &mut SealClientCG,
        rng: &mut RNG,
    ) -> Result<(Output<AdditiveShare<P>>), bincode::Error> {
        // TODO: Add batch size

        // let preprocess_time = timer_start!(|| "Client preprocessing");

        let mut client_share_next = Input::zeros(output_dims);
        let ct_vec = client_cg.preprocess(&input.to_repr());
        // timer_end!(preprocess_time);

        // Send layer_i randomness for processing by server.
        // let send_time = timer_start!(|| "Sending input");
        let sent_message = OfflineClientMsgSend::new(&ct_vec);
        crate::bytes::serialize(writer, &sent_message)?;
        // timer_end!(send_time);

        // let rcv_time = timer_start!(|| "Receiving Result");
        let enc_result: OfflineClientMsgRcv = crate::bytes::deserialize(reader)?;
        // timer_end!(rcv_time);

        // let post_time = timer_start!(|| "Post-processing");
        
        // Decrypt + reshape resulting ciphertext and free C++ allocations
        client_cg.decrypt(enc_result.msg());
        client_cg.postprocess(&mut client_share_next);

        // Should be equal to -(L*r1 - s)
        assert_eq!(client_share_next.dim(), output_dims);
        // timer_end!(post_time);

        Ok(client_share_next)
    }

    pub fn online_client_protocol<W: Write + Send>(
        writer: &mut IMuxSync<W>,
        x_s: &Input<AdditiveShare<P>>,
        layer: &LinearLayerInfo<AdditiveShare<P>, FixedPoint<P>>,
        next_layer_input: &mut Output<AdditiveShare<P>>,
    ) -> Result<(), bincode::Error> {

        match layer {
            LinearLayerInfo::Conv2d { .. } | LinearLayerInfo::FullyConnected => {
                let sent_message = MsgSend::new(x_s);
                crate::bytes::serialize(writer, &sent_message)?;
            }
            _ => {
                layer.evaluate_naive(x_s, next_layer_input);
                for elem in next_layer_input.iter_mut() {
                    elem.inner.signed_reduce_in_place();
                }
            }
        }

        Ok(())
    }

    pub fn online_server_protocol<R: Read + Send>(
        reader: &mut IMuxSync<R>,
        layer: &LinearLayer<AdditiveShare<P>, FixedPoint<P>>,
        output_rerandomizer: &Output<P::Field>,
        input_derandomizer: &Input<P::Field>,
        output: &mut Output<AdditiveShare<P>>,
    ) -> Result<(), bincode::Error> {
        let start = timer_start!(|| "Linear online protocol");
        // Receive client share and compute layer if conv or fc
        let mut input: Input<AdditiveShare<P>> = match &layer {
            LinearLayer::Conv2d { .. } | LinearLayer::FullyConnected { .. } => {
                let recv: MsgRcv<P> = crate::bytes::deserialize(reader).unwrap();
                recv.msg()
            }
            _ => Input::zeros(input_derandomizer.dim()),
        };
        // 进行上一层ReLU层的OTP解密
        input.randomize_local_share(input_derandomizer);
        *output = layer.evaluate(&input);
        // 计算完成后添加随机向量
        output.zip_mut_with(output_rerandomizer, |out, s| {
            *out = FixedPoint::randomize_local_share(out, s)
        });
        timer_end!(start);
        Ok(())
    }

    // ClientA的密文与Server的随机向量相乘
    pub fn online_client_protocol_2<
        'a,
        R: Read + Send,
        W: Write + Send,
    >(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        input: Input<FixedPoint<P>>,
        output_dims: (usize, usize, usize, usize),
        client_cg: &mut SealClientCG,
    ) -> Result<(Output<AdditiveShare<P>>), bincode::Error> {
        let input_dims = input.dim();
        let mut input_row = Input::zeros((1,input_dims.1,input_dims.2,input_dims.3));
        let mut client_share_next = Input::zeros(output_dims);
        let mut client_share_next_row = Input::zeros((1,output_dims.1,output_dims.2,output_dims.3));
        for row in 0..input_dims.0{
            for col in 0..input_dims.1{
                input_row[[0,col,0,0]] = input[[row,col,0,0]];
            }
            let input_ct_vec = client_cg.preprocess(&input_row.to_repr());
        
            // 将加密后的输入写入writer,供server读取
            let sent_message = OfflineClientMsgSend::new(&input_ct_vec);
            crate::bytes::serialize(writer, &sent_message)?;
            // timer_end!(send_time);

            // 从reader中接收计算好的结果
            let enc_result: OfflineClientMsgRcv = crate::bytes::deserialize(reader)?;

            client_cg.decrypt(enc_result.msg());
            client_cg.postprocess(&mut client_share_next_row);
            for out_col in 0..output_dims.1{
                client_share_next[[row,out_col,0,0]] = client_share_next_row[[0,out_col,0,0]];
            }
        }
        Ok(client_share_next)
    }

    //构造出来的层与client端的A相乘
    pub fn online_server_protocol_2<R: Read + Send, W: Write + Send>(
        reader: &mut IMuxSync<R>,
        writer: &mut IMuxSync<W>,
        server_randomness: &mut Input<P::Field>,
        input_dims: (usize, usize, usize, usize),
        output_dims: (usize, usize, usize, usize),
        server_cg: &mut SealServerCG,
    ) -> Result<(), bincode::Error> {
        // TODO: Add batch size
        let start_time = timer_start!(|| "Server linear offline protocol");
        // let preprocess_time = timer_start!(|| "Preprocessing");

        let mut server_randomness_c = Output::zeros(output_dims);
        server_randomness_c
            .iter_mut()
            .zip(server_randomness)
            .for_each(|(e1, e2)| *e1 = e2.into_repr().0);
        let mut server_randomness_c_row = Output::zeros((1,output_dims.1,output_dims.2,output_dims.3));
        for row in 0..input_dims.0{
            for out_col in 0..output_dims.1{
                server_randomness_c_row[[0,out_col,0,0]] = server_randomness_c[[row,out_col,0,0]];
            }
            // Preprocess filter rotations and noise masks
            server_cg.preprocess(&server_randomness_c_row);
            // Receive client Enc(r_i)
            // let rcv_time = timer_start!(|| "Receiving Input");
            let client_share: OfflineServerMsgRcv = crate::bytes::deserialize(reader)?;
            let client_share_i = client_share.msg();
            // Compute client's share for layer `i + 1`.
            // That is, compute -Lr + s
            let enc_result_vec = server_cg.process(client_share_i);

            let sent_message = OfflineServerMsgSend::new(&enc_result_vec);
            crate::bytes::serialize(writer, &sent_message)?;
        }
        
        timer_end!(start_time);
        Ok(())
    }

    // 用来做aggregation(AX)的旧方法
    pub fn online_server_protocol_3<R: Read + Send>(
        reader: &mut IMuxSync<R>,
        layer: &LinearLayer<AdditiveShare<P>, FixedPoint<P>>,
        output_rerandomizer: &Output<P::Field>,
        input_derandomizer: &Input<P::Field>,
        output: &mut Output<AdditiveShare<P>>,
    ) -> Result<(), bincode::Error> {
        let start = timer_start!(|| "Linear online protocol");
        // Receive client share and compute layer if conv or fc
        let mut input: Input<AdditiveShare<P>> = match &layer {
            LinearLayer::Conv2d { .. } | LinearLayer::FullyConnected { .. } => {
                let recv: MsgRcv<P> = crate::bytes::deserialize(reader).unwrap();
                recv.msg()
            }
            _ => Input::zeros(input_derandomizer.dim()),
        };
        
        // 减去随机向量
        let one = FixedPoint::<P>::one();
        let mut s:Input<AdditiveShare<P>> = Input::zeros(input.dim());
        for out in &mut s.iter_mut(){
            out.inner *= one;
        };
        s.zip_mut_with(input_derandomizer, |out, s|{
            *out = FixedPoint::randomize_local_share(out, s);
        });
        for i in &mut s {
            i.inner.signed_reduce_in_place();
        }
        for row in 0..input.dim().0{
            for col in 0..input.dim().1{
                input[[row,col,0,0]] = input[[row,col,0,0]] - s[[row,col,0,0]];
            }
        }
        *output = layer.evaluate(&input);
        // 计算完成后添加随机向量
        output.zip_mut_with(output_rerandomizer, |out, s| {
            *out = FixedPoint::randomize_local_share(out, s)
        });
        timer_end!(start);
        Ok(())
    }
    
    // 用来做线性预测LP层的操作
    pub fn online_server_protocol_4<R: Read + Send>(
        reader: &mut IMuxSync<R>,
        layer: &LinearLayer<AdditiveShare<P>, FixedPoint<P>>,
        output_rerandomizer: &Output<P::Field>,
        input: &Input<AdditiveShare<P>>,
        output: &mut Output<AdditiveShare<P>>,
        rows: f64,
    ) -> Result<(), bincode::Error> {
        let start = timer_start!(|| "Linear online protocol");
        
        *output = layer.evaluate(input);
        
        // 计算完成后添加随机向量
        let mut s:Input<AdditiveShare<P>> = Input::zeros(output.dim());
        let one = FixedPoint::<P>::one();
        for out in &mut s.iter_mut(){
            out.inner *= one;
        };
        s.zip_mut_with(output_rerandomizer, |out, s|{
            *out = FixedPoint::randomize_local_share(out, s);
        });
        for i in &mut s {
            i.inner.signed_reduce_in_place();
        }
        for col in 0..output.dim().1{
            output[[0,col,0,0]].inner += s[[0,col,0,0]].inner * rows.into(); 
        }
        // println!("&&&&&&{:?}",output);
        timer_end!(start);
        Ok(())
    }

    // 用来做aggregation(AX)的新方法
    pub fn online_server_protocol_5<R: Read + Send>(
        reader: &mut IMuxSync<R>,
        input: Input<AdditiveShare<P>>,
        layer: &LinearLayer<AdditiveShare<P>, FixedPoint<P>>,
        output_rerandomizer: &Output<P::Field>,
        input_derandomizer: &Input<P::Field>,
        output: &mut Output<AdditiveShare<P>>,
    ) -> Result<(), bincode::Error> {
        let start = timer_start!(|| "Linear online protocol");
        // Receive client share and compute layer if conv or fc
        // let mut input: Input<AdditiveShare<P>> = match &layer {
        //     LinearLayer::Conv2d { .. } | LinearLayer::FullyConnected { .. } => {
        //         let recv: MsgRcv<P> = crate::bytes::deserialize(reader).unwrap();
        //         recv.msg()
        //     }
        //     _ => Input::zeros(input_derandomizer.dim()),
        // };
        // // 进行上一层ReLU层的OTP解密
        // input.randomize_local_share(input_derandomizer);
        *output = layer.evaluate(&input);
        // 计算完成后添加随机向量
        output.zip_mut_with(output_rerandomizer, |out, s| {
            *out = FixedPoint::randomize_local_share(out, s)
        });
        timer_end!(start);
        Ok(())
    }

}
