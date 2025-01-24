use crate::*;
use neural_network::{ndarray::Array4, tensors::Input, NeuralArchitecture};
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use std::{
    cmp,
    sync::atomic::{AtomicUsize, Ordering}, collections::BTreeMap,
};

const RANDOMNESS: [u8; 32] = [
    0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4, 0x76,
    0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a, 0x52, 0xd2,
];

pub fn softmax(x: &Input<TenBitExpFP>) -> Input<TenBitExpFP> {
    let mut max: TenBitExpFP = x[[0, 0, 0, 0]];
    x.iter().for_each(|e| {
        max = match max.cmp(e) {
            cmp::Ordering::Less => *e,
            _ => max,
        };
    });
    let mut e_x: Input<TenBitExpFP> = x.clone();
    e_x.iter_mut().for_each(|e| {
        *e = f64::from(*e - max).exp().into();
    });
    let e_x_sum = 1.0 / f64::from(e_x.iter().fold(TenBitExpFP::zero(), |sum, val| sum + *val));
    e_x.iter_mut().for_each(|e| *e *= e_x_sum.into());
    return e_x;
}

pub fn run(
    network: NeuralNetwork<TenBitAS, TenBitExpFP>,
    architecture: NeuralArchitecture<TenBitAS, TenBitExpFP>,
    ax_s: BTreeMap<usize,Array4<f64>>,
    a_s: BTreeMap<usize,Array4<f64>>,
    x_s: BTreeMap<usize,Array4<f64>>,
    classes: BTreeMap<usize,usize>,
) {
    let i = 0;
    let mut correct =0;
    let mut server_rng = ChaChaRng::from_seed(RANDOMNESS);
    let mut client_rng = ChaChaRng::from_seed(RANDOMNESS);
    let base_port = 10095;
    let server_addr = "c:20095"; 
    let mut client_output = Output::zeros((1, 2, 0, 0));
    let mut data_trans = 0;
    let offline_time = timer_start!(|| "ToTal offline time");

    let (client_state) =
    crossbeam::thread::scope(|s| {
        // let scoped_server_state = s.spawn(|_| nn_server_offline(&server_addr, &network, &mut server_rng));
        let client_state = s
            .spawn(|_| {
                nn_client_offline(
                    &server_addr,
                    &architecture,
                    &mut client_rng,
                )
            })
            .join()
            .unwrap();
            
        // let server_state = scoped_server_state.join().unwrap();
        client_state
    })
    .unwrap();
    timer_end!(offline_time);
    
    let online_time = timer_start!(|| "ToTal online time");
    for i in 0..ax_s.len() {
    // for i in 300..411 {
        let port_off = i % 50;
        let server_addr = format!("server_IP:{}", base_port + port_off);
        crossbeam::thread::scope(|s| {
            // let server_output = s.spawn(|_| nn_server_online(&server_state, &server_addr, &network, &mut server_rng));
            client_output = s
                .spawn(|_| {
                    nn_client_online(
                        &client_state,
                        &server_addr,
                        &architecture,
                        (ax_s[&i].clone()).into(),
                        (a_s[&i].clone()).into(),
                        (x_s[&i].clone()).into(),
                        &mut client_rng,
                    )
                })
                .join()
                .unwrap();
            
            // data_trans += server_output.join().unwrap();
        })
        .unwrap();
        // let sm = softmax(&client_output);
        let max = client_output.iter().map(|e| f64::from(*e)).fold(0. / 0., f64::max);
        let index = client_output.iter().position(|e| f64::from(*e) == max).unwrap() as usize;        
        if index == classes[&i] {
            correct += 1;
        }
        println!("The current progress is:{},{},{}",i+1,correct,server_addr);
    }
    timer_end!(online_time);    
    println!("{}",correct);

}
