use clap::{App, Arg, ArgMatches};
use experiments::minionn::construct_minionn;
use neural_network::{ndarray::Array4, npy::NpyData};
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use std::{io::Read, path::Path};
use std::collections::BTreeMap;

const RANDOMNESS: [u8; 32] = [
    0x11, 0xe0, 0x8f, 0xbc, 0x89, 0xa7, 0x34, 0x01, 0x45, 0x86, 0x82, 0xb6, 0x51, 0xda, 0xf4, 0x76,
    0x5d, 0xc9, 0x8d, 0xea, 0x23, 0xf2, 0x90, 0x8f, 0x9d, 0x03, 0xf2, 0x77, 0xd3, 0x4a, 0x52, 0xd2,
];

fn get_args() -> ArgMatches<'static> {
    App::new("minionn-accuracy")
        .arg(
            Arg::with_name("weights")
                .short("w")
                .long("weights")
                .takes_value(true)
                .help("Path to weights")
                .required(true),
        )
        .arg(
            Arg::with_name("images")
                .short("i")
                .long("images")
                .takes_value(true)
                .help("Path to test images")
                .required(true),
        )
        .get_matches()
}

fn main() {
    let mut rng = ChaChaRng::from_seed(RANDOMNESS);
    let args = get_args();
    let weights = args.value_of("weights").unwrap();
    let images = args.value_of("images").unwrap();



    // // Build network
    let mut network = construct_minionn(None, 1, 99, &mut rng);
    let architecture = (&network).into();

    // // Load network weights
    network.from_numpy(&weights).unwrap();

    // Open all images, classes, and classification results
    let data_dir = Path::new(&images);
    let mut ax_s = BTreeMap::new();
    let mut a_s = BTreeMap::new();
    let mut x_s = BTreeMap::new();
    let mut classes = BTreeMap::new();
    let mut buf = vec![];
    
    for entry in data_dir.read_dir().unwrap() {
        let path = entry.unwrap().path().to_str().unwrap().to_owned();
        let tmp: Vec<&str> = path.split('_').collect();
        let num:usize = tmp[2].parse().unwrap();
        let row:usize = tmp[3].parse().unwrap();
        let class:usize = tmp[4].get(0..1).unwrap().parse().unwrap();

        
        let data_path = "";//################### the path of data
        let input_dim = 7;

        if path.starts_with([data_path,"AX_"].concat().as_str()) {
            buf = vec![];
            std::fs::File::open(Path::new(&path))
                .unwrap()
                .read_to_end(&mut buf)
                .unwrap();
            let image_vec: Vec<f64> = NpyData::from_bytes(&buf).unwrap().to_vec();
            let ax = Array4::from_shape_vec((row, input_dim, 1, 1), image_vec).unwrap();
            ax_s.insert(num, ax);
            classes.insert(num, class);
        } else if path.starts_with([data_path,"A_"].concat().as_str()) {
            buf = vec![];
            std::fs::File::open(Path::new(&path))
                .unwrap()
                .read_to_end(&mut buf)
                .unwrap();
            let image_vec: Vec<f64> = NpyData::from_bytes(&buf).unwrap().to_vec();
            let a = Array4::from_shape_vec((row, row, 1, 1), image_vec).unwrap();
            a_s.insert(num, a);
        } else if path.starts_with([data_path,"X_"].concat().as_str()) {
            buf = vec![];
            std::fs::File::open(Path::new(&path))
                .unwrap()
                .read_to_end(&mut buf)
                .unwrap();
            let image_vec: Vec<f64> = NpyData::from_bytes(&buf).unwrap().to_vec();
            let x = Array4::from_shape_vec((row, input_dim, 1, 1), image_vec).unwrap();
            x_s.insert(num, x);
        }
    }
    experiments::validation::validate::run(network, architecture, ax_s, a_s, x_s, classes);
}