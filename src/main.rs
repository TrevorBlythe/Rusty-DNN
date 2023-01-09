mod ment;
use ment::layers::*;
use ment::*;
use std::vec;

fn main() {
    //initialize a model
    let mut net = Net::new(
        vec![
            FC::new(2, 5),
            Sig::new(5),
            FC::new(5, 5),
            Sig::new(5),
            FC::new(5, 1),
        ],
        1,
        0.1,
    );

    //loop through the XOR dataset 1000 times and train on it.
    let mut x = 0;
    while x < 5000 {
        net.forward_data(&vec![1.0, 0.0]);
        net.backward_data(&vec![1.0]);

        net.forward_data(&vec![1.0, 1.0]);
        net.backward_data(&vec![0.0]);

        net.forward_data(&vec![0.0, 1.0]);
        net.backward_data(&vec![1.0]);

        net.forward_data(&vec![0.0, 0.0]);
        net.backward_data(&vec![0.0]);
        x += 1;
    }

    //test if the network is doing it right.
    println!("{:?}", net.forward_data(&vec![1.0, 0.0])); //should output ~1
    println!("{:?}", net.forward_data(&vec![0.0,1.0])); //should output ~1
    println!("{:?}", net.forward_data(&vec![0.0,0.0])); //should output ~0
    println!("{:?}", net.forward_data(&vec![1.0,1.0])); //should output ~0
    net.print_layers();
}
