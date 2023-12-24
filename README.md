# Rust DNN

Create Modular Deep Neural Networks in Rust easy

# In progress

If literally anyone stars this project I will add convolutional layers, more activations, and deconv layers.
If this project get 20 stars I add everything

# Installation

After running

```
cargo add Rust_Simple_DNN
```

Then you must put these in your rust code

```rust
use Rust_Simple_DNN::rdnn::layers::*;
use Rust_Simple_DNN::rdnn::*;
```

# Mini tutorial

This is how you make a neural network that looks like this
<br>
<img src="network.png" alt="image-alt-text-check-github-to-see-image" width="300"/>

Use this code to make it:

```rust
//FC layers are dense layers.
//Sig layers are sigmoid activation
let mut net = Net::new(
        vec![
            FC::new(3, 4), //input 3, output 4
            Sig::new(4), //sigmoid, input 4 output 4

            FC::new(4, 4),
            Sig::new(4), //sigmoid

            FC::new(4, 1),// input 4 output 1
            Sig::new(1), //sigmoid
        ],
        1, //batch size
        0.1, //learning rate
    );
    //"net" is the variable representing your entire network
```

<br>
<br>
This is how you propagate data through the network:

```rust
net.forward_data(&vec![1.0, 0.0, -69.0]); //returns the output vector
```

After propagating some data through, you can then also backpropagate some like this:

```rust
 net.backward_data(&vec![0.0]); //a vector of what you want the nn to output
```

The network will automatically store and apply the gradients, so to train the network, all you need to do is repeatedly forward and backpropagate your data

```rust
let mut x = 0;

    while x < 5000 {
        net.forward_data(&vec![1.0, 0.0, 0.0]);
        net.backward_data(&vec![1.0]);

        net.forward_data(&vec![1.0, 1.0, 0.0]);
        net.backward_data(&vec![0.0]);

        net.forward_data(&vec![0.0, 1.0, 0.0]);
        net.backward_data(&vec![1.0]);

        net.forward_data(&vec![0.0, 0.0, 0.0]);
        net.backward_data(&vec![0.0]);
        x += 1;
    }

//at this point its trained
```

This is Pytorch if it wasn't needlessly complicated be like hahahaha
