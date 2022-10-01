import argparse

def config_parser():
    parser = argparse.ArgumentParser(description = "Parser generating parameter for the program")
    parser.add_argument("--num_epochs", action = "store", default = 2000, type = int)
    parser.add_argument("--batch_size", action = "store", default = 128, type = int)
    parser.add_argument("--num_samples", action = "store", default = 2000, type = int)
    parser.add_argument("--num_seeds", action = "store", default = 5, type = int)
    parser.add_argument("--total_epoch", action = "store", default = 30, type = int)
    parser.add_argument("--pickle_type", action = "store", default = 1, type = int)
    parser.add_argument("--hidden_size", action = "store", default = "32 32 10", help = "hidden size for mlp, input as a string with \
        space between layer")   
    parser.add_argument("--activation_id", action = "store", default = 1, type = int, help = "activation for NN, 0 for \
        nn.Tanh(), 1 for nn.ReLU(), 2 for nn.LeakyReLU(), 3 for nn.ELU()")
    
    return parser