# Policy Gradients for Atari Games in Tensorflow

Based on Karpathy Blog Post "From Pong to Pixels"

### Structure

*models/* Contains all models for training. Currently only one based on Karpathy's blog post, which is the REINFORCE algorithm with a several layer CNN. Will add deep deterministic policy gradients.

*logs/* Contains results. Might be empty due to Github size constraints.

*train.py* Main training script. Use `python train.py --help` to get all the flags. Will add logging and checkpointing in the future. Using the flag `--game GAME` you can try playing any game in the Open AI gym catalog.

*utils.py* Assorted utilities for training.

### Requirements

- Tensorflow (clearly) and all of its requirements
- tflearn to simplify boilerplate code
- tqdm (for progress bars)