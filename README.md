# Adversarial-generation-and-training
## Generates adversarial mnist examples and also does adversarial training to improve accuracy over adversarial examples

### This is an implementation of the paper "Explaining and harnessing adversarial examples." arXiv preprint arXiv:1412.6572 (2014).

fastGradient.py trains a CNN network over mnist examples and then uses the trained model to generate adversarial examples.
The test accuracy for the network over MNIST is 99% whereas the accuracy over adversarial examples drops to 60% even though the two sets of images look visually the same.



