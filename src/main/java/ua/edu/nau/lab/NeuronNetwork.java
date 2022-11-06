package ua.edu.nau.lab;

import java.util.Random;
public class NeuronNetwork {

    Neuron[] neurons;

    public NeuronNetwork(int inputSize, int[] layersSize) {
        neurons = new Neuron[layersSize.length];
        Random r = new Random(1234);
        for (int i = 0; i < layersSize.length; i++) {
            int inSize = i == 0 ? inputSize : layersSize[i - 1];
            neurons[i] = new Neuron(inSize, layersSize[i], r);
        }
    }

    public Neuron getNeuron(int idx) {
        return neurons[idx];
    }

    public double[] run(double[] input) {
        double[] actIn = input;
        for (int i = 0; i < neurons.length; i++) {
            actIn = neurons[i].run(actIn);
        }
        return actIn;
    }

    public void train(double[] input, double[] targetOutput, double learningRate, double momentum) {
        double[] calcOut = run(input);
        double[] error = new double[calcOut.length];
        for (int i = 0; i < error.length; i++) {
            error[i] = targetOutput[i] - calcOut[i]; // negative error
        }
        for (int i = neurons.length - 1; i >= 0; i--) {
            error = neurons[i].train(error, learningRate, momentum);
        }
    }


}
