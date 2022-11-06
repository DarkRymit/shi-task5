package ua.edu.nau.lab;

import java.util.Arrays;
import java.util.Random;

public class Neuron {

    double[] output;
    double[] input;
    double[] weights;
    double[] dweights;
    boolean isSigmoid = true;

    public Neuron(int inputSize, int outputSize, Random r) {
        output = new double[outputSize];
        input = new double[inputSize + 1];
        weights = new double[(1 + inputSize) * outputSize];
        dweights = new double[weights.length];
        initWeights(r);
    }

    public void setIsSigmoid(boolean isSigmoid) {
        this.isSigmoid = isSigmoid;
    }

    public void initWeights(Random r) {
        for (int i = 0; i < weights.length; i++) {
            weights[i] = (r.nextDouble() - 0.5f) * 4f;
        }
    }

    public double[] run(double[] in) {
        System.arraycopy(in, 0, input, 0, in.length);
        input[input.length - 1] = 1;
        int offs = 0;
        Arrays.fill(output, 0);
        for (int i = 0; i < output.length; i++) {
            for (int j = 0; j < input.length; j++) {
                output[i] += weights[offs + j] * input[j];
            }
            if (isSigmoid) {
                output[i] = (1 / (1 + Math.exp(-output[i])));
            }
            offs += input.length;
        }
        return Arrays.copyOf(output, output.length);
    }

    public double[] train(double[] error, double learningRate, double momentum) {
        int offs = 0;
        double[] nextError = new double[input.length];
        for (int i = 0; i < output.length; i++) {
            double d = error[i];
            if (isSigmoid) {
                d *= output[i] * (1 - output[i]);
            }
            for (int j = 0; j < input.length; j++) {
                int idx = offs + j;
                nextError[j] += weights[idx] * d;
                double dw = input[j] * d * learningRate;
                weights[idx] += dweights[idx] * momentum + dw;
                dweights[idx] = dw;
            }
            offs += input.length;
        }
        return nextError;
    }

}
