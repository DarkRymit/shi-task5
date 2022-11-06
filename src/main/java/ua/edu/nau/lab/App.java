package ua.edu.nau.lab;

import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;

public class App {
    public static void main(String[] args) {
        List<Double> data = List.of(1.92, 4.01, 1.48, 5.45, 1.56, 5.42, 1.28, 4.34, 1.51, 5.49, 1.32, 4.00, 0.49, 4.19,
                1.53);

        double[][] train = IntStream.range(0, data.size() - 2)
                .mapToObj(i -> new double[]{data.get(i), data.get(i + 1), data.get(i + 2)})
                .toList()
                .toArray(new double[0][]);

        double[][] result = IntStream.range(3, data.size())
                .mapToObj(i -> new double[]{data.get(i)})
                .toList()
                .toArray(new double[0][]);

        NeuronNetwork neuronNetwork = new NeuronNetwork(3, new int[]{14, 1});
        neuronNetwork.getNeuron(1).setIsSigmoid(false);
        Random r = new Random();
        int en = 10000;
        for (int e = 0; e < en; e++) {

            for (int i = 0; i < result.length; i++) {
                int idx = r.nextInt(result.length);
                neuronNetwork.train(train[idx], result[idx], 0.05, 0.6);
            }
            if ((e + 1) % 1000 == 0) {
                System.out.printf("%d період%n", e + 1);
                System.out.println("Тестове\tНейроної мережі");
                for (int i = result.length - 2; i < result.length; i++) {
                    double[] t = train[i];
                    double[] s = result[i];
                    System.out.printf("%.2f\t%.2f%n", s[0], neuronNetwork.run(t)[0]);
                }
            }
        }
    }
}
