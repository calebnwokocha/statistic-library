/*
 * AUTHOR: CALEB PRINCEWILL NWOKOCHA
 * SCHOOL: THE UNIVERSITY OF MANITOBA
 * DEPARTMENT: COMPUTER SCIENCE
 */

import java.util.Arrays;

public class Main {
    public static void main (String[] args) {
        double[] datasetA = new double[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        double[] datasetB = new double[] {2, 4, 6 ,8 ,10, 12, 14, 16, 18, 20};
        double[][] sampleSpaces = new double[][] {datasetA, datasetB};

        Statistics statistics = new Statistics();
        double[] datasetAProbabilities = statistics.probabilities(datasetA);
        double[] datasetBProbabilities = statistics.probabilities(datasetB);
        double[][] probabilities = new double[][] {datasetAProbabilities, datasetBProbabilities};

        double correlation = statistics.correlation(sampleSpaces, probabilities);

        System.out.println("Correlation of " + Arrays.toString(datasetA) +
                " and " + Arrays.toString(datasetB) + " is " + correlation);
    }
}
