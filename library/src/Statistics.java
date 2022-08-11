/*
 * Author: Caleb Nwokocha
 * School: The University of Manitoba
 * Department: Computer Science
 */

import java.util.Arrays;
import org.apache.commons.lang3.ArrayUtils;

public class Statistics extends Algebra {
    // MINIMUM
    public double minimum(double... sampleSpace) {
        Arrays.sort(sampleSpace);
        double min = sampleSpace[0];
        return min;
    }

    // MAXIMUM
    public double maximum(double... sampleSpace) {
        Arrays.sort(sampleSpace);
        double max = sampleSpace[sampleSpace.length - 1];
        return max;
    }

    // MEAN
    public double mean(double... sampleSpace) {
        double sum = sum(sampleSpace);
        double mean = 0;

        if (sampleSpace.length > 0) {
            mean = sum / sampleSpace.length;
        }

        return mean;
    }

    // MEDIAN
    public double median(double... sampleSpace) {
        double median = 0.0;
        double pair1;
        double pair2;
        double[] centerPair;

        Arrays.sort(sampleSpace);

        if (sampleSpace.length % 2 == 0) {
            pair1 = sampleSpace[(sampleSpace.length + 1) / 2];
            pair2 = sampleSpace[(sampleSpace.length + 2) / 2];
            centerPair = new double[]{pair1, pair1};
            median = this.mean(centerPair);
        } else {
            median = sampleSpace[(sampleSpace.length + 1) / 2];
        }

        return median;
    }

    // MODE
    public double mode (double... sampleSpace) {
        double mode = 0.0;
        int maxCount = 0;
        int count;

        for (double sample1 : sampleSpace) {
            count = 0;

            for (double sample2 : sampleSpace) {
                if (sample2 == sample1)
                    count += 1;
            }

            if (count > maxCount) {
                maxCount = count;
                mode = sample1;
            }
        }

        return mode;
    }

    // LOWER-QUARTILE MEDIAN
    public double Q1 (double... sampleSpace) {
        double median = this.median(sampleSpace);
        double[] event = new double[sampleSpace.length];
        double Q1;
        int j = 0;

        for (double sample : sampleSpace) {
            if (sample < median) {
                event[j] = sample;
                j += 1;
            }
        }

        Arrays.sort(event);
        Q1 = this.median(event);

        return Q1;
    }

    // UPPER-QUARTILE MEDIAN
    public double Q3 (double... sampleSpace) {
        double median = this.median(sampleSpace);
        double[] event = new double[sampleSpace.length];
        double Q3;
        int j = 0;

        for (double sample : sampleSpace) {
            if (sample > median) {
                event[j] = sample;
                j += 1;
            }
        }

        Arrays.sort(event);
        Q3 = this.median(event);

        return Q3;
    }

    // OUTLIER
    public boolean isOutlier (double[] sampleSpace, double outCome) {
        double interQuartileRange = this.Q3(sampleSpace) -
                this.Q1(sampleSpace);

        boolean isOutlier = outCome < this.Q1(sampleSpace) - (1.5 * interQuartileRange) ||
                outCome > this.Q3(sampleSpace) + (1.5 * interQuartileRange);

        return isOutlier;
    }

    // EVENT PROBABILITY
    public double probability (double[] sampleSpace, double... event) {
        double frequency = 0.0;
        double probability = 0.0;

        if (sampleSpace.length > 0) {
            Arrays.sort(event);

            for (int i = 0; i < event.length; i++) {
                try {
                    if (event[i] == event[i + 1]) {
                        event = ArrayUtils.remove(event, i + 1);
                        i -= 1;
                    }
                } catch (ArrayIndexOutOfBoundsException e) {
                    e.printStackTrace();
                }
            }

            for (double outcome : event) {
                for (double sample : sampleSpace){
                    if (sample == outcome) {
                        frequency = sum(frequency, 1.0);
                    }
                }

                probability = frequency / sampleSpace.length;
            }
        }

        return probability;
    }

    // EVENT CONDITIONAL PROBABILITY
    public double probability (double[] sampleSpace, double[] event, double[] givenEvent) {
        double givenEventProbability = this.probability(sampleSpace, givenEvent);
        double conditionalProbability = 0.0;

        if (givenEventProbability > 0.0) {
            conditionalProbability = Math.abs((sum(givenEventProbability,
                    this.probability(sampleSpace, event)) - this.probability(sampleSpace,
                    new double[][]{givenEvent, event}, "or"))
                    / givenEventProbability);
        } else {
            conditionalProbability = this.probability(sampleSpace, event);
        }

        return conditionalProbability;
    }

    // EVENTS PROBABILITY
    public double probability (double[] sampleSpace, double[][] events, String operator) {
        double[] allEvents = new double[0];
        double probability = 0.0;

        if (sampleSpace.length > 0) {
            if ("and".equals(operator)) {
                probability = this.probability(sampleSpace, events[0]);

                for (int i = 1; i < events.length; i++) {
                    probability = multiply(probability,
                            this.probability(sampleSpace, events[i], events[i - 1]));
                }
            } else if ("or".equals(operator)) {
                for (double[] event : events) {
                    allEvents = ArrayUtils.addAll(allEvents, event);
                }

                probability = this.probability(sampleSpace, allEvents);
            } else if ("not".equals(operator)) {
                for (double[] event : events) {
                    allEvents = ArrayUtils.addAll(allEvents, event);
                }

                probability = 1 - this.probability(sampleSpace, allEvents);
            }
        }

        return probability;
    }

    // EVENT OUTCOMES PROBABILITIES
    public double[] probabilities (double[] sampleSpace) {
        double[] probabilities = new double[sampleSpace.length];
        Arrays.fill(probabilities, (double) 1 / probabilities.length);
        return probabilities;
    }

    // EVENTS OUTCOMES PROBABILITIES
    public double[][] probabilities (double[][] sampleSpaces) {
        double[][] probabilities;
        int highestSampleSize = 0;

        // Find the highest sample size in sampleSpaces.
        for (double[] sampleSpace : sampleSpaces) {
            highestSampleSize = (int) multiply(sum(sum(highestSampleSize,
                    sampleSpace.length), Math.abs(sum(highestSampleSize,
                    -sampleSpace.length))), 0.5);
        }

        probabilities = new double[sampleSpaces.length][highestSampleSize];

        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] = probabilities(sampleSpaces[i]);
        }

        return probabilities;
    }

    // EVENT EXPECTATION
    public double expectedValue (double[] sampleSpace, double[] probabilities) {
        double expectedValue = (Double) multiply(sampleSpace, probabilities);
        return expectedValue;
    }

    // EVENTS EXPECTATIONS
    // The expectation of a matrix (which rows are sample spaces) is the
    // weighted average of its column elements.
    public double[] expectedValues (double[][] sampleSpaces, double[][] probabilities) {
        double[][] transposedSampleSpaces = transpose(sampleSpaces);
        double[][] transposedProbabilities = probabilities(transpose(probabilities));
        double[] expectedValues = new double[transposedSampleSpaces.length];

        for (int i = 0; i < expectedValues.length; i++) {
            expectedValues[i] = this.expectedValue(transposedSampleSpaces[i],
                    transposedProbabilities[i]);
        }

        return expectedValues;
    }

    public double[][] expectedValues (double[][][] sampleSpaces, double[][][] probabilities) {
        double[][] events = new double[sampleSpaces.length][];
        double[][] eventsProbabilities = new double[probabilities.length][];
        double[][] expectedValues;
        int j = 0;
        int highestSampleSpaceMatrixRow = 0;

        // Find the highest sample space matrix row in sampleSpaces.
        for (double[][] sampleSpaceMatix : sampleSpaces) {
            highestSampleSpaceMatrixRow = (int) multiply(sum(sum(highestSampleSpaceMatrixRow,
                    sampleSpaceMatix.length) + Math.abs(sum(highestSampleSpaceMatrixRow,
                    -sampleSpaceMatix.length))), 0.5);
        }

        expectedValues = new double[highestSampleSpaceMatrixRow][];

        for (int k = 0; k < highestSampleSpaceMatrixRow; k++) {
            for (int i = 0; i < sampleSpaces.length; i++) {
                events[i] = sampleSpaces[i][j];
                try {
                    eventsProbabilities[i] = probabilities[i][j];
                } catch (ArrayIndexOutOfBoundsException e) {
                    e.printStackTrace();
                }
            }

            expectedValues[k] = expectedValues(events, eventsProbabilities);

            j += 1;
        }

        return expectedValues;
    }

    // EVENT VARIANCE
    public double variance (double[] sampleSpace, double expectedValue) {
        double squaredSum = 0.0;
        double variance = 0.0;

        if (sampleSpace.length > 1) {
            for (double sample : sampleSpace) {
                squaredSum += Math.pow(sample - expectedValue, 2);
            }

            variance = squaredSum / (sampleSpace.length - 1);
        }

        return variance;
    }

    // EVENTS VARIANCES
    public double[] variances (double[][] sampleSpaces, double[] expectedValues) {
        double[] variances = new double[sampleSpaces.length];

        for (int i = 0; i < variances.length; i++) {
            variances[i] = this.variance(sampleSpaces[i], expectedValues[i]);
        }

        return variances;
    }

    // EVENT STANDARD DEVIATION
    public double standardDeviation (double[] sampleSpace, double expectedValue) {
        double standardDeviation = Math.sqrt(this.variance(sampleSpace, expectedValue));
        return standardDeviation;
    }

    // EVENTS STANDARD DEVIATIONS
    public double[] standardDeviations (double[][] sampleSpaces, double[] expectedValues) {
        double[] standardDeviations = new double[sampleSpaces.length];

        for (int i = 0; i < standardDeviations.length; i++) {
            standardDeviations[i] = this.standardDeviation(sampleSpaces[i], expectedValues[i]);
        }

        return standardDeviations;
    }


    // OUTCOME Z-SCORE
    public double zScore (double[] sampleSpace, double[] probabilities, double outCome) {
        double expectedValue = this.expectedValue(sampleSpace, probabilities);
        double standardDeviation = this.standardDeviation(sampleSpace, expectedValue);
        double zScore = 0.0;

        if (standardDeviation > 0.0) {
            zScore = (outCome - expectedValue) / standardDeviation;
        }

        return zScore;
    }

    // EVENT OUTCOMES Z-SCORES
    public double[] zScores (double[] sampleSpaces, double[] probabilities, double[] event) {
        double[] zScores = new double[sampleSpaces.length];
        double expectedValue = this.expectedValue(sampleSpaces, probabilities);
        double standardDeviation = this.standardDeviation(sampleSpaces, expectedValue);

        if (standardDeviation > 0.0) {
            for (int i = 0; i < sampleSpaces.length; i++) {
                zScores[i] = (event[i] - expectedValue) / standardDeviation;
            }
        }

        return zScores;
    }

    // EVENTS OUTCOMES Z-SCORES
    public double[][] zScores (double[][] sampleSpaces, double[][] probabilities, double[][] events) {
        double[][] zScores;
        int highestSampleSize = 0;

        // Find the highest sample size in sampleSpaces.
        for (double[] sampleSpace : sampleSpaces) {
            highestSampleSize = (int) multiply(sum(sum(highestSampleSize,
                    sampleSpace.length) + Math.abs(sum(highestSampleSize, sampleSpace.length))),
                    0.5);
        }

        zScores = new double[sampleSpaces.length][highestSampleSize];

        for (int i = 0; i < zScores.length; i++) {
            zScores[i] = this.zScores(sampleSpaces[i], probabilities[i], events[i]);
        }

        return zScores;
    }

    // EVENTS CORRELATION
    public double correlation (double[][] sampleSpaces, double[][] probabilities) {
        double[][] zScores = zScores(sampleSpaces, probabilities, sampleSpaces);
        double product = (Double) multiply(zScores);
        double correlation = product / (zScores[0].length - 1);
        return correlation;
    }

    // ENTROPY
    public double entropy (double... sampleSpace) {
        double[] probabilities = this.probabilities(sampleSpace);
        return 0.0;
    }
}

