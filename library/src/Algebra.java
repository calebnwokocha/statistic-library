/*
 * Author: Caleb Nwokocha
 * School: The University of Manitoba
 * Department: Computer Science
 */

import java.util.*;
import java.lang.reflect.Array;

public class Algebra {

    // SCALAR SUMMATION
    public double sum(double... scalars) {
        double sum = 0.0;

        if (scalars.length > 1) {
            for (double scalar : scalars) {
                sum += scalar;
            }
        } else if(scalars.length == 1) {
            sum = scalars[0];
        }

        return sum;
    }

    // SCALAR MULTIPLICATION
    public double multiply(double... scalars){
        double product = 1.0;

        // Before multiplication, check if there are more
        // than one scalars in the scalars array.
        if (scalars.length > 1) {
            for (double scalar : scalars) {
                product *= scalar;
            }
        } else if (scalars.length == 1) {
            product = scalars[0];
        } else {
            product = 0.0;
        }

        return product;
    }

    // VECTOR SUMMATION
    public double[] sum(double[]... vectors) {
        double[] sum = new double[]{0.0};
        int highestVectorSize = 0;

        // Before summation, check if there are more than
        // one vectors in the vectors array.
        if (vectors.length > 1) {
            // Using Dr. Dan Kalman formulas for maximum
            // and minimum ranks between two vectors.
            // max(r, s) = [r + s + abs(r - s)] / 2
            // min(r, s) = [r + s - abs(r - s)] / 2
            for (double[] row : vectors) {
                highestVectorSize = (int) this.multiply(this.sum(
                        this.sum(highestVectorSize, row.length) +
                                Math.abs(this.sum(highestVectorSize,
                                        -row.length))), 0.5); // max
            }
            sum = new double[highestVectorSize];

            for (int i = 0; i < sum.length; i++) {
                for (double[] vector : vectors) {
                    try {
                        sum[i] = this.sum(sum[i], vector[i]);
                    } catch (ArrayIndexOutOfBoundsException e) {
                        e.printStackTrace();
                    }
                }
            }
        } else if (vectors.length == 1) {
            sum = vectors[0];
        }

        return sum;
    }

    // VECTOR MULTIPLICATION
    public Object multiply(double[]... vectors){
        double product = 0.0;
        double scalarProduct = 1.0; // Cannot be 0.0 because it's used for multiplication.

        // Before multiplication, check if there are more than
        // one vectors in the vectors array.
        if (vectors.length > 1) {
            // Read vector row elements
            for (int i = 0; i < vectors[0].length; i++) {
                // of each vector in the vectors array.
                for (double[] vector : vectors) {
                    // Store the multiplication product of each
                    // vector row element in scalarProduct.
                    try {
                        scalarProduct = this.multiply(scalarProduct, vector[i]);
                    } catch (ArrayIndexOutOfBoundsException e) {
                        scalarProduct = 0.0;
                        break;
                    }
                }
                // Store sum of scalarProduct in product.
                product += scalarProduct;
                // Reset scalarProduct.
                scalarProduct = 1.0;
            }
        } else if (vectors.length == 1) {
            return vectors[0];
        }

        return product;
    }

    // MATRIX SUMMATION
    public double[][] sum(double[][]... matrices) {
        double[][] sum = new double[][]{{0.0}};
        int highestRowSize = 0;
        int highestColumnSize = 0;

        // Before summation, check if there are more than
        // one matrices in the matrices array.
        if (matrices.length > 1) {
            for (double[][] row : matrices) {
                highestRowSize = (int) this.multiply(this.sum(
                        this.sum(highestRowSize, row.length) +
                                Math.abs(this.sum(highestRowSize,
                                        -row.length))), 0.5);

                for (double[] value : row) {
                    highestColumnSize = (int) this.multiply(this.sum(
                            this.sum(highestColumnSize, value.length) +
                                    Math.abs(this.sum(highestColumnSize,
                                            -value.length))), 0.5);
                }
            }

            sum = new double[highestRowSize][highestColumnSize];

            for (int i = 0; i < sum.length; i++) {
                for (int j = 0; j < sum[i].length; j++) {
                    for (double[][] matrix : matrices) {
                        try {
                            sum[i][j] = this.sum(sum[i][j], matrix[i][j]);
                        } catch (ArrayIndexOutOfBoundsException e) {
                            e.printStackTrace();
                        }
                    }
                }
            }

        } else if (matrices.length == 1) {
            sum = matrices[0];
        }

        return sum;
    }

    // MATRIX TRANSPOSITION
    public double[][] transpose(double[][] matrix) {
        double[][] transposedMatrix = new double[][]{{0}};
        int highestColumnSize = 0;

        try {
            // Before transposition, check if there are more
            // than one elements in the matrices array.
            if (matrix.length > 0) {
                for (double[] row : matrix) {
                    highestColumnSize = (int) this.multiply(this.sum(
                            this.sum(highestColumnSize, row.length) +
                                    Math.abs(this.sum(highestColumnSize,
                                            -row.length))), 0.5);
                }

                transposedMatrix = new double[highestColumnSize][matrix.length];

                for (int i = 0; i < matrix.length; i++) {
                    for (int j = 0; j < matrix[i].length; j++) {
                        transposedMatrix[j][i] = matrix[i][j];
                    }
                }
            }
        } catch (ArrayIndexOutOfBoundsException e) {
            e.printStackTrace();
        }

        return transposedMatrix;
    }

    // MATRIX MULTIPLICATION
    public double[][] multiply(double[][]... matrices){
        double[][] product = new double[][]{{0}};
        double[][] transposedMatrix;
        double[] transposedMatrixColElements;
        double[] nextMatrixColElements;
        double[][] tempProduct = new double[][]{{0}};
        int highestColumnSize = 0;

        // Before multiplication, check if there are more than
        // one matrices in the matrices array.
        if (matrices.length > 1) {
            // Do for every matrix in the matrices array:
            for (int n = 0; n < matrices.length; n++) {
                try {
                    for (int i = 0; i < matrices[n + 1].length; i++) {
                        highestColumnSize = (int) this.multiply(this.sum(
                                this.sum(highestColumnSize,
                                        matrices[n + 1][i].length) +
                                        Math.abs(this.sum(highestColumnSize,
                                                -matrices[n + 1][i].length))), 0.5);
                    }
                    // tempProduct (or temporary product) is a temporary array for
                    // storing intermediate products of matrices in matrices 3-D array.
                    // The row rank for tempProduct is the row rank of previous matrix,
                    // and it column rank is the highest column rank of next matrix.
                    tempProduct = new double[matrices[n].length][highestColumnSize];
                } catch (ArrayIndexOutOfBoundsException e) {break;}

                // Transpose the previous matrix and store it in
                // the transposedMatrix array.
                transposedMatrix = this.transpose(matrices[n]);

                // Intialize transposedMatrixColElements to store column elements of
                // the transposed matrix.
                transposedMatrixColElements = new double[transposedMatrix.length];

                // Intialize nextMatrixColElements for storing column elements of
                // the next matrix.
                try {
                    nextMatrixColElements = new double[matrices[n + 1].length];
                } catch (ArrayIndexOutOfBoundsException e) {break;}

                // For every row in tempProduct array:
                for (int i = 0; i < tempProduct.length; i++) {

                    // For every index in the transposedMatrixColElements array:
                    for (int l = 0; l < transposedMatrixColElements.length; l++) {
                        // Store the column elements of the transposed matrix
                        // in each index.
                        transposedMatrixColElements[l] = transposedMatrix[l][i];
                    }

                    // Every column in tempProduct array
                    for (int j = 0; j < tempProduct[0].length; j++) {
                        // and every row element in nextMatrixColElements array,
                        for (int k = 0; k < nextMatrixColElements.length; k++) {
                            try {
                                // Store column element of the next matrix (in
                                // the matrices array) in nextMatrixColElements using
                                // indices k and j because their values correspond
                                // to values of index of column elements of next matrix.
                                nextMatrixColElements[k] = matrices[n + 1][k][j];
                            } catch (ArrayIndexOutOfBoundsException e) {
                                nextMatrixColElements[k] = 0.0;
                            }
                        }

                        // transposedMatrixColElements and nextMatrixColElements are vectors,
                        // therefore, multiply them as vectors using the multiply function. This function
                        // returns a scalar that is then stored in i,j index of tempProduct.
                        tempProduct[i][j] = (double) this.multiply(transposedMatrixColElements,
                                nextMatrixColElements);
                    }
                }

                try {
                    // Substitute the next matrix M(n + 1) in the with product of
                    // previous matrices M(n) and M(n + 1).
                    matrices[n + 1] = tempProduct;
                } catch (ArrayIndexOutOfBoundsException e) {break;}
            }

            // Result is the final temporary product.
            product = tempProduct;

        } else if (matrices.length == 1) {
            product = matrices[0];
        }

        return product;
    }

    // TENSOR SUMMATION
    public double[][][] sum (double[][][]... tensors) {
        double[][][] sum = new double[tensors[0].length]
                [tensors[0][0].length][tensors[0][0][0].length];

        if (tensors.length > 1) {
            for (int i = 0; i < sum.length; i++) {
                for (double[][][] tensor : tensors) {
                    sum[i] = sum(sum[i], tensor[i]);
                }
            }
        } else {
            sum = tensors[0];
        }

        return sum;
    }

    // TENSOR MULTIPLICATION
    public double[][][] multiply (double[][][]... tensors) {
        double[][][] product = new double[0][][];

        if (tensors.length > 1) {
            // TODO:
            // Implement tensor multiplication algorithm.

        } else {
            product = tensors[0];
        }

        return product;
    }

    // SCALAR AND VECTOR MULTIPLICATION
    public Object multiply(double[] scalars, double[][] vectors) {
        double scalarProduct = this.multiply(scalars);
        Object vectorProduct = this.multiply(vectors);
        Double[] vectorProductToDouble;
        Object scalarVectorProduct;

        if (vectorProduct.getClass().isArray()) {
            vectorProductToDouble = new Double[Array.getLength(vectorProduct)];
            for (int i = 0; i < vectorProductToDouble.length; i++) {
                vectorProductToDouble[i] = multiply((Double) Array.get(vectorProduct, i),
                        scalarProduct);
            }

            scalarVectorProduct = vectorProductToDouble;
        } else {
            scalarVectorProduct = this.multiply(scalarProduct, (double) vectorProduct);
        }

        return scalarVectorProduct;
    }

    // VECTOR AND SCALAR MULTIPLICATION
    public Object multiply(double[][] vectors, double[] scalars) {
        return multiply(scalars, vectors);
    }

    // SCALAR AND MATRIX MULTIPLICATION
    public double[][] multiply(double[] scalars, double[][][] matrices) {
        double scalarProduct = this.multiply(scalars);
        double[][] matrixProduct = this.multiply(matrices);
        double[][] scalarMatrixProduct = new double[matrixProduct.length]
                [matrixProduct[0].length];

        scalarMatrixProduct[0][0] = 0.0;

        for (int i = 0; i < matrixProduct.length; i++) {
            for (int j = 0; j < matrixProduct[i].length; j++) {
                scalarMatrixProduct[i][j] = this.multiply(scalarProduct, matrixProduct[i][j]);
            }
        }

        return scalarMatrixProduct;
    }

    // MATRIX AND SCALAR MULTIPLICATION
    public double[][] multiply(double[][][] matrices, double[] scalars) {
        return this.multiply(scalars, matrices);
    }

    // TENSOR AND SCALAR MULTIPLICATION
    public double[][][] multiply(double[][][][] tensor, double[] scalars) {
        double[][][] product = multiply(tensor);
        for (int i = 0; i < tensor.length; i++) {
            product[i] = this.multiply(new double[][][]{product[i]}, scalars);
        }
        return product;
    }

    // SCALAR AND TENSOR MULTIPLICATION
    public double[][][] multiply(double[] scalars, double[][][][] tensor) {
        return multiply(tensor, scalars);
    }

    // MATRIX AND VECTOR MULTIPLICATION
    public Object multiply(double[][][] matrices, double[][] vectors) {
        double[][] matrixProduct = this.multiply(matrices);
        Object vectorProduct = this.multiply(vectors);
        double[][] vectorMatrix = new double[vectors[0].length][1];
        double[][] matrixVectorProduct = {{0.0}};
        double[] reducedVectorProduct;

        if (vectorProduct.getClass().isArray()) {
            for (int i = 0; i < vectorMatrix.length; i++) {
                vectorMatrix[i][0] = (Double) Array.get(vectorProduct, i);
            }

            matrixVectorProduct = this.multiply(matrixProduct, vectorMatrix);
            reducedVectorProduct = new double[matrixVectorProduct.length];

            for (int i = 0; i < reducedVectorProduct.length; i++) {
                reducedVectorProduct[i] = matrixVectorProduct[i][0];
            }

            return reducedVectorProduct;
        } else {
            matrixVectorProduct = this.multiply(new double[][][]{matrixProduct},
                    new double[] {(Double) vectorProduct});
            return matrixVectorProduct;
        }
    }

    // VECTOR AND MATRIX MULTIPLICATION
    public double[][] multiply(double[][] vectors, double[][][] matrices) {
        double[][] matrixProduct = this.multiply(matrices);
        Object vectorProduct = this.multiply(vectors);
        double[][] vectorMatrix = new double[vectors[0].length][1];
        double[][] vectorMatrixProduct = {{0.0}};

        if (vectorProduct.getClass().isArray()) {
            for (int i = 0; i < vectorMatrix.length; i++) {
                vectorMatrix[i][0] = (Double) Array.get(vectorProduct, i);
            }

            vectorMatrixProduct = this.multiply(vectorMatrix, matrixProduct);
        } else {
            vectorMatrixProduct = this.multiply(new double[]{(Double) vectorProduct},
                    new double[][][]{matrixProduct});
        }

        return vectorMatrixProduct;
    }

    // SCALAR, VECTOR, AND MATRIX MULTIPLICATION
    public double[][] multiply(double[] scalars, double[][] vectors, double[][][] matrices) {
        double scalarProduct = this.multiply(scalars);
        Object vectorProduct = this.multiply(vectors);
        double[][] matrixProduct = this.multiply(matrices);
        Object scalarVectorProduct = this.multiply(new double[]{scalarProduct},
                new double[][]{(double[]) vectorProduct});
        double[][] totalProduct = new double[0][];

        if (scalarVectorProduct.getClass().isArray()) {
            totalProduct = this.multiply(new double[][] {(double[]) scalarVectorProduct},
                    new double[][][] {matrixProduct});
        } else {
            totalProduct = this.multiply(new double[]{(Double) scalarVectorProduct},
                    new double[][][]{matrixProduct});
        }

        return totalProduct;
    }

    // SCALAR, MATRIX, AND VECTOR MULTIPLICATION
    public Object multiply(double[] scalars, double[][][] matrices, double[][] vectors) {
        double scalarProduct = this.multiply(scalars);
        Object vectorProduct = this.multiply(vectors);
        double[][] matrixProduct = this.multiply(matrices);
        double[][] scalarMatrixProduct = this.multiply(new double[]{scalarProduct},
                new double[][][]{matrixProduct});
        Object totalProduct = new double[0][];

        if (vectorProduct.getClass().isArray()) {
            totalProduct = this.multiply(new double[][][] {scalarMatrixProduct},
                    new double[][] {(double[]) vectorProduct});
        } else {
            totalProduct = this.multiply(new double[][][] {scalarMatrixProduct},
                    new double[] {(Double) vectorProduct});
        }

        return totalProduct;
    }

    // VECTOR, SCALAR, AND MATRIX MULTIPLICATION
    public double[][] multiply(double[][] vectors, double[] scalars, double[][][] matrices) {
        double scalarProduct = this.multiply(scalars);
        Object vectorProduct = this.multiply(vectors);
        double[][] matrixProduct = this.multiply(matrices);
        Object vectorScalarProduct = this.multiply(new double[][]{(double[]) vectorProduct},
                new double[]{scalarProduct});
        double[][] totalProduct = new double[0][];

        if (vectorScalarProduct.getClass().isArray()) {
            totalProduct = this.multiply(new double[][] {(double[]) vectorScalarProduct},
                    new double[][][] {matrixProduct});
        } else {
            totalProduct = this.multiply(new double[]{(Double) vectorScalarProduct},
                    new double[][][]{matrixProduct});
        }

        return totalProduct;
    }

    // VECTOR, MATRIX, AND SCALAR MULTIPLICATION
    public double[][] multiply(double[][] vectors, double[][][] matrices, double[] scalars) {
        double scalarProduct = this.multiply(scalars);
        Object vectorProduct = this.multiply(vectors);
        double[][] matrixProduct = this.multiply(matrices);
        double[][] vectorMatrixProduct = new double[0][];
        double[][] totalProduct = new double[0][];

        if (vectorProduct.getClass().isArray()) {
            vectorMatrixProduct = this.multiply(new double[][] {(double[]) vectorProduct},
                    new double[][][] {matrixProduct});
        } else {
            vectorMatrixProduct = this.multiply(new double[] {(Double) vectorProduct},
                    new double[][][] {matrixProduct});
        }

        totalProduct = this.multiply(new double[][][]{vectorMatrixProduct},
                new double[]{scalarProduct});

        return totalProduct;
    }

    // MATRIX, VECTOR, AND SCALAR MULTIPLICATION
    public Object multiply(double[][][] matrices, double[][] vectors, double[] scalars) {
        double scalarProduct = this.multiply(scalars);
        Object vectorProduct = this.multiply(vectors);
        double[][] matrixProduct = this.multiply(matrices);
        Object matrixVectorProduct = new double[0][];
        Object totalProduct = new double[0][];

        if (vectorProduct.getClass().isArray()) {
            matrixVectorProduct = this.multiply(new double[][][] {matrixProduct},
                    new double[][] {(double[]) vectorProduct});
        } else {
            matrixVectorProduct = this.multiply(new double[][][] {matrixProduct},
                    new double[] {(Double) vectorProduct});
        }

        if (matrixVectorProduct.getClass().getComponentType().isArray()) {
            totalProduct = this.multiply(new double[][][]{(double[][]) matrixVectorProduct},
                    new double[]{scalarProduct});
        } else {
            totalProduct = this.multiply(new double[][]{(double[]) matrixVectorProduct},
                    new double[]{scalarProduct});
        }

        return totalProduct;
    }

    // MATRIX, SCALAR, AND VECTOR MULTIPLICATION
    public Object multiply(double[][][] matrices, double[] scalars, double[][] vectors) {
        double scalarProduct = this.multiply(scalars);
        Object vectorProduct = this.multiply(vectors);
        double[][] matrixProduct = this.multiply(matrices);
        double[][] matrixScalarProduct = this.multiply(new double[][][]{matrixProduct},
                new double[]{scalarProduct});
        Object totalProduct = new double[0][];

        if (vectorProduct.getClass().isArray()) {
            totalProduct = this.multiply(new double[][][] {matrixScalarProduct},
                    new double[][] {(double[]) vectorProduct});
        } else {
            totalProduct = this.multiply(new double[][][] {matrixScalarProduct},
                    new double[] {(Double) vectorProduct});
        }

        return totalProduct;
    }

    // L^p NORM
    public double norm (double[] vector, int power) {
        double norm = 0.0;

        for (double value : vector) {
            norm = this.sum(norm, Math.pow(Math.abs(value), power));
        }
        norm = Math.pow(norm, 1.0 / power);

        return norm;
    }

    // FROBENIUS NORM
    public double norm (double[][] matrix) {
        double norm = 0.0;

        for (double[] row : matrix) {
            for (int j = 0; j < matrix[0].length; j++) {
                norm = this.sum(norm, Math.pow(row[j], 2));
            }
        }
        norm = Math.sqrt(norm);

        return norm;
    }

    // DIAGONAL MATRIX
    public double[][] diag (double[] vector, double side, boolean isConstant) {
        double[][] diag = new double[vector.length][vector.length];

        for (int i = 0; i < diag.length; i++) {
            diag[i][i] = vector[i];
            for (int j = 0; j < diag[i].length; j++) {
                if (j != i) {
                    diag[i][j] = side;
                    if (!isConstant) {
                        side = sum(side, 1);
                    }
                }
            }
        }

        return diag;
    }

    // MATRIX TRACE
    public double trace (double[][] matrix) {
        double trace = 0.0;

        for (int i = 0; i < matrix.length; i++) {
            try {
                trace += matrix[i][i];
            } catch (ArrayIndexOutOfBoundsException e) {break;}
        }

        return trace;
    }

    // QR DECOMPOSITION
    public double[][][] QR (double[][] matrix) {
        double[][][] QR;
        double[] givenVector = new double[2];
        double vectorAngle;
        double[][] givenRotation = new double[2][2];
        int highestColumnSize = 0;
        double[][] tempQ;
        double[] qDiag;
        double[][] Q;
        double[][] R;
        int k = 0;
        int p = 0;

        // Find the highest column size in the input matrix.
        for (double[] row : matrix) {
            highestColumnSize = (int) this.multiply(this.sum(
                    this.sum(highestColumnSize + row.length) +
                            Math.abs(this.sum(highestColumnSize,
                                    -row.length))), 0.5);
        }

        // Make tempQ a square matrix by padding according to
        // dimensions the input matrix.
        if (matrix.length > highestColumnSize) {
            highestColumnSize += matrix.length - highestColumnSize;
            tempQ = new double[matrix.length][highestColumnSize];
        } else {
            tempQ = new double[matrix.length + (highestColumnSize - matrix.length)]
                    [highestColumnSize];
        }

        // This is the initial diagonal vector of tempQ.
        qDiag = new double[tempQ.length];
        Arrays.fill(qDiag, 1);

        // Initial Q value.
        Q = this.diag(qDiag, 0.0, false);

        for (int n = 0; n < highestColumnSize; n++) {
            for (int i = matrix.length - 1; i > p; i--) {
                try {
                    givenVector[0] = matrix[i - 1][n];
                    givenVector[1] = matrix[i][n];
                } catch (ArrayIndexOutOfBoundsException e) {
                    e.printStackTrace();
                }

                vectorAngle = Math.atan((givenVector[1]) / givenVector[0]);

                // Populate givenRotation matrix.
                givenRotation[0][0] = Math.cos(vectorAngle);
                givenRotation[0][1] = -1 * Math.sin(vectorAngle);
                givenRotation[1][0] = Math.sin(vectorAngle);
                givenRotation[1][1] = Math.cos(vectorAngle);

                // tempQ is initialized as a diagonal matrix and identity
                // of the padded square matrix representation of the input matrix.
                tempQ = this.diag(qDiag, 0.0, false);

                // Update special elements of temporary Q.
                try {
                    tempQ[tempQ.length - 2 - k][3 - k] = givenRotation[0][0];
                    tempQ[tempQ.length - 2 - k][4 - k] = givenRotation[1][0];
                    tempQ[tempQ.length - 1 - k][3 - k] = givenRotation[0][1];
                    tempQ[tempQ.length - 1 - k][4 - k] = givenRotation[1][1];
                } catch (ArrayIndexOutOfBoundsException e) {
                    e.printStackTrace();
                }

                // Q is determined recursively by dot-product of all tempQ transpose.
                Q = this.multiply(Q, this.transpose(tempQ));

                // New matrix dot-product, could be referred as temporary R.
                matrix = this.multiply(tempQ, matrix);

                k += 1;
            }

            k = 0;
            p += 1;
        }

        R = matrix;
        QR = new double[][][]{Q, R};

        return QR;
    }

    // MATRIX DETERMINANT
    public double det (double[][] matrix) {
        double det = 1.0;
        double[][][] QR = null;
        int lowestColumnSize = matrix[0].length;

        // Find the lowest column size in the input matrix.
        for (double[] row : matrix) {
            lowestColumnSize = (int) this.multiply(this.sum(
                    this.sum(lowestColumnSize, row.length) -
                            Math.abs(this.sum(lowestColumnSize,
                                    -row.length))), 0.5);
        }

        // Check if the input matrix is a square matrix.
        if (matrix.length == lowestColumnSize) {
            QR = this.QR(matrix);

            for (int i = 0; i < QR[1].length; i++) {
                det = this.multiply(det, QR[1][i][i]);
            }
        } else {
            det = 0.0;
        }

        return det;
    }
}