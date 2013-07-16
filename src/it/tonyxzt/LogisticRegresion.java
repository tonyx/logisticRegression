package it.tonyxzt;

import java.util.Random;

/**
 * Created with IntelliJ IDEA.
 * User: Tonyx
 * Date: 15/07/2013
 * Time: 21:47
 * To change this template use File | Settings | File Templates.
 */
public class LogisticRegresion {
    public static double[] logisticTrain(int[][] mapfeaturesValues, double[] expected) {

        double[] theta = new double[mapfeaturesValues[0].length];
        Random random = new Random(System.currentTimeMillis());

        for (int i=0;i<theta.length;i++) {
            theta[i] =  random.nextBoolean()?random.nextDouble():-random.nextDouble();
        }
        double[] grad = new double[theta.length];

        double alpha = 0.1;
        for (int i=0;i<1000;i++) {
            for (int l=0;l<grad.length;l++) {
                grad[l] = 0.0;
            }
            for (int k=0;k<mapfeaturesValues.length;k++) {
                for (int j=0;j<grad.length;j++) {
                    grad[j] = grad[j]  + (logisticFunction(scalarProduct(theta,mapfeaturesValues[k]))-expected[k])*mapfeaturesValues[k][j];
                }
            }
            for (int m=0;m<theta.length;m++) {
                theta[m] = theta[m] -alpha*grad[m]/mapfeaturesValues.length;
            }
        }
        return theta;
    }

    public static double logisticFunction(double x) {
        return (1.0/(1.0+((double)(Math.exp(-(long)x)))));
    }

    public static double scalarProduct(double[] firstArray, int secondArray[]) {
        double toReturn = 0;
        for(int i=0;i<firstArray.length;i++) {
            toReturn+=firstArray[i]*secondArray[i];
        }
        return toReturn;
    }

}
