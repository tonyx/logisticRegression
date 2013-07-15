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
    public static double[] logisticTrain(int[][] mapfeaturesValues, double[] featuresExpectedValues) {

        double[] randomFeaturesParameter = new double[mapfeaturesValues[0].length];
        Random random = new Random(System.currentTimeMillis());

        for (int i=0;i<randomFeaturesParameter.length;i++) {
            randomFeaturesParameter[i] =  random.nextBoolean()?random.nextDouble():-random.nextDouble();
        }
        double[] grad = new double[randomFeaturesParameter.length];

        double alpha = 0.1;
        for (int i=0;i<100000;i++) {
            for (int l=0;l<grad.length;l++) {
                grad[l] = 0.0;
            }
            for (int k=0;k<mapfeaturesValues.length;k++) {
                for (int j=0;j<grad.length;j++) {
                    grad[j] = grad[j]  + (logisticFunction(scalarProduct(randomFeaturesParameter,mapfeaturesValues[k]))-featuresExpectedValues[k])*mapfeaturesValues[k][j];
                }
            }
            for (int m=0;m<randomFeaturesParameter.length;m++) {
                randomFeaturesParameter[m] = randomFeaturesParameter[m] -alpha*grad[m]/mapfeaturesValues.length;
            }
        }
        return randomFeaturesParameter;
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
