package test.it.tonyxzt;

import it.tonyxzt.LogisticRegresion;
import junit.framework.Assert;
import org.junit.Test;

/**
 * Created with IntelliJ IDEA.
 * User: Tonyx
 * Date: 15/07/2013
 * Time: 21:50
 * To change this template use File | Settings | File Templates.
 */
public class LogisticRegressionTest {
    @Test
    public void logisticRegressionTest() {
        // given
        int numberOfFeatures = 4;
        double[] theta;
        int mapfeaturesValues[][] = new int[5][numberOfFeatures];
        mapfeaturesValues[0] = new int[]{1,100,80,10};
        mapfeaturesValues[1] = new int[]{1,60,70,20};
        mapfeaturesValues[2] = new int[]{1,30,40,30};
        mapfeaturesValues[3] = new int[]{1,15,20,50};
        mapfeaturesValues[4] = new int[]{1,5,8,60};

        double featuresExpectedValues[] = new double[]{1.0,0.0,0.0,0.0,1.0};

        // when
        theta = LogisticRegresion.logisticTrain( mapfeaturesValues, featuresExpectedValues);

        // then
        for(int k = 0;k<mapfeaturesValues.length;k++) {
            Assert.assertEquals(featuresExpectedValues[k], LogisticRegresion.logisticFunction(LogisticRegresion.scalarProduct(theta, mapfeaturesValues[k])), 0.1);
        }
    }

    @Test
    public void moreComplexLogisticRegressionTest() {
        // given
        int numberOfFeatures = 3;
        double[] theta;
        int parameters[][] = new int[7][numberOfFeatures];
        parameters[0] = new int[]{1,100,80};
        parameters[1] = new int[]{1,60,70};
        parameters[2] = new int[]{1,30,40};
        parameters[3] = new int[]{1,15,20};
        parameters[4] = new int[]{1,5,8};
        parameters[5] = new int[]{1,5,90};
        parameters[6] = new int[]{1,0,130};

        // and
        int extendedParameters[][] = new int[7][10];

        // wrap to quadratic and cubic parameters to learn also non linearly separable classifiers
        for (int i=0;i<parameters.length;i++) {
            extendedParameters[i][0] = parameters[i][0]; // =1
            extendedParameters[i][1] = parameters[i][1];
            extendedParameters[i][2] = parameters[i][2];

            extendedParameters[i][3] = parameters[i][1]^2;
            extendedParameters[i][4] = parameters[i][2]^2;
            extendedParameters[i][5] = parameters[i][1]*parameters[i][2];
            extendedParameters[i][6] = parameters[i][1]^3;
            extendedParameters[i][7] = parameters[i][2]^3;
            extendedParameters[i][8] = parameters[i][1]^3*parameters[i][2]^2;
            extendedParameters[i][9] = parameters[i][2]^3*parameters[i][1]^2;
        }
        double featuresExpectedValues[] = new double[]{1.0,0.0,1.0,0.0,1.0,0.0,0.0};

        // when
        theta = LogisticRegresion.logisticTrain( extendedParameters, featuresExpectedValues);

        // then
        for(int k = 0;k<extendedParameters.length;k++) {
            Assert.assertEquals(featuresExpectedValues[k], LogisticRegresion.logisticFunction(LogisticRegresion.scalarProduct(theta,extendedParameters[k])), 0.1);
        }
    }
}
