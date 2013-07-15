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
        int valueOfFeatureParameters[][] = new int[7][numberOfFeatures];
        valueOfFeatureParameters[0] = new int[]{1,100,80};
        valueOfFeatureParameters[1] = new int[]{1,60,70};
        valueOfFeatureParameters[2] = new int[]{1,30,40};
        valueOfFeatureParameters[3] = new int[]{1,15,20};
        valueOfFeatureParameters[4] = new int[]{1,5,8};
        valueOfFeatureParameters[5] = new int[]{1,5,90};
        valueOfFeatureParameters[6] = new int[]{1,0,130};

        // and
        int extendedFeatures[][] = new int[7][10];

        // wrap to quadratic and cubic parameters to learn also non linearly separable classifiers
        for (int i=0;i<valueOfFeatureParameters.length;i++) {
            extendedFeatures[i][0] = valueOfFeatureParameters[i][0]; // =1
            extendedFeatures[i][1] = valueOfFeatureParameters[i][1];
            extendedFeatures[i][2] = valueOfFeatureParameters[i][2];

            extendedFeatures[i][3] = valueOfFeatureParameters[i][1]^2;
            extendedFeatures[i][4] = valueOfFeatureParameters[i][2]^2;
            extendedFeatures[i][5] = valueOfFeatureParameters[i][1]*valueOfFeatureParameters[i][2];
            extendedFeatures[i][6] = valueOfFeatureParameters[i][1]^3;
            extendedFeatures[i][7] = valueOfFeatureParameters[i][2]^3;
            extendedFeatures[i][8] = valueOfFeatureParameters[i][1]^3*valueOfFeatureParameters[i][2]^2;
            extendedFeatures[i][9] = valueOfFeatureParameters[i][2]^3*valueOfFeatureParameters[i][1]^2;
        }
        double featuresExpectedValues[] = new double[]{1.0,0.0,1.0,0.0,1.0,0.0,0.0};

        // when
        theta = LogisticRegresion.logisticTrain( extendedFeatures, featuresExpectedValues);

        // then
        for(int k = 0;k<extendedFeatures.length;k++) {
            Assert.assertEquals(featuresExpectedValues[k], LogisticRegresion.logisticFunction(LogisticRegresion.scalarProduct(theta,extendedFeatures[k])), 0.1);
        }
    }
}
