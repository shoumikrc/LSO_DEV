package Clustering;

import java.util.HashMap;
import java.util.Random;

import utilities.DataStructureConversions;
import utilities.Logging;
import utilities.Logging.LogLevel;
import weka.classifiers.rules.DecisionTableHashKey;
import weka.clusterers.SimpleKMeans;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class KMeans 
{
	
	public double [][] InitializeKMeansPP(double [][] data, int K, int numIter)
	{
		if( data.length <= 0  ) return null;
		
		int L = data[0].length;
		
		//Logging.println("Total Points " + data.length, LogLevel.DEBUGGING_LOG);
		
		Instances ins = DataStructureConversions.ToWekaInstances(data); 
		Instances centroidsWeka = null;
		
		try{
			SimpleKMeans skm = new SimpleKMeans();
			skm.setNumClusters(K);
			skm.setMaxIterations(numIter);
			skm.setInitializeUsingKMeansPlusPlusMethod(true); 
			
			skm.buildClusterer( ins );
			
			centroidsWeka = skm.getClusterCentroids();
		}
		catch(Exception exc)
		{
			Logging.println(exc.getMessage(), LogLevel.ERROR_LOG);			
		}
		
		double [][] centroids = DataStructureConversions.FromWekaInstances(centroidsWeka);
		return centroids;
	}
	
	public double [][] InitializeKMeansPP(double [][] data, int K)
	{
		return InitializeKMeansPP(data, K, 1);
	}
	
	  public Instances kMeansPlusPlusInit(Instances data, int K) throws Exception
	  {
		Instances m_ClusterCentroids = null;
		EuclideanDistance ed = new EuclideanDistance();
	  	int m_NumClusters = K;
		  
	    Random randomO = new Random();
	    HashMap<DecisionTableHashKey, String> initC = new HashMap<DecisionTableHashKey, String>();

	    // choose initial center uniformly at random
	    int index = randomO.nextInt(data.numInstances());
	    m_ClusterCentroids.add(data.instance(index));
	    
	    DecisionTableHashKey hk = new DecisionTableHashKey(data.instance(index),
	        data.numAttributes(), true);
	    initC.put(hk, null);

	    int iteration = 0;
	    int remainingInstances = data.numInstances() - 1;
	    if (m_NumClusters > 1) {
	      // proceed with selecting the rest

	      // distances to the initial randomly chose center
	      double[] distances = new double[data.numInstances()];
	      double[] cumProbs = new double[data.numInstances()];
	      for (int i = 0; i < data.numInstances(); i++) {
	        distances[i] = ed.distance(data.instance(i),
	            m_ClusterCentroids.instance(iteration));
	      }

	      // now choose the remaining cluster centers
	      for (int i = 1; i < m_NumClusters; i++) {

	        // distances converted to probabilities
	        double[] weights = new double[data.numInstances()];
	        System.arraycopy(distances, 0, weights, 0, distances.length);
	        Utils.normalize(weights);

	        double sumOfProbs = 0;
	        for (int k = 0; k < data.numInstances(); k++) {
	          sumOfProbs += weights[k];
	          cumProbs[k] = sumOfProbs;
	        }

	        cumProbs[data.numInstances() - 1] = 1.0; // make sure there are no
	                                                 // rounding issues

	        // choose a random instance
	        double prob = randomO.nextDouble();
	        for (int k = 0; k < cumProbs.length; k++) {
	          if (prob < cumProbs[k]) {
	            Instance candidateCenter = data.instance(k);
	            hk = new DecisionTableHashKey(candidateCenter,
	                data.numAttributes(), true);
	            if (!initC.containsKey(hk)) {
	              initC.put(hk, null);
	              m_ClusterCentroids.add(candidateCenter);
	            } else {
	              // we shouldn't get here because any instance that is a duplicate
	              // of
	              // an already chosen cluster center should have zero distance (and
	              // hence
	              // zero probability of getting chosen) to that center.
	              System.err.println("We shouldn't get here....");
	            }
	            remainingInstances--;
	            break;
	          }
	        }
	        iteration++;

	        if (remainingInstances == 0) {
	          break;
	        }

	        
	        
	        // prepare to choose the next cluster center.
	        // check distances against the new cluster center to see if it is closer
	        for (int k = 0; k < data.numInstances(); k++) {
	          if (distances[k] > 0) {
	            double newDist = ed.distance(data.instance(k),
	                m_ClusterCentroids.instance(iteration));
	            if (newDist < distances[k]) {
	              distances[k] = newDist;
	            }
	          }
	        }
	      }
	    }
	    
	    return m_ClusterCentroids;
	  }

}
