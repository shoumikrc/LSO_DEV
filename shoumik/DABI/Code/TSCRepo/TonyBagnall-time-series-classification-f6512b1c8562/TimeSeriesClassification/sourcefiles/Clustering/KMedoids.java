/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package Clustering;

import DataStructures.DataInstance;
import DataStructures.DataSet;
import TimeSeries.DTW;
import TimeSeries.DistanceOperator;
import utilities.Logging;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Implementation of the KMedoids algorithm
 * @author josif
 */
public class KMedoids 
{
    // singleton implementation
    
    private static KMedoids instance = null;
    
    public KMedoids()
    {
        // by default dtw is the distance operator
        distanceOperator = DTW.getInstance();
        //distanceOperator = new EuclideanDistance();
        // random number generator
        rand = new Random();
        
        // set default number of iterations
        numIterations = 500;
        
        historyOfMedoidCandidates = new ArrayList<Integer>();
    }
    
    public static KMedoids getInstance()
    {
        if( instance == null )
            instance = new KMedoids();
        
        return instance;
    }
    
    // random number generator
    Random rand;
    
    // the distance operator
    public DistanceOperator distanceOperator;
    
    // the dataset
    DataSet ds;
    int numInstances;
    
    // the predefined number of clusters
    int numClusters;
       
    // the number of iterations
    int numIterations; 
    
    List<Integer> historyOfMedoidCandidates;
    
    public double ConfigurationLoss(List<Integer> clusterConfiguration)
    {
        double loss = 0;
        
        for( int i = 0; i < numInstances; i++ )
        {
            int medoidIndex = clusterConfiguration.get(i);
            
            // if medoid index of the instance is greater than or eq to zero,
            // i.e. instance is not medoid itself, then add loss
            if(medoidIndex >= 0)
            {
                loss += distanceOperator.CalculateDistance(
                    ds.instances.get(i), 
                    ds.instances.get(medoidIndex));
            }
        }
         
        return loss;
    }
    
    public List<Integer> InitializeMedoids()
    {
        boolean [] selectedMedoidsFlags = new boolean[numInstances];
        for( int i = 0; i < numInstances; i++ ) selectedMedoidsFlags[i] = false;
        
        List<Integer> medoids = new ArrayList<Integer>();
        
        for( int i = 0; i < numClusters; i++)
        {
            int randomInstanceIndex = -1;
            
            // pick a previously nonselected medoid index
            do
            {
                randomInstanceIndex = rand.nextInt( numInstances );
            }
            while(selectedMedoidsFlags[randomInstanceIndex]);
            
            // add it to the list and set its flag as selected
            medoids.add(randomInstanceIndex);
            selectedMedoidsFlags[randomInstanceIndex] = true;
        }     
        
        Logging.println("Initialized medoids", Logging.LogLevel.DEBUGGING_LOG);
        
        historyOfMedoidCandidates.clear();
        
        return medoids;
    }
    
    public int FindClosestMedoid(int instanceId, List<Integer> medoids)
    {
        // search for the closest medoid 
        int closestMedoid=-1;
        double closestMedoidDistance = Double.MAX_VALUE;

        for(int j = 0; j < numClusters; j++)
        {
            int medoidIndex = medoids.get(j);

            // if the instance is itself a medoid 
            // then break and assign -1 as the medoid
            if(medoidIndex == instanceId)
            {
                closestMedoid = -1;
                break;
            }
            else
            {
                double distance = distanceOperator.CalculateDistance(
                        ds.instances.get(instanceId), 
                        ds.instances.get(medoidIndex));

                if( distance <= closestMedoidDistance)
                {
                    closestMedoidDistance = distance;
                    closestMedoid = j;
                }
            }
        }
        
        return closestMedoid;
    }
    
    /*
     * assign the cluster configuration, i.e assign the closeset medoid 
     * for every instance
     */
    public List<Integer> CreateConfiguration(List<Integer> medoids)
    {
        List<Integer> clusterConfiguration = new ArrayList<Integer>();
        
        // for every instance 
        for( int i = 0; i < numInstances; i++ )
        {
            int closestMedoid = FindClosestMedoid(i, medoids);
            
            clusterConfiguration.add(closestMedoid);
        } 
        
        return clusterConfiguration;
    }
    
    /*
     * Create candidate medoids by randomly swaping a medoid with a non medoid
     */
    public List<Integer> CreateNewMedoids(List<Integer> medoids)
    {
        List<Integer> candidateMedoids = new ArrayList<Integer>();
        for(int i = 0; i < numClusters; i++) candidateMedoids.add(medoids.get(i));
        
        int medoidCandidate = 0;
        
        // pick a medoid candidate
        // iterate until you find one which is not already a medoid        
        do
        {
            medoidCandidate = rand.nextInt(numInstances);
        }
        while(medoids.indexOf(medoidCandidate) >= 0 
                || historyOfMedoidCandidates.indexOf(medoidCandidate) >= 0);
        
        // swap the candidate with a random medoid
        int medoidIndexToSwap= rand.nextInt(numClusters);
        
        //Logging.println("Swapping: " + medoids.get(medoidIndexToSwap) + " with " + medoidCandidate, Logging.LogLevel.DEBUGGING_LOG);
        
        candidateMedoids.set(medoidIndexToSwap, medoidCandidate);
        
        // add the medoid to the history of medoid candidates
        historyOfMedoidCandidates.add(medoidCandidate); 
        
        return candidateMedoids;
    }
    
    /*
     * cluster the dataset using the Partitioning Around Medoids (PAM) algorithm
     */
    public List<Integer> Cluster(DataSet dataSet, int numberOfClusters)
    {
        numClusters = numberOfClusters;
        return Cluster(dataSet);
    }
    
    public List<Integer> Cluster(DataSet dataSet)
    {
        ds = dataSet;
        numInstances = ds.instances.size();
        //dataSet.ReadNominalTargets();
        //numClusters = numInstances/4;//dataSet.nominalLabels.size();
        numIterations = numClusters*numInstances;
        
        Logging.println("Num instances: " + numInstances, Logging.LogLevel.INFORMATIVE_LOG);
        Logging.println("Auto-assigned: Num clusters: " + numClusters + ", num iterations: " + numIterations, Logging.LogLevel.INFORMATIVE_LOG);
     
        
        List<Integer> currentMedoids = InitializeMedoids();
        double currentLoss = ConfigurationLoss(CreateConfiguration(currentMedoids));
        

        Logging.print("Initial Medoids:", Logging.LogLevel.DEBUGGING_LOG);
        Logging.println(currentMedoids, Logging.LogLevel.DEBUGGING_LOG);
        
        // swap one element 
        
        for(int i = 0; i < numIterations; i++)
        {
            // swap one medoid with a nonmedoid
            List<Integer> newMedoids = CreateNewMedoids(currentMedoids);
            
            
            // create the cluster configuration for the new candidate medoids
            List<Integer> newClusterConfig = CreateConfiguration(newMedoids);
            
            // check the loss for the candidate medoids
            double newLoss = ConfigurationLoss(newClusterConfig);
            
            // if there is improvement keep it
            if(newLoss < currentLoss)
            {
                currentLoss = newLoss;
                currentMedoids = new ArrayList<Integer>(newMedoids);
                 
                historyOfMedoidCandidates.clear();
                
                Logging.println("Improvement: Epoch " + i + ", Loss: " + newLoss, Logging.LogLevel.DEBUGGING_LOG);
                
                Logging.print("New Medoids:", Logging.LogLevel.DEBUGGING_LOG);
                Logging.println(currentMedoids, Logging.LogLevel.DEBUGGING_LOG);
            }
            else
            {
                Logging.print(".", Logging.LogLevel.DEBUGGING_LOG);
               // Logging.println("Non Improvement: Epoch " + i + ", Loss: " + newLoss + ", Best: " + currentLoss, Logging.LogLevel.ERROR_LOG);
                
                if( historyOfMedoidCandidates.size() > numInstances - numClusters )
                {
                   Logging.println("Stopping. Already tried "  + historyOfMedoidCandidates.size() + " swappings.", Logging.LogLevel.ERROR_LOG);
                   break;
                }
            }
        }
        
        return currentMedoids;        
    }
    
    public double Classify(DataSet trainSet, DataSet testSet)
    {
        double errorRate = 1;
        
        List<DataInstance> medoids = new ArrayList<DataInstance>();
        
        trainSet.ReadNominalTargets();
        
        // cluster each label filtered subset and get the medoids for each 
        // label-cluster
        for(int l = 0; l < trainSet.nominalLabels.size(); l++)
        {
            DataSet labelFilteredDataset = trainSet.FilterByLabel(trainSet.nominalLabels.get(l));
            
            System.out.println("Clustering labeled-dataset: " + trainSet.nominalLabels.get(l));
            
            List<Integer> labelFilteredMedoids = Cluster(labelFilteredDataset); 
            
            for( int i = 0; i < labelFilteredMedoids.size(); i++ )
            {
                medoids.add( labelFilteredDataset.instances.get(labelFilteredMedoids.get(i)) ); 
            }
        }        
        
        // classify based on the medoids distance
        for(int i = 0; i < testSet.instances.size(); i++)
        {
            // search for the closest medoid 
            int closestMedoid=-1;
            double closestMedoidDistance = Double.MAX_VALUE;

            for(int j = 0; j < medoids.size(); j++)
            {
                double distance = distanceOperator.CalculateDistance(
                        testSet.instances.get(i), 
                        medoids.get(j)); 

                if( distance <= closestMedoidDistance)
                {
                    closestMedoidDistance = distance;
                    closestMedoid = j;
                }
            }
            
            if( testSet.instances.get(i).target != medoids.get(closestMedoid).target)
            {
                errorRate++;
            }        
        }
        
        errorRate /= testSet.instances.size();
        
        return errorRate;
    }
    
}
