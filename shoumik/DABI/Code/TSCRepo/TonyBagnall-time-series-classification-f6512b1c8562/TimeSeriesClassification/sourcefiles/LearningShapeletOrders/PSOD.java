
/*
Copyright (c) 2019, Shoumik Roychoudhury, DABI, Temple University
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither DABI, Temple Univeristy nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
 
package LearningShapeletOrders;

/**
 *
 * @author shoumik PSOD
 */

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
//import TimeSeries.SAXRepresentation;
import utilities.StatisticalUtilities;
import DataStructures.MultivariateDataset;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.Collections;
import java.io.FileNotFoundException;
import multiVariateTS.thresholdFromShapelets_GeneralOrder;

public class PSOD {
    
    public MultivariateDataset dataSet;
    // the length of the shapelet we are searching for
    int [] shapeletLengths;
    int candidateClass;
    //double alpha = -1;
    // the epsilon parameter to prune candidates being epsilon close to a 
    // rejected or accepted shapelet
    public double [] epsilon;
		
    // the percentile for the distribution of distances between pairs of segments
    // is used as epsilon
    public int percentile;
		
    // list of accepted and rejected shapelets, separate per channel
    List<List<double []>> acceptedList;
    List<List<double []>> rejectedList;
    
    
    //List<List<double []>> DistancesMatrix;
    List<String> featureName;
    List <String> featureClass ;
    //List<List<double []>> beforeMatrix;
    //List<List<double []>> afterMatrix;
    //List<List<double []>> featureMatrix;
    //List<List<double []>> finalFeatureMatrix;
    List<List<double []>> startTime;
    //List<List<double []>> midTime;
    //HashMap<String, int[]> orderS;
    //HashMap<String, int[]> orderS_binaryDis;
    
    public String trainSetPath, testSetPath;
    
    // the paa ratio, i.e. 0.25 reduces the length of series by 1/4 
    public double paaRatio ;
    
    // the histogram contains lists of frequency columns
    List< double [] > distancesShapelets = new ArrayList<double []>();
    //List< double [] > finalFeatureMatrix = new ArrayList<double []>();
    List< double [] > normedfinalFeatureMatrix = new ArrayList<double []>();
    //List< double [] > finalOrderMatrix = new ArrayList<double []>();
    //List< double [] > altfinalFeatureMatrix = new ArrayList<double []>();
    List< Double > orderClassInfo = new ArrayList<Double>();
    List< Double > orderConfidenceInfo = new ArrayList<Double>();
    List< double []> orderInstanceInfo = new ArrayList<double[]>();
       
    // the current classification error of the frequencies histogram
    double currentTrainError = 1;
    double [][] seriesDistancesMatrix;
    double [][] instanceProb;
    double [][] instanceProbAlt;
    
    double [] before;
    double [] value;
    double [] after;
    
    // logs on the number of acceptances and rejections
    public int numAcceptedShapelets, numRejectedShapelets, numRefusedShapelets, numAcceptedOrders;
    
    //public int flag = 0;
    public long trainTime, testTime; 
    public boolean normalizeData;
    int flag = -1;
    
    // random number generator
    Random rand = new Random();
        
    PSOD (){
        rand.setSeed(1);
        normalizeData = false;
    }
    public void LoadData(){
	dataSet = new MultivariateDataset(trainSetPath, testSetPath, normalizeData);
	// apply the PAA
	//SAXRepresentation sr = new SAXRepresentation();
	//dataSet = sr.generatePAA(dataSet, paaRatio);
    }
    public double Search(String place, int trial, double alpha) throws Exception{
                acceptedList = new ArrayList<List<double[]>>(); 
                rejectedList = new ArrayList<List<double[]>>(); 
                startTime = new ArrayList<List<double[]>>(); 
                featureName = new ArrayList<String>();
                featureClass = new ArrayList<String>();
                //midTime = new ArrayList<List<double[]>>(); 
                // initialize the lists per each channel
                for(int channelIdx = 0; channelIdx < dataSet.numChannels; channelIdx++){
                        acceptedList.add( new ArrayList<double[]>() );
                        rejectedList.add( new ArrayList<double[]>() );
                        startTime.add( new ArrayList<double[]>() );
                        //midTime.add( new ArrayList<double[]>() );
                }
                
                numAcceptedShapelets = numRejectedShapelets = numRefusedShapelets = numAcceptedOrders = 0;
		    	
            // check 10%,20%, 30% or different shapelet lengths    	
                shapeletLengths = new int[1]; 
                shapeletLengths[0] = (int)(0.10*dataSet.avgLength);
                //shapeletLengths[1] = (int)(0.20*dataSet.avgLength); 
                //shapeletLengths[2] = (int)(0.30*dataSet.avgLength);
                epsilon = new double[dataSet.numChannels];
                for(int channelIdx = 0; channelIdx < dataSet.numChannels; channelIdx++){
                    
                        epsilon[channelIdx] = EstimateEpsilon(channelIdx);
                        //System.out.println("channel=" + channelIdx + ", epsilon=" + epsilon[channelIdx] );
                }
    	
            // set distances matrix to 0.0
                seriesDistancesMatrix = new double[dataSet.numTrain][dataSet.numTrain];
                for(int i = 0;  i < dataSet.numTrain; i++)
                	for(int j = i+1;  j < dataSet.numTrain; j++)
                		seriesDistancesMatrix[i][j] = 0.0;
                instanceProb = new double[dataSet.numTrain+dataSet.numTest][dataSet.numLabels];
                instanceProbAlt = new double[dataSet.numTrain+dataSet.numTest][dataSet.numLabels];
                for(int i = 0;  i < dataSet.numTrain+dataSet.numTest; i++)
			for(int j = i+1;  j < dataSet.numLabels; j++){
				instanceProb[i][j] = 0.0;
                                instanceProbAlt[i][j] = 0.0;
                        }
                
                
            // evaluate all the words of all series
                int numTotalCandidates = dataSet.numTrain*dataSet.minLength*shapeletLengths.length;
                double initialError = Double.MAX_VALUE;
                //System.out.println("numTotalCandidates=" + numTotalCandidates );  

                for( int candidateIdx = 0; candidateIdx < numTotalCandidates; candidateIdx++){
    		// select a random series
                    //System.out.println("Candidate=" + candidateIdx );
                        int i = rand.nextInt(dataSet.numTrain);
    		// select a random channel
                        int channel = dataSet.numChannels <= 1 ? 0: rand.nextInt(dataSet.numChannels);     
    		// select a random shapelet length 
                        int shapeletLength = shapeletLengths[ rand.nextInt(shapeletLengths.length) ];
        	// select a random segment of the i-th series where the shapelet can be located 
                        int maxTimeIndex = dataSet.timeseries[i][channel].length - shapeletLength + 1;
        	// avoid cases where the shapelet length is longer than the series
        	// because we cannot extract a candidate from that series
                        if( maxTimeIndex <= 0)
                                continue;
    		
        	// pick a start of the time indices
                        int j = rand.nextInt(maxTimeIndex);
                        
                        /*int i = 12;
                        int channel = 0;
                        int shapeletLength = 93;
                        int maxTimeIndex = 377;
                        int j = 349;*/
                        
                
        	// set the candidate shapelets
                        double [] candidateShapelet = new double[shapeletLength];
                        for(int k = 0; k < shapeletLength; k++)
                                candidateShapelet[k] = dataSet.timeseries[i][channel][j + k];
			candidateClass = dataSet.labels[i];
    		// evaluate the shapelet
                        
                        EvaluateShapelet(candidateShapelet, channel,candidateIdx, place, trial, alpha);
                        //System.err.println(orderClassInfo.size());
                        //initialError == currentTrainError;
                        if(candidateIdx == 0)
                            initialError = currentTrainError;
                        else if(candidateIdx % 10000 == 0)
                                if(initialError == currentTrainError ){
                                    break;
                                }
                                else
                                    initialError = currentTrainError;
    
                      		
                    //if( candidateIdx % 2000 == 0) 
                           //System.out.println("Iteration:" + " " + candidateIdx + "," + " Current train error:" + " " + currentTrainError + "," + " Accepted shapelets:" + " "  + numAcceptedShapelets + "," + " Accepted orders:" + " "  + numAcceptedOrders +"," + " Rejected Shapelets:" + " "  + numRejectedShapelets + "," + " Refused Shapalets:" + " " + numRefusedShapelets); 
                    
                    if(candidateIdx > 1000  && currentTrainError == 0.0)
                        break;
                    
                     //if (candidateIdx == 1 )
                        //break;
                   
                }
                //if(numAcceptedShapelets >1){
                    //altfinalFeatureMatrix.addAll(distancesShapelets);
                    //altfinalFeatureMatrix.addAll(finalOrderMatrix);
                //}
                
                //System.err.println(orderClassInfo.size());
                //normedfinalFeatureMatrix = zNormalization(altfinalFeatureMatrix, dataSet.numTrain, dataSet.numTrain +dataSet.numTest, altfinalFeatureMatrix.size());
             return currentTrainError;
    
    }
    public void EvaluateShapelet(double [] candidate, int channel,int candidateIdx, String place, int trial, double alpha) throws Exception{
    	
                //ArrayList<Double> values = new ArrayList<Double>();
                double [][] distancesCandidate = ComputeDistances(candidate, channel);
                //double [][] normalizedDistancecandidate = thresholdFromShapelets_GeneralOrder.normalization(distancesCandidate, dataSet.numTrain, dataSet.numTrain+dataSet.numTest,numAcceptedShapelets);
                double [] dis = new double[dataSet.numTrain + dataSet.numTest];
                double [] trainDis = new double[dataSet.numTrain];
               // double [] sortedDis = new double[dataSet.numTrain];
                double [] tstart = new double[dataSet.numTrain + dataSet.numTest];
                double [] tmid = new double[dataSet.numTrain + dataSet.numTest];
                int [] trainLabel = new int[dataSet.numTrain];
                //double [] thresholdShapelet = new double[2];
                
                for(int i=0; i<dataSet.numTrain+dataSet.numTest; i++ ){
                        dis[i] = distancesCandidate[i][0];
                        tstart[i] = distancesCandidate[i][1];
                        tmid[i] = distancesCandidate[i][2];
                }
                //double [] normalizedDis = normalization(dis, dataSet.numTrain, dataSet.numTrain+dataSet.numTest);
                
                //for(int i=0; i<dataSet.numTrain+dataSet.numTest; i++ ){
                    //dis[i] = normalizedDis[i];
                //}
                
                for(int i = 0;  i < dataSet.numTrain; i++){
                        trainLabel[i] = dataSet.labels[i];
                        trainDis[i] = dis[i]; 
                        
                }
                //for(int n=0; n< dataSet.numTrain; n++){
                        //values.add(dis[n]);
                //}
                //Collections.sort(values);
//                thresholdShapelet = thresholdFromShapelets.threshold(values, trainDis, trainLabel);  // Two class
                //thresholdShapelet = thresholdFromShapelets.threshold_multiclass(values, trainDis, trainLabel, dataSet.classLabels); // ??
                
                // if the lists are both empty or the candidate is previously not been considered 
                // then give it a chance
                if(numAcceptedShapelets == 0){
                             
                    //sanity check for candidate
                    if( !FoundInList(candidate, acceptedList.get(channel), channel) &&  
                            !FoundInList(candidate, rejectedList.get(channel), channel)){
                            AddCandidateDistancesToDistancesMatrix(dis);
                            //temp_finalFeatureMatrix.add(dis);
                            // compute error
                            double newTrainError = ComputeTrainError(alpha); 
    		
                            if( newTrainError < currentTrainError){
                                // accept the word, which improves the error
                                acceptedList.get(channel).add(candidate);
                                //DistancesMatrix.get(channel).add(dis);
                               // featureMatrix.get(channel).add(dis);
                                //finalFeatureMatrix.get(channel).add(dis);
                                startTime.get(channel).add(tstart);
                                //midTime.get(channel).add(tmid);
    			        // add the distances of the shapelet to a list
                                // will be used for testing
                                distancesShapelets.add(dis);
                                //finalFeatureMatrix.add(dis);
                                featureName.add("S_0" );
                                featureClass.add(Integer.toString(candidateClass));
    			        // set the new error as the current one
                                currentTrainError = newTrainError;
                                // increase the counter of accepted words
                                numAcceptedShapelets++;
                            }   
                            else{
                                // the word doesn't improve the error, therefore is rejected
                                rejectedList.get(channel).add(candidate); 
    			        // finally remove the distances from the distance matrix
                                RemoveCandidateDistancesToDistancesMatrix(dis);
                                //temp_finalFeatureMatrix.remove(dis);
    			        // increase the counter of rejected words
                                numRejectedShapelets++;
                            }
                        }
                        else{ // word already was accepted and/or rejected before 
                        numRefusedShapelets++;
                        }
                }
                else if(numAcceptedShapelets>0){
                    double newTemporaryTrainError = Double.MAX_VALUE;
                
                    if( !FoundInList(candidate, acceptedList.get(channel), channel) &&  
                            !FoundInList(candidate, rejectedList.get(channel), channel) ){
                    
                        AddCandidateDistancesToDistancesMatrix(dis);
                        //temp_finalFeatureMatrix.add(dis);
                        // compute error
                        double newTrainErrorWithoutNewOrder = ComputeTrainError(alpha);
                        newTemporaryTrainError = newTrainErrorWithoutNewOrder; 
                        flag = 0;
                        startTime.get(channel).add(tstart);
                        //midTime.get(channel).add(tmid);
                        // DistancesMatrix.get(channel).add(dis);
                        // featureMatrix.get(channel).add(dis);
                        double[][] TimeMatrix = new double[dataSet.numTrain+dataSet.numTest][numAcceptedShapelets+1];
                        int colIndex =0;
                        for(int dim=0; dim < dataSet.numChannels; dim++){
                                int k = startTime.get(dim).size();  //number of shapelets
                                for(int j=0; j< k; j++){
                                    //ps.println(colIndex+"\tS_"+dim+"_"+j);
                                    for(int i=0; i<dataSet.numTrain+dataSet.numTest; i++ ){
                                    	TimeMatrix[i][colIndex] = startTime.get(dim).get(j)[i];
                                        //candidateMatrix[i][colIndex] = startTime.get(dim).get(j)[i];
                                        //newFeatureMatrix[i][colIndex] = candidateMatrixSoFar.get(dim).get(j)[i];
                                    }
                                    colIndex = colIndex + 1;
                                    //System.out.println("ColIndex" + " " + colIndex);
                                }
                        }
                    
                        boolean binary = false;
                        double[] temporal_order =  new double[dataSet.numTrain+dataSet.numTest];
                        double[] temporal_time_diff =  new double[dataSet.numTrain+dataSet.numTest];
                        String temporal_order_name = null;
                        double temporal_order_class = 0;
                        double temporal_order_confidence=0;
                        double temp1= 0, temp2 = 0, temp_child = -1;
                        int s_index = numAcceptedShapelets;
//                        double [] Labels = new double[dataSet.numLabels];
//                        for (int k = 0;k<dataSet.numLabels;k++){
//                            Labels[k] = dataSet.classLabels[k];
//                        }   
                        for(int i=0; i < numAcceptedShapelets; i++){
                                double [] tempOrderClass  = new double [1];
                                double [] tempOrderConfidence = new double[1]; 
                                double numOrderOccurs = 0;
                                //double numTrainClassExamples =0;
                                double numOrderClass = 0;
                                double[] timeDiff = new double[dataSet.numTrain+dataSet.numTest];
                                double[] trainTimeDiff = new double[dataSet.numTrain];
                                ArrayList<Double> trainTdiff = new ArrayList<Double>();
                                String order_name=null;
                                double precisionOfOrder = Double.MAX_VALUE;
				
                                double [] value_train = new double[dataSet.numTrain];
//                                before = new double[dataSet.numTrain+dataSet.numTest];
//				                after = new double[dataSet.numTrain+dataSet.numTest];
                                value = new double[dataSet.numTrain+dataSet.numTest];
                                
                                for(int n = 0; n <dataSet.numTrain+dataSet.numTest; n++){
					                timeDiff[n] = TimeMatrix[n][i] - TimeMatrix[n][numAcceptedShapelets];
                                }
                                for(int n = 0; n <dataSet.numTrain; n++){
                                    trainTdiff.add(timeDiff[n]);
                                    trainTimeDiff[n] = timeDiff[n];
                                }
                                Collections.sort(trainTdiff);
                                double time_value = thresholdFromShapelets_GeneralOrder.threshold_order(trainTdiff, trainTimeDiff, trainLabel, dataSet.classLabels)[1];
                                double leftChild = thresholdFromShapelets_GeneralOrder.threshold_order(trainTdiff, trainTimeDiff, trainLabel, dataSet.classLabels)[2];
                                double orderClass = thresholdFromShapelets_GeneralOrder.threshold_order(trainTdiff, trainTimeDiff, trainLabel, dataSet.classLabels)[3];
                                //System.err.println("Leftchild : " +   " "  + leftChild);
                                //int[] value = new int[dataSet.numTrain+dataSet.numTest];
                                //if(left_child ==1){
                                if(leftChild == 1){
                                    //int[] value = new int[dataSet.numTrain+dataSet.numTest];
                                    for(int n = 0; n <dataSet.numTrain+dataSet.numTest; n++){
                                    	if(TimeMatrix[n][i] - TimeMatrix[n][numAcceptedShapelets] <= time_value){
                                        	value[n] = 1;
                                        }
                                        else {
                                        	value[n] = 0;
                                            }
                                    }
                                        double numTrainClassExamples=0;
                                        for(int n = 0; n <dataSet.numTrain; n++){
                                            if(trainLabel[n] == orderClass)
                                                numTrainClassExamples+=1;
                                            if(value[n] == 1){
                                                numOrderOccurs += 1;
                                                if(trainLabel[n] == orderClass)
                                                    numOrderClass+=1;
                                                
                                            }
                                                
                                            
                                        }
                                        order_name = "S"+i+"-"+"S"+s_index+"<="+time_value;
                                        precisionOfOrder = (double)(numOrderClass/numOrderOccurs)*(numOrderClass/numTrainClassExamples);
                                        //orders.put(name, value);
                                }
                                else if(leftChild == 0){
                                    //int[] value = new int[dataSet.numTrain+dataSet.numTest];
                                    for(int n = 0; n <dataSet.numTrain+dataSet.numTest; n++){
                                    	if(TimeMatrix[n][i] - TimeMatrix[n][numAcceptedShapelets] > time_value){
                                        	value[n] = 1;
                                        }else{
                                        	value[n] = 0;
                                        }
                                    }
                                    double numTrainClassExamples=0;
                                        for(int n = 0; n <dataSet.numTrain; n++){
                                            if(trainLabel[n] == orderClass)
                                                numTrainClassExamples+=1; 
                                            if(value[n] == 1){
                                                numOrderOccurs += 1;
                                                if(trainLabel[n] == orderClass)
                                                    numOrderClass+=1;
                                            }
                                    }
                                    order_name = "S"+i+"-"+"S"+s_index+">"+time_value;
                                    precisionOfOrder = (double)(numOrderClass/numOrderOccurs)*(numOrderClass/numTrainClassExamples);
                                    //orders.put(name, value);
                                }    
                                
                                 tempOrderClass[0] = orderClass;
                                 tempOrderConfidence[0] = precisionOfOrder;
                                 for(int n = 0; n <dataSet.numTrain; n++){
                                    value_train[n] = value[n];
                                }
                                orderClassInfo.add(orderClass);                        
                                orderConfidenceInfo.add(precisionOfOrder);
                                orderInstanceInfo.add(value);
                                                     
                          
                                //order.add()
                                //AddCandidateDistancesToDistancesMatrix(value_train);
                                //temp_finalFeatureMatrix.add(before);
                                double newTrainErrorWithNewOrderBefore = ComputeTrainError(alpha);
                                if(newTrainErrorWithNewOrderBefore<newTemporaryTrainError){
                                        newTemporaryTrainError = newTrainErrorWithNewOrderBefore;
                                        for(int m=0; m< dataSet.numTrain+dataSet.numTest; m++){
                                            temporal_order[m] = value[m];
                                            temporal_time_diff[m]= timeDiff[m];
                                           
                                        }
                                        temporal_order_name = order_name;
                                        temporal_order_class = orderClass;
                                        temporal_order_confidence = precisionOfOrder;
                                        temp1 = numOrderClass;
                                        temp2 = numOrderOccurs;
                                        temp_child = leftChild;
                                        flag = 1;
                                }
                                orderClassInfo.remove(orderClassInfo.size()-1);
                                orderConfidenceInfo.remove(orderConfidenceInfo.size()-1);
                                orderInstanceInfo.remove(orderInstanceInfo.size()-1);
                                //RemoveCandidateDistancesToDistancesMatrix(value_train);
                                
                        
                        }
                    
                        if(newTemporaryTrainError<currentTrainError){
                            if(flag == 1){
                                //for(int n = 0; n <dataSet.numTrain; n++)
                                    //before[n] = beforeMatrix.get(counterBefore-1).get(counterBefore-1)[n];
                                
                                //AddCandidateDistancesToDistancesMatrix(temporal_order);
                                //temp_finalFeatureMatrix.add(temporal_order);
                                acceptedList.get(channel).add(candidate);
                                binary = true;
                                distancesShapelets.add(dis);
                                //finalFeatureMatrix.add(dis);
                                //finalFeatureMatrix.add(temporal_order);
                                //finalOrderMatrix.add(temporal_order);
                                //DistancesMatrix.get(channel).add(before);
                               // featureMatrix.get(channel).add(temporal_order);
                                featureName.add("S_"+Integer.toString(s_index));
                                featureClass.add(Integer.toString(candidateClass));
                                featureName.add(temporal_order_name);
                                featureClass.add(Double.toString(temporal_order_class));
                                //ps.print(temporal_order_name);
                                //ps.println();
                                numAcceptedOrders++;
                                orderClassInfo.add(temporal_order_class);
                                //System.err.print("Accepted general order Class" + " " + temporal_order_class + ": ");
                                //System.err.println(" " + temp1 + " " +  temp2 + " "+ temp_child);
                                orderConfidenceInfo.add(temporal_order_confidence);
                                orderInstanceInfo.add(temporal_order);
                                //System.err.println(numAcceptedShapelets);
                                //System.err.println(temporal_order_name);
                                //System.err.println(numAcceptedOrders);
                                numAcceptedShapelets++;
//                                for(int m = 0;m<numAcceptedShapelets;m++){
//                                    //System.err.println(TimeMatrix[0][m]);
//                                }
//                                for(int m = 0;m<dataSet.numTrain+dataSet.numTest;m++){
//                                    //System.err.println(temporal_time_diff[m]);
//                                }
                                
                            }
                            if(flag == 0){
                                acceptedList.get(channel).add(candidate);
                                binary = true;
                                distancesShapelets.add(dis);
                                //finalFeatureMatrix.add(dis);
                                featureName.add("S_"+Integer.toString(s_index));
                                featureClass.add(Integer.toString(candidateClass));
                                //String shapelet = featureName.
                                //ps.print(temporal_order_name);
                                //ps.println();
                                numAcceptedShapelets++;
                                
                            }
                            currentTrainError = newTemporaryTrainError;
                        }
                    
                    if(binary == false){
                        rejectedList.get(channel).add(candidate); 
                        //DistancesMatrix.get(channel).remove(dis);
                        startTime.get(channel).remove(tstart);
                        //featureMatrix.get(channel).remove(dis);
    			// finally remove the distances from the distance matrix
                        RemoveCandidateDistancesToDistancesMatrix(dis);
                       
                        //temp_finalFeatureMatrix.remove(dis);
    			// increase the counter of rejected words
                        numRejectedShapelets++;
                    }
                   
                }
                else { // word already was accepted and/or rejected before 
                        numRefusedShapelets++;
                }
                
                
            }
             
                
    }
    public double ComputeTrainError(double alpha){
                
                double numMissClassifications = 0;
                double predictedLabel = 0;
                //int numShapelets = distancesShapelets.size();
                //System.err.println("no of shapelets:" + numShapelets);
                //String outfile = place + "trial_"+trial+"_predictionWithoutOrder.txt";
                //FileOutputStream fos2 = new FileOutputStream(outfile);
                //PrintStream ps2 = new PrintStream(fos2);
                double [] Distances = new double[dataSet.numTrain];
                //double [] actualDistances = new double[dataSet.numTrain];
                double [] Labels = new double[dataSet.numLabels];
                double [][] probij = new double[dataSet.numTrain][dataSet.numTrain]; 
                double [][] euclidDistance = new double [dataSet.numTrain][dataSet.numTrain]; 
                double [][] Prob = new double[dataSet.numTrain][dataSet.numLabels];
                // for every test instance 
                for (int k = 0;k<dataSet.numLabels;k++){
                    Labels[k] = dataSet.classLabels[k];
                }
                
                
                for(int i = 0; i < dataSet.numTrain; i++){
                        double realLabel = dataSet.labels[i];
                        for(int k = 0;k<dataSet.numTrain;k++){
                                Distances[k] = Double.MAX_VALUE; 
                        }                      
                       
                        // iterate through training instances and find the closest neighbours
                        for(int j = 0; j < dataSet.numTrain; j++){
                               if( i == j) continue;
                               // compute the distance between the train and test instances
                               // in the shapelet-transformed space
                                                           
                                //double distance = 0;
                                //for(int k = 0; k < numShapelets; k++){
                                Distances[j] = ( i < j ? seriesDistancesMatrix[i][j] : seriesDistancesMatrix[j][i] );
                                        //distance += error*error; 
                                        //distance+=Math.abs(error);
                                        // stop measuring the distance if it already exceeds the nearest distance so far
                                        //if(distance > nearestDistance)
                                                //break;
                                
                                euclidDistance[i][j] = Math.sqrt(Distances[j]);
                                //System.err.println(realLabel + " " + euclidDistance[i][j]);
                              // euclidDistance[i][j] =Distances[j];
                        }
                               
                        double softMaxNormalization = 0;
                        for(int j = 0; j < dataSet.numTrain; j++){
                                if( i == j) continue;
                                softMaxNormalization += Math.exp(alpha*euclidDistance[i][j]);
                        }
                        
                        for(int j = 0; j < dataSet.numTrain; j++){
                                if( i == j) probij[i][j] = 0.0;
                                probij[i][j] = Math.exp(alpha*euclidDistance[i][j])/softMaxNormalization;
                        }
                        //predictedLabel=Labels[0];
                        for(int j = 0; j < dataSet.numLabels; j++){
                            double sum = 0;
                            for(int k = 0;k<dataSet.numTrain;k++){
                                if( i == k) continue;
                                    if (Labels[j] == dataSet.labels[k]){
                                            sum += probij[i][k];
                                    } 
                            }
                            Prob[i][j] = sum;
                        
                            instanceProb[i][j] = Prob[i][j];
                            instanceProbAlt[i][j] = Prob[i][j];
                            
                        }
                        if (orderClassInfo.size()>0){
                            double [] orderConfidencProduct = new double[dataSet.numLabels];
                            for(int k = 0;k<dataSet.numLabels;k++){
                            	boolean hasorder = false;
                                ArrayList<Double> orderprobs = new ArrayList<Double>();
                               for(int j =0;j<orderInstanceInfo.size();j++){  // no. of orders
                                  if(orderInstanceInfo.get(j)[i] == 1){  // has this order
                                        double LabelOfOrder = orderClassInfo.get(j);
                                        double confidenceOfOrder = orderConfidenceInfo.get(j);
                                        if(LabelOfOrder == dataSet.classLabels[k]){
                                        	hasorder = true;
                                            orderprobs.add(confidenceOfOrder);
                                        }
                                  }   
                                }
                               orderConfidencProduct[k] = computehasOrderProb(orderprobs);
                               
                               if(hasorder){
                            	   
                            	   instanceProb[i][k] = instanceProb[i][k]*orderConfidencProduct[k];
                               }else{
                            	   instanceProb[i][k] = instanceProb[i][k]*((double) 1.0 / dataSet.numLabels);
                               }
                              
                            }
                            
                        }
                        double maxProbability = instanceProb[i][0];
                        predictedLabel=Labels[0];
                        for(int j = 0; j < dataSet.numLabels; j++){
                            if(j > 0)
                                if(instanceProb[i][j]>maxProbability){
                                    maxProbability = instanceProb[i][j];
                                    predictedLabel = Labels[j];
                                }
                        
                        }
                        //System.err.println("without orders:  Example : " +  " " + i + " realLabel: "+ realLabel + " prediction: "+ predictedLabel);
                        //for(int k = 0; k < dataSet.numLabels; k++){
                            //System.err.println(" Probability of class: " +  " " + dataSet.classLabels[k] + " " + instanceProb[i][k]);
                        //}
                       //System.out.println();
                        if( realLabel != predictedLabel ){
                                numMissClassifications += 1.0;
                                //System.out.println("Example : " +  " " + i);
                        }
                        
                }
                return (double) numMissClassifications / (double) dataSet.numTest;       
        }
    public static double computehasOrderProb(ArrayList<Double> orderprobs) {
		double prob = 0;
		for(int k = 1; k<= orderprobs.size(); k++){
			prob = prob + Math.pow(-1, k-1)* value(orderprobs, k);
		}
//		System.out.println(prob);
		return prob;
	}
    public static double value(ArrayList<Double> orderprobs, int k) {
		double sum = 0;
		int[] nums = new int[orderprobs.size()];
		for(int i=0; i< orderprobs.size(); i++){
			nums[i] = i;
		}
		List<int[]> combines = new ArrayList<int[]>();
		Permutation.printCombination(nums, nums.length, k, combines);
		for(int i=0; i< combines.size(); i++){
			double value = 1;
			for(int j =0; j< k;j++){
//				System.out.print(combines.get(i)[j]+" ");
				value = value * orderprobs.get(combines.get(i)[j]);
			}
			sum = sum + value;
			
		}
//		System.out.println(sum);
		
		return sum;
	}
    public double ComputeTestErrorUsingShapeletTransform(String place, String name,int trial, double alpha) throws FileNotFoundException{
                
                double numMissClassifications = 0;
                String outfile30 = place + "trial_"+ trial+"_predictionsShapeletOnly.txt";
                FileOutputStream fos30 = new FileOutputStream(outfile30);
                PrintStream ps30 = new PrintStream(fos30);
                double predictedLabel = 0;
		int numShapelets = distancesShapelets.size();
                //System.err.println("no of shapelets:" + numShapelets);
                //String outfile = place + "trial_"+trial+"_predictionWithoutOrder.txt";
                //FileOutputStream fos2 = new FileOutputStream(outfile);
                //PrintStream ps2 = new PrintStream(fos2);
		double [] Distances = new double[dataSet.numTrain];
                double [] actualDistances = new double[dataSet.numTrain];
                double [] Labels = new double[dataSet.numLabels];
                double [][] probij = new double[dataSet.numTest][dataSet.numTrain]; 
                double [][] euclidDistance = new double [dataSet.numTest][dataSet.numTrain]; 
                double [][] Prob = new double[dataSet.numTest][dataSet.numLabels];
                // for every test instance 
                for (int k = 0;k<dataSet.numLabels;k++){
                    Labels[k] = dataSet.classLabels[k];
                }
                
                
                for(int i = dataSet.numTrain; i < dataSet.numTrain+dataSet.numTest; i++){
                        double realLabel = dataSet.labels[i];
                        //double realLabel = dataSet.labels[i];
                        //for(int k = 0;k<dataSet.numLabels;k++){
                            //Labels[k] = -1; 
                            //Distances[k] = Double.MAX_VALUE; 
                        //}
                        //double nearestExample = -1;
                        //double nearestLabel = 0;
                        //double nearestDistance = Double.MAX_VALUE;
                         
                       
                // iterate through training instances and find the closest neighbours
                        for(int j = 0; j < dataSet.numTrain; j++){
                                //if( i == j+dataSet.numTrain-1) continue;
                        // compute the distance between the train and test instances
                        // in the shapelet-transformed space
                            
                                
                                double distance = 0;
                                for(int k = 0; k < numShapelets; k++){
                                        double error = distancesShapelets.get(k)[i] - distancesShapelets.get(k)[j];
                                        distance += error*error; 
                                        //distance+=Math.abs(error);
                                        // stop measuring the distance if it already exceeds the nearest distance so far
                                        //if(distance > nearestDistance)
                                                //break;
                                }
                                euclidDistance[i-dataSet.numTrain][j] = Math.sqrt(distance);
                                
                                
                                
                        }
                        double softMaxNormalization = 0;
                        
                        for(int j = 0; j < dataSet.numTrain; j++){
                                //if( i == j+dataSet.numTrain) continue;
                                softMaxNormalization += Math.exp(alpha*euclidDistance[i-dataSet.numTrain][j]);
                        }
                        
                        for(int j = 0; j < dataSet.numTrain; j++){
                                //if( i == j+dataSet.numTrain) probij[i-dataSet.numTrain][j] = 0.0;
                                probij[i-dataSet.numTrain][j] = Math.exp(alpha*euclidDistance[i-dataSet.numTrain][j])/softMaxNormalization;
                        }
                        
                            
                        for(int j = 0; j < dataSet.numLabels; j++){
                            double sum = 0;
                            for(int k = 0;k<dataSet.numTrain;k++){
                                    if (Labels[j] == dataSet.labels[k]){
                                            sum += probij[i-dataSet.numTrain][k];
                                    } 
                            }
                            Prob[i-dataSet.numTrain][j] = sum;
                            instanceProbAlt[i][j] = Prob[i-dataSet.numTrain][j];
                            
                        }
                        
                        predictedLabel=Labels[0];
                        double maxProbability = instanceProbAlt[i][0];
                        for(int j = 0; j < dataSet.numLabels; j++){
                            if(j > 0)
                                if(instanceProbAlt[i][j]>maxProbability){
                                    maxProbability = instanceProbAlt[i][j];
                                    predictedLabel = Labels[j];
                                }
                        
                        }
                        
                       ps30.print("without orders:  Example : " +  " " + i + " realLabel: "+ realLabel + " prediction: "+ predictedLabel);
                        for(int k = 0; k < dataSet.numLabels; k++){
                            ps30.print(" Probability of class: " +  " " + dataSet.classLabels[k] + " " + instanceProbAlt[i][k]);
                        }
                        ps30.println();
                        if( realLabel != predictedLabel ){
                                numMissClassifications += 1.0;
                                //System.out.println("Example : " +  " " + i);
                        }
                        
                }
                
                
         ps30.print("Only shapelets, error is "+(double) numMissClassifications / (double) dataSet.numTest);
         ps30.close();
         return (double) numMissClassifications / (double) dataSet.numTest;       
        }
    public double ComputeTestErrorUsingOrder(String place, String name,int trial, double alpha) throws FileNotFoundException{
                double numMissClassifications = 0;
                double predictedLabel = 0;
                int numShapelets = distancesShapelets.size();
                String outfile31 = place + "trial_"+ trial+"_predictionsShapeletAndOrders.txt";
                FileOutputStream fos31 = new FileOutputStream(outfile31);
                PrintStream ps31 = new PrintStream(fos31);
                double [] Labels = new double[dataSet.numLabels];
                double [][] probij = new double[dataSet.numTest][dataSet.numTrain]; 
                double [][] euclidDistance = new double [dataSet.numTest][dataSet.numTrain]; 
                double [][] Prob = new double[dataSet.numTest][dataSet.numLabels];
                // for every test instance 
                for (int k = 0;k<dataSet.numLabels;k++){
                    Labels[k] = dataSet.classLabels[k];
                }
                
                
                for(int i = dataSet.numTrain; i < dataSet.numTrain+dataSet.numTest; i++){
                        double realLabel = dataSet.labels[i];
                        //double realLabel = dataSet.labels[i];
                        //for(int k = 0;k<dataSet.numLabels;k++){
                            //Labels[k] = -1; 
                            //Distances[k] = Double.MAX_VALUE; 
                        //}
                        //double nearestExample = -1;
                        //double nearestLabel = 0;
                        //double nearestDistance = Double.MAX_VALUE;
                         
                       
                // iterate through training instances and find the closest neighbours
                        for(int j = 0; j < dataSet.numTrain; j++){
                                //if( i == j+dataSet.numTrain-1) continue;
                        // compute the distance between the train and test instances
                        // in the shapelet-transformed space
                            
                                
                                double distance = 0;
                                for(int k = 0; k < numShapelets; k++){
                                        double error = distancesShapelets.get(k)[i] - distancesShapelets.get(k)[j];
                                        distance += error*error; 
                                        //distance+=Math.abs(error);
                                        // stop measuring the distance if it already exceeds the nearest distance so far
                                        //if(distance > nearestDistance)
                                                //break;
                                }
                                
                                euclidDistance[i-dataSet.numTrain][j] = Math.sqrt(distance);
                               // euclidDistance[i-dataSet.numTrain][j] =distance;
                                
                                
                        }
                        double softMaxNormalization = 0;
                        
                        for(int j = 0; j < dataSet.numTrain; j++){
                                //if( i == j+dataSet.numTrain) continue;
                                softMaxNormalization += Math.exp(alpha*euclidDistance[i-dataSet.numTrain][j]);
                        }
                        
                        for(int j = 0; j < dataSet.numTrain; j++){
                                //if( i == j+dataSet.numTrain) probij[i-dataSet.numTrain][j] = 0.0;
                                probij[i-dataSet.numTrain][j] = Math.exp(alpha*euclidDistance[i-dataSet.numTrain][j])/softMaxNormalization;
                        }
                        
                        for(int j = 0; j < dataSet.numLabels; j++){
                            double sum = 0;
                            for(int k = 0;k<dataSet.numTrain;k++){
                                    if (Labels[j] == dataSet.labels[k]){
                                            sum += probij[i-dataSet.numTrain][k];
                                    } 
                            }
                            Prob[i-dataSet.numTrain][j] = sum;
                            instanceProb[i][j] = Prob[i-dataSet.numTrain][j];
                            
                                                       
                        }
                        
                        if (orderClassInfo.size()>0){
                            
                            double [] orderConfidencProduct = new double[dataSet.numLabels];
                            for(int k = 0;k<dataSet.numLabels;k++){
                            	boolean hasorder = false;
                                ArrayList<Double> orderprobs = new ArrayList<Double>();
                               for(int j =0;j<orderInstanceInfo.size();j++){  // no. of orders
                                  if(orderInstanceInfo.get(j)[i] == 1){  // has this order
                                        double LabelOfOrder = orderClassInfo.get(j);
                                        double confidenceOfOrder = orderConfidenceInfo.get(j);
                                        if(LabelOfOrder == dataSet.classLabels[k]){
                                        	hasorder = true;
                                            orderprobs.add(confidenceOfOrder);
                                        }
                                  }   
                                }
                               orderConfidencProduct[k] = computehasOrderProb(orderprobs);
                               //ps31.print("Probability of order with class" + " " + LabelOfOrder + " " + orderConfidencProduct[k]);
                               if(hasorder){
                            	   
                            	   instanceProb[i][k] = instanceProb[i][k]*orderConfidencProduct[k];
                               }else{
                            	   instanceProb[i][k] = instanceProb[i][k]*((double) 1.0 / dataSet.numLabels);
                               }
                              
                            }
                           
                        }
                        double maxProbability = instanceProb[i][0];
                        predictedLabel=Labels[0];
                        for(int k = 0; k < dataSet.numLabels; k++){
                            if(k > 0)
                                if(instanceProb[i][k]>maxProbability){
                                    maxProbability = instanceProb[i][k];
                                    predictedLabel = Labels[k];
                                }
                        
                        }
                        ps31.print("with orders:  Example: " +  " " + i + " realLabel: "+ realLabel + " prediction: "+ predictedLabel);
                        for(int k = 0; k < dataSet.numLabels; k++){
                            ps31.print(" " +  "Probability of class: " +  " " + dataSet.classLabels[k] + " " + instanceProb[i][k] + " ");
                        }
                        ps31.println();
                        if( realLabel != predictedLabel ){
                                numMissClassifications += 1.0;
                                //System.out.println("Example : " +  " " + i);
                        }
                        
                }
                
                
        ps31.print("Shapelets + orders, error is "+(double) numMissClassifications / (double) dataSet.numTest);
        ps31.close();
        return (double) numMissClassifications / (double) dataSet.numTest;       
    }
    public void AddCandidateDistancesToDistancesMatrix(double [] candidateDistances){
		double diff = 0;
		for(int i = 0;  i < dataSet.numTrain; i++)
			for(int j = i+1;  j < dataSet.numTrain; j++){
				diff = candidateDistances[i]-candidateDistances[j];				
				seriesDistancesMatrix[i][j] += diff*diff;
                                //seriesDistancesMatrix[i][j] += Math.abs(diff);     /// if this line has been changed, then check line 716
			}
	}
    public void RemoveCandidateDistancesToDistancesMatrix(double [] candidateDistances){
		double diff = 0;
		
		for(int i = 0;  i < dataSet.numTrain; i++)
			for(int j = i+1;  j < dataSet.numTrain; j++){
				diff = candidateDistances[i]-candidateDistances[j];				
                                seriesDistancesMatrix[i][j] -= diff*diff;
				//seriesDistancesMatrix[i][j] -= Math.abs(diff);  /// if this line has been changed, then check line 706
			}
	}
    private double [][] ComputeDistances(double [] candidate, int channel){
                            
            double [][] distancesCandidate = new double[dataSet.numTrain + dataSet.numTest][3];
            double diff = 0, distanceToSegment = 0, minDistanceSoFar = Double.MAX_VALUE;
            double start = -1; 
            double mid = -1;
		
            for( int i = 0; i < dataSet.numTrain + dataSet.numTest; i++ ){ 
			// if the candidate is longer than the series then slide the series
			// accross the canidate
            	if( candidate.length > dataSet.timeseries[i][channel].length ){
            		minDistanceSoFar = Double.MAX_VALUE; 
			for(int j = 0; j < candidate.length - dataSet.timeseries[i][channel].length + 1; j++){
            			distanceToSegment = 0; 
            			for(int k = 0; k < dataSet.timeseries[i][channel].length; k++){
                                            
					diff = candidate[j + k] - dataSet.timeseries[i][channel][k];  
					distanceToSegment += diff*diff;
					//distanceToSegment+=Math.abs(diff);
//					if( distanceToSegment > minDistanceSoFar ) 
//						break; 
            			} 
					
            			//distanceToSegment = distanceToSegment/(double) dataSet.timeseries[i][channel].length;
            			if( distanceToSegment < minDistanceSoFar) {
            				minDistanceSoFar = distanceToSegment; 
                                        start = j;
                                        mid  = j -1 + (double) candidate.length/2;  // ????  (double) (j+ dataSet.timeseries[i][channel].length-1)/2;
                                }
            		}
				
            		distancesCandidate[i][0] = Math.sqrt(minDistanceSoFar/(double) dataSet.timeseries[i][channel].length);
                        distancesCandidate[i][1] = start;
                        distancesCandidate[i][2] = mid;
            	}
            	else {// slide the candidate accorss the series and keep the smallest distance
			
            		minDistanceSoFar = Double.MAX_VALUE; 
				
            		for( int j = 0; j < dataSet.timeseries[i][channel].length-candidate.length+1; j++ ){
            			distanceToSegment = 0; 
					
            			for(int k = 0; k < candidate.length; k++){
            				diff = candidate[k]- dataSet.timeseries[i][channel][j + k];  
            				distanceToSegment += diff*diff; 
						
            				// if the distance of the candidate to this segment is more than the best so far
            				// at point k, skip the remaining points
//            				if( distanceToSegment > minDistanceSoFar ) 
//            					break; 
            			} 
					
            			//distanceToSegment = distanceToSegment/(double) candidate.length;
            			if( distanceToSegment < minDistanceSoFar){ 
            				minDistanceSoFar = distanceToSegment; 
            				start = j;
            				mid = j -1 + (double) candidate.length/2;
            			}
                        } 
                        distancesCandidate[i][0] = Math.sqrt(minDistanceSoFar/(double) candidate.length); ;
                        //distancesCandidate[i][0] = minDistanceSoFar;
                        distancesCandidate[i][1] = start;
                        distancesCandidate[i][2] = mid;
                }
            }
		
		return distancesCandidate;		
	}
    public boolean FoundInList(double [] candidate, List<double[]> list, int channel){
		double diff = 0, distance = 0; 
		int shapeletLength = candidate.length; 
		
		for(double [] shapelet : list) 
		{ 
			// avoid comparing against shapelets of other lengths
			if(shapelet.length != candidate.length) 
				continue; 
			
			distance = 0;
			for(int k = 0; k < shapeletLength; k++)
			{
				diff = candidate[k]- shapelet[k]; 
				distance += diff*diff; 
				
				// if the distance so far exceeds epsilon then stop
				if( (distance/shapeletLength) > epsilon[channel] ) 
					break; 
				
			}
			
			if( (distance/shapeletLength) < epsilon[channel] ) 
					return true;			
		}
		
		return false;
	}
    public double EstimateEpsilon(int channel){
		// return 0 epsilon if no pruning is requested, i.e. percentile=0
		if (percentile == 0)
			return 0;
		
		int numPairs = dataSet.numTrain*dataSet.minLength; 
		
				
		double [] distances = new double[numPairs]; 
		
		int seriesIndex1 = -1, pointIndex1 = -1, seriesIndex2 = -1, pointIndex2 = -1;
		double pairDistance = 0, diff = 0;
		int shapeletLength = 0;
		
		DescriptiveStatistics stat = new DescriptiveStatistics();
		
		for(int i = 0; i < numPairs; i++)
		{
			shapeletLength = shapeletLengths[ rand.nextInt(shapeletLengths.length) ];
			
			seriesIndex1 = rand.nextInt( dataSet.numTrain );
			int maxPoint1 = dataSet.timeseries[seriesIndex1][channel].length - shapeletLength + 1;
			seriesIndex2 = rand.nextInt( dataSet.numTrain );
			int maxPoint2 = dataSet.timeseries[seriesIndex2][channel].length - shapeletLength + 1;
			
			// avoid series having length less than the shapeletLength
			if( maxPoint1 <= 0 || maxPoint2 <= 0)
				continue;
			
			pointIndex1 = rand.nextInt( maxPoint1 ); 
			pointIndex2 = rand.nextInt( maxPoint2 ); 
			
			pairDistance = 0;
			for(int k = 0; k < shapeletLength; k++)
			{
				diff = dataSet.timeseries[seriesIndex1][channel][pointIndex1 + k] 
						- dataSet.timeseries[seriesIndex2][channel][pointIndex2 + k]; 
				pairDistance += diff*diff;
			}  
			
			distances[i] = pairDistance/(double)shapeletLength;	
			
			stat.addValue(distances[i]); 
			
		} 
		
		return stat.getPercentile(percentile); 
	} 
    public static void main(String [] args) throws Exception{
        System.out.println("Learn General Shapelet orders Cross Validation");
        String maindirectory = "C:\\shoumik\\DABI\\datasets\\TSCProblems2018\\synthetic\\PSOD\\";
        File file = new File(maindirectory);
        String[] names  = file.list();
        String[] percent = new String[] {"0", "15", "25" , "35", "50", "75", "90"};
        //String[] names = Arrays.copyOfRange(names_raw, 1, names_raw.length);
        
        for(String name : names){
            //for(int par = 0; par < percent.length; par++){
            System.out.println("DATASET : " + name);
            Random rand = new Random(1);
            double [] Alpha = { -10, -100};
            double bestAlpha = 0;
            double nextAlpha = 0;
            int restart = 0;
            double trainError = 1;
            //////////////////////////////Start internal crossvalidation//////////////////////////////////////
            /*for (int i = 0;i<Alpha.length;i++){
                double []validationError = new double[4];
                int count = 0;
                String trainDirectory = maindirectory + name+  "\\train\\";
                File fileTrain = new File(trainDirectory);
                String[] namesTrain  = fileTrain.list();
            
                for(String nameTrain : namesTrain){
            
                    String sp = File.separator;
                    //System.out.println("TRAINING DATASET : " + nameTrain);
                    args = new String[] { 
				"trainFile=" + trainDirectory + nameTrain + sp + nameTrain + "_TRAIN",
				"testFile=" + trainDirectory + nameTrain + sp + nameTrain + "_TEST",
				"paaRatio=1", 
				"percentile=35", //+ percent[par], 
				"numTrials=5" 
			};
                
                    String place = maindirectory + name + sp;
		
            // initialize variables
                    String dir = "", ds = name;
                    int percentile = 0, numTrials = 5;
                    double paaRatio = 0.0;
		
            // set the paths of the train and test files
                    String trainSetPath = "";
                    String testSetPath = "";
		
            // parse command line arguments
                for (String arg : args){
                    String[] argTokens = arg.split("=");
                    if (argTokens[0].compareTo("dir") == 0)
                        dir = argTokens[1];
                    else if (argTokens[0].compareTo("trainFile") == 0)
                        trainSetPath = argTokens[1];
                    else if (argTokens[0].compareTo("testFile") == 0) 
                        testSetPath = argTokens[1];
                    else if (argTokens[0].compareTo("paaRatio") == 0) 
                        paaRatio = Double.parseDouble(argTokens[1]);
                    else if (argTokens[0].compareTo("percentile") == 0)
                        percentile = Integer.parseInt(argTokens[1]);
                    else if (argTokens[0].compareTo("numTrials") == 0)
                        numTrials = Integer.parseInt(argTokens[1]);				
                }
                PSOD ssd1 = new PSOD();
                int startp = rand.nextInt();
                ssd1.rand.setSeed(startp);
                ssd1.trainSetPath = trainSetPath;
                ssd1.testSetPath = testSetPath; 
                ssd1.percentile = percentile;
                ssd1.paaRatio = paaRatio;
                ssd1.normalizeData = true;  
                ssd1.LoadData();
                double crossValTrainError = ssd1.Search(place, 0, bestAlpha);
            
            /*while(ssd1.numAcceptedShapelets <= 1){
                    System.err.println("RESTART due to 1 or 0 Shapelet");
                     ssd1= new PSOD();
                     startp = rand.nextInt();
                     ssd1.rand.setSeed(startp);
                     ssd1.trainSetPath = trainSetPath;
                     ssd1.testSetPath = testSetPath; 
                     ssd1.percentile = percentile;
                     ssd1.paaRatio = paaRatio;
                     ssd1.normalizeData = true;  
                     ssd1.LoadData(); 
                     ///startMethodTime = System.currentTimeMillis(); 
                     crossValTrainError = ssd1.Search(place, 0, bestAlpha);
                     //elapsedMethodTime = System.currentTimeMillis() - startMethodTime;
                }*/
            
                /*double validationerrorRateWithOrder = ssd1.ComputeTestErrorUsingOrder(place, nameTrain, 0, bestAlpha);
                validationError[count] = validationerrorRateWithOrder;
                count = count+1;
                               
                
                }
                double validationErrorWithOrder = StatisticalUtilities.Mean(validationError);
                //System.err.println("Validation Error: " + " " + validationErrorWithOrder +  " " + "Alpha" +  " " + Alpha[i]);
                if(validationErrorWithOrder<trainError){
                    trainError = validationErrorWithOrder;
                    bestAlpha = Alpha[i];
                }
                if(validationErrorWithOrder==trainError){
                    nextAlpha = Alpha[i];
                }
                
                
            }*/
            //System.err.println("Best Alpha = " + " " + bestAlpha);
            ////////////////////// End Internal Cross validation/////////////////////////////////////////
            bestAlpha = -10;
            String sp = File.separator;
            args = new String[] { 
				"trainFile=" + maindirectory + name + sp + name + "_TRAIN",
				"testFile=" + maindirectory + name + sp + name + "_TEST",
				"paaRatio=1", 
				"percentile=35", //+percent[par], 
				"numTrials=10" 
			};
		
            
            String place = maindirectory + name + sp;
		
            // initialize variables
            String dir = "", ds = name;
            int percentile = 0, numTrials = 5;
            double paaRatio = 0.0;
		
            // set the paths of the train and test files
            String trainSetPath = "";
            String testSetPath = "";
		
            // parse command line arguments
            for (String arg : args){
                String[] argTokens = arg.split("=");
                if (argTokens[0].compareTo("dir") == 0)
                    dir = argTokens[1];
                else if (argTokens[0].compareTo("trainFile") == 0)
                    trainSetPath = argTokens[1];
                else if (argTokens[0].compareTo("testFile") == 0) 
                    testSetPath = argTokens[1];
                else if (argTokens[0].compareTo("paaRatio") == 0) 
                    paaRatio = Double.parseDouble(argTokens[1]);
                else if (argTokens[0].compareTo("percentile") == 0)
                    percentile = Integer.parseInt(argTokens[1]);
                else if (argTokens[0].compareTo("numTrials") == 0)
                    numTrials = Integer.parseInt(argTokens[1]);				
            }

		
            // run the algorithm a number of times times
            //double [] erroRatesWithOnlyOrders = new double[numTrials];
            double [] errorRatesWithShapelets = new double[numTrials];
            //double [] errorRatesWithShapeletsProbability = new double[numTrials];
            double [] errorRatesWithOrder = new double[numTrials];
            double [] trainTimes = new double[numTrials]; 
            double [] totalTimes = new double[numTrials]; 
            double [] numAccepted = new double[numTrials]; 
            double [] numRefused = new double[numTrials];
            double [] numRejected = new double[numTrials];
            double [] numAcceptedFeatures = new double[numTrials];
            double [] numOrders = new double[numTrials];
            
                    
            for(int trial = 0; trial < numTrials; trial++){
                
                String outfile7 = place + "trial_"+trial+"_shapeletsOnly_ContinousDis_GeneralOrders.txt";
                String outfile1 = place + "trial_"+trial+"_shapelets_ContinousDis_GeneralOrders.txt";
                //String outfile = place + "trial_"+trial+"_features_ContinousDis_GeneralOrders.txt";
                FileOutputStream fos1 = new FileOutputStream(outfile1);
                FileOutputStream fos7 = new FileOutputStream(outfile7);
                //FileOutputStream fos = new FileOutputStream(outfile);
                PrintStream ps7 = new PrintStream(fos7);
                PrintStream ps1 = new PrintStream(fos1);
                //PrintStream ps = new PrintStream(fos);
                       	
                PSOD ssd = new PSOD();
                
                int startp = rand.nextInt();
                ssd.rand.setSeed(startp);
                ssd.trainSetPath = trainSetPath;
                ssd.testSetPath = testSetPath; 
                ssd.percentile = percentile;
                ssd.paaRatio = paaRatio;
                ssd.normalizeData = true;  
                ssd.LoadData();
                
                //System.err.println("Trial Number:" +  " "+ trial );
                long startMethodTime = System.currentTimeMillis(); 
                double trialTrainError = ssd.Search(place, trial, bestAlpha);
                double elapsedMethodTime = System.currentTimeMillis() - startMethodTime;
               
                while(ssd.numAcceptedShapelets <= 1){
                    //System.err.println("RESTART due to 1 or 0 Shapelet");
                    restart = restart + 1;
                     ssd = new PSOD();
                     startp = rand.nextInt();
                     ssd.rand.setSeed(startp);
                     ssd.trainSetPath = trainSetPath;
                     ssd.testSetPath = testSetPath; 
                     ssd.percentile = percentile;
                     ssd.paaRatio = paaRatio;
                     ssd.normalizeData = true;  
                     ssd.LoadData();
                     if (restart>5){
                            bestAlpha = nextAlpha;
                     }
                     startMethodTime = System.currentTimeMillis();
                     trialTrainError = ssd.Search(place, trial, bestAlpha);
                     elapsedMethodTime = System.currentTimeMillis() - startMethodTime;
                }
                
                for(int num = 0;num<ssd.featureName.size();num++){
                    ps1.print(ssd.featureName.get(num) +  " " + " Class: " +  " " + ssd.featureClass.get(num));
                    ps1.println();
                }
                ps1.close();
                                   
                
                //System.err.println(ssd.orderClassInfo.size());
                //double errorRateWithOnlyOrder = ssd.ComputeTestErrorOnlyOrders(place, name, trial);
                double errorRateWithShapelet = ssd.ComputeTestErrorUsingShapeletTransform(place, name, trial, bestAlpha);
                //System.out.println(" Final train error:" + " " + ssd.currentTrainError + "," + " Accepted shapelets:" + " "  + ssd.numAcceptedShapelets + "," + " Accepted orders:" + " "  + ssd.numAcceptedOrders +"," + " Rejected Shapelets:" + " "  + ssd.numRejectedShapelets + "," + " Refused Shapalets:" + " " + ssd.numRefusedShapelets); 
                double errorRateWithOrder = ssd.ComputeTestErrorUsingOrder(place, name, trial, bestAlpha);
                //System.out.println("Test accuracy with Shapeletes only " + (1 - errorRateWithShapelet)*100);
                //System.out.println("Test accuracy with Shapeletes + Orders " + (1 - errorRateWithOrder)*100);
                double testTime = System.currentTimeMillis() - startMethodTime;  
                errorRatesWithShapelets[trial] = errorRateWithShapelet;
                errorRatesWithOrder[trial] = errorRateWithOrder;
                trainTimes[trial] = elapsedMethodTime/1000; // in second
                totalTimes[trial] = testTime/1000; // in second
                numAccepted[trial] = ssd.numAcceptedShapelets;
                numRefused[trial] = ssd.numRefusedShapelets;
                numRejected[trial] = ssd.numRejectedShapelets;
                numAcceptedFeatures[trial] = ssd.numAcceptedShapelets+ssd.numAcceptedOrders;
                numOrders[trial] = ssd.numAcceptedOrders;
                
						                                                  
                //double percent = 0.125;
                int numInstance =  ssd.dataSet.numTrain +ssd.dataSet.numTest;
                String outfile2 = place +name +"_numTrain_numTest.txt";
                FileOutputStream fos2 = new FileOutputStream(outfile2);
                PrintStream ps2 = new PrintStream(fos2);
                String outfile3 = place + "trial_"+ trial+"_time.txt";
                FileOutputStream fos3 = new FileOutputStream(outfile3);
                PrintStream ps3 = new PrintStream(fos3);
                ps2.print(ssd.dataSet.numTrain + " ");
                ps2.print(ssd.dataSet.numTest);
                ps2.println();
                //System.out.println();
                //int numShape = ssd.finalFeatureMatrix.size();
                                
                /*for( int i = ssd.dataSet.numTrain;i<numInstance;i++){
                
                        ps2.println();
                        for(int c = 0;c<numShape;c++){
                        ps2.print(ssd.finalFeatureMatrix.get(c)[i]+ " ");
                }
                }*/
                //ps2.println();
                //ps2.println();
                /*int altnumShape = ssd.altfinalFeatureMatrix.size();
                for( int i = 0;i<numInstance;i++){
                        ps.print(ssd.dataSet.labels[i] +  " "                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     );
                        for(int c = 0;c<altnumShape;c++){
                            ps.print(ssd.altfinalFeatureMatrix.get(c)[i] + " ");
                        }
                        ps.println();
                }
                ps.close();*/
                                 
                int numShap = ssd.distancesShapelets.size();
                for( int i = 0;i<numInstance;i++){
                        ps7.print(ssd.dataSet.labels[i] +  " ");
                        for(int c = 0;c<numShap;c++){
                            ps7.print(ssd.distancesShapelets.get(c)[i] + " ");
                        }
                        ps7.println();
                }
                ps7.close();
                                 
                for(int dim=0; dim < ssd.dataSet.numChannels; dim++){
                        int k = ssd.startTime.get(dim).size();  //number of shapelets
                        //ps.println(colIndex+"\tS_"+dim+"_"+j);
                        for(int i=0; i<ssd.dataSet.numTrain+ssd.dataSet.numTest; i++ ){
                            for(int j=0; j< k; j++){
                                ps3.print(ssd.startTime.get(dim).get(j)[i] + " ");
                                                
                                        //candidateMatrix[i][colIndex] = startTime.get(dim).get(j)[i];
                                        //newFeatureMatrix[i][colIndex] = candidateMatrixSoFar.get(dim).get(j)[i];
                            }
                                            ps3.println();
                        }
                }
                                //ps3.close();
                                /*int numShape = ssd.temp_finalFeatureMatrix.size();
                                for(int b = 0;b<ssd.dataSet.numTrain;b++){
                                    ps2.println();
                                    for(int c = 0;c<numShape;c++)
                                        ps2.print(ssd.temp_finalFeatureMatrix.get(c)[b] + " ");
                                }
                                ps2.println();
                                ps2.println();
                                for(int b = 0;b<ssd.dataSet.numTrain+ssd.dataSet.numTest;b++){
                                   ps2.println();
                                    for(int c = 0;c<numShape;c++)
                                        ps2.print(ssd.finalFeatureMatrix.get(c)[b] + " ");
                                    
                                }*/
                ps2.close();
                ps3.close();
                                    
            
                                	    
            } 

            String outfile = place +name +"_generalOrdes_Results.txt";
            FileOutputStream fos = new FileOutputStream(outfile);  
            PrintStream ps = new PrintStream(fos);
                        
            //double accuracyWithOnlyOrder = 1 - StatisticalUtilities.Mean(erroRatesWithOnlyOrders);
            //double stdWithOnlyOrder = StatisticalUtilities.StandardDeviation(erroRatesWithOnlyOrders);
                       
            double accuracyWithShapelets = 1 - StatisticalUtilities.mean(errorRatesWithShapelets, false);
            double stdWithShapelets = StatisticalUtilities.standardDeviation(errorRatesWithShapelets, false, accuracyWithShapelets);
                        
            double accuracyWithOrder = 1 - StatisticalUtilities.mean(errorRatesWithOrder, false);
            double stdWithOrder = StatisticalUtilities.standardDeviation(errorRatesWithOrder, false, accuracyWithOrder);
            
            double trainTime = StatisticalUtilities.mean(trainTimes, false);
            double trainStd = StatisticalUtilities.standardDeviation(trainTimes, false, trainTime);
            
            double refused = StatisticalUtilities.mean(numRefused, false);
            double refusedStd = StatisticalUtilities.standardDeviation(numRefused, false, refused);
            
            double accepted = StatisticalUtilities.mean(numAccepted, false);
            double acceptedStd = StatisticalUtilities.standardDeviation(numAccepted, false, accepted);
            
            double rejected = StatisticalUtilities.mean(numRejected, false);
            double rejectedStd = StatisticalUtilities.standardDeviation(numRejected, false, rejected);
            
            double orders = StatisticalUtilities.mean(numOrders, false);
            double ordersStd = StatisticalUtilities.standardDeviation(numOrders, false, orders);
            //double accuracyWithShapeletsProbability = 1 - StatisticalUtilities.Mean(errorRatesWithShapeletsProbability);
            //double stdWithShapeletsProbability = StatisticalUtilities.StandardDeviation(errorRatesWithShapeletsProbability);
        
            //ps.print(accuracyWithOnlyOrder + "\u00B1" + stdWithOnlyOrder);
            //ps.println();
            ps.print(accuracyWithShapelets + "\u00B1" + stdWithShapelets);
            ps.println();
            ps.print(accuracyWithOrder + "\u00B1" + stdWithOrder);
            ps.println();
            ps.print("Average Training Time");
            ps.println();
            ps.print(trainTime + "\u00B1" + trainStd);
            ps.println();
            ps.print("Best Alpha:" + " " + bestAlpha);
            ps.close();
            
                        
            //System.out.println(ds + " " +"Accuracy with only Orders" + " " + accuracyWithOnlyOrder*100 + "\u00B1" + stdWithOnlyOrder*100 );
            //System.out.println(ds + " " +"Accuracy with Shapelets only" + " " + accuracyWithShapelets*100 + "\u00B1" + stdWithShapelets*100 );
            System.out.println(ds + " " +"Accuracy With Order" + " " + accuracyWithOrder*100 + "\u00B1" + stdWithOrder*100 );//+  " " + "Percentile" + " " + percent[par]);
            System.out.println(ds + " " +"Training Time" + " " + trainTime + "\u00B1" + trainStd );
            //System.out.println(ds + " " +"% Refused shapelets" + " " + 100*refused/(refused+rejected+accepted));
            //System.out.println(ds + " " +"% Rejected shapelets" + " " + 100*rejected/(refused+rejected+accepted));
            System.out.println(ds + " " +"% Acceped shapelets" + " " + 100*accepted/(refused+rejected+accepted));
            System.out.println(ds + " " +"Number of shapelets-orders" + " " + orders + "\u00B1" + ordersStd );

//System.out.println(ds + " " +"Accuracy With shapelets (Probability)" + " " + accuracyWithShapeletsProbability*100 + "\u00B1" + stdWithShapeletsProbability*100 );
       /* System.out.println(
        		ds + " " + paaRatio + ", " + percentile + ", " + numTrials + ", " + "Error Rate:" + " " +
        				StatisticalUtilities.Mean(errorRates) + "\u00B1" +
        				StatisticalUtilities.StandardDeviation(errorRates) + " " + "Mean Time:" + " " +
        				StatisticalUtilities.Mean(trainTimes) + "\u00B1" + 
        				StatisticalUtilities.StandardDeviation(trainTimes) + " " +"Total Time:" + " " +
        				StatisticalUtilities.Mean(totalTimes)  + "\u00B1" + 
        				StatisticalUtilities.StandardDeviation(totalTimes) + " " + "mean Accepted Shapelets:" + " " +
        				StatisticalUtilities.Mean(numAccepted) + "\u00B1" + 
    					StatisticalUtilities.StandardDeviation(numAccepted)  ); */
        
        
        //} percent block
        }
    }
    
}
