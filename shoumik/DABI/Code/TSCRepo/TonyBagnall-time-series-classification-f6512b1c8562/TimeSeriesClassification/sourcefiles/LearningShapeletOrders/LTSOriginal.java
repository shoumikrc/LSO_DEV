/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package LearningShapeletOrders;


import org.happy.commons.concurrent.loops.ForEachTask_1x0;
import org.happy.commons.concurrent.loops.Parallel_1x0;
import java.io.IOException;
import java.io.File;
import utilities.Logging;
import utilities.Logging.LogLevel;
import java.io.FileNotFoundException;
import utilities.InstanceTools;
import weka.core.Instances;
import DataStructures.DataSet;
import DataStructures.Matrix;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import utilities.StatisticalUtilities;
import static utilities.StatisticalUtilities.calculateSigmoid;
import weka.clusterers.SimpleKMeans;

/**
 *
 * @author shoumik
 * Paper Title: Learning Temporal Dependency Among Pairwise Shapelets
 * Authors:Shoumik Roychoudhury, Fang Zhou, Zoran Obradovic
 * Code: Shoumik Roychoudhury
 */
public class LTSOriginal {
    //public MultivariateDataset dataSet;
    public String trainSetPath, testSetPath;
    public boolean normalizeData;
    public double paaRatio ;
    //public double[][] train2D;
    //public double[][] test2D;
    
    public int ITrain, ITest; // number of training and testing instances
    // length of a time-series 
    public int Q;
    // length of shapelet
    public int L[];
    public int L_min;
    // number of latent patterns
    public int K;
    //maximum number of orders
    public int H;
    // scales of the shapelet length
    public int R; 
    // number of classes
    public int C;
    // number of segments
    public int J[];
    // time series data and the label 
    public Matrix T;
    public Matrix Y, Y_b;
    
    // shapelets
    double Shapelets[][][];
    // the softmax parameter
    public double alpha;
    // accumulate the gradients
    double GradHistShapelets[][][];
    double GradHistW[][][];
    //double GradHistW_h[][][];
    double GradHist_u[];
    double GradHistBiasW[];
    
    public int maxIter;
    // the learning rate
    public double eta; 
    public int kMeansIter;
    // the regularization parameters
    public double lambdaW;
    //public double lambdaW_k;
    //public double lambdaW_h;
    public double delta[];
    public double u[];
    public List<Double> nominalLabels;
		
    // structures for storing the precomputed terms
    double D[][][][];
    double E[][][][];
    //double E_alt[][][][];
    double startTime[][][][];
    double M[][][];
    double B[][][];
    double G[][][];
    double Psi[][][]; 
    double sigY[][]; 
   

    Random rand = new Random();
	
    List<Integer> instanceIdxs;
    List<Integer> rIdxs;
    
    // classification weights
    double W[][][];
    //double W_h[][][];
    double biasW[];
    
    //public List<Double> nominalLabels;
    
    
    public void CreateOneVsAllTargets(){
        C = nominalLabels.size(); 
		
	Y_b = new Matrix(ITrain+ITest, C);
		
		// initialize the extended representation  
        for(int i = 0; i < ITrain+ITest; i++){
            // firts set everything to zero
            for(int c = 0; c < C; c++)  
                    Y_b.set(i, c, 0);
            
            // then set the real label index to 1
            int indexLabel = nominalLabels.indexOf( Y.get(i, 0) ); 
            Y_b.set(i, indexLabel, 1.0); 
        } 

    }
    
    public void Initialize(){
        // avoid K=0 
		if(K == 0) 
			K = 1;
		
		
		// set the labels to be binary 0 and 1, needed for the logistic loss
		CreateOneVsAllTargets();
		
		
		// initialize the shapelets (complete initialization during the clustering)
		Shapelets = new double[R][][];
		
		GradHistShapelets = new double[R][][];
		
		// initialize the number of shapelets and the length of the shapelets 
		J = new int[R]; 
		L = new int[R];
		// set the lengths of shapelets and the number of segments
		// at each scale r
		int totalSegments = 0;
		for(int r = 0; r < R; r++)
		{
			L[r] = (r+1)*L_min;
			J[r] = Q - L[r];
			
			totalSegments += ITrain*J[r]; 
		}
		
		// set the total number of shapelets per scale as a rule of thumb 
		// to the logarithm of the total segments
		if( K < 0)
			K = (int) Math.log(totalSegments) * (C-1); 
                        //System.out.println(K);
		//Logging.println("Original LTS");
		//Logging.println("ITrain="+ITrain + ", ITest="+ITest + ", Q="+Q + ", Classes="+C, LogLevel.DEBUGGING_LOG);
		//Logging.println("K="+K + ", L_min="+ L_min + ", R="+R, LogLevel.DEBUGGING_LOG);
		//Logging.println("eta=" + eta + ", maxIter="+ maxIter, LogLevel.DEBUGGING_LOG);
		//Logging.println("lambdaW="+lambdaW + ", alpha="+ alpha, LogLevel.DEBUGGING_LOG);
		//Logging.println("totalSegments="+totalSegments + ", K="+ K, LogLevel.DEBUGGING_LOG);
		
		// initialize an array of the sizes
		rIdxs = new ArrayList<Integer>();
		for(int r = 0; r < R; r++)
				rIdxs.add(r);
		
		// initialize shapelets
                
		InitializeShapeletsKMeans();
		
		
		// initialize the terms for pre-computation
		D = new double[ITrain+ITest][R][K][];
		E = new double[ITrain+ITest][R][K][];
		
		for(int i=0; i <ITrain+ITest; i++)
			for(int r = 0; r < R; r++)
				for(int k = 0; k < K; k++)
				{
					D[i][r][k] = new double[J[r]];
					E[i][r][k] = new double[J[r]];
				}
		
		// initialize the placeholders for the precomputed values
		M = new double[ITrain+ITest][R][K];
		Psi = new double[ITrain+ITest][R][K];
		sigY = new double[ITrain+ITest][C];
		
		// initialize the weights
		
		W = new double[C][R][K];
		biasW = new double[C];
		
		GradHistW = new double[C][R][K];
		GradHistBiasW = new double[C];
		
		
		
		for(int c = 0; c < C; c++)
		{
			for(int r = 0; r < R; r++)
				for(int k = 0; k < K; k++)
				{
					W[c][r][k] = 2*rand.nextDouble()-1;
					GradHistW[c][r][k] = 0;
				}
			
			biasW[c] = 2*rand.nextDouble()-1;
			GradHistBiasW[c] = 0;
		}
	
		// precompute the M, Psi, sigY, used later for setting initial W
		for(int i=0; i < ITrain+ITest; i++)
			PreCompute(i); 
		
		// store all the instances indexes for
		instanceIdxs = new ArrayList<Integer>();
		for(int i = 0; i < ITrain; i++)
				instanceIdxs.add(i);
		// shuffle the order for a better convergence
		Collections.shuffle(instanceIdxs, rand); 
		
		//Logging.println("Initializations Completed!", LogLevel.DEBUGGING_LOG);
    }
    public void PreCompute(int i){
        // precompute terms
	for(int r = 0; r < R; r++){
            for(int k = 0; k < K; k++){
                for(int j = 0; j < J[r]; j++){
                    // precompute D
                    D[i][r][k][j] = 0;
                    double err = 0;
					
                    for(int l = 0; l < L[r]; l++){
                        err = T.get(i, j+l)- Shapelets[r][k][l];
                        //err = T.get(i, j+l)-getMean- Shapelets[r][k][l];
                        //System.out.print(Shapelets[r][k][l] + " ");                                 
                        D[i][r][k][j] += err*err;
                        //startTime[i][r][k][j] = j+l;
                    }
					
                    D[i][r][k][j] /= (double)L[r];
                        // precompute E
                    E[i][r][k][j] = Math.exp(alpha * D[i][r][k][j]);
                    
		}
				
		// precompute Psi 
		Psi[i][r][k] = 0; 
		for(int j = 0; j < J[r]; j++) 
                    Psi[i][r][k] +=  Math.exp( alpha * D[i][r][k][j] );
				
		// precompute M 
		M[i][r][k] = 0;
                //B[i][r][k] = 0;
				
		for(int j = 0; j < J[r]; j++){
                    M[i][r][k] += D[i][r][k][j]* E[i][r][k][j];
                    //B[i][r][k] += startTime[i][r][k][j] * E[i][r][k][j];
                }
                    
                    M[i][r][k] /= Psi[i][r][k];
                    //B[i][r][k] /= Psi[i][r][k];
                    //B[i][r][k] = Math.floor(B[i][r][k]);
            }
            
            //Precompute the order space
            int orderIndex = 0;
            
                //System.out.println("OrderIndex: " +  " " + orderIndex + " " + H);
                for(int k1 = 0;k1<K-1;k1++){
                    //System.out.println(k1);
                    for(int k2 = 1;k2 < K;k2++){
                        //System.out.println(k2);
                        if(k1<k2 && orderIndex < H){
                            //System.out.println("k1: " + k1 + " " + "k2: " + k2);
                            G[i][r][orderIndex] = B[i][r][k1] - B[i][r][k2];
                            orderIndex=orderIndex+1;
                        }
                        
                    }
                }
            
            
                
	}
		
	for(int c = 0; c < C; c++)
            sigY[i][c] = calculateSigmoid( Predict(i, c) ); 
    }
    // predict the label value vartheta_i
    public double Predict(int i, int c){
        //double Y_hat_ic = biasW[c];
        double Y_hat_ic = 0;
        //double y_hat_ic_shapelets = 0;
        //double y_hat_ic_orders = 0;
                
                // Shapelet space space
        for(int r = 0; r < R; r++)
            for(int k = 0; k < K; k++)
                Y_hat_ic += M[i][r][k] * W[c][r][k];
		
        
                
        //Y_hat_ic = delta[c]*y_hat_ic_shapelets + (1- delta[c])*y_hat_ic_orders;
	return Y_hat_ic;
    }
    public void InitializeShapeletsKMeans(){
        // a multi-threaded parallel implementation of the clustering
	// on thread for each scale r, i.e. for each set of K shapelets at
	// length L_min*(r+1)
	Parallel_1x0.ForEach(rIdxs, new ForEachTask_1x0<Integer>(){
            public void iteration(Integer r){
                try {
                    //Logging.println("Initialize Shapelets: r="+r+", J[r]="+J[r]+", L[r]="+L[r], LogLevel.DEBUGGING_LOG);
                    
                    double [][] segmentsR = new double[ITrain*J[r]][L[r]];
                    
                    for(int i= 0; i < ITrain; i++)
                        for(int j= 0; j < J[r]; j++)
                            for(int l = 0; l < L[r]; l++)
                                segmentsR[i*J[r] + j][l] = T.get(i, j+l);
                    
                    // normalize segments
                    for(int i= 0; i < ITrain; i++) 
                        for(int j= 0; j < J[r]; j++)
                            for(int l = 0; l < L[r]; l++)
                                segmentsR[i*J[r] + j] = utilities.StatisticalUtilities.normalize(segmentsR[i*J[r] + j]);
                    //segmentsR[i*J[r] + j] = StatisticalUtilities.Normalize(segmentsR[i*J[r] + j]);
                    
                    Instances ins = InstanceTools.toWekaInstances(segmentsR);
                    SimpleKMeans kmeans = new SimpleKMeans();
                    kmeans.setNumClusters(K);
                    kmeans.setMaxIterations(100);
                    kmeans.setInitializeUsingKMeansPlusPlusMethod(true);
                    kmeans.buildClusterer( ins );
                    Instances centroidsWeka = kmeans.getClusterCentroids();
                    Shapelets[r] =  InstanceTools.fromWekaInstancesArray(centroidsWeka,false);
                    //Shapelets[r] = kmeans.InitializeKMeansPP(segmentsR, K, 100);
                    
                    // initialize the gradient history of shapelets
                    GradHistShapelets[r] = new double[K][ L[r] ];
                    for(int k= 0; k < K; k++)
                        for(int l = 0; l < L[r]; l++)
                            GradHistShapelets[r][k][l] = 0.0;
                    
                    
                    if( Shapelets[r] == null)
                        System.out.println("P not set");
                } catch (Exception ex) {
                    Logger.getLogger(LearnShapeletOrders.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
		});
	}
    public double[] Learn(String predictfile) throws FileNotFoundException{
        // initialize the data structures
	Initialize();
				
	List<Double> lossHistory = new ArrayList<Double>();
	lossHistory.add(Double.MIN_VALUE);
		
	// apply the stochastic gradient descent in a series of iterations
	for(int iter = 0; iter <= maxIter; iter++){
            // learn the latent matrices
            LearnF(); 
			
            // measure the loss
            if( iter % 200 == 0){
                double mcrTrain = GetMCRTrainSet();
                double mcrTest[] = GetMCRTestSet(predictfile); 
				
                double lossTrain = AccuracyLossTrainSet();
                double lossTest = AccuracyLossTestSet();
				
				
                lossHistory.add(lossTrain);
				
                //Logging.println("It=" + iter + ", alpha= "+alpha+", lossTrain="+ lossTrain + ", lossTest="+ lossTest  +
                              //", MCRTrain=" + mcrTrain + ", MCRTest=" +mcrTest[0] //+ ", SVM=" + mcrSVMTest
                              //, LogLevel.DEBUGGING_LOG);
				
                //System.out.println( eta/Math.sqrt(GradHistBiasW[0]) );
                //System.out.println( eta/Math.sqrt(GradHistW[0][1][5]) );
				
                // if divergence is detected start from the beggining 
                // at a lower learning rate
                if( Double.isNaN(lossTrain) || mcrTrain == 1.0 ){
                    iter = 0;
                    eta /= 3;
                    lossHistory.clear();
                    Initialize();
                    //Logging.println("Divergence detected. Restarting at eta=" + eta, LogLevel.DEBUGGING_LOG);
                }
				
                if( lossHistory.size() > 100 ) 
                    if( lossTrain > lossHistory.get( lossHistory.size() - 2  )  )
                       	break;
            }
	}
		
	return GetMCRTestSet(predictfile); 
    }
    private double[] GetMCRTestSet(String predictfile)  throws FileNotFoundException{
        int numErrors = 0;
                 
        double[] Predict = new double[ ITrain+ITest];
        double[] Prob1 = new double[ ITrain+ITest];
        double[] Prob2 = new double[ ITrain+ITest];
	//FileOutputStream fos2 = new FileOutputStream(predictfile);
        //PrintStream ps2 = new PrintStream(fos2);
	for(int i = ITrain; i < ITrain+ITest; i++){
            PreCompute(i);
		
            double max_Y_hat_ic = Double.MIN_VALUE;
            int label_i = -1; 
			
            for(int c = 0; c < C; c++){
                double Y_hat_ic = calculateSigmoid( Predict(i, c) );
				
		if(Y_hat_ic > max_Y_hat_ic){
                    max_Y_hat_ic = Y_hat_ic; 
                    label_i = (int)Math.ceil(c);
		}
            }
			
            if( nominalLabels.indexOf(Y.get(i)) != label_i ) 
                numErrors++;
                        
        }
        //ps2.close();
                    
                    
                    
        //System.out.println("Sensi = " +sensitivity);
        //System.out.println("WAccuracy = " +weighted_Accuracy);
        //System.out.println("TP = "+TP);
        // System.out.println("TN = "+TN);
        //System.out.println("FP = "+FP);
        //System.out.println("FN = "+FN);    
	return new double[]{(double)numErrors/(double)ITest};
    }
    public double GetMCRTrainSet(){
        int numErrors = 0;
		
	for(int i = 0; i < ITrain; i++){
	
            PreCompute(i);
			
            double max_Y_hat_ic = Double.MIN_VALUE;
            int label_i = -1; 
		
            for(int c = 0; c < C; c++){
                double Y_hat_ic = calculateSigmoid( Predict(i, c) );
				
		if(Y_hat_ic > max_Y_hat_ic){
                    max_Y_hat_ic = Y_hat_ic; 
                    label_i = (int)Math.ceil(c);
		}
            }
            
			
            if( nominalLabels.indexOf(Y.get(i)) != label_i ) 
                numErrors++;
        }
		
	return (double)numErrors/(double)ITrain;
    }
    public double AccuracyLoss(int i, int c){
        double Y_hat_ic = Predict(i, c);
	double sig_y_ic = calculateSigmoid(Y_hat_ic);
		
	return -Y_b.get(i,c)*Math.log( sig_y_ic ) - (1-Y_b.get(i, c))*Math.log(1-sig_y_ic); 
    }
    // compute the accuracy loss of the train set
    public double AccuracyLossTrainSet(){
            double accuracyLoss = 0;
		
            for(int i = 0; i < ITrain; i++){
                PreCompute(i);
		
		for(int c = 0; c < C; c++)
                    accuracyLoss += AccuracyLoss(i, c);
            }
		
            return accuracyLoss;
	}
    // compute the accuracy loss of the train set
    public double AccuracyLossTestSet(){
	double accuracyLoss = 0;
		
	for(int i = ITrain; i < ITrain+ITest; i++){
            PreCompute(i);
			
            for(int c = 0; c < C; c++) 
                accuracyLoss += AccuracyLoss(i, c); 
	}
		return accuracyLoss;
    }
      public void LearnF(){
         
		// parallel implementation of the learning, one thread per instance
		// up to as much threads as JVM allows
		Parallel_1x0.ForEach(instanceIdxs, new ForEachTask_1x0<Integer>() 
		{
			public void iteration(Integer i)
		    {
				double regWConst = ((double)2.0*lambdaW) / ((double) ITrain*C);
				
				double tmp2 = 0, tmp1 = 0, dLdY = 0, dMdS=0, gradS_rkl = 0, gradBiasW_c = 0, gradW_crk = 0; 
				double eps = 0.000001;
				
				for(int c = 0; c < C; c++)
				{
					PreCompute(i);
					//double Y_hat_ic = Predict(i, c);
					dLdY = -(Y_b.get(i, c) - sigY[i][c]);
                                        //dLdY = (-Y_b.get(i, c)*Math.exp(- Y_b.get(i, c)*Y_hat_ic))/(1+Math.exp(- Y_b.get(i, c)*Y_hat_ic));
					
					for(int r = 0; r < R; r++)
					{
						for(int k = 0; k < K; k++)
						{
							// gradient with respect to W_crk
							gradW_crk = dLdY*M[i][r][k] + regWConst*W[c][r][k];
							
							// add gradient square to the history
							GradHistW[c][r][k] += gradW_crk*gradW_crk;
							
							// update the weights
							W[c][r][k] -= (eta / ( Math.sqrt(GradHistW[c][r][k]) + eps))*gradW_crk; 
							
							tmp1 = ( 2.0 / ( (double) L[r] * Psi[i][r][k]) );
							
							for(int l = 0; l < L[r]; l++) 
							{
								tmp2=0;
								for(int j = 0; j < J[r]; j++)
									tmp2 += E[i][r][k][j]*(1 + alpha*(D[i][r][k][j] - M[i][r][k]))*(Shapelets[r][k][l] - T.get(i, j+l));
								
								gradS_rkl =  dLdY*W[c][r][k]*tmp1*tmp2;
								
								// add the gradient to the history
								GradHistShapelets[r][k][l] += gradS_rkl*gradS_rkl;
								
								Shapelets[r][k][l] -= (eta / ( Math.sqrt(GradHistShapelets[r][k][l]) + eps)) 
															* gradS_rkl;
								
							}				
						}
					}
		
					// the gradient 
					gradBiasW_c = dLdY;
                                        
					
					// add the gradient to the history
					GradHistBiasW[c] += gradBiasW_c*gradBiasW_c;
					
					biasW[c] -= (eta / ( Math.sqrt(GradHistBiasW[c]) + eps))*gradBiasW_c; 
                                        
				}			 	
		    }
		
		});
		
    }
    public void PrintShapeletsAndWeights(String outputfile) throws FileNotFoundException{
                FileOutputStream fos = new FileOutputStream(outputfile);
                PrintStream ps = new PrintStream(fos);
		for(int r = 0; r < R; r++){
			for(int k = 0; k < K; k++){
				//System.out.print("Shapelets("+r+","+k+")= [ ");
                                //ps.print("Shapelets("+r+","+k+")= [ ");
				
				for(int l = 0; l < L[r]; l++){
					//System.out.print(Shapelets[r][k][l] + " ");
                                        ps.print(Shapelets[r][k][l] + " ");
				}
				
				//System.out.println("]");
                                ps.println();
			}
		}

		//for(int c = 0; c < C; c++){
			for(int r = 0; r < R; r++){
				//System.out.print("W("+c+","+r+")= [ ");
                                //ps.print("W("+c+","+r+")= [ ");
				
                                for(int k = 0; k < K; k++){
					//System.out.print(W[c][r][k] + " ");
                                        // ps.print(W[c][r][k] + " ");
                                }
				
				//System.out.print(biasW[c] + " ");
                                //ps.print(biasW[c] + " ");
				//System.out.println("]");
                                //ps.println("]");
			}
		//}
                ps.close();
	}   
    
    public static void main(String [] args) throws FileNotFoundException, IOException{
        System.out.println("LTS");
        //main outer directory for Dataset Selection
        String maindirectory = "C:\\shoumik\\DABI\\datasets\\TSCProblems2018\\synthetic\\syntheticLTS\\";
        String sp = File.separator;
        File file = new File(maindirectory);
        String[] datasets  = file.list();
        //String[] names = Arrays.copyOfRange(names_raw, 1, names_raw.length);
        
        for(String name : datasets){
            System.out.println("DATASET : " + name);
            
            
            //Load the initial sample data
            Instances test = utilities.ClassifierTools.loadData(maindirectory+name+sp+name+"_TEST.arff");
            Instances train = utilities.ClassifierTools.loadData(maindirectory+name+sp+name+"_TRAIN.arff"); 
            String place = maindirectory + name + sp;
            
            
            //File handling preliminaries
            String outfile = place +name +"_LTS_Results.txt";
            FileOutputStream fos = new FileOutputStream(outfile);  
            PrintStream ps = new PrintStream(fos);
            
            //String outfile4 = place + name+"_numTrain_numTest.txt";
            //FileOutputStream fos4 = new FileOutputStream(outfile4);
            //PrintStream ps4 = new PrintStream(fos4);
            //ps4.print("seed " + " numTrain " + "numTest" );
            //ps4.println();
            int numofSeeds = 10;
            int counter = 0;
            double [] meanAccuracy = new double[numofSeeds]; 
            double [] meanTrainAccuracy = new double[numofSeeds];
            double [] trainTimes = new double[numofSeeds];
            //Create resample datasets for seeds 0 - 10 
            for(long seed = 0;seed<numofSeeds;seed++ ){
                System.out.println("seed:" + seed);
                                              
                //Java class object
                LTSOriginal lso = new LTSOriginal(); 
                
                //## Resample the dataset with same distribution of classes          
                Instances[] resampled = InstanceTools.resampleTrainAndTestInstances(train, test, seed);
                
                //convert the training set into a 2D Matrix
                //lso.train2D = InstanceTools.fromWekaInstancesArray(resampled[0], true);
                //lso.test2D =  InstanceTools.fromWekaInstancesArray(resampled[1], true);
                
                DataSet trainSet = new DataSet(resampled[0]);
                DataSet testSet = new DataSet(resampled[1]);
                
                
                // normalize the data instance
		//trainSet.NormalizeDatasetInstances();
		//testSet.NormalizeDatasetInstances();
                
                
                // predictor variables T
                Matrix T = new Matrix();
                T.LoadDatasetFeatures(trainSet, false);
                T.LoadDatasetFeatures(testSet, true);
                // outcome variable O
                Matrix O = new Matrix();
                O.LoadDatasetLabels(trainSet, false);
                O.LoadDatasetLabels(testSet, true);
                
                
                
                // set the time series and labels
                lso.T = T;
                lso.Y = O;
                //System.out.println(lso.T);
                
                // initialize the sizes of data structures
                lso.ITrain = trainSet.GetNumInstances();  
                lso.ITest = testSet.GetNumInstances();
                lso.Q = T.getDimColumns();
                
                

                // set the learn rate and the number of iterations and other hyper parameters
                double L = 0.1;
                //double lambdaW_k  = 0.1;
                //double lambdaW_h  = 0.1;
                double lambdaW = 0.01;
                int maxEpochs   = 1000;
                double K = -1;
                double eta = 0.1;
                double alpha = -100;
                int R = 3;
                
                // set predefined parameters if none set
                if(R < 0) R = 3;
                if(L < 0) L = 0.15;
                if(eta < 0) eta = 0.01;
                if(alpha > 0) alpha = -30;
                if(maxEpochs < 0) maxEpochs = 1000;
                
                
                
                lso.maxIter = maxEpochs;
                // set te number of patterns
                lso.K = (int)(K*T.getDimColumns());
                lso.L_min = (int)(L*T.getDimColumns());
                lso.R = R;
                // set the regularization parameter
                lso.lambdaW = lambdaW;
  
                lso.eta = eta;  
                lso.alpha = alpha; 
                trainSet.ReadNominalTargets();
                lso.nominalLabels =  new ArrayList<Double>(trainSet.nominalLabels);
                             
                // learn the model
                long startMethodTime = System.currentTimeMillis(); 
                lso.Learn(outfile);
                double elapsedMethodTime = System.currentTimeMillis() - startMethodTime;
                trainTimes[counter] = elapsedMethodTime/1000; // in second
                double trainSetError = lso.GetMCRTrainSet();
                double trainSetAccuracy = 1 - trainSetError;
                double [] arrayRet = lso.GetMCRTestSet(outfile);
                double accuracy = 1 - arrayRet[0];
                System.out.println("Train Set Accuracy for seed " + seed + ":" + trainSetAccuracy);
                meanTrainAccuracy[counter] = trainSetAccuracy;
                System.out.println("Test Set Accuracy for seed " + seed + ":" + accuracy);
                meanAccuracy[counter] = accuracy;
                counter++;
                
                
                ////File handling processing preliminaries
                //String outfile2 = place + "seed_"+seed+"_shapeletsOnly.txt";
                //String outfile3 = place + "seed_"+seed+"_shapeletsAndOrders.txt";
                //FileOutputStream fos2 = new FileOutputStream(outfile2);
                //FileOutputStream fos3 = new FileOutputStream(outfile3);
                //PrintStream ps2 = new PrintStream(fos2);
                //PrintStream ps3 = new PrintStream(fos3);
                
                //int numInstance = lso.ITrain+lso.ITest;
                //ps4.print(seed + " ");
                //ps4.print(lso.ITrain + " ");
                //ps4.print(lso.ITest);
                //ps4.println();
                
                

                
                  
            }
            //ps4.close();
            double accuracyMean = StatisticalUtilities.mean(meanAccuracy, false);
            double stdMean = StatisticalUtilities.standardDeviation(meanAccuracy, false, accuracyMean);
            
            double trainAccuracyMean = StatisticalUtilities.mean(meanTrainAccuracy, false);
            double trainstdMean = StatisticalUtilities.standardDeviation(meanTrainAccuracy, false, trainAccuracyMean);

            double trainTime = StatisticalUtilities.mean(trainTimes,false);
            double trainStd = StatisticalUtilities.standardDeviation(trainTimes, false, trainTime);
            
            System.out.println(trainAccuracyMean + "\u00B1" + trainstdMean);
            System.out.println(accuracyMean + "\u00B1" + stdMean);
            System.out.println(trainTime + "\u00B1" + trainStd);
            ps.print("Average Train set Accuracy: " + trainAccuracyMean + "\u00B1" + trainstdMean);
            ps.print("\nAverage Test set Accuracy: " + accuracyMean + "\u00B1" + stdMean);
            ps.print("\nTraining Time: " + trainTime + "\u00B1" + trainStd);
            ps.close();
            
   
        }
        
        
            
    }
    
}

