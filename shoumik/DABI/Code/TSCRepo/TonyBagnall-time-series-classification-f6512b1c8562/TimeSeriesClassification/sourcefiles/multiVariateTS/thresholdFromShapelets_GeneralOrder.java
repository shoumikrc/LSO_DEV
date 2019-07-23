package multiVariateTS;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

public class thresholdFromShapelets_GeneralOrder {
	
    public static double[] threshold(ArrayList<Double> values, double[] distance, int[] trainlabels){
		
		int times = 10;
		int gap = (int) trainlabels.length/times;
		double maxIG = -1;
		double chosenvalue = -1;
		
		
		double ig_parent = readTime.IG_Parent(trainlabels);
		for(int i=1; i< times; i++){
			double v = values.get(i*gap);
			ArrayList<Integer> left = new ArrayList<Integer>();
			ArrayList<Integer> right = new ArrayList<Integer>();
			for(int k=0; k< distance.length; k++){
				if(distance[k] <= v){
					left.add(trainlabels[k]);
				}else{
					right.add(trainlabels[k]);
				}
			}
			double ig_leftchild = ((double) left.size()/distance.length )*readTime.IG_Child(left) ;
			double ig_rightchild = ((double) right.size()/distance.length)*readTime.IG_Child(right); 
			double ig_child = ig_leftchild + ig_rightchild;
			double ig = ig_parent - ig_child;
            if(ig > maxIG){
				maxIG = ig;
				chosenvalue = v;
				
			}
			
		}
		
		double[] IG_thre = new double[2];
		IG_thre[0] = maxIG;   // IG value;
		IG_thre[1] = chosenvalue;  // dis threshold
		return IG_thre;
	}
    
    public static double[] threshold_order(ArrayList<Double> values, double[] distance, int[] trainlabels, int [] Labels){
		
		int times = 10;
		double gap = (double) trainlabels.length/times;
		double maxIG = -1;
		double chosenvalue = -1;
		int leftchild = -1;
                
        double countMax = -100;  
        double selectedLabel = -100;
                
		
//		double ig_parent = readTime.IG_Parent(trainlabels);
		double ig_parent = readTime.entropy_Parent_multclass(trainlabels, Labels);
		for(int i=1; i< times; i++){
            double [] countLabels = new double[Labels.length];      
			double v = values.get((int)(i*gap));
			ArrayList<Integer> left = new ArrayList<Integer>();
			ArrayList<Integer> right = new ArrayList<Integer>();
                        
			for(int k=0; k< distance.length; k++){
				if(distance[k] <= v){
					left.add(trainlabels[k]);
                                        
				}else{
					right.add(trainlabels[k]);                                       
				}
                                                               
			}
//            double entrop_leftchild = readTime.IG_Child(left);
//            double entrop_rightchild = readTime.IG_Child(right);
            
            double entrop_leftchild = readTime.entropy_child_multiclass(left, Labels);
            double entrop_rightchild = readTime.entropy_child_multiclass(right, Labels);
            
			double ig_leftchild = ((double) left.size())/distance.length *entrop_leftchild ;
			double ig_rightchild = ((double) right.size()/distance.length)*entrop_rightchild; 
			double ig_child = ig_leftchild + ig_rightchild;
            double ig = ig_parent - ig_child;
			if(ig > maxIG){
                                
				maxIG = ig;
				chosenvalue = v;
				if(entrop_leftchild - entrop_rightchild  < 0){
					leftchild = 1;
                    for (int l = 0;l<left.size();l++){
                        for (int k =0;k<Labels.length;k++){
                            if(left.get(l) == Labels[k]){
                                countLabels[k]+=1;
                            }
                        }
                    } 
				}else{
					leftchild = 0;
                    for (int l = 0;l<right.size();l++){
                        for (int k =0;k<Labels.length;k++){
                            if(right.get(l) == Labels[k]){
                                countLabels[k]+=1;
                            }
                        }
                    } 
				}
				
				countMax = countLabels[0];
				selectedLabel = Labels[0];
                for (int k = 1;k<Labels.length;k++){
                        if(countLabels[k] > countMax){
                        	selectedLabel = Labels[k];
                        	countMax = countLabels[k];
                        }
                }
			}
		}
		
		double[] IG_thre = new double[4];
		IG_thre[0] = maxIG;   // IG value;
		IG_thre[1] = chosenvalue;  // dis threshold
		IG_thre[2] = leftchild;
        IG_thre[3] = selectedLabel;//class of the order
		return IG_thre;
	}
    
    
    public static double[] threshold_order_position( double[] TimePosition, int[] trainlabels, int [] Labels){
		
		int times = 2;
		double[] gaps = new double[2];
		gaps[0] = -1; gaps[1] = 0; 
		double maxIG = -1;
		double chosenvalue = -1;
		int leftchild = -1;
                
                double countMax = -100;  
                double selectedLabel = -100;
                
		
//		double ig_parent = readTime.IG_Parent(trainlabels);
		double ig_parent = readTime.entropy_Parent_multclass(trainlabels, Labels);
		for(int i=0; i< times; i++){
                double [] countLabels = new double[Labels.length];      
			double v = gaps[i];
			ArrayList<Integer> left = new ArrayList<Integer>();
			ArrayList<Integer> right = new ArrayList<Integer>();
                        
			for(int k=0; k< TimePosition.length; k++){
				if(TimePosition[k] <= v){
					left.add(trainlabels[k]);
                                        
				}else{
					right.add(trainlabels[k]);                                       
				}
                                                               
			}
//            double entrop_leftchild = readTime.IG_Child(left);
//            double entrop_rightchild = readTime.IG_Child(right);
            
            double entrop_leftchild = readTime.entropy_child_multiclass(left, Labels);
            double entrop_rightchild = readTime.entropy_child_multiclass(right, Labels);
            
			double ig_leftchild = ((double) left.size())/TimePosition.length *entrop_leftchild ;
			double ig_rightchild = ((double) right.size()/TimePosition.length)*entrop_rightchild; 
			double ig_child = ig_leftchild + ig_rightchild;
            double ig = ig_parent - ig_child;
			if(ig > maxIG){
                                
				maxIG = ig;
				chosenvalue = v;
				if(entrop_leftchild - entrop_rightchild  < 0){
					leftchild = 1;
                    for (int l = 0;l<left.size();l++){
                        for (int k =0;k<Labels.length;k++){
                            if(left.get(l) == Labels[k]){
                                countLabels[k]+=1;
                            }
                        }
                    } 
				}else{
					leftchild = 0;
                    for (int l = 0;l<right.size();l++){
                        for (int k =0;k<Labels.length;k++){
                            if(right.get(l) == Labels[k]){
                                countLabels[k]+=1;
                            }
                        }
                    } 
				}
				
				countMax = countLabels[0];
				selectedLabel = Labels[0];
                for (int k = 1;k<Labels.length;k++){
                        if(countLabels[k] > countMax){
                        	selectedLabel = Labels[k];
                        	countMax = countLabels[k];
                        }
                }
			}
		}
		
		double[] IG_thre = new double[4];
		IG_thre[0] = maxIG;   // IG value;
		IG_thre[1] = chosenvalue;  // dis threshold
		IG_thre[2] = leftchild;
                IG_thre[3] = selectedLabel;//class of the order
		return IG_thre;
	}
	
	/*public static HashMap<String, int[]> getGelOrders_threshold_binaryDis(List<List<double []>> startTime, int TotalNumShapelets, int InstancesNum, int dimension, double orderthreshold, int[] labels, String place, int trial, int trainNum, int[][] shapeletExist) throws IOException{
		
//		String outfile = place + "trial_"+trial+"_shapelets_index.txt";
//		FileOutputStream fos = new FileOutputStream(outfile);  
//		PrintStream ps = new PrintStream(fos);
		
		///// concatenate the matrix into a giant TstartMartix
		double[][] TimeMatrix = new double[InstancesNum][TotalNumShapelets];
		int colIndex =0;		
		for(int dim=0; dim < dimension; dim++){
			int k = startTime.get(dim).size();  //number of shapelets
			for(int j=0; j< k; j++){
//				ps.println(colIndex+"\tS_"+dim+"_"+j);
				for(int i=0; i<InstancesNum; i++ ){
					if(shapeletExist[i][colIndex] == 1){
						TimeMatrix[i][colIndex] = startTime.get(dim).get(j)[i];
					}else{
						TimeMatrix[i][colIndex] = -1;  /// shapelet does not exist
					}
					
				}
				colIndex = colIndex + 1;
			}
		}
//		ps.close();
		
		HashMap<String, int[]> orders = new HashMap<String, int[]>();  /// save order information
		int[] trainlabels = new int[trainNum];  // labels of training data
		
		for(int i=0; i< trainNum; i++){
			trainlabels[i] = labels[i];
		}
                
                  
		
		
		// get information gain value from training data
		for(int i=0; i< TotalNumShapelets; i++){
			for(int j=i+1; j< TotalNumShapelets; j++){
				double[] timeDiff = new double[trainNum];
				ArrayList<Double> tdiff = new ArrayList<Double>();
				for(int n = 0; n <trainNum; n++){
					if(TimeMatrix[n][i] > 0 &&  TimeMatrix[n][j] > 0){
						timeDiff[n] = TimeMatrix[n][i] - TimeMatrix[n][j];
						tdiff.add(timeDiff[n]);
					}else{
						timeDiff[n] = Double.MAX_VALUE;
						tdiff.add(timeDiff[n]);
					}
				}
				
				Collections.sort(tdiff); 
				double ig_order = threshold_order(tdiff, timeDiff, trainlabels)[0];
				double time_value = threshold_order(tdiff, timeDiff, trainlabels)[1];
				double leftchild = threshold_order(tdiff, timeDiff, trainlabels)[2];
						
				if(ig_order >= orderthreshold){
					int[] value = new int[InstancesNum];
					if(leftchild == 1){
						for(int n = 0; n <InstancesNum; n++){
							if( TimeMatrix[n][i] > 0 &&  TimeMatrix[n][j] > 0 && TimeMatrix[n][i] - TimeMatrix[n][j] <= time_value){
								value[n] = 1;
							}else{
								value[n] = 0;
							}
						}
						String name = "S"+i+"-"+"S"+j+"<="+time_value;
						orders.put(name, value);
					}else{
						for(int n = 0; n <InstancesNum; n++){
							if( TimeMatrix[n][i] > 0 &&  TimeMatrix[n][j] > 0 && TimeMatrix[n][i] - TimeMatrix[n][j] > time_value){
								value[n] = 1;
							}else{
								value[n] = 0;
							}
						}
						String name = "S"+i+"-"+"S"+j+">"+time_value;
						orders.put(name, value);
					}
					
				}
				
				
			}
		}
	
		if(orders.size() > 0){
			System.err.println("trail "+trial+ " has general binary Orders");
		}
		
		return orders;
		
	}*/
	
	/*public static HashMap<String, int[]> getGelOrders_thred_continuous(List<List<double []>> startTime, int TotalNumShapelets, int InstancesNum, int dimension, double orderthreshold, int[] labels, String place, int trial, int trainNum) throws IOException{
		
//		String outfile = place + "trial_"+trial+"_shapelets_index.txt";
//		FileOutputStream fos = new FileOutputStream(outfile);  
//		PrintStream ps = new PrintStream(fos);
		
		///// concatenate the matrix into a giant TstartMartix
		double[][] TimeMatrix = new double[InstancesNum][TotalNumShapelets];
		int colIndex =0;		
		for(int dim=0; dim < dimension; dim++){
			int k = startTime.get(dim).size();  //number of shapelets
			for(int j=0; j< k; j++){
//				ps.println(colIndex+"\tS_"+dim+"_"+j);
				for(int i=0; i<InstancesNum; i++ ){
					TimeMatrix[i][colIndex] = startTime.get(dim).get(j)[i];
				}
				colIndex = colIndex + 1;
			}
		}
//		ps.close();
		
		HashMap<String, int[]> orders = new HashMap<String, int[]>();  /// save order information
		int[] trainlabels = new int[trainNum];  // labels of training data
		
		for(int i=0; i< trainNum; i++){
			trainlabels[i] = labels[i];
		}
		
		// get information gain value from training data
		for(int i=0; i< TotalNumShapelets; i++){
			for(int j=i+1; j< TotalNumShapelets; j++){
				double[] timeDiff = new double[trainNum];
				ArrayList<Double> tdiff = new ArrayList<Double>();
				for(int n = 0; n <trainNum; n++){
					timeDiff[n] = TimeMatrix[n][i] - TimeMatrix[n][j];
					tdiff.add(timeDiff[n]);
				}
				
				Collections.sort(tdiff); 
				double ig_order = threshold_order(tdiff, timeDiff, trainlabels)[0];
				double time_value = threshold_order(tdiff, timeDiff, trainlabels)[1];
				double leftchild = threshold_order(tdiff, timeDiff, trainlabels)[2];
						
				if(ig_order >= orderthreshold){
					if(leftchild == 1){
						int[] value = new int[InstancesNum];
						for(int n = 0; n <InstancesNum; n++){
							if(TimeMatrix[n][i] - TimeMatrix[n][j] <= time_value){
								value[n] = 1;
							}else{
								value[n] = 0;
							}
						}
						String name = "S"+i+"-"+"S"+j+"<="+time_value;
						orders.put(name, value);
					}else{
						int[] value = new int[InstancesNum];
						for(int n = 0; n <InstancesNum; n++){
							if(TimeMatrix[n][i] - TimeMatrix[n][j] > time_value){
								value[n] = 1;
							}else{
								value[n] = 0;
							}
						}
						String name = "S"+i+"-"+"S"+j+">"+time_value;
						orders.put(name, value);
					}
					
				}
				
				
				
			}
		}
		if(orders.size() > 0){
			System.err.println("trial "+trial+ " has General continuous Orders");
		}
		
		return orders;
		
		
		
	}*/

	/*public static void FindThresholds_generalOrder (List<List<double []>> DistancesMatrix, List<List<double []>> startTime, int TotalNumShapelets, int InstancesNum,int trainNum, int dimensions, int[] labels, String place, int trial, int TSlength) throws Exception{
		
		int[][] exist = new int [InstancesNum][TotalNumShapelets];
		double[] thresholdShapelet = new double[TotalNumShapelets];  // save the threshold for each shapelet
		ArrayList<Double> igs = new ArrayList<Double>();  // save the shapelet threshold
		
		double[][]  oridisMatrix = new double[InstancesNum][TotalNumShapelets];
		int colIndex = 0;
		//// add shapelet distance infor into the disMatrix 
		for(int dim = 0; dim < dimensions; dim++ ){
			int k = DistancesMatrix.get(dim).size();  //number of shapelets
//			System.out.println("dimensions: "+dimensions);
//			System.out.println("k: "+k);
//			System.out.println("dim: "+dim+" shapletnum: "+DistancesMatrix.get(dim).size()+" instancenum: "+DistancesMatrix.get(dim).get(0).length);
			for(int j=0; j< k; j++){
				for(int i=0; i<InstancesNum; i++ ){
					double distance =  DistancesMatrix.get(dim).get(j)[i];
//					System.err.println("colIndex: "+colIndex);
					oridisMatrix[i][colIndex] = distance;
				}
				colIndex = colIndex+1;
			}
			
		}
		
		/// normalize distance 
		double[][] disMatrix = new double[InstancesNum][TotalNumShapelets];
		disMatrix= normalization(oridisMatrix, trainNum, InstancesNum, TotalNumShapelets);
		
                //double[][] disMatrix = oridisMatrix;
                
		///// calculate the threshold for each shapelet
		int[] trainlabels = new int[trainNum];
		for(int i=0; i< trainNum; i++){
			trainlabels[i] = labels[i];
		}
		
		for(int i =0; i< TotalNumShapelets; i++ ){
			ArrayList<Double> values = new ArrayList<Double>();
			double[] distance = new double[trainNum];
			for(int n=0; n< trainNum; n++){
				values.add(disMatrix[n][i]);
				distance[n] = disMatrix[n][i];
			}
			Collections.sort(values);  // sort the distance from small to large
			thresholdShapelet[i] = threshold(values, distance, trainlabels)[1];
			igs.add(threshold(values, distance, trainlabels)[0]);
//			System.out.println("dis threshold "+i+"\t"+thresholdShapelet[i]);
//			System.out.println("ig values: "+threshold(values, distance, trainlabels)[0]);
		}
		
		///// record whether the shapelet exist in the instance or not, and save in the exist matrix
		for (int i=0; i< TotalNumShapelets; i++){
			double thred = thresholdShapelet[i];
			for(int n=0; n< InstancesNum; n++){
				if(disMatrix[n][i] <= thred){
					exist[n][i] =1;
				}else{
					exist[n][i] = 0;
				}
			}
		}
		
		/////// sort the ig values
           
		Collections.sort(igs);
		double threOrder;
		if(igs.size() > 1){
            threOrder = igs.get(igs.size()-1);
        }else{
            threOrder = igs.get(igs.size()-1);
        }
        
                
		System.err.println("general threOrder "+threOrder);
		
		/// binary order 
		HashMap<String, int[]> orderS_binaryDis = new HashMap<String, int[]>();
		//orderS_binaryDis = getGelOrders_threshold_binaryDis( startTime, TotalNumShapelets, InstancesNum, dimensions, threOrder, labels,  place, trial, trainNum, exist);
		concatenateAtts_binaryDis(orderS_binaryDis,  exist, TotalNumShapelets, InstancesNum,  dimensions,  trial, labels, place);
		
		
		/// continuous distance + order
		 HashMap<String, int[]> orderS_continuous = new HashMap<String, int[]>();
		 //orderS_continuous = getGelOrders_thred_continuous(startTime,  TotalNumShapelets,  InstancesNum,  dimensions, threOrder, labels, place,trial, trainNum);
		 concatenateAtts_continuousDis(orderS_continuous, disMatrix, TotalNumShapelets, InstancesNum, dimensions, trial,labels, place);
			
		 
		 HashMap<String, double[]> continuousOrders = new HashMap<String, double[]>();
		 continuousOrders = getContinuousOrders(startTime,  TotalNumShapelets,  InstancesNum,  dimensions, threOrder, labels, place,trial, trainNum, TSlength);
		 concatenateAtts_continuousOrders(continuousOrders, disMatrix, TotalNumShapelets, InstancesNum, dimensions, trial,labels, place);

	}

	private static void concatenateAtts_continuousOrders(HashMap<String, double[]> orders,
			double[][] disMatrix, int TotalNumShapelets, int InstancesNum, int dimensions, int trial, int[] labels, String place) throws IOException {
		int orderNum = orders.keySet().size();
		int attNum = TotalNumShapelets + orderNum;
		double[][] atts =  new double[InstancesNum][attNum];
		

		String outfile = place + "trial_"+trial+"_attributeGeneralNames_ContinousDis_ContinuousOrders.txt";
		FileOutputStream fos = new FileOutputStream(outfile);  
		PrintStream ps = new PrintStream(fos);
		
		int colIndex = 0;	
		for(colIndex = 0; colIndex< TotalNumShapelets; colIndex++){
			ps.println("S_"+colIndex);
			for(int i=0; i<InstancesNum; i++){
				atts[i][colIndex] = disMatrix[i][colIndex];
			}
		}
		
		/// add order infor into the attribute matrix
		Iterator iterator = orders.keySet().iterator();

		while (iterator.hasNext()) {
		   String key = iterator.next().toString();
		   double[] value = orders.get(key);

		   ps.println(key);
		   for(int i=0; i<value.length; i++){
			   atts[i][colIndex] = value[i];
		   }
		   colIndex = colIndex+1;
		}
		ps.close();
		
		// save shapelets and orders into file
		outfile = place + "trial_"+trial+"_attr_shapelts_Generalorders_Matrix_ContinousDis_ContinuousOrders.txt";
		fos = new FileOutputStream(outfile);  
		ps = new PrintStream(fos);
		
		for(int i=0; i< InstancesNum; i++ ){
			ps.print(labels[i]);
			for( int j = 0; j< attNum; j++){
				ps.print(","+atts[i][j]);
			}
			ps.println();
		}
		ps.close();	
		
	}

	private static HashMap<String, double[]> getContinuousOrders(List<List<double[]>> startTime, int TotalNumShapelets,
			int InstancesNum, int dimension, double threOrder, int[] labels, String place, int trial, int trainNum,
			int TSlength) {
		
		double[][] TimeMatrix = new double[InstancesNum][TotalNumShapelets];
		int colIndex =0;		
		for(int dim=0; dim < dimension; dim++){
			int k = startTime.get(dim).size();  //number of shapelets
			for(int j=0; j< k; j++){
//				ps.println(colIndex+"\tS_"+dim+"_"+j);
				for(int i=0; i<InstancesNum; i++ ){
					TimeMatrix[i][colIndex] = startTime.get(dim).get(j)[i];
				}
				colIndex = colIndex + 1;
			}
		}
//		ps.close();
		
		HashMap<String, double[]> orders = new HashMap<String, double[]>();  /// save order information
		int[] trainlabels = new int[trainNum];  // labels of training data
		
		for(int i=0; i< trainNum; i++){
			trainlabels[i] = labels[i];
		}
		
		// get information gain value from training data
		for(int i=0; i< TotalNumShapelets; i++){
			for(int j=i+1; j< TotalNumShapelets; j++){
				double[] timeDiff = new double[trainNum];
				ArrayList<Double> tdiff = new ArrayList<Double>();
				for(int n = 0; n <trainNum; n++){
					timeDiff[n] = (double) (TimeMatrix[n][i] - TimeMatrix[n][j])/(TSlength*2);
					tdiff.add(timeDiff[n]);
				}
				
				Collections.sort(tdiff); 
				//double ig_order = threshold_order(tdiff, timeDiff, trainlabels)[0];
				//double time_value = threshold_order(tdiff, timeDiff, trainlabels)[1];
				
				System.err.println(ig_order);
						
				if(ig_order >= threOrder){

					double[] value = new double[InstancesNum];
					for(int n = 0; n <InstancesNum; n++){
						value[n] = (double) (TimeMatrix[n][i] - TimeMatrix[n][j])/(TSlength*2);
					}
					String name = "S"+i+"-"+"S"+j+"<="+time_value;
					orders.put(name, value);
				}
				
				
				
			}
		}
		if(orders.size() > 0){
			System.err.println("trail "+trial+ " has  continuous Orders");
		}
		
		return orders;
		
	}

	public static void concatenateAtts_continuousDis(HashMap<String, int[]> orders, double[][] DistancesMatrix, int TotalNumShapelets, int InstancesNum, int dimensions, int trial, int[] labels, String place) throws IOException{
		
		int orderNum = orders.keySet().size();
		int attNum = TotalNumShapelets + orderNum;
		double[][] atts =  new double[InstancesNum][attNum];
		

		String outfile = place + "trial_"+trial+"_attributeGeneralNames_ContinousDis.txt";
		FileOutputStream fos = new FileOutputStream(outfile);  
		PrintStream ps = new PrintStream(fos);
		
		int colIndex = 0;	
		for(colIndex = 0; colIndex< TotalNumShapelets; colIndex++){
			ps.println("S_"+colIndex);
			for(int i=0; i<InstancesNum; i++){
				atts[i][colIndex] = DistancesMatrix[i][colIndex];
			}
		}
		
		/// add order infor into the attribute matrix
		Iterator iterator = orders.keySet().iterator();

		while (iterator.hasNext()) {
		   String key = iterator.next().toString();
		   int[] value = orders.get(key);

		   ps.println(key);
		   for(int i=0; i<value.length; i++){
			   atts[i][colIndex] = value[i];
		   }
		   colIndex = colIndex+1;
		}
		ps.close();
		
		// save shapelets and orders into file
		outfile = place + "trial_"+trial+"_attr_shapelts_Generalorders_Matrix_ContinousDis.txt";
		fos = new FileOutputStream(outfile);  
		ps = new PrintStream(fos);
		
		for(int i=0; i< InstancesNum; i++ ){
			ps.print(labels[i]);
			for( int j = 0; j< attNum; j++){
				ps.print(","+atts[i][j]);
			}
			ps.println();
		}
		ps.close();	
		
	}

	public static void concatenateAtts_binaryDis(HashMap<String, int[]> orders,  int[][] ShapeletExistMatrix, int TotalNumShapelets, int InstancesNum, int dimensions, int trial, int[] labels, String place) throws IOException{
		
		int orderNum = orders.keySet().size();
		int attNum = TotalNumShapelets + orderNum;
		int[][] atts =  new int[InstancesNum][attNum];
		

		String outfile = place + "trial_"+trial+"_attributeGeneralNames_binaryDis.txt";
		FileOutputStream fos = new FileOutputStream(outfile);  
		PrintStream ps = new PrintStream(fos);
		
		int colIndex = 0;
		//// add shapelet existence infor into the attributes matrix
		for(int j=0; j < TotalNumShapelets; j++ ){
			for(int i=0; i< InstancesNum; i++){
				atts[i][j] = ShapeletExistMatrix [i][j];
			}
			colIndex = colIndex+1;
			ps.println("S_"+j);
		}
		
		/// add order infor into the attribute matrix
		Iterator iterator = orders.keySet().iterator();

		while (iterator.hasNext()) {
		   String key = iterator.next().toString();
		   int[] value = orders.get(key);

		   ps.println(key);
		   for(int i=0; i<value.length; i++){
			   atts[i][colIndex] = value[i];
		   }
		   colIndex = colIndex+1;
		}
		ps.close();
		
		// save shapelets and orders into file
		outfile = place + "trial_"+trial+"_attr_shapelts_Generalorders_Matrix_binaryDis.txt";
		fos = new FileOutputStream(outfile);  
		ps = new PrintStream(fos);
		
		for(int i=0; i< InstancesNum; i++ ){
			ps.print(labels[i]);
			for( int j = 0; j< attNum; j++){
				ps.print(","+atts[i][j]);
			}
			ps.println();
		}
		ps.close();
		
	}

	public static double[][] normalization(double[][] oridisMatrix, int trainNum, int instancesNum, int totalNumShapelets) {
		double[][] normed = new double[instancesNum][totalNumShapelets];
		for(int k = 0; k< totalNumShapelets; k++){
			double min = Double.MAX_VALUE;
			double max = Double.MIN_VALUE;
			for(int i=0; i<trainNum; i++ ){
				if(oridisMatrix[i][k] < min){
					min = oridisMatrix[i][k];
				}
				if(oridisMatrix[i][k] > max ){
					max = oridisMatrix[i][k];
				}
			}
			
			for(int i=0; i< instancesNum; i++){
				double value = (oridisMatrix[i][k] - min)/(max - min);
				if(value > 1){
					value = 1;
				}
				if(value < 0){
					value = 0;
				}
				normed[i][k] = value;
			}
			
		}
		
		return normed;
	}*/
	
}
