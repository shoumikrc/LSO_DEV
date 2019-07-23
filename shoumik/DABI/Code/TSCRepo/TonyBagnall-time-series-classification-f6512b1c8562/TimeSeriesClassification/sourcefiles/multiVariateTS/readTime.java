package multiVariateTS;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

public class readTime {
	
	public static double IG_Parent( int[] labels ){
		double value = 0;
		int pos = 0;
		int neg = 0;
		for(int i=0; i< labels.length;i++){
			if(labels[i] > 0){
				pos++;
			}else{
				neg ++;
			}
		}
		int n = labels.length;
		value = -(((double)pos/n)*(Math.log((double)pos/n)/Math.log(2))  + ((double)neg/n)*(Math.log((double) neg/n)/Math.log(2)) );
		
		return value;
	}
		
	
	public static double entropy_Parent_multclass( int[] labels, int [] Labels){
		double value = 0;
		int [] classCount = new int[Labels.length];
		for(int i=0; i< labels.length;i++){
			for(int k = 0;k < Labels.length; k++){
				if(labels[i] == (int) Labels[k]){
					classCount[k] ++;
				}
			}
		}
		int n = labels.length;

		for(int k=0; k< Labels.length; k++){
			value = value + (double)classCount[k]/n * (Math.log((double)classCount[k]/n)/Math.log(2));
		}
		value = -value;
		
		return value;
	}
	
	
	public static double IG_Child( ArrayList<Integer> child ){
		double value = 0;
		int pos = 0;
		int neg = 0;
		for(int i=0; i< child.size();i++){
			if(child.get(i)> 0){
				pos++;
			}else{
				neg ++;
			}
		}
		int n = child.size();
		
		if((double) pos/n == 0){
			value = -( ((double) neg/n)*(Math.log((double) neg/n)/Math.log(2))  );
		}else if((double) neg/n == 0){
			value = -(((double) pos/n)*(Math.log((double) pos/n)/Math.log(2)) );		
		}else{
			value = -(((double) pos/n)*(Math.log((double) pos/n)/Math.log(2))  + ((double) neg/n)*(Math.log((double) neg/n)/Math.log(2))  );
			
		}
		
		return value;
	}
	
	public static double entropy_child_multiclass( ArrayList<Integer> child, int [] Labels  ){
		double value = 0;
		
		int [] classCount = new int[Labels.length];
		for(int i=0; i< child.size();i++){
			for(int k =0 ; k< Labels.length; k++){
				if(child.get(i) ==  Labels[k]){
					classCount[k] ++;
				}
			}
		}
		
		int n = child.size();
		
		for(int k=0; k< Labels.length; k++){
			if(classCount[k] > 0){
				value = value + (double)classCount[k]/n * (Math.log((double)classCount[k]/n)/Math.log(2));
			}
			
		}
		value = -value;
		
		return value;
	}
	
	public static double InformationGain(int[] atts, int[] labels){
		double value = 0;
		ArrayList<Integer> left = new ArrayList<Integer>();
		ArrayList<Integer> right = new ArrayList<Integer>();
		for(int i=0; i< atts.length; i++){
			if(atts[i]>0){
				left.add(labels[i]);
			}else{
				right.add(labels[i]);
			}
		}
		if(left.size() > 0 & right.size() > 0){
			double ig_child = ((double) left.size())/labels.length *IG_Child(left) + ((double) right.size()/labels.length)*IG_Child(right);
			double ig_parent = IG_Parent(labels);
			
			value = ig_parent - ig_child;
		}else{
			value = 0;
		}
		
			
		return value;
	}
	


	
	public static HashMap<String, int[]> getOrders_percentile(List<List<double []>> startTime, int TotalNumShapelets, int InstancesNum, int dimension, double percentile, int[] labels, String place, int trial, int trainNum) throws IOException{
		
		String outfile = place + "trial_"+trial+"_shapelets_index.txt";
		FileOutputStream fos = new FileOutputStream(outfile);  
		PrintStream ps = new PrintStream(fos);
		
		///// concatenate the matrix into a giant TstartMartix
		double[][] TimeMatrix = new double[InstancesNum][TotalNumShapelets];
		int colIndex =0;		
		for(int dim=0; dim < dimension; dim++){
			int k = startTime.get(dim).size();  //number of shapelets
			for(int j=0; j< k; j++){
				ps.println(colIndex+"\tS_"+dim+"_"+j);
				for(int i=0; i<InstancesNum; i++ ){
					TimeMatrix[i][colIndex] = startTime.get(dim).get(j)[i];
				}
				colIndex = colIndex + 1;
			}
		}
		
		HashMap<String, int[]> orders = new HashMap<String, int[]>();  /// save order information
		ArrayList<Double> IG = new ArrayList<Double>();
		double[][] IG_before_values = new double[TotalNumShapelets][TotalNumShapelets];  //save information gain computed from train data
		double[][] IG_after_values = new double[TotalNumShapelets][TotalNumShapelets]; 
		int[] trainlabels = new int[trainNum];  // labels of training data
		
		for(int i=0; i< trainNum; i++){
			trainlabels[i] = labels[i];
		}
		
		
		// get information gain value from training data
		for(int i=0; i< TotalNumShapelets; i++){
			for(int j=i+1; j< TotalNumShapelets; j++){
				int[] before = new int[trainNum];
				int[] after = new int[trainNum];
				for(int n = 0; n <trainNum; n++){
					if( TimeMatrix[n][i] <= TimeMatrix[n][j] &  TimeMatrix[n][i] > 0  & TimeMatrix[n][j] > 0 ){  //shapelet i is before shapelet j
						before[n] = 1;
						after[n] = 0;
					}else if( TimeMatrix[n][i] > TimeMatrix[n][j] &  TimeMatrix[n][i] > 0  & TimeMatrix[n][j] > 0 ){
						before[n] = 0;
						after[n] = 1;
					}else{
						before[n] = 0;
						after[n] = 0;
					}
				}
				
				double informationGainValue = InformationGain(before, trainlabels); /// compute the entropy value
			    if(informationGainValue > 0){
			    	 IG.add(informationGainValue);
			    }
				IG_before_values[i][j] = informationGainValue;
				
				informationGainValue = InformationGain(after, trainlabels);
				if(informationGainValue > 0){
			    	 IG.add(informationGainValue);
			    }	
				IG_after_values[i][j] = informationGainValue;
			}
		}
		
		
		Collections.sort(IG); 
		System.out.println("continousDis ig size: "+IG.size());
		
		if(IG.size() > 1){
			double threshold = IG.get((int)(IG.size()*(1-percentile)));
//			System.out.println("continousDis ig threshold: "+threshold);
			
			
			for(int i=0; i< TotalNumShapelets; i++){
				for(int j=i+1; j< TotalNumShapelets; j++){
					
					if( i!=j){ 
						int[] before = new int[InstancesNum];
						int[] after = new int[InstancesNum];
						for(int n = 0; n <InstancesNum; n++){
							if( TimeMatrix[n][i] <= TimeMatrix[n][j] & TimeMatrix[n][i] > 0  & TimeMatrix[n][j] > 0 ){  //shapelet i is before shapelet j
								before[n] = 1;
								after[n] = 0;
							}else if( TimeMatrix[n][i] > TimeMatrix[n][j] & TimeMatrix[n][i] > 0  & TimeMatrix[n][j] > 0 ){
								before[n] = 0;
								after[n] = 1;
							}else{
								before[n] = 0;
								after[n] =  0;
							}
							if(before[n] + after[n] > 1){
								System.err.println("Two 1s: "+before[n]+"\t"+after[n]);
							}
						}
						
						if(IG_before_values[i][j] >= threshold ){
							String name = i +"<=" +j;
							orders.put(name, before);
						}else if(IG_after_values[i][j] >= threshold ){
							String name = i+">"+j;
							orders.put(name, after);
						}else if(IG_before_values[i][j] >= threshold & IG_after_values[i][j] >= threshold){
							System.err.println("Two rules can not be shown at the same time.");
						}
						
					}
				}
			}
			
//			return orders;
		}else{
			System.err.println("continuous order size: "+orders.size());
		}
		
		return orders;
		
		
		
	}


public static HashMap<String, int[]> getOrders_percentile_binaryDis(List<List<double []>> startTime, int TotalNumShapelets, int InstancesNum, int dimension, double percentile, int[] labels, String place, int trial, int trainNum, int[][] shapeletExist) throws IOException{
		
		String outfile = place + "trial_"+trial+"_shapelets_index.txt";
		FileOutputStream fos = new FileOutputStream(outfile);  
		PrintStream ps = new PrintStream(fos);
		
		///// concatenate the matrix into a giant TstartMartix
		double[][] TimeMatrix = new double[InstancesNum][TotalNumShapelets];
		int colIndex =0;		
		for(int dim=0; dim < dimension; dim++){
			int k = startTime.get(dim).size();  //number of shapelets
			for(int j=0; j< k; j++){
				ps.println(colIndex+"\tS_"+dim+"_"+j);
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
		
		HashMap<String, int[]> orders = new HashMap<String, int[]>();  /// save order information
		ArrayList<Double> IG = new ArrayList<Double>();
		double[][] IG_before_values = new double[TotalNumShapelets][TotalNumShapelets];  //save information gain computed from train data
		double[][] IG_after_values = new double[TotalNumShapelets][TotalNumShapelets];  //save information gain computed from train data
		int[] trainlabels = new int[trainNum];  // labels of training data
		
		for(int i=0; i< trainNum; i++){
			trainlabels[i] = labels[i];
		}
		
		
		// get information gain value from training data
		for(int i=0; i< TotalNumShapelets; i++){
			for(int j=i+1; j< TotalNumShapelets; j++){
				int[] before = new int[trainNum];
				int[] after = new int[trainNum];
				for(int n = 0; n <trainNum; n++){
					if( TimeMatrix[n][i] <= TimeMatrix[n][j] &  TimeMatrix[n][i] > 0  & TimeMatrix[n][j] > 0 ){  //shapelet i is before shapelet j
						before[n] = 1;
						after[n] = 0;
					}else if( TimeMatrix[n][i] > TimeMatrix[n][j] &  TimeMatrix[n][i] > 0  & TimeMatrix[n][j] > 0 ){
						before[n] = 0;
						after[n] = 1;
					}else{
						before[n] = 0;
						after[n] = 0;
					}
				}
				
				double informationGainValue = InformationGain(before, trainlabels); /// compute the entropy value
				if(informationGainValue > 0){
			    	 IG.add(informationGainValue);
			    }	
				IG_before_values[i][j] = informationGainValue;
				
				informationGainValue = InformationGain(after, trainlabels);
				if(informationGainValue > 0){
			    	 IG.add(informationGainValue);
			    }	
				IG_after_values[i][j] = informationGainValue;
			}
		}
		
		Collections.sort(IG);  
		System.out.println("binary ig size: "+IG.size());
		if(IG.size() > 1){
			double threshold = IG.get((int)(IG.size()*(1-percentile)));
//			System.out.println("binary ig threshold: "+threshold);
			
			for(int i=0; i< TotalNumShapelets; i++){
				for(int j=i+1; j< TotalNumShapelets; j++){
					
					if( i!=j){ 
						int[] before = new int[InstancesNum];
						int[] after = new int[InstancesNum];
						for(int n = 0; n <InstancesNum; n++){
							if( TimeMatrix[n][i] <= TimeMatrix[n][j] & TimeMatrix[n][i] > 0  & TimeMatrix[n][j] > 0 ){  //shapelet i is before shapelet j
								before[n] = 1;
								after[n] = 0;
							}else if( TimeMatrix[n][i] > TimeMatrix[n][j] & TimeMatrix[n][i] > 0  & TimeMatrix[n][j] > 0 ){
								before[n] = 0;
								after[n] = 1;
							}else{
								before[n] = 0;
								after[n] =  0;
							}
						}
						
						if(IG_before_values[i][j] >= threshold){
							String name = i +"<=" +j;
							orders.put(name, before);
						}else if(IG_after_values[i][j] >= threshold){
							String name = i+">"+j;
							orders.put(name, after);
						}else if(IG_before_values[i][j] >= threshold & IG_after_values[i][j] >= threshold){
							System.err.println("Two rules can not be shown at the same time.");
						}
						
					}
				}
			}
			
//			return orders;
		}else{
			System.err.println("binary order size: "+orders.size());
		}
		return orders;
		
	}

	
	
	
}
