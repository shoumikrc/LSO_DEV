package DataStructures;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.Arrays;
//import TimeSeries.SAXRepresentation;
import utilities.Logging;


public class MultivariateDataset
{
	// the multivariate time series 
	public double[][][] timeseries;
	// the labels of the time series 
	public int[] labels;
	// the number of labels 
	public int numLabels;
        public int[] classLabels;
	// the number of train and test series 
	public int numTrain, numTest;
	// the number of channels 
	public int numChannels;
	
	// the length of the shortest series
	public int minLength = Integer.MAX_VALUE;
	public int maxLength = Integer.MIN_VALUE;
	public int avgLength = 0; 
	
	// the set of labels and channels 
	Set<Integer> labelsSet = new HashSet<Integer>(); 
	Set<Integer> channelsSet = new HashSet<Integer>(); 
	
	public MultivariateDataset( )
	{
	}
	
	public MultivariateDataset(String train, String test, boolean normalize)
	{
		BufferedReader br = null;
		 
		try
		{
			String line = null;
			// series length
			int length = 0;
 
			//Count the number of instances and streams
			br = new BufferedReader(new FileReader(train));
 
			while ((line = br.readLine()) != null)
			{
				String[] splits = line.split(","); 
				
				numTrain = Integer.parseInt(splits[0]); 
				labelsSet.add( Integer.parseInt(splits[1]) );
				channelsSet.add( Integer.parseInt(splits[2]) );
				length = Integer.parseInt( splits[3] );  
				
				if(length < minLength )
					minLength = length;
				if(length > maxLength )
					maxLength = length;
				
				avgLength += length;
			}
			br.close();
			// the count is one more than index which starts at 0
			numTrain++; 
			
			br = new BufferedReader(new FileReader(test));
 
			numTest = 0;
			
			while ((line = br.readLine()) != null)
			{		
				String[] splits = line.split(","); 
				
				numTest = Integer.parseInt(splits[0]);
				labelsSet.add( Integer.parseInt(splits[1]) );
				channelsSet.add( Integer.parseInt(splits[2]) );
				length = Integer.parseInt( splits[3] ); 
				
				if(length < minLength )
					minLength = length;
				if(length > maxLength )
					maxLength = length;
				
				avgLength += length;
			} 
			br.close();
			
			// the count is one more than index which starts at 0
			numTest++;
			
			numLabels = labelsSet.size();
                        //classLabels = labelsSet.to
			numChannels = channelsSet.size();
			
			avgLength /= (numTrain+numTest)*numChannels; 
			
			//System.out.println("Dataset Info: numTrain=" + numTrain + ", numTest=" + numTest + 
					//", numLabels=" + numLabels+ ", numChannels=" + numChannels + 
					//", minLength=" + minLength + ", maxLength=" + maxLength + ", avgLength=" + avgLength );   
			
			// initialize the time series
			timeseries = new double[numTrain+numTest][numChannels][]; 
			labels = new int[numTrain+numTest];
                        classLabels = new int[numLabels];
                        int index = 0;

                        for( Integer i : labelsSet ) {
                                classLabels[index++] = i; //note the autounboxing here
                        }
			
			// read the training file
			int instanceIdx = 0, label = 0, channel = 0; 
			
			br = new BufferedReader(new FileReader(train));
			while ((line = br.readLine()) != null)
			{
				String[] splits = line.split(","); 
				
				instanceIdx = Integer.parseInt( splits[0] ); 
				label = Integer.parseInt( splits[1] ) ;
				channel = Integer.parseInt( splits[2] );
				length = Integer.parseInt( splits[3] );  
				
				// read the time series
				timeseries[instanceIdx][channel] = new double[length];				
				for(int len=0; len < length; len++)
					timeseries[instanceIdx][channel][len] = Double.parseDouble( splits[4 + len] ); 
				
				// set the labels
				labels[instanceIdx] = label;
			}
			br.close();
			
			// read the test file
			instanceIdx = 0;
			label = 0;
			channel = 0;
			length = 0;
			
			br = new BufferedReader(new FileReader(test));
			while ((line = br.readLine()) != null)
			{
				String[] splits = line.split(","); 
				
				instanceIdx = Integer.parseInt( splits[0] ); 
				label = Integer.parseInt( splits[1] ) ;
				channel = Integer.parseInt( splits[2] );
				length = Integer.parseInt( splits[3] );  
				// read the time series
				timeseries[numTrain + instanceIdx][channel] = new double[length];
				
				for(int len=0; len < length; len++)
					timeseries[numTrain + instanceIdx][channel][len] = Double.parseDouble( splits[4 + len] ); 
				
				// set the labels
				labels[numTrain + instanceIdx] = label; 
			}
			br.close(); 
			
		}
		catch (IOException e) 
		{
			e.printStackTrace();
		}
		finally
		{
			try
			{
				if (br != null)
					br.close();
			}
			catch (IOException ex)
			{
				ex.printStackTrace();
			}
		}
		
		if(normalize)
		{
			for(int i = 0; i < this.timeseries.length; i++)
				for(int channel = 0; channel < this.timeseries[i].length; channel++)
					this.timeseries[i][channel] = normalize( this.timeseries[i][channel] ); 
		} 
		
//		int[] length = this.getLength();
//		this.normalizedLength = new double[length.length];
//		double max = 0;
//		
//		for(int i = 0; i < this.numInstancesTrain; i++)
//		{
//			max = Math.max(max, length[i]);
//		}
//		for(int i = 0; i < length.length; i++)
//		{
//			this.normalizedLength[i] = (double) length[i] / max;
//		}
		
	}
	
	public int[] getLength()
	{
		int[] result = new int[this.numTest + this.numTrain];
		for(int i = 0; i < result.length; i++)
		{
			result[i] = this.timeseries[i][0].length;
		}
		return result;
	}
	
	/**
	 * Create multivariate subsequence from timeseries i starting at position j over all streams with length L.
	 * @param i Timeseries Index (1..N)
	 * @param j Starting position in T_i
	 * @param L Length of the subsequence
	 * @return normalize(T_{i,j:j+l-1})
	 */
	public double[][] subsequence(int i, int j, int L)
	{
		double[][] subsequence = new double[this.numChannels][L];
		
		for(int s = 0; s < this.numChannels; s++)
			subsequence[s] = Arrays.copyOfRange(this.timeseries[i][s], j, j + L);
		
		return subsequence;
	}
	
	/**
	 * Create normalized subsequence from timeseries i starting at position j in stream s with length L.
	 * @param i Timeseries Index (1..N)
	 * @param j Starting position in T_i
	 * @param s Stream index
	 * @param L Length of the subsequence
	 * @return normalize(T_{i,s,j:j+l-1})
	 */
	public double[] subsequence(int i, int j, int s, int L)
	{
		double[] subsequence = new double[L];
		
		for(int l = 0; l < L; l++)
			subsequence[l] = this.timeseries[i][s][j + l];
		
		return this.normalize(subsequence);
	}
	
	private double[] normalize(double[] series)
	{
		double normalizedSeries[] = new double[ series.length ]; 
		
		double mean = this.mean(series);  
		double sd = this.sd(series, mean); 

		if(sd == 0)
			sd = 1;

		for(int i = 0; i < series.length; i++)
			normalizedSeries[i] = (series[i] - mean) / sd;
                
                /*for(int h = 0;h<normalizedSeries.length;h++){
                    System.err.print(series[h] + ",");
                    System.err.print(normalizedSeries[h] + ",");
                }
                System.err.println();*/
		
		return normalizedSeries;
	}
	
	public double maxGlobalValue()
	{
		int lengths[] = getLength();
		double maxVal = Double.MIN_VALUE;
		
		for(int i = 0; i < numTrain+numTest; i++)
			for(int n = 0; n < numChannels; n++)
				for(int j = 0; j < lengths[i]; j++)
					if(  timeseries[i][n][j] > maxVal )
						maxVal =  timeseries[i][n][j];
		
		return maxVal;
	}
	
	public double minGlobalValue()
	{
		int lengths[] = getLength();
		double minVal = Double.MAX_VALUE;
		
		for(int i = 0; i < numTrain+numTest; i++)
			for(int n = 0; n < numChannels; n++)
				for(int j = 0; j < lengths[i]; j++)
					if( timeseries[i][n][j] < minVal )
						minVal = timeseries[i][n][j];
		
		return minVal;
	}
	
	public double meanGlobalValue()
	{
		int lengths[] = getLength();
		double meanVal = 0;
		double numVals = 0;
		
		for(int i = 0; i < numTrain+numTest; i++)
			for(int n = 0; n < numChannels; n++)
				for(int j = 0; j < lengths[i]; j++)
				{
					meanVal += timeseries[i][n][j];
					numVals++;
				}
		
		return meanVal / (double) numVals; 
	}
	
	public void removeGlobalMean()
	{
		int lengths[] = getLength();
		double meanVal = meanGlobalValue();
		
		for(int i = 0; i < numTrain+numTest; i++)
			for(int n = 0; n < numChannels; n++)
				for(int j = 0; j < lengths[i]; j++)
				{
					timeseries[i][n][j] -= meanVal;
				}
	}
	
	
	public void setMaxAbsoluteValueToOne()
	{
		int lengths[] = getLength();
		double maxVal = maxGlobalValue();
		
		for(int i = 0; i < numTrain+numTest; i++) 
			for(int n = 0; n < numChannels; n++)
				for(int j = 0; j < lengths[i]; j++)
						timeseries[i][n][j] /= maxVal;
	
	}
	
	public double mean(double[] series)
	{
		double mean = 0;
		for(int i = 0; i < series.length; i++)
			mean += series[i];
		return mean / series.length;
	}
	
	public double mean(int[] series)
	{
		double mean = 0;
		for(int i = 0; i < series.length; i++)
			mean += series[i];
		return (double)mean / (double)series.length;
	}
	
	public int min(int[] series)
	{
		int min = Integer.MAX_VALUE;
		for(int i = 0; i < series.length; i++)
			if( min > series[i])
				min = series[i];
		
		return min;
	}
	
	public int max(int[] series)
	{
		int max = Integer.MIN_VALUE;
		for(int i = 0; i < series.length; i++)
			if( max < series[i])
				max = series[i];
		
		return max;
	}
	
	public double sd(double[] series, double mean)
	{
		double sd = 0;
		for(int i = 0; i < series.length; i++)
			sd += Math.pow(series[i] - mean, 2);
		return Math.sqrt(sd / series.length);
	}
	
	
	public static void main(String [] args)
	{
		String train = "C:\\Users\\josif\\Downloads\\HMP_Dataset\\HMP_TRAIN";  
		String test = "C:\\Users\\josif\\Downloads\\HMP_Dataset\\HMP_TEST";  
		boolean normalize = false;
		
		MultivariateDataset mvd = new MultivariateDataset(train, test, normalize);
		
	}
}
