/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package TimeSeries;

import DataStructures.DataInstance;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import utilities.Logging;
import utilities.Logging.LogLevel;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;

/**
 *
 * @author Josif Grabocka
 */
public class DTW extends DistanceOperator
{
    public static DTW instance = null;
    
    public String dsName = "";
    
    private DTW()
    {
        
    }
    
    public static DTW getInstance()
    {
        if(instance == null)
            instance = new DTW();
        
        return instance;
    }
    
    /** the warping window of DTW */
    public static int m_warpingwindow = Integer.MAX_VALUE;

    /*
     * Calculate the distance between two time instances
     */
    @Override
    public double CalculateDistance(DataInstance ts1, DataInstance ts2) 
    {
        
        double warpingWindow = getWarpingWindow(dsName);
        m_warpingwindow = (int)Math.ceil(ts1.features.size() * warpingWindow);
        
        int m = ts1.features.size();
        int n = ts2.features.size();

        double df[][] = new double[m][n];

        df[0][0] = ts1.features.get(0).distanceSquare(ts2.features.get(0));
        for (int i = 1; i < m; i++) 
        {
                df[i][0] = 
                        df[i-1][0] + ts1.features.get(i).distanceSquare(ts2.features.get(0));
        }
        for (int j = 1; j < n; j++) 
        {
                df[0][j] = 
                        df[0][j-1] + ts1.features.get(0).distanceSquare(ts2.features.get(j));
        }

        for (int i = 1; i < m; i++) 
        {
                for (int j = 1; j < n; j++) 
                {
                        if (((i < j) && (j-i) <= m_warpingwindow) || 
                                        ((i >= j) && (i-j) <= m_warpingwindow)) 
                        {
                                double min1 = Math.min(df[i][j-1], df[i-1][j]);
                                double min2 = Math.min(min1, df[i-1][j-1]);
                                df[i][j] = min2 + ts1.features.get(i).distanceSquare(ts2.features.get(j));
                        }
                        else 
                        {
                               df[i][j] = Double.MAX_VALUE;
                        }
                }
        }
        
        return df[m - 1][n - 1];
    }
    
    private double Cost( double val1, double val2 )
    {
        //return Math.abs(val1-val2);
        
        return (val1-val2)*(val1-val2);
    }
    
    
    // calculate the warping path of two given time series
    public double CalculateDistance(List<Double> ts1, List<Double> ts2) 
    {
        int n = ts1.size(), m = ts2.size();
        
        // initialize a warping path
        List<List<Integer>> warpingPath = new ArrayList<List<Integer>>();
        for(int i = 0; i < n; i++)
            warpingPath.add(new ArrayList<Integer>()); 
        
        double [][] costMatrix = new double[n][m]; 
        // initialize first cost cell
        costMatrix[0][0] = Cost(ts1.get(0), ts2.get(0));
        
        // compute distances in first row
        for(int i = 1; i < n; i++)
            costMatrix[i][0] = costMatrix[i-1][0] + Cost(ts1.get(i), ts2.get(0));
        // pre compute distances in first column
        for(int j = 1; j < m; j++)
            costMatrix[0][j] = costMatrix[0][j-1] + Cost(ts1.get(0), ts2.get(j));
        
        // compute the cost matrix
        for(int i = 1; i < n; i++)
        {
            for(int j = 1; j < m; j++)
            {
                double min = Math.min(costMatrix[i-1][j], costMatrix[i][j-1]);
                min = Math.min(costMatrix[i-1][j-1], min);
                
                costMatrix[i][j] = min + Cost( ts1.get(i), ts2.get(j) );
            }
        }
        
        return costMatrix[n-1][m-1];
    }
    
    // calculate the warping path of two given time series
    public double CalculateDistance(double [] ts1, double [] ts2) 
    {
        int n = ts1.length, m = ts2.length;
        
        // initialize a warping path
        //List<List<Integer>> warpingPath = new ArrayList<List<Integer>>();
        //for(int i = 0; i < n; i++)
        //    warpingPath.add(new ArrayList<Integer>()); 
        
        double [][] costMatrix = new double[n][m]; 
        // initialize first cost cell
        costMatrix[0][0] = Cost(ts1[0], ts2[0]);
        
        // compute distances in first row
        for(int i = 1; i < n; i++)
            costMatrix[i][0] = costMatrix[i-1][0] + Cost(ts1[i], ts2[0]);
        // pre compute distances in first column
        for(int j = 1; j < m; j++)
            costMatrix[0][j] = costMatrix[0][j-1] + Cost(ts1[0], ts2[j]);
        
        // compute the cost matrix
        for(int i = 1; i < n; i++)
        {
            for(int j = 1; j < m; j++)
            {
                double min = Math.min(costMatrix[i-1][j], costMatrix[i][j-1]);
                min = Math.min(costMatrix[i-1][j-1], min);
                
                costMatrix[i][j] = min + Cost( ts1[i], ts2[j] );
            }
        }
         
        return costMatrix[n-1][m-1];
    }
    
    
    public double CalculateDistance(Instance ts1, Instance ts2) 
    {
        return CalculateDistance(
                new DataInstance("", ts1), 
                new DataInstance("", ts2));
    }
    
    // calculate the warping path of two given time series
    public List<List<Integer>> CalculateWarpingPath(List<Double> ts1, List<Double> ts2) 
    {
        // initialize a warping path
        List<List<Integer>> warpingPath = new ArrayList<List<Integer>>();
        for(int i = 0; i < ts1.size(); i++)
            warpingPath.add(new ArrayList<Integer>()); 
        
        double [][] costMatrix = new double[ts1.size()][ts2.size()]; 
        
        for(int i = 0; i < ts1.size(); i++)
             for(int j = 0; j < ts2.size(); j++)
            	 costMatrix[i][j] = Double.MAX_VALUE;
        

        // initialize first cost cell
        costMatrix[0][0] = Cost(ts1.get(0), ts2.get(0));
        
        // compute distances in first row
        for(int i = 1; i < ts1.size(); i++)
            costMatrix[i][0] = costMatrix[i-1][0] + Cost(ts1.get(i), ts2.get(0));
        // pre compute distances in first column
        for(int j = 1; j < ts2.size(); j++)
            costMatrix[0][j] = costMatrix[0][j-1] + Cost(ts1.get(0), ts2.get(j));
        
        // compute the cost matrix
        for(int i = 1; i < ts1.size(); i++)
        {
            for(int j = 1; j < ts2.size(); j++)
            {
                double min = Math.min(costMatrix[i-1][j], costMatrix[i][j-1]);
                min = Math.min(costMatrix[i-1][j-1], min);
                
                costMatrix[i][j] = min + Cost( ts1.get(i), ts2.get(j) );
            }
        }
        
        
        int i = ts1.size()-1, j = ts2.size()-1; 
        
        // build the warping path 
        while(i >= 0 && j >= 0 )
        {
        	// add the current warping path indices
        	warpingPath.get(i).add(j);
        	
        	// check if we reached the 
        	if(i == 0 && j == 0)
        	{
        		break;
        	}
        	else if( i == 0)
        	{
        		j--;
        	}
        	else if( j == 0 )
        	{
        		i--;
        	}
        	else
        	{
	        	// get the distances to the neighbour cells forward
	            double d_left = costMatrix[i-1][j],
	                d_down = costMatrix[i][j-1],
	                d_downLeft = costMatrix[i-1][j-1];
	        
	            // add the one having the smallest distance to the warping path
	            if( d_downLeft <= d_down && d_downLeft <= d_left )
	            {
	                i--;
	                j--;
	            }
                else if( d_left <= d_down && d_left <= d_downLeft )
	            {
	                i--;
	            }
	            else if( d_down <= d_left && d_down <= d_downLeft )
	            {
	                j--;
	            }
	            else
	            {
	            	Logging.println("CalculateWarpingPath:: No warping between: " + i + " " + j, LogLevel.ERROR_LOG);
	            	Logging.println("dl: " + d_downLeft + " d: " + d_down + " l: " + d_left, LogLevel.ERROR_LOG);
	            	
	            }
        	}
        }
        
        return warpingPath;
    }

    // calculate the warping path of two given time series
    public List<List<Integer>> CalculateWarpingPath(List<Double> ts1, List<Double> ts2, int warpingWindow) 
    {
    	
    	if( warpingWindow < 0)
    		return CalculateWarpingPath(ts1, ts2);
    	
        // initialize a warping path
        List<List<Integer>> warpingPath = new ArrayList<List<Integer>>();
        for(int i = 0; i < ts1.size(); i++)
            warpingPath.add(new ArrayList<Integer>()); 
        
        double [][] costMatrix = new double[ts1.size()][ts2.size()]; 
        
        // set cost matrix to infinity
        for(int i = 1; i < ts1.size(); i++)
        	for(int j = 1; j < ts2.size(); j++)
        		costMatrix[i][j] = Double.MAX_VALUE; 
        
        // initialize first cost cell
        costMatrix[0][0] = Cost(ts1.get(0), ts2.get(0));
        
        // compute distances in first row
        for(int i = 1; i < warpingWindow; i++)
            costMatrix[i][0] = costMatrix[i-1][0] + Cost(ts1.get(i), ts2.get(0));
        // pre compute distances in first column
        for(int j = 1; j < warpingWindow; j++)
            costMatrix[0][j] = costMatrix[0][j-1] + Cost(ts1.get(0), ts2.get(j));
        
        
    	for (int i = 1; i < ts1.size(); i++) 
    	{
			for (int j = 1; j < ts1.size(); j++) 
			{
				if (((i < j) && (j-i) <= warpingWindow) || 
						((i >= j) && (i-j) <= warpingWindow)) 
				{
					double min1 = Math.min(costMatrix[i][j-1], costMatrix[i-1][j]);
					double min2 = Math.min(min1, costMatrix[i-1][j-1]);
					
					costMatrix[i][j] = 
						min2 + Cost( ts1.get(i), ts2.get(j) );
				}
				else 
				{
					costMatrix[i][j] = Double.MAX_VALUE;
				}
			}
		}
        
        // add warping path for point 0 of ts1 to be 0 of ts2 
        
        
        int i = ts1.size()-1, j = ts2.size()-1; 
        
        // build the warping path 
        while(i >= 0 && j >= 0 )
        {
        	// add the current warping path indices
        	warpingPath.get(i).add(j);
        	
        	// check if we reached the 
        	if(i == 0 && j == 0)
        	{
        		break;
        	}
        	else if( i == 0)
        	{
        		j--;
        	}
        	else if( j == 0 )
        	{
        		i--;
        	}
        	else
        	{
	        	// get the distances to the neighbour cells forward
	            double d_left = costMatrix[i-1][j],
	                d_down = costMatrix[i][j-1],
	                d_downLeft = costMatrix[i-1][j-1];
	        
	            // add the one having the smallest distance to the warping path
	            if( d_downLeft <= d_down && d_downLeft <= d_left )
	            {
	                i--;
	                j--;
	            }
	            else if( d_down <= d_left && d_down <= d_downLeft )
	            {
	                j--;
	            }
	            else if( d_left <= d_down && d_left <= d_downLeft )
	            {
	                i--;
	            }
	            else
	            {
	            	Logging.println("CalculateWarpingPath:: No warping between: " + i + " " + j, LogLevel.ERROR_LOG);
	            	// anyway default on the diagonal
	            	i--;
	                j--;
	            }
        	}
        }
        
        return warpingPath;
    }
    

    
    
    // predefined warping window settings for the datasets
    // from the Keogh's time series collection site
    public double getWarpingWindow(String dsName)
    {
        // dont use the best warping window and instead
        // set a fixed limit of 10% warping
        if(true)
        return 1.0;
        
        double ww = 0.1;
        
        if(dsName.compareTo("50words") == 0)
        {
            ww = 0.06;
        }
        else if(dsName.compareTo("Adiac") == 0)
        {
            ww = 0.03;
        }
        else if(dsName.compareTo("Beef") == 0)
        {
            ww = 0.00;
        }
        else if(dsName.compareTo("CBF") == 0)
        {
            ww = 0.11;
        }
        else if(dsName.compareTo("Coffee") == 0)
        {
            ww = 0.03;
        }
        else if(dsName.compareTo("Cricket_X") == 0)
        {
            ww = 0.07;
        }
        else if(dsName.compareTo("Cricket_Y") == 0)
        {
            ww = 0.17;
        }
        else if(dsName.compareTo("Cricket_Z") == 0)
        {
            ww = 0.07;
        }
        else if(dsName.compareTo("DiatomSizeReduction") == 0)
        {
            ww = 0.01;
        }
        else if(dsName.compareTo("ECG200") == 0)
        {
            ww = 0.01; 
        }
        else if(dsName.compareTo("ECGFiveDays") == 0)
        {
            ww = 0.00;
        }
        else if(dsName.compareTo("FaceAll") == 0)
        {
            ww = 0.03;
        }
        else if(dsName.compareTo("FaceFour") == 0)
        {
            ww = 0.02;
        }
        else if(dsName.compareTo("FacesUCR") == 0)
        {
            ww = 0.12;
        }
        else if(dsName.compareTo("Fish") == 0)
        {
            ww = 0.04;
        }
        else if(dsName.compareTo("Gun_Point") == 0)
        {
            //ww = 0.00;
            ww = 0.01;
        }
        else if(dsName.compareTo("ItalyPowerDemand") == 0)
        {
            ww = 0.00;
        }
        else if(dsName.compareTo("Lighting2") == 0)
        {
            ww = 0.06;
        }
        else if(dsName.compareTo("Lighting7") == 0)
        {
            ww = 0.05;
        }
        else if(dsName.compareTo("MedicalImages") == 0)
        {
            //ww = 0.20;
            ww = 0.10;
        }
        else if(dsName.compareTo("MoteStrain") == 0)
        {
            ww = 0.01;
        }
        else if(dsName.compareTo("OliveOil") == 0)
        {
            ww = 0.01;
        }
        else if(dsName.compareTo("OSULeaf") == 0)
        {
            ww = 0.07;
        }
        else if(dsName.compareTo("SonyAIBORobotSurface") == 0)
        {
            ww = 0.00;
        }
        else if(dsName.compareTo("SonyAIBORobotSurfaceII") == 0)
        {
            ww = 0.00;
        }
        else if(dsName.compareTo("SwedishLeaf") == 0)
        {
            ww = 0.02;
        }
        else if(dsName.compareTo("Symbols") == 0)
        {
            ww = 0.08;
        }
        else if(dsName.compareTo("synthetic_control") == 0)
        {
            ww = 0.06;
        }
        else if(dsName.compareTo("Trace") == 0)
        {
            ww = 0.03;
        }
        else if(dsName.compareTo("TwoLeadECG") == 0)
        {
            ww = 0.05;
        }
        else if(dsName.compareTo("Two_Patterns") == 0)
        {
            ww = 0.04;
        }
        else if(dsName.compareTo("uWaveGestureLibrary_X") == 0)
        {
            ww = 0.04;
        }
        else if(dsName.compareTo("uWaveGestureLibrary_Y") == 0)
        {
            ww = 0.04;
        }
        else if(dsName.compareTo("uWaveGestureLibrary_Z") == 0)
        {
            ww = 0.06;
        }
        else if(dsName.compareTo("wafer") == 0)
        {
            ww = 0.01; 
        }
        else if(dsName.compareTo("WordsSynonyms") == 0)
        {
            ww = 0.08;
        }
        else if(dsName.compareTo("ChlorineConcentration") == 0)
        {
            ww = 0.00;
        }
        else if(dsName.compareTo("CinC_ECG_torso") == 0)
        {
            ww = 0.01;
        }
        else if(dsName.compareTo("InlineSkate") == 0)
        {
            ww = 0.14;
        } 
        else if(dsName.compareTo("MALLAT") == 0)
        {
            ww = 0.00;
        }
        
        
        return ww;
    }

    

}

