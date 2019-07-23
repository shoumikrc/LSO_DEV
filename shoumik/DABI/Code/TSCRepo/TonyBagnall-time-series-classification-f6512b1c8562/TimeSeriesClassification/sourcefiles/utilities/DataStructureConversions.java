/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package utilities;

import java.util.ArrayList;
import java.util.List;

import utilities.Logging.LogLevel;

import DataStructures.DataInstance;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instances;

/**
 *
 * @author Josif Grabocka
 */
public class DataStructureConversions 
{
    public static double[] ListToArrayDouble( List<Double> list)
    {
        double [] array = new double[list.size()];
        
        for(int i = 0; i < list.size(); i++)
        { 
            array[i] = list.get(i);
        }
        
        return array;
    }
    
    public static List<Double> ArrayToListDouble( double[] array)
    {
        List<Double> list = new ArrayList<Double>();
        
        for(int i = 0; i < array.length; i++)
        { 
            list.add(array[i]);
        }
        
        return list;
    }
    
    public static Instances ToWekaInstances(double [][] data)
    {
    	Instances wekaInstances = null;
    	
    	if(data.length <= 0) return wekaInstances;
    	
    	int dimRows = data.length;
    	int dimColumns = data[0].length;
    	
    	//Logging.println("Converting " + dimRows + " and " + dimColumns + " features.", LogLevel.DEBUGGING_LOG);
		
		// create a list of attributes features + label
        FastVector attributes = new FastVector(dimColumns);
        for( int i = 0; i < dimColumns; i++ )
            attributes.addElement( new Attribute("attr" + String.valueOf(i+1)) );
        
        // add the attributes 
        wekaInstances = new Instances("", attributes, dimRows);
        
        // add the values
        for( int i = 0; i < dimRows; i++ )
        {
            double [] instanceValues = new double[dimColumns];
            
            for( int j = 0; j < dimColumns; j++ )                         
                instanceValues[j] = data[i][j];
            
            wekaInstances.add( new DenseInstance(1.0, instanceValues) );
        }
    
        return wekaInstances;
    }
    
    public static double[][] FromWekaInstances(Instances ds)
    {
    	int numFeatures = ds.numAttributes();
    	int numInstances = ds.numInstances();
            
    	//Logging.println("Converting " + numInstances + " instances and " + numFeatures + " features.", LogLevel.DEBUGGING_LOG);
		
        double [][] data = new double[numInstances][numFeatures]; 
        
        for(int i = 0; i < numInstances; i++)
        	for(int j = 0; j < numFeatures; j++)
        		data[i][j] = ds.get(i).value(j); 
        
        return data;
        		
    }
}
