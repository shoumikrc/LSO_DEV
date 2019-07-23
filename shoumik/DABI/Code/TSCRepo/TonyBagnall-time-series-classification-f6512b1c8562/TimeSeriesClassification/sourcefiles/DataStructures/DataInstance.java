package DataStructures;

import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;
import javax.imageio.ImageIO;
import weka.core.Instance;

/*
 * In classical scenarios an instance is a set of concrete values for the attributes
 * however in our case we set a FeaturePoint object instead of a simple value in order 
 * to make the model more generic, s.th. every point can have for instance other
 * attributes such as time, meaning the feature vector can be sequentially ordered.
 * 
 * Such property is useful for time series where we need to know the time of a point
 * feature 
 */
public class DataInstance 
{
	public String name;
	/*
	 * The feature vector of the instance
	 */
	public List<FeaturePoint> features;
	
	/*
	 * The label of the instance described as a real number 
	 */
	public double target;
	
	/*
	 * constructors
	 */
	public DataInstance() 
	{
		features = new ArrayList<FeaturePoint>();
		target = 0;
		name = "-";
	}
	
	public DataInstance( DataInstance ins)
	{
		features = new ArrayList<FeaturePoint>();
		for(FeaturePoint value : ins.features)
			features.add( new FeaturePoint(value) );
		
		target =  ins.target;
		name = new String(ins.name);
	}
	
        /*
         * Initialize from a weka instance
         */
        public DataInstance(String instanceName, Instance instance)
	{
		features = new ArrayList<FeaturePoint>();
                
		for(int i=0; i< instance.numAttributes()-1; i++)
			features.add( new FeaturePoint(instance.value(i) ) );
		
		target = instance.value(instance.numAttributes()-1); 
		name = instanceName;
	}
	
	/*
	 * Get the minimum value of the instance features' values
	 */
	
	public double GetMinFeatureValue()
	{
		double minimum = Double.POSITIVE_INFINITY;
		
		for( int i = 0; i < features.size(); i++ )
		{
			if( features.get(i).status != FeaturePoint.PointStatus.MISSING 
					&& features.get(i).value < minimum  )
			{
				minimum = features.get(i).value;
			}
		}
		
		return minimum;
	}
	
	/*
	 * Get the maximum value of the instance features' values
	 */
	public double GetMaxFeatureValue()
	{
		double maximum = Double.NEGATIVE_INFINITY;
		
		for( int i = 0; i < features.size(); i++ )
		{
			if( features.get(i).status != FeaturePoint.PointStatus.MISSING 
					&& features.get(i).value > maximum  )
			{
				maximum = features.get(i).value;
			}
		}
		
		return maximum;
	}
	
	/*
	 * get the mean value of the features
	 */
	public double GetFeaturesMean()
	{
		double sum = 0;
		
		for( int i = 0; i < features.size(); i++ )
		{
			sum += features.get(i).value;
		}
		
		return sum / features.size();
	}
	
	/*
	 * Get the standard deviation of the feature values
	 */
	
	public double GetFeaturesStandardDeviation(double mean)
	{
		double sumSquaresDiffs = 0;
		
		for( int i = 0; i < features.size(); i++ )
		{
			sumSquaresDiffs += Math.pow( features.get(i).value - mean, 2);
		}
		
		return Math.sqrt( sumSquaresDiffs / (features.size()-1) );
	}
	
	/*
	 * Normalize feature values
	 */
	public void NormalizeFeaturesValues()
	{
		double mean = GetFeaturesMean();
		double std = GetFeaturesStandardDeviation(mean);
		
		for(int i = 0; i < features.size(); i++) 
		{
			features.get(i).value = (features.get(i).value-mean)/std; 
		}
	}
	
	public void RemoveMean()
	{
		double mean = GetFeaturesMean();
		
		for( int i = 0; i < features.size(); i++ )
		{ 
            features.get(i).value = (features.get(i).value - mean);
		}
	}
	
	/*
	 * Scale the values by a parameter scale
	 */
	public void Scale(double scaleFactor)
	{
		for(int i = 0; i < features.size(); i++)
		{
			features.get(i).value *= scaleFactor;
		}
	}
        
        
        /*
         * Check for equality between data instances, meaning all feature values
         * and label are equal
         */
        public boolean IsEqual( DataInstance di )
        {
            // if the number of features differs then the instances aren't equal
            if( features.size() != di.features.size())
            {   
                return false;
            }
            else
            {
                // if any two pairs of feature values of the same feature index
                // is not equal then the instances are not equal
                for(int i = 0; i < features.size(); i++)
                {
                    double distance = Math.abs(features.get(i).value - di.features.get(i).value);
                    
                    if( distance > 0.000001  )
                        return false; 
                }
                
                // if the labels differ then the instances aren't equal
                //if( label != di.label )
                  //  return false;
            }
            
            // if all tests passed then the instances are equal
            return true;
        }
        
        // select a list of neighbour indexes on either left or right of the missing point
        public List<Integer> GetNeighboursIndexes(int indexMissing, boolean directionToRight,int numPoints)
        {
            List<Integer> neighborIndexes = new ArrayList<Integer>();

            int advanceStep = directionToRight ? 1 : -1;
            int numTotalPoints = features.size();

            int i = indexMissing + advanceStep;

            for( ; i >= 0 && i < numTotalPoints; i += advanceStep )
            {
                if( features.get(i).status == FeaturePoint.PointStatus.PRESENT ) 
                {
                    neighborIndexes.add(i);
                    
                    if( neighborIndexes.size() > numPoints )
                        break;
                }
            }

        return neighborIndexes;
    }  
	
    public BufferedImage ConvertToImage(String name, double dsMax, double dsMin)
    {
        int width = features.size(); 
        int height = 1;
        
        BufferedImage seriesImage = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        
        // We need its raster to set the pixels' values.
        WritableRaster raster = seriesImage.getRaster();
        
        //dsMin = GetMinFeatureValue();
        //dsMax = GetMaxFeatureValue();
        
        // Put the pixels on the raster, using values between 0 and 255.
        for(int w=0;w<width;w++)
        {
            double val = features.get(w).value;
            
            double imVal = ((val-dsMin)/(dsMax-dsMin))*255;
            
            System.out.println("min:" + dsMin + ", max: " + dsMax + 
                    ", val: " + val + ", imVal: " + imVal);
            
            raster.setSample(w,0,0,imVal); 
        }
        
        
        return seriesImage;
    }
    
    public List<Double> GetFeatureValues()
    {
        List<Double> featureValues = new ArrayList<Double>();
        
        for( FeaturePoint fp :  features )
        {
            featureValues.add( fp.value );
        }
        
        return featureValues;
    }
	
}
