package DataStructures;

import java.util.*;

import utilities.Logging;

/*
 * A point of a time series including the coordinates and status of presence
 * 
 * For time series data the X-axis is interpreted as time and Y as values
 */
public class FeaturePoint 
{
	/*
	 * FeaturePoint coordinate
	 */
	public double value; 
	
	// in special cases the time series is multivariate
	// therefore we include a flag and a placeholders for this case 
	boolean isMultivariate;
	public List<Double> values;
	
	/*
	 * The point presence status, used to denote whether a point 
	 * is set missing for experimental purpose or syntetically 
	 * imputed/interpolated 
	 */
	public enum PointStatus { PRESENT, MISSING };
	
	public PointStatus status;
	
	/*
	 * Constructor
	 */
	public FeaturePoint()
	{
		value = 0;
		status = PointStatus.PRESENT;
		
		isMultivariate = false;
		values = null;
	}
	
	public FeaturePoint(FeaturePoint p)
	{
		value = p.value;
		status = p.status;
	}
        
    public FeaturePoint(double v)
	{
		value = v;
		status = PointStatus.PRESENT;
	}
        
    public double distanceSquare(FeaturePoint p)
    {
        if( status == PointStatus.MISSING || p.status == PointStatus.MISSING )
        {
            Logging.println("Point: Distance of missing points requested!", Logging.LogLevel.PRODUCTION_LOG);
            return Double.MAX_VALUE;
        }
        
        double distance = 0;
        
        // for single-valued time series 
        if( ! isMultivariate)
        {
        	distance = Math.pow(value - p.value, 2);
        }
        // for multivariate time series
        else 
        {
        	for( int i = 0; i < values.size(); i++ )
        		distance += Math.pow(values.get(i) - p.values.get(i), 2);
        }
        
        return distance;
    }
	
}
