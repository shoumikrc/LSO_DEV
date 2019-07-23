/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package TimeSeries;

import DataStructures.DataInstance;

/**
 * A distance operator abstract class useful for implementing
 * 
 * @author Josif Grabocka
 */
public abstract class DistanceOperator 
{
    public abstract double CalculateDistance(DataInstance timeSeries1, DataInstance timeSeries2);
    
    //public abstract double CalculateDistance(List<Double> timeSeries1, List<Double> timeSeries2);
    
            
}
