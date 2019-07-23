/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package DataStructures;

/**
 *
 * @author Josif Grabocka
 */
public class Coordinate 
{
    public int x, y;
    public double value;
    
    public Coordinate( int X, int Y )
    {
        x = X;
        y = Y;
        value = 0;
    }
    
    public Coordinate( int X, int Y, double V )
    {
        x = X;
        y = Y;
        value = V;
    }
}
