/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package DataStructures;

/**
 *
 * @author Josif Grabocka
 */
public class Tripple 
{
    public int row, col;
    public double value;
    
    public Tripple( int R, int C )
    {
        row = R;
        col = C;
        value = 0;
    }
    
    public Tripple( int R, int C, double V )
    {
        row = R;
        col = C;
        value = V;
    }
}
