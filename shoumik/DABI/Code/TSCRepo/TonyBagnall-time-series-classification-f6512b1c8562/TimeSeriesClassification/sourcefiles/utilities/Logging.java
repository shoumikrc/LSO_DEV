package utilities;

import java.io.PrintStream;
import java.util.List;

import DataStructures.Matrix;

public class Logging 
{
	public static LogLevel currentLogLevel = LogLevel.DEBUGGING_LOG; 

	public enum LogLevel{
            DEBUGGING_LOG, 
            INFORMATIVE_LOG, 
            WARNING_LOG, 
            ERROR_LOG, 
            PRODUCTION_LOG};

	public static void print(String text, LogLevel logLevel)
	{
		if( logLevel.compareTo(currentLogLevel) >= 0)
		{
			String methodTrace = Thread.currentThread().getStackTrace()[2].toString();
			System.out.print(methodTrace + " : " + text);
		}
	}
	
	public static void println(String text, LogLevel logLevel)
	{
		if( logLevel.compareTo(currentLogLevel) >= 0)
		{
			String methodTrace = Thread.currentThread().getStackTrace()[2].toString();
			System.out.println(methodTrace + ":  " + text);
		}
	}
	
	public static void println(String text)
	{
		println(text, LogLevel.DEBUGGING_LOG);
	}
	
    public static void print(List<Integer> arr, LogLevel logLevel)
	{
            if( logLevel.compareTo(currentLogLevel) >= 0)
            {
                String text = "";

                for(int i = 0; i < arr.size(); i++)
                    text += arr.get(i) + " ";
                
                System.out.print(text);
            }
	}
    
    public static void println(double [] arr)
  	{
    	println(arr, LogLevel.DEBUGGING_LOG);
  	} 
    
    public static void print(double [] arr)
  	{
    	print(arr, LogLevel.DEBUGGING_LOG);
  	} 
    
    public static void print(double [] arr, LogLevel logLevel)
	{
            if( logLevel.compareTo(currentLogLevel) >= 0)
            {
                String text = "";

                for(int i = 0; i < arr.length; i++)
                    text += arr[i] + " ";
                
                System.out.print(text);
            }
	}
    
    public static void println(double [] arr, LogLevel logLevel)
	{
            if( logLevel.compareTo(currentLogLevel) >= 0)
            {
                String text = "";

                for(int i = 0; i < arr.length; i++)
                    text += arr[i] + " ";
                
                System.out.println(text);
            }
	}
    
    // print the array of integers
    public static void print(int [] arr, LogLevel logLevel)
	{
            if( logLevel.compareTo(currentLogLevel) >= 0)
            {
                String text = "";

                for(int i = 0; i < arr.length; i++)
                    text += arr[i] + " ";
                
                Logging.print(text, logLevel);
            }
	}
    
    // println of an array
    public static void println(int [] arr, LogLevel logLevel)
	{
            if( logLevel.compareTo(currentLogLevel) >= 0)
            {
                String text = "";

                for(int i = 0; i < arr.length; i++)
                    text += arr[i] + " ";
                
                Logging.println(text, logLevel);
            }
	}
    
    public static void println(int [] arr)
	{
    	println(arr, LogLevel.DEBUGGING_LOG);
	}
    
    public static void print(int [] arr)
	{
    	print(arr, LogLevel.DEBUGGING_LOG);
	}
    
    public static void print(Matrix m, LogLevel logLevel)
	{
            if( logLevel.compareTo(currentLogLevel) >= 0)
            {
                String text = "";

                for(int i = 0; i < m.getDimRows(); i++)
                {
                	text += i + ": ";
                	
                	for(int j = 0; j < m.getDimColumns(); j++)
                	{
                		text += m.get(i, j) + " ";
                	}
                	
                	text += "\n";
                }
                
                System.out.print(text);
            }
	}
        
    public static void println(List<Integer> arr, LogLevel logLevel)
	{
            if( logLevel.compareTo(currentLogLevel) >= 0)
            {
                String text = "";

                for(int i = 0; i < arr.size(); i++)
                    text += arr.get(i) + " ";
                
                System.out.println(text);
            }
	}
    
    public static void print(double [][] matrix, PrintStream ps, LogLevel logLevel)
	{
    	int n = matrix.length;
    	int m = matrix[0].length; 
    		
    	ps.println(n + " " + m );
    	
    	for(int i = 0; i < n; i++)
    	{
    		for(int j=0; j < m; j++)
    		{
    			ps.print(matrix[i][j] + " ");
    		} 
    		
    		ps.print("\n");
    	}
	}
    
    public static void print(int [][] matrix, PrintStream ps)
   	{
       	int n = matrix.length;
       	int m = matrix[0].length; 
       		
       	ps.println(n + " " + m );
       	
       	for(int i = 0; i < n; i++)
       	{
       		for(int j=0; j < m; j++)
       		{
       			ps.print(matrix[i][j] + " ");
       		} 
       		
       		ps.print("\n");
       	}
   	}
        
}
