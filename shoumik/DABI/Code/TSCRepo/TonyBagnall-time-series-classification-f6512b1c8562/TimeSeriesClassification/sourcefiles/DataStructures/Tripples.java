package DataStructures;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.StringTokenizer;

import utilities.Logging;

public class Tripples 
{
	public List<Tripple> cells;
	
	// set of row and col ids
	public Set<Integer> rowIds;
	public Set<Integer> colIds;
	
	// constructor
	public Tripples()
	{
		cells = new ArrayList<Tripple>();
		rowIds = new HashSet<Integer>();
		colIds = new HashSet<Integer>();
	}
	
	// constructor using file
	public Tripples(String file)
	{
		ReadTripples(file, false);
	}
	
	public Tripples(Tripples trps)
	{
		cells = trps.cells;
		rowIds = trps.rowIds;
		colIds = trps.colIds;
		
	}
	
	// read sequentially all the lines of a file and read the tripples
 	public void ReadTripples(String fileName, boolean append)
     {
 		// if appending not required then 
 		 if(!append)
 		 {
 			cells = new ArrayList<Tripple>();
 		 	rowIds = new HashSet<Integer>();
 		 	colIds = new HashSet<Integer>();
 		 }
		 
		 try
		 {
		     BufferedReader br = new BufferedReader( new FileReader(fileName) );
		
		     String line = null;
		
		     while( (line = br.readLine()) != null )
		     {
		         StringTokenizer tokenizer = new StringTokenizer(line, "\t ,;");
		
		         int x = Integer.parseInt(tokenizer.nextToken());
		         int y = Integer.parseInt(tokenizer.nextToken());
		         double value = Double.parseDouble(tokenizer.nextToken());
		         
		         rowIds.add(x);
		         colIds.add(y);
		
		         cells.add( new Tripple(x,y,value) );
		     }
		 }
		 catch(Exception exc)
		 {
		     exc.printStackTrace();
		 }
		 
		 Logging.println("File:" + fileName + ", Loaded: " + cells.size() + " cells, " 
				 		+ "Num rows: " + rowIds.size() + 
				 		", Num cols: " + colIds.size() , 
				 		Logging.LogLevel.DEBUGGING_LOG);
         
     }
}
