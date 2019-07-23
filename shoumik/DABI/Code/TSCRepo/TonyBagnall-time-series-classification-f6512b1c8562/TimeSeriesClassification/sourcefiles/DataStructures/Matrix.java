package DataStructures;

import java.util.List;
import java.util.Random;
//import java.util.StringTokenizer;

//import DataStructures.FeaturePoint.PointStatus;
////import Experiments.RegressionMain;
import utilities.GlobalValues;
import utilities.Logging;
import utilities.Logging.LogLevel;


import java.io.*;
import java.text.DecimalFormat;
import java.util.ArrayList;

import libsvm.svm_node;
import libsvm.svm_problem;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instances;

/*
 * a 2d Matrix 
 */
public class Matrix 
{
	/*
	 * The 2d array storing the cell values
	 */
	public double[][] cells;
	private int dimRows, dimColumns;
	
        // the mean value of cells
        double mean;
        
	
	/*
	 * Constructors
	 */
	public Matrix()
	{
		cells = null;
		dimRows = dimColumns = 0;
        mean = 0;
	}
        
    // option = "features" => load features of the dataset
    // option = "labels" => load labels of the dataset
    public Matrix(DataSet ds, String option)
	{
            cells = null;
            dimRows = dimColumns = 0;

            if( option.compareTo("features") == 0)
                LoadDatasetFeatures(ds, false);
            else if( option.compareTo("labels") == 0)
                LoadDatasetLabels(ds, false); 
            else
                System.err.println("Matrix constructor: Invalid option: " + option);
	}
	
	/*
	 * create a matrix as a clone copy of parameter
	 */
	public Matrix(Matrix m)
	{
		if( m != null)
        {
			dimColumns = m.dimColumns;
			dimRows = m.dimRows;
			
			cells = new double[dimRows][dimColumns];
			
			for( int r = 0; r < dimRows; r++ )
                for( int c = 0; c < dimColumns; c++ )
                	cells[r][c] = m.get(r, c);
            
        }
		else
		{
			Logging.println("Matrix constructor :: Cannot copy empty matrix", LogLevel.ERROR_LOG);
		}
	}
	
	/* 
	 * Initialize a matrix with parameter dimensions
	 */
	public Matrix(int rows, int columns)
	{
            dimRows = rows;
            dimColumns = columns;

            cells = new double[dimRows][dimColumns];
			
            for( int r = 0; r < dimRows; r++ )
                for( int c = 0; c < dimColumns; c++ )
                	cells[r][c] = GlobalValues.MISSING_VALUE;
	}
	
	/*
	 * Randomly initialize the values of cells
	 */
	public void RandomlyInitializeCells(double minValue, double maxValue)
	{
		//System.out.println("Initializing matrix with values between " + minValue + " and " + maxValue);
		
		if( cells != null )
		{
                    Random randomGenerator = new Random();
                    double scale = maxValue - minValue;

                    /* 
                     * nextDouble generates a value between 0 and 1 which should
                     * scaled up to the range of parameter values min to max 
                     */

                    for(int r = 0; r < dimRows; r++)
                            for(int c = 0; c < dimColumns; c++)
                                    cells[r][c] = minValue + randomGenerator.nextDouble() * scale;

		}
		else
		{
			System.err.println("Cells array of matrix not initialized");
		}
	}
	
	
	/*
	 * Return the value of a cell, or POSITIVE_INFINITY if the matrix is not initialized
	 */
	public double get(int i, int j)
	{
        return cells[i][j]; 
	}
	
	/*
	 * Syntatic sugar notation means get the 0-th element of the i-th row
	 */
	public double get(int i)
	{
        return get(i,0);
	}
	
	/*
	 * Set the value of a cell
	 */
	public void set(int i, int j, double value)
	{
        cells[i][j] = value;
        
	}
	
	/*
	 * Get the dimensions of rows and columns
	 */
	
	public int getDimRows()
	{
            return dimRows;
	}
	
	public int getDimColumns()
	{
            return dimColumns;
	}
	
	/*
	 * Get minimum / maximum values of the matrix cells
	 */
	
	public double getMinValue()
	{
            return searchExtremeValue(true);
	}
	
	public double getMaxValue()
	{
            return searchExtremeValue(false);
	}
	
        /*
         * Get the mean value of the matrix 
         * - compute once and use subsequently
         * if changes are made to the matrix then call
         * the compute mean value method
         */
        
        public double getMeanValue()
        {
            if( mean == 0)
            {
                ComputeMeanValue();
            }
            
            return mean;
        }
        
        
        /*
         * computes the mean value of the matrix
         */
        public void ComputeMeanValue()
        {
            mean = 0;
            int noObserved = 0;
            
            for(int r = 0; r < getDimRows(); r++)
            {
                for(int c = 0; c < getDimColumns(); c++)
                {
                    double value = get(r,c);
                    
                    if(value != GlobalValues.MISSING_VALUE)
                    {
                        mean += value;
                        noObserved++;
                    }
                }
            }
            
            mean /= noObserved;
            
        }
        
        public double GetColumnMean(int c)
        {
            mean = 0;
            int noObserved = 0;
            
            for(int r = 0; r < getDimRows(); r++)
            {
            	double value = get(r,c);
                    
                if(value != GlobalValues.MISSING_VALUE)
                {
                    mean += value;
                    noObserved++;
                }
            
            } 
            
            return mean / noObserved; 
        }
        
        public double GetRowMean(int r)
        {
            mean = 0; 
            int noObserved = 0; 
            
            for(int c = 0; c < getDimColumns(); c++)
            {
            	double value = get(r,c);
                   
                if(value != GlobalValues.MISSING_VALUE)
                {
                    mean += value;
                    noObserved++;
                }
            } 
            
            return mean / noObserved; 
        }
        
	/*
	 * helper function to find an extreme value in the matrix
	 * either minimum or maximum
	 */
	private double searchExtremeValue(boolean isMinimum)
	{
            double extremeValue = GlobalValues.MISSING_VALUE;

            if( cells != null )
            {
                extremeValue = isMinimum ? Double.MAX_VALUE : Double.MIN_VALUE;

                for(int r = 0; r < dimRows; r++)
                {
                    for(int c = 0; c < dimColumns; c++)
                    {
                        double val = cells[r][c];

                        if( isMinimum)
                        {
                            if( val < extremeValue )
                            {
                                extremeValue = val;
                            }
                        }
                        else
                        {
                            if( val > extremeValue )
                            {
                                extremeValue = val;
                            }
                        }
                    }
                }
            }
            else
            {
                    System.err.println("Cells array of matrix not initialized");
            }

            return extremeValue;
	}
	
	/*
	 * Set one value for all the matrix cells
	 * i.e. like all 1, or all 0
	 */
	
	public void SetUniqueValue(double v)
	{
            if(cells != null)
            {
                for(int r = 0; r < dimRows; r++)
                    for(int c = 0; c < dimColumns; c++)
                        cells[r][c] = v;
            }
            else
            {
                System.err.println("Cells array of matrix not initialized");
            }
	}
	
	/*
	 * Set a parameter column to have value v
	 */
	public void SetColumnValue(int col, double v)
	{
            if(cells != null)
            {
                for(int r = 0; r < dimRows; r++)
                    cells[r][col] = v; 
            }
            else 
            {
                System.err.println("Cells array of matrix not initialized");
            }
	}
	
	/*
	 * Set a parameter row to have value v
	 */
	public void SetRowValue(int row, double v)
	{
            if(cells != null)
            {
                for(int c = 0; c < dimColumns; c++)
                    cells[row][c] = v;
            }
            else
            {
                System.err.println("Cells array of matrix not initialized");
            }
	} 
	
        /*
         * Get the row of a matrix
         */
        
        public double [] getRow(int r)
        {
        	double [] row = null;
            
            if( r < 0 || r > dimRows )
            {
                Logging.println("Matrix get row: " + r + "out of range", Logging.LogLevel.ERROR_LOG);
            }
            else
            {
                row = cells[r];
            }
            
            return row;
        }

        /*
         * Get the column c of a matrix
         */
        
        public double [] getCol(int c)
        {
        	double [] col = new double[dimRows];
        	
        	for(int i = 0; i < dimRows; i++)
        		col[i] = cells[i][c];
           
            return col;
        }
        
        public void setCol(int c, double values[])
        {
        	
        	for(int i = 0; i < dimRows; i++)
        		cells[i][c] = values[i];
           
            
        }
        
	/* 
	 * get the column sum and the row sum
	 */
	
	public double getColumnSum(int c) 
	{
            double colSum = 0;

            for(int r = 0; r < dimRows; r++)
                    colSum += cells[r][c];

            return colSum;
	}
	
	public double getRowSum(int r)
	{
            double rowSum = 0;

            for(int c = 0; c < dimColumns; c++)
                    rowSum += cells[r][c];

            return rowSum;
	}
	
	/* 
	 * get the column squared sum and the row squared sum
	 */
	
	public double getColumnSquaresSum(int c)
	{
            double colSum = 0;

            for(int r = 0; r < dimRows; r++)
                    colSum += Math.pow( cells[r][c], 2 );

            return colSum;
	}
	
	public double getRowSquaresSum(int r)
	{
            double rowSum = 0;

            for(int c = 0; c < dimColumns; c++)
                    rowSum += Math.pow( cells[r][c], 2 );

            return rowSum;
	}
        
        // get the total square sum of all elements
        public double getSquaresSum()
        {
            double sum = 0;
            
            for(int i = 0; i < dimRows; i++)
            {
                sum += getRowSquaresSum(i);
            }
            
            return sum;
        }
	
	/*
	 * Load the feature values of a time series dataset into the matrix,
	 * ommiting of course the labels
	 */
	public void AppendMatrix(Matrix m)
	{
        if( m.getDimRows() == 0 )
        	return;
		
    	dimColumns = m.dimColumns;

        // create enlarged allocation
        double [][] newCells = new double[dimRows + m.dimRows][dimColumns];
        
        // copy the previous cells
        for(int r = 0; r < dimRows; r++)
            for(int c = 0; c < dimColumns; c++)
            	newCells[r][c] = cells[r][c];
            	
        // set cells to point to new cells
        cells = newCells;

		int startIndex = dimRows,
				endIndex = dimRows + m.dimRows;

        for(int r = startIndex; r < endIndex; r++)
            for(int c = 0; c < dimColumns; c++)
                cells[r][c] = m.get(r-startIndex, c);
        
        dimRows = dimRows + m.dimRows;
        
	}
	
	// set a specific row
	public void SetRow( int r, double [] values)
	{
		cells[r] = values;
	}
	 
	
	public void LoadDatasetFeatures(DataSet dataSet, boolean append)
	{
            // read the rows dimensions as the number of time series
            if( dataSet.instances.size() > 0 )
            {
            	int startIndex = 0, endIndex = 0;
            	
                // if it is not an append call or the cells are not initialized 
                if( append == false )
                { 
                    dimRows = dataSet.instances.size();
                    dimColumns = dataSet.instances.get(0).features.size();
                    
                    cells = new double[dimRows][dimColumns];

                    startIndex = 0;
                    endIndex = dimRows;
                }
                else // if append is true
                {
                	dimColumns = dataSet.numFeatures;
                	
                    int dsNumInstances = dataSet.instances.size();

                    // create enlarged allocation
                    double [][] newCells = new double[dimRows + dsNumInstances][dimColumns];
                    
                    // copy the previous cells
                    for(int r = 0; r < dimRows; r++)
                        for(int c = 0; c < dimColumns; c++)
                        	newCells[r][c] = cells[r][c];
                        	
                    // set cells to point to new cells
                    cells = newCells;
                    
                    startIndex = dimRows;
                    endIndex = dimRows + dsNumInstances;
                    
                    dimRows = dimRows + dsNumInstances;
                }

                //System.out.println("\nLoading matrix from time series of dimensions ("+dimRows+","+dimColumns+")");
                
                for(int r = startIndex; r < endIndex; r++)
                {
                    for(int c = 0; c < dimColumns; c++)
                    {
                        FeaturePoint p = dataSet.instances.get(r-startIndex).features.get(c);
                        double pointValue = 0; 

                        if( p.status != FeaturePoint.PointStatus.MISSING)
                            pointValue = p.value;
                        else
                            pointValue = GlobalValues.MISSING_VALUE;

                        cells[r][c] = pointValue;
                    }
                }
                
            }		
	}
	
	
	/*
	 * Load into the matrix the labels of a time series dataset
	 */
	public void LoadDatasetLabels(DataSet dataSet, boolean append)
	{
            // read the rows dimensions as the number of time series
            if( dataSet.instances.size() > 0 )
            {
            	int startIndex=0, endIndex = 0;
            	
                // if it is not an append call or the cells are not initialized 
                if( append == false)
                {
                    dimRows = dataSet.instances.size();
                    dimColumns = 1;

                    cells = new double[dimRows][dimColumns];
                    startIndex = 0;
                    endIndex = dimRows;
                }
                else
                {
                	int dsNumInstances = dataSet.instances.size();

                    // create enlarged allocation
                    double [][] newCells = new double[dimRows + dsNumInstances][dimColumns];
                    
                    // copy the previous cells
                    for(int r = 0; r < dimRows; r++)
                        for(int c = 0; c < dimColumns; c++)
                        	newCells[r][c] = cells[r][c];
                        	
                    // set cells to point to new cells
                    cells = newCells;
                    
                    startIndex = dimRows;
                    endIndex = dimRows + dsNumInstances;    
                    
                    dimRows = dimRows + dsNumInstances;
                }

                for(int r = startIndex; r < endIndex; r++)
                    cells[r][0] = dataSet.instances.get(r-startIndex).target; 
                
                
            }
	}
	
	@Override
	public String toString()
	{
		String str = "";
		
		for(int r = 0; r < dimRows; r++)
		{
			for(int c = 0; c < dimColumns; c++)
			{
				str += get(r,c) + " ";
			}
			
			str += "\n";
		}
		
		return str;
	}	
	
        public void SaveToFile( String file)
        {
            File f = (new File(file));
            f.delete();
            
            try 
            {
                BufferedWriter out = new BufferedWriter(new FileWriter(f));
                DecimalFormat df = new DecimalFormat("#.####");
                
                if(cells != null)
                {
                    for(int r = 0; r < dimRows; r++)
                    {
                    	for(int c = 0; c < dimColumns; c++)
                        {
                            out.write( df.format(cells[r][c]) );
                            
                            if( c < dimColumns-1) out.write(",");
                        }
                        
                        out.write("\n");
                    }
                }
                
                out.flush();
                out.close();
                
            } 
            catch (IOException e) 
            {
                e.printStackTrace();
            }
        }
        
        /*
         * Check if the cell is missing
         */
        public boolean isMissing(int r, int c)
        {
            return get(r,c) == GlobalValues.MISSING_VALUE;
        }
        
        public int SearchRow(List<Double> row)
        {
        	int index = -1;
        	
        	for(int r = 0; r < dimRows; r++)
        	{
        		boolean isEqual = true;
        		
        		for(int c = 0; c < dimColumns; c++)
        			if( get(r,c) != row.get(c) )
        			{
        				isEqual = false;
        				break;
        						
        			}
        		
        		if(isEqual)
        		{
        			index = r;
        			break;
        		}
        	}
        	
        	return index;
        }
        
       /*
        * The dot product of Row1 and Row2 vectors
        */
        public double RowDotProduct(int row1, int row2)
        {
    		// the dot product
    		double dp = 0; 
    		for(int c = 0; c < dimColumns; c++) 
    		{
    			if( get(row1, c) != GlobalValues.MISSING_VALUE && 
    					get(row2, c) != GlobalValues.MISSING_VALUE )
    						dp += get(row1, c)*get(row2, c);
    		}
    		
    		return dp;
        }
        
        public double RowEuclideanDistance(int row1, int row2)
    	{
    		// the dot product
    		double dist = 0;
    		for(int c = 0; c < dimColumns; c++)
    		{
    			if( get(row1, c) != GlobalValues.MISSING_VALUE && 
    					get(row2, c) != GlobalValues.MISSING_VALUE )
    			{
    				double diff = get(row1, c) - get(row2, c);
    				dist += diff*diff;
    			}
    		}
    		return dist;
    	}
        
        public double ColumnEuclideanDistance(int col1, int col2)
    	{
    		// the dot product
    		double dist = 0;
    		for(int r = 0; r < dimRows; r++)
    		{
    			double diff = get(r, col1) - get(r, col2);
    			dist += diff*diff;
    		}
    		return dist;
    	}
        
        public void ScaleColumnsToUnitVector()
    	{
    		// the dot product
    		for(int c = 0; c < dimRows; c++)
    		{
    			double columnNorm = 0;
	    		for(int r = 0; r < dimRows; r++)
	    			columnNorm += get(r, c)*get(r, c);
	    		
	    		columnNorm=Math.sqrt(columnNorm);
	    		
	    		for(int r = 0; r < dimRows; r++)
	    			set(r, c, get(r,c)/columnNorm );
	    	}
	    		
    		
    	}
        
        
        public double ColumnDotProduct(int col1, int col2)
    	{
    		// the dot product
    		double dp = 0;
    		for(int r = 0; r < dimRows; r++)
    			dp += get(r, col1)*get(r, col2);
    		
    		return dp;
    	}
        
        
    /*
     * Load the regression data, which in difference from classical 
     * datasets which are suited for classification, doesn't have 
     * The regression dataset types are
     */
    public void LoadRegressionData( List<List<Double>> data)
    {
    	LoadRegressionData(data, false);
    }
    
    // load the regression data, ommit the target (last variable) if required
    public void LoadRegressionData( List<List<Double>> data, boolean ommitTarget )
    {
    	dimRows = data.size();
    	
    	if( dimRows > 0)
    	{
    		dimColumns = data.get(0).size();
    		
    		// check whether we want to 
    		if( ommitTarget )
    			dimColumns = dimColumns - 1;
    		
    		cells = new double[dimRows][dimColumns];
    		
    		
    		for(int i = 0; i < dimRows; i++)
    			for(int j = 0; j < dimColumns; j++)
    				cells[i][j] = data.get(i).get(j);
    			
    	}
    }
    
    public Instances ToWekaInstances()
    {
    	Instances wekaInstances = null;
		
		// create a list of attributes features + label
        FastVector attributes = new FastVector(dimColumns);
        for( int i = 0; i < dimColumns; i++ )
            attributes.addElement( new Attribute("attr" + String.valueOf(i+1)) );
        
        // add the attributes 
        wekaInstances = new Instances("", attributes, dimRows);
        
        wekaInstances.setClassIndex(dimColumns-1); 

        // add the values
        for( int i = 0; i < dimRows; i++ )
        {
            //Instance ins = new Instance(numFeatures+1);
            double [] instanceValues = new double[dimColumns];
            
            for( int j = 0; j < dimColumns; j++ )                         
            {
                try
                {
                    if( cells[i][j] == GlobalValues.MISSING_VALUE )
                        instanceValues[j] = weka.core.Utils.missingValue();
                    else
                        instanceValues[j] = cells[i][j];
                }
                catch(Exception exc)
                {
                    //System.out.println( i + ", " + j ); 
                }
            }
            
            wekaInstances.add( new DenseInstance(1.0, instanceValues) );
            
        }
        
        
    	
    	return wekaInstances;
    }
    
    public svm_problem ToLibSvmProblem()
	{
		  svm_problem data = new svm_problem();
	        
	        data.l = getDimRows();
	        data.x = new svm_node[getDimRows()][];
	                
	        // iterate through all the instances
	        for(int i = 0; i < getDimRows(); i++)
	        {
	            
	            List<svm_node> nodes = new ArrayList<svm_node>();
	            
	            // iterate through the feature values of the instance and create nodes
	            // for the nonmissing values only 
	            for(int j = 0; j < getDimColumns(); j++)
	            {
	            	if( cells[i][j] != GlobalValues.MISSING_VALUE )
	            	{
	            	    svm_node node = new svm_node();
		                node.index = j+1;
		                node.value = cells[i][j]; 
		                
		                nodes.add(node);
	            	}
	            } 
	            
	            // flush the list as array
	            data.x[i] = new svm_node[nodes.size()];
	            for(int k = 0; k < nodes.size(); k++)
	            {
	            	data.x[i][k] = nodes.get(k);
	            }
	            
	        }
		
		return data;
	}
    
    // load a weka dataset into the matrix
    public void LoadWekaInstances(Instances ins)
    {
    	
    }
    
    public void SaveTripples(String tripplesFile)
    {
    	try
        {
            File f = new File(tripplesFile);
            
            // delete the existing and create a new file
            if( f.exists() )
            	f.delete();
            f.createNewFile();
            
            BufferedWriter out = new BufferedWriter(new FileWriter(f));
            
            for( int i = 0; i < getDimRows(); i++ )
            {            
                for( int j = 0; j < getDimColumns(); j++ )
                {
                    out.write( i + "," + j + "," + cells[i][j] + "\n");
                }
                
            }
            
            out.flush();
            out.close();
        }
        catch(Exception exc)
        {
            System.err.println(exc.getMessage());
        }
    }
    
    public double GetSparsityRatio()
    {
    	double sparseCells = 0, totalCells = 0;
    	
    	for(int i = 0; i < getDimRows(); i++)
			for(int j = 0; j < getDimColumns(); j++)
			{
				if( get(i, j) == GlobalValues.MISSING_VALUE ||  get(i, j) == 0 )
				{
					sparseCells += 1.0;
				}
				
				totalCells += 1.0;
			}		
    	
    	return sparseCells/totalCells;
    }
    
    
    
       
}
