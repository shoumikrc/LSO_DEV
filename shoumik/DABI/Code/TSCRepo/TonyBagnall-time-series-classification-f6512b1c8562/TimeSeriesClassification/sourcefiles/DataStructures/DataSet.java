package DataStructures;

import java.io.File;
import java.io.FileFilter;
import java.io.FileReader;
import java.io.LineNumberReader;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

import utilities.GlobalValues;
import utilities.Logging;
import utilities.Logging.LogLevel;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instances;
import DataStructures.FeaturePoint.PointStatus;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectOutputStream;
import java.io.ObjectStreamClass;
import java.io.Serializable;
import java.util.Collections;

import libsvm.svm_node;
import libsvm.svm_problem;
import weka.core.DenseInstance;
import weka.core.converters.ConverterUtils.DataSink;


/*
 * A dataset described as a list of instances:(featureVector,label)
 */
public class DataSet implements Serializable
{
	
	public String name;
	
	/*
	 * A list of instances
	 */
	public List<DataInstance> instances;
        
    public int numFeatures;
	
	public List<Double> nominalLabels;
	
	/*
	 * Constructors
	 */
	public DataSet() 
	{
		instances = new ArrayList<DataInstance>();
		nominalLabels =  new ArrayList<Double>();
        numFeatures = 0;
        
        name = "dummyDataSet";
	}
        
    public DataSet(Instances wekaInstances) 
	{
		instances = new ArrayList<DataInstance>();
		
		nominalLabels =  new ArrayList<Double>();
                
    	LoadWekaInstances(wekaInstances);
            
    	ReadNominalTargets();
	}
	
	public DataSet( DataSet dataSet )
	{
            name = dataSet.name;
            
            if(name == null) name = "";
            
            instances = new ArrayList<DataInstance>();
            nominalLabels = new ArrayList<Double>();

            numFeatures = dataSet.numFeatures;

            for( DataInstance ins : dataSet.instances )
            {
                    instances.add( new DataInstance(ins) );
            }
            
            ReadNominalTargets();
	}
	

	/*
	 * Gets the overall minimum value among all features
	 */
	public double GetOverallMinimum()
	{
		double minimum = Double.POSITIVE_INFINITY;
		
		for( DataInstance ins : instances )
		{
			double min = ins.GetMinFeatureValue();
			
			if(min < minimum)
			{
				minimum = min;
			}
		}
		
		return minimum;
	}
	
	
	/*
	 * Get the maximum value among all features of the dataset
	 */
	
	public double GetOverallMaximum()
	{
		double maximum = Double.NEGATIVE_INFINITY;
		
		for( DataInstance ins : instances )
		{
			double max = ins.GetMaxFeatureValue();
			
			if(max > maximum)
			{
				maximum = max;
			}
		}
		
		return maximum;
	}
	
	
	
	/*
	 * Load a folder containing data set files
	 */
	public void LoadDataSetFolder(File dataSetFolder)
	{
            if( dataSetFolder == null )
            {
            	Logging.println("Load folder:: null folder file parameter.", Logging.LogLevel.ERROR_LOG);		
                return;
            }
            
            if( !dataSetFolder.exists() )
            {
            	Logging.println("Load folder:: folder doesnt exist", Logging.LogLevel.ERROR_LOG);		
                return;
            }

            Logging.println("Loading:" + dataSetFolder.getPath(), Logging.LogLevel.PRODUCTION_LOG);
            name = dataSetFolder.getName();

            instances = new ArrayList<DataInstance>();

            DataFileNameFilter filter = new DataFileNameFilter();

            File[] timeSeriesFilesList = null;

            if (dataSetFolder != null) 
            {
                timeSeriesFilesList = dataSetFolder.listFiles(filter);

                for( File timeSeriesFile : timeSeriesFilesList )
                {
                        LoadDataSetFile(timeSeriesFile);
                }
                
                ReadNominalTargets();
            } 
            else 
            {
                    System.err.println("Cannot open data directory!");
                    System.exit(-1);
            }
	}
	
	/*
	 * Load UCR dataset file as a dataset
	 */
	public void LoadDataSetFile(File dataSetFile)
	{
        try
        {
            name = dataSetFile.getName();
//            LineNumberReader lr = new LineNumberReader(new BufferedReader( new FileReader(dataSetFile)));
            LineNumberReader lr = new LineNumberReader(new InputStreamReader(new BufferedInputStream(new FileInputStream(dataSetFile))));
            
            // Read each line and tokenize it to read the time series data
            String line = null;

            int lineNumber = 1;
            
            int numTotalCells = 0, numSparseCells = 0;
            
            while ( (line = lr.readLine()) != null )
            { 
                DataInstance ins = new DataInstance();
                ins.name = String.valueOf(instances.size());

                String delimiters = "\t ,;";
                StringTokenizer tokenizer = new StringTokenizer(line, delimiters);
                

                // the first value of the line is the label in cases of classification
                String labelToken = null;
                if(tokenizer.hasMoreElements())
                	labelToken = tokenizer.nextToken();
                else
                {
                	Logging.println("Error: read " + name +  ", line=" + lineNumber, LogLevel.ERROR_LOG);
                	break;
                }
                
        		if(labelToken.compareTo("?")==0)
        			ins.target = GlobalValues.MISSING_VALUE;
        		else
        			ins.target = Double.parseDouble(labelToken);
                
                ins.features = new ArrayList<FeaturePoint>();
                double time = 0.0;
                

                // read the time series featureVector
                while( tokenizer.hasMoreTokens() )
                {
                        FeaturePoint p = new FeaturePoint();
                        String token = tokenizer.nextToken();
                        
                        if( token.toUpperCase().compareTo("NAN") == 0 
                        		|| token.compareTo("?") == 0 || token.compareTo("") == 0)
                        {
                        	p.status = PointStatus.MISSING;
                        	numSparseCells++;
                        }
                        else
                        {
                            p.value = new Double(token).doubleValue();
                            p.status = PointStatus.PRESENT;
                
                        }                        

                        numTotalCells++;
                        
                        ins.features.add(p);

                        time++;
                }

                instances.add(ins);  
                
                if( instances.size() <= 1)
                {
                    // the feature size is recorded by the first instance added
                    numFeatures = ins.features.size();
                }
                else
                {
                    // check if there is a different features size for other instances
                    if( numFeatures != ins.features.size() )
                    {
                    	/*
                        Logging.println("Load UCR file: Different number of features on instance: " + String.valueOf(lineNumber)
                                           + ", " + String.valueOf(numFeatures) + " to " + String.valueOf(ins.features.size()), 
                                Logging.LogLevel.ERROR_LOG);
                        */
                    	
                        if(ins.features.size() > numFeatures)
                        	numFeatures = ins.features.size(); 
                    }
                }
                
                lineNumber++;
            }
            
            lr.close();
            	
            //Logging.println("Sparse cells " + numSparseCells + " out of total cells " + numTotalCells, 
              //      Logging.LogLevel.ERROR_LOG);
        }
        catch(Exception exc)
        {
                exc.printStackTrace();
                
               
        }
        

	}
	
	
	/*
	 * Normalize a dataset by scaling everypoint up to a maximum of 1
	 */
	
	public void NormalizeZeroToOne()
	{
		for(DataInstance ins : instances )
        {
			double max = ins.GetMaxFeatureValue();        
			
			for(  FeaturePoint fp : ins.features )
			{
				fp.value = fp.value/max;
			}
        }
	}
	
	public void NormalizeDatasetInstances()
	{
            
            for(DataInstance ins : instances )
            {
                    ins.NormalizeFeaturesValues();
            }
                   
	}
	
	public void NormalizeDatasetFeatures()
	{
		for(int f = 0; f < numFeatures; f++)
		{
			double mean = 0.0, stdev = 0.0;
			int countNonSparse = 0;
		
			// compute mean
			for(int i = 0; i < instances.size(); i++)
			{
				FeaturePoint fp = instances.get(i).features.get(f);
				
				if( fp.status == PointStatus.PRESENT )
				{
					mean = mean + fp.value;
					countNonSparse++;
				}
			}
			
			mean = mean / countNonSparse;

			// compute standard deviation
			for(int i = 0; i < instances.size(); i++)
			{
				FeaturePoint fp = instances.get(i).features.get(f);
				
				if( fp.status == PointStatus.PRESENT )
				{
					double diff = mean - fp.value; 
					stdev += diff * diff;
				}
			}
			
			stdev = Math.sqrt( stdev / countNonSparse-1 );
			
			System.out.println("Feature " + f + ", mean=" + mean + ", stdev=" + stdev);

			// normalize values
			for(int i = 0; i < instances.size(); i++)
			{
				FeaturePoint fp = instances.get(i).features.get(f);
				
				if( fp.status == PointStatus.PRESENT )
				{
					double newVal = stdev > 0 ?
								(fp.value- mean)/stdev :
									fp.value- mean;
					
					//instances.get(i).features.set(f, new FeaturePoint(newVal));
				}
			}
			
			
		}
        
	}
	
	/*
	 * Gather a list of the nominal labels 
	 */
	
	public void ReadNominalTargets()
	{
		if( instances.size() > 0 )
		{
			nominalLabels = new ArrayList<Double>();
			
			for(DataInstance ins : instances)
			{	 
                boolean alreadyAdded = false;
                        
                for(int i = 0; i < nominalLabels.size(); i++)
                {
                    if( nominalLabels.get(i) == ins.target )
                    {
                         alreadyAdded = true;
                         break;
                    }
                }
                
                if(!alreadyAdded)
                {
                    nominalLabels.add( ins.target );
                }
			}
                        
            Collections.sort(nominalLabels); 
		}
	}
	
	/*
	 * Export the database as a Weka DataSet
	 */
	
	public Instances ToWekaInstances()
	{
		Instances wekaInstances = null;
		
		if( instances != null && instances.size() > 0 )
		{
                    // create a list of attributes features + label
                    FastVector attributes = new FastVector(numFeatures+1);

                    // add the numeric feature attributes 
                    for( int i = 0; i < numFeatures; i++ )
                    {
                        attributes.addElement( new Attribute("attr" + String.valueOf(i+1)) );
                    } 
                    
                	ReadNominalTargets();

                    // add the label attribute as a nominal attribute
                    FastVector labels = new FastVector(nominalLabels.size());
                    for( int i = 0; i < nominalLabels.size(); i++ )
                    {
                        labels.addElement( String.valueOf(nominalLabels.get(i)) ); 
                    }

                    Attribute labelAttribute = new Attribute("class", labels);
                    attributes.addElement( labelAttribute );
                    
                    // add the attributes 
                    wekaInstances = new Instances(name, attributes, instances.size());
                    // set the class attribute by defining its index
                    // the index of the class feature is the noFeatures one 
                    wekaInstances.setClass(labelAttribute);

                    // add the values
                    for( int i = 0; i < instances.size(); i++ )
                    {
                        //Instance ins = new Instance(numFeatures+1);
                        double [] instanceValues = new double[numFeatures+1];

                        for( int j = 0; j < numFeatures; j++ )                         
                        {
                            try
                            {
                                if( instances.get(i).features.get(j).status == PointStatus.MISSING )
                                    instanceValues[j] = weka.core.Utils.missingValue();
                                else
                                    instanceValues[j] = instances.get(i).features.get(j).value;
                            }
                            catch(Exception exc)
                            {
                                //System.out.println( i + ", " + j ); 
                            }
                        }

                        String label = "";
                        
                        try
                        {
                            //int labelIndex = labelAttribute.indexOfValue( String.valueOf( instances.get(i).label )); 
                            //instanceValues[numFeatures] = Double.parseDouble(labelAttribute.value(labelIndex));
                            instanceValues[numFeatures] = labelAttribute.indexOfValue( String.valueOf( instances.get(i).target ));
                            
                            //ins.setValue(labelAttribute, labelAttribute.indexOfValue(label) instances.get(i).label); 
                            
                        }
                        catch(Exception exc)
                        {
                            Logging.println(exc.getMessage(), Logging.LogLevel.PRODUCTION_LOG);
                            Logging.println("Error adding label value: " + label, Logging.LogLevel.PRODUCTION_LOG);
                        }
                        
                        // add the instance values
                        //wekaInstances.add( ins );
                        wekaInstances.add( new DenseInstance(1.0, instanceValues) );
                        
                        /*
                        System.out.println( 
                                instances.get(i).label + 
                                " str " + String.valueOf(instances.get(i).label) +
                                " weka " + wekaInstances.lastInstance().classValue());
                        */
                    }
		}		
	
		return wekaInstances;
	}
	
	/*
	 * convert the dataset to a libsvm problem
	 */
	public svm_problem ToLibSvmProblem()
	{
		  svm_problem data = new svm_problem();
	        
	        data.l = instances.size();
	        data.x = new svm_node[instances.size()][];
	        data.y = new double[instances.size()];
	                
	        // iterate through all the instances
	        for(int i = 0; i < instances.size(); i++)
	        {
	            DataInstance ins = instances.get(i);
	            
	            List<svm_node> nodes = new ArrayList<svm_node>();
	            
	            // iterate through the feature values of the instance and create nodes
	            // for the nonmissing values only 
	            for(int j = 0; j < numFeatures; j++)
	            {
	            	if( instances.get(i).features.get(j).status != PointStatus.MISSING )
	            	{
	            	    svm_node node = new svm_node();
		                node.index = j+1;
		                node.value = ins.features.get(j).value; 
		                
		                nodes.add(node);
	            	}
	            } 
	            
	            // flush the list as array
	            data.x[i] = new svm_node[nodes.size()];
	            for(int k = 0; k < nodes.size(); k++)
	            {
	            	data.x[i][k] = nodes.get(k);
	            }
	            
	            data.y[i] = ins.target;	            
	            
	        }
		
		return data;
	}
	
        // save dataset to arff format
	public void SaveToArffFile(String file)
        {
            Instances wekaInstances = ToWekaInstances();
            
             try
             {
                 File f = new File(file);
                 if(f.exists()) f.delete();

                 DataSink.write(file, wekaInstances);
                 
                 Logging.println("Saved arff: " + file, Logging.LogLevel.ERROR_LOG);
             }
             catch(Exception exc)
             {
                 exc.printStackTrace();
             }
        }
        
	/*
	 * Load a dataset from two matrixes of data and labels
	 */
	
    public void LoadMatrixes(Matrix data, Matrix labels)
    {
        instances = new ArrayList<DataInstance>();
        nominalLabels = new ArrayList<Double>();
        
        LoadMatrixes(data, labels, 0, data.getDimRows());
    }
    
    public void LoadMatrixes(Matrix data, Matrix labels, int startRow, int endRow)
    {
		if( data.getDimRows() == labels.getDimRows() )
		{
            name = "";
            
            if( instances == null )
            {
                instances = new ArrayList<DataInstance>();
            }
            
            numFeatures = data.getDimColumns();

            for(int row = startRow; row < endRow; row++)
            {
                    DataInstance instance = new DataInstance();
                    instance.name = String.valueOf(row);

                    for(int col = 0; col < numFeatures; col++) 
                    {
                            FeaturePoint p = new FeaturePoint();
                            p.value = data.get(row, col);
                            
                            if( data.isMissing(row, col) )
                                p.status = PointStatus.MISSING;
                            else
                                p.status = PointStatus.PRESENT;  

                            instance.features.add( p );
                    }

                    instance.target = Math.rint(labels.get(row, 0));

                    instances.add(instance);
            }			
		}
    }
        
    // save the dataset to a file
    public void SaveToFile(String file)
    {
        int numTimeSeries = instances.size();
        
        try
        {
            File f = new File(file);
            f.delete(); 
            
            BufferedWriter out = new BufferedWriter(new FileWriter(f));
            
            for( int i = 0; i < numTimeSeries; i++ )
            {            
                out.write( instances.get(i).target + " ");
                        
                for( int j = 0; j < numFeatures; j++ )
                {
                    //out.write( df.format(instances.get(i).features.get(j).value) + " ");
                    out.write( String.valueOf( instances.get(i).features.get(j).value ) );
                    
                    if( j < numFeatures-1) 
                    {
                        out.write(" ");
                    }
                    else
                    {
                        out.write("\n");
                    }
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
        
        
    /*
     * Load a weka instances dataset
     */
    public void LoadWekaInstances(Instances ds)
    {
    	name = ds.relationName();
        
    	if(name == null) name = "";
    	
        nominalLabels = new ArrayList<Double>();
        ReadNominalTargets();
	
        numFeatures = ds.numAttributes()-1;
            
        instances = new ArrayList<DataInstance>();
        
        for(int i = 0; i < ds.numInstances(); i++ )
            instances.add(new DataInstance(String.valueOf(i), ds.instance(i) ) );
    }
        
        /*
         * Append instances of a dataset to the end of the current instances
         */
        public void AppendDataSet(DataSet dataSet)
        {
            if( instances == null || instances.size() <= 0)
            {
                Logging.println("Appending dataset to an already empty dataset!", Logging.LogLevel.PRODUCTION_LOG);
            }
            else if(numFeatures != dataSet.numFeatures)
            {
                   Logging.println("Append dataset: Num Features don't match: " 
                           + numFeatures + " " + dataSet.numFeatures, Logging.LogLevel.PRODUCTION_LOG);
            } 
            else
            {
                // append the instances in the end of the existing lists
                for( DataInstance dataInstance : dataSet.instances )
                {
                    instances.add( new DataInstance( dataInstance ) );
                }
                
                ReadNominalTargets();
            }
        }
        
        /*
         * Check if a dataset contains a data instance
         * returns the index of the first match or -1 otherwise
         */
        public int Search( DataInstance ins )
        {
           for(int i = 0; i < instances.size(); i++ )
           {
               if( ins.IsEqual(instances.get(i)) )
               {
                   return i;
               }
           }
           
           // if it wasn't equal to any instance then report it to not
           // be found
           return -1;
        }
        
        /*
         * Search for multiple matches
         */
        public List<Integer> SearchMultiMatch( DataInstance ins )
        {
           List<Integer> multiMatches = new ArrayList<Integer>();
           
           for(int i = 0; i < instances.size(); i++ )
           {
               if( ins.IsEqual(instances.get(i)) )
               {
                   multiMatches.add(i);
               }
           }
           
           // if it wasn't equal to any instance then report it to not
           // be found
           return multiMatches;
        }
        
        /*
         * Remove the first occurrence of a data instance from the list
         */
        public void Remove(DataInstance di)
        {
            int match = -1;            
        }
        
        /*
         * Remove any duplicate instances 
         */
        public void RemoveDuplicateInstances()
        {
            List<DataInstance> duplicateFreeInstances = new ArrayList<DataInstance>();
            
            for(int i = 0; i < instances.size(); i++)
            {
                boolean alreadyInserted = false;
                
                for(int j = 0; j < duplicateFreeInstances.size(); j++)
                {
                    if(duplicateFreeInstances.get(j).IsEqual( instances.get(i) )  )
                    {
                        alreadyInserted = true;
                        break;
                    }
                }
                
                if(!alreadyInserted)
                {
                    duplicateFreeInstances.add( new DataInstance( instances.get(i) ) ); 
                }
                else 
                {
                    Logging.println("Removed: " + (i+1), Logging.LogLevel.DEBUGGING_LOG);
                }
            }
            
            instances = duplicateFreeInstances;
        }
        
        /*
         * 
         */
        public DataSet GetSubset(int fromIndex, int toIndex)
        {
            DataSet subset = new DataSet();
            subset.name = name;
            subset.numFeatures = numFeatures;
            
            for(int i = fromIndex; i < toIndex; i++)
            {
                subset.instances.add( instances.get(i) ); 
            }
            
            return subset;
        }
        
        
        public DataSet FilterByLabel(double filterLabel)
        {
            DataSet filteredSubset = new DataSet();
            filteredSubset.name = name;
            filteredSubset.numFeatures = numFeatures;
            
            for(int i = 0; i < instances.size(); i++)
            {
                if( instances.get(i).target == filterLabel )
                {
                    filteredSubset.instances.add( instances.get(i) ); 
                }
            }
            
            return filteredSubset;
        }

        public double get(int i, int j)
        {
        	return instances.get(i).features.get(j).value;
        }
        
        public int GetNumInstances()
        {
        	return instances.size();
        }
        
}
/*
 * A file filter for the names of the UCR time series collection files
 */
class DataFileNameFilter implements FileFilter {
	public boolean accept(File pathname) {
		if (pathname.getName().contains("train") || 
			pathname.getName().contains("test") ||
			pathname.getName().contains("TRAIN") ||
			pathname.getName().contains("TEST") ||
			pathname.getName().contains("merged") ) {
			//System.out.println("Reading file: " + list[i].getName());
			return true;
		}
		else {
			return false;
		}
	}
}
