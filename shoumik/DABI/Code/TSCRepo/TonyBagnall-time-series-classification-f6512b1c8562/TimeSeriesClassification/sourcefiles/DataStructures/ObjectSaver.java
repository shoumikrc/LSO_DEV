package DataStructures;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import utilities.Logging;

/*
 * Save objects to files and load back objects from files
 */

public class ObjectSaver 
{

    
    // save the object to a file
    public void SaveToObjectFile(Object o, String fileName)
    {
    	try 
    	{
			ObjectOutputStream oos = new ObjectOutputStream( new FileOutputStream( fileName) );
			oos.writeObject(o);
			oos.close();
		} 
    	catch (Exception e) 
		{
			Logging.println(e.getMessage(), Logging.LogLevel.ERROR_LOG);
			e.printStackTrace();
		}
    }
    
    // load from object file
    public Object LoadFromObjectFile(String fileName)
    {
    	Object o = null;
    	try 
    	{
			ObjectInputStream ois = new ObjectInputStream( new FileInputStream( fileName) );
			o = ois.readObject();
			ois.close();
		} 
    	catch (Exception e) 
		{
			Logging.println(e.getMessage(), Logging.LogLevel.ERROR_LOG);
			e.printStackTrace();
		}
    	
    	return o;
    }
    
}
