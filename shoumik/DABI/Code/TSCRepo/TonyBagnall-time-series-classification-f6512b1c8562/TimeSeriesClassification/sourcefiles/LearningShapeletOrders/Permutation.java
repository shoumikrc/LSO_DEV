package LearningShapeletOrders;

import java.util.ArrayList;
import java.util.List;

public class Permutation {
	
	static void combinationUtil(int arr[], int data[], int start,
            int end, int index, int r, List<int[]> combines)
{
// Current combination is ready to be printed, print it
		if (index == r)
		{
//			for (int j=0; j<r; j++)
//                System.out.print(data[j]+" ** ");
// 
            int[] save = new int[data.length];
            for (int j=0; j<r; j++)
            	save[j] = data[j];
            int n = combines.size();
			combines.add(n, save);
//			for(int i=0; i< combines.size(); i++){
//				for(int j=0; j< combines.get(i).length; j++){
//					System.err.print(combines.get(i)[j] + " $$");
//				}
//				System.err.println();
//			}
			return;
			
		}

// replace index with all possible elements. The condition
// "end-i+1 >= r-index" makes sure that including one element
// at index will make a combination with remaining elements
// at remaining positions
		for (int i=start; i<=end && end-i+1 >= r-index; i++)
		{
		data[index] = arr[i];
		combinationUtil(arr, data, i+1, end, index+1, r, combines);
		}
}

// The main function that prints all combinations of size r
// in arr[] of size n. This function mainly uses combinationUtil()
static void printCombination(int arr[], int n, int r, List<int[]> combines)
{
// A temporary array to store all combination one by one
	int data[]=new int[r];

	// Print all combination using temprary array 'data[]'
	combinationUtil(arr, data, 0, n-1, 0, r, combines);
}

/*Driver function to check for above function*/
public static void main (String[] args) {
	int arr[] = {0, 1, 2, 3, 4};
//	ArrayList<int[]> combines = new ArrayList<int[]>();
	int r = 4;
	int n = arr.length;
//	printCombination(arr, n, r, combines);
	
	ArrayList<Double> probs = new ArrayList<Double>();
	probs.add(0.2);
	probs.add(0.4);
	probs.add(0.6);
	
	//LearnGeneralOrderedShapeletsKNNProbabilisticOutputVersion2.computehasOrderProb(probs);
}

}
