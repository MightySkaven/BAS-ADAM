package Ensemble_SI;
import java.util.Arrays;
import java.util.Comparator;
//import java.util.Random;


public class DE_Param{


    public int DeNum = 10;        // numbers of individuals
    public int DeDim = 21;        // dimensions
    public int trial=5;		// the number of trials with different random initial
	public int  MaxGen=30;		// Maximum generation for optimization, quite large, 
	

	public double de_dw;		// the differential weight [0,2], a chinese paper recommend to [0.5, 2], 0.5 fly;
	public double de_cr;		// the crossover rate [0,1]
	public double de_cp;		// the crossover probability [0,1]
	
	public double de_lambda;

	public double[] MaxLim;	// Maximum limitation 
	public double[] MinLim; // Minimum limitation
	public double[] LimScale; // limit scale between max and min
	
    public double total_round; // total round for average round
    
	// for random getting numbers 
	public long random = System.currentTimeMillis();
	
}
