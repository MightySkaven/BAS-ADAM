package Ensemble_SI;
import java.util.Arrays;
import java.util.Comparator;
//import java.util.Random;


public class BAS_Param{


    public int BASPopNum = 10;        // numbers of BAS population, no use for single general BAS algorithm and Adam_BAS algorithm
    public int BASDim = 21;        // dimensions
    public int MaxGen=30;		// Maximum generation for optimization, quite large, 
	

	public double bas_delta0 = 0.1;		//  0.95
	public double bas_eps=0.01;		// the eps value for d, avoiding the zero for d
	
	public double bas_antlength0 = 0.1;  // for original bas, d0 is set 2.0; for github program, d0 is step/c = 1/5 = 0.2
	// 11/18, antlength0 =2, no optimization; 
	public double bas_delta = 0.5; // step size for BAS. for original bas, delta is set 0.5; for github program, delta is set as 1
	
	public double regular_lambda;         // lambda*(pu2+b2), regulation item
	 

    public double total_round; // total round for average round
    
    // the initial variables for adam
    public double m0;
    public double v0;
    public double theta0;
    

}