package Ensemble_SI;

public class PSO_Param
{
	public int PopNum;		// Number of populations
	public double SearchRange;// Particle initial search range
	public double w;		// Inertia constant
	public double c1;		// Individual confidence factor
	public double c2;		// Swarm confidence factor
	public double Rao;		// Coefficient for PPSO algorithm
	public double GDFadingFactor;	//GD PSO fading factor
	public int Opt_Type;	// Optimizer type: 0 for maximize; 1 for minimize
	public int MaxGen;		// Maximum generation for optimization
	public int N;			// Length of vector to be optimized
	public double[] MaxLim;	// Maximum limitation 
	public double[] MinLim; // Minimum limitation
	public double ErrLim;	// Error tolerance
	
	public CommonRecomm_NEW.OptimizeAlgorithmSet PSOApprType;  // add by CJ on 2021/08/01 for introducing the pso type into pso_optimizer
	//public PSOApproach PSOMethodType;	// PSO method type
	
    public double total_round; // total round for average round, added by CJ
}