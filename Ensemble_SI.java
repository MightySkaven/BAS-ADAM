package Ensemble_SI;
import java.io.IOException;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Random;

import Ensemble_SI.BAS_Param;
import Ensemble_SI.CommonRecomm_NEW;
import Ensemble_SI.RTuple;
import Ensemble_SI.CommonRecomm_NEW.IndexSort;
import Ensemble_SI.CommonRecomm_NEW.LossFunSet;
import Ensemble_SI.CommonRecomm_NEW.TrainErrorSet;

import java.util.Arrays;


import java.util.Date;
import java.text.DateFormat;
import java.text.SimpleDateFormat;


public class Ensemble_SI extends CommonRecomm_NEW {

    public Ensemble_SI( ) throws NumberFormatException, IOException {
        super();

        // TODO Auto-generated constructor stub
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //      The below codes are for PSO optimization
    //      PSO_optimizer() is for optimizing one pu,bu or qi,ci via extending the input as a K particle swarm
    //      train_BatchPSO() is for mini_batch PSO algorithm, which calling PSO_optimizer sequentially as each pu,bu or qi,ci
    //      train_pso_from_yuanye() is for the first layer of HPL—— the PLFA model
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Optimizer based on Evolutionary Algorithm (2D cost function: L=(r - x.y - b - c) ^2 + lambda * (x.^2 + b.^2))
    public void PSO_Optimizer(ArrayList<RTuple> TrainData, PSO_Param Param, double[] x, double[][] y, double[][] c, int[] Gen) throws IOException
    {
        double[] MaxLim = new double[Param.N];
        double[] MinLim = new double[Param.N];
        double[] LimScale = new double[Param.N];
        for (int l = 0; l < Param.N; l++)
        {
            MaxLim[l] = Param.MaxLim[l];
            MinLim[l] = Param.MinLim[l];
            LimScale[l] = MaxLim[l] - MinLim[l];
        }
        int gen = 0;
        double Err = 0;
        double Err1 = 0;
        int DataSize = TrainData.size();

        // Init population and parameters
        double[][] x_Pops = new double[Param.PopNum][Param.N];		// Swarm location of x
        double[][] v_Pops = new double[Param.PopNum][Param.N];
        double[][] x_PBest = new double[Param.PopNum][Param.N];		// Best Swarm location of x
        double[] F_PBest = new double[Param.PopNum];				// Best swarm fitness of x
        double[] x_GBest = new double[Param.N];						// Global best swarm location of x
        double F_GBest;												// Best global fitness of P and x
        double[][] G_Pops = new double[Param.PopNum][Param.N];		// Gradient of swarm

        double[][] x_Pops1 = new double[Param.PopNum][Param.N];		// Swarm location of x
        double[][] v_Pops1 = new double[Param.PopNum][Param.N];
        double[][] x_PBest1 = new double[Param.PopNum][Param.N];	// Best Swarm location of x
        double[] F_PBest1 = new double[Param.PopNum];				// Best swarm fitness of x
        double[] x_GBest1 = new double[Param.N];					// Global best swarm location of x
        double F_GBest1;											// Best global fitness of P and x
        double[] tempC = new double[DataSize];
        double[] Rij = new double[DataSize];

        for (int k = 0; k < DataSize; k++)
        {
            tempC[k] = 0;
            for (int l = 0; l < c[k].length; l++)
                tempC[k] += c[k][l];
            Rij[k] = TrainData.get(k).dRating - tempC[k];
        }

        Random random = new Random(System.currentTimeMillis());
        F_GBest = Double.MAX_VALUE;
        F_GBest1 = Double.MAX_VALUE;

        // initialize the pu,bu or qi, ci 's particle swarm while fixing other paras
        // random the specific one pu,bu to K particles
        for (int i = 0; i < Param.PopNum; i++)
        {
            //double RandNum = random.nextDouble();
            double RandNum = random.nextDouble()*2 - 1.0;

            // the 1st particle equals to the original one
            if (i == 0)
                for (int l = 0; l < Param.N; l++)
                {
                    x_PBest[i][l] = x_PBest1[i][l] = x_Pops[i][l] = x_Pops1[i][l] = x[l];
                    //x_PBest[i][l] = x_Pops[i][l] = MinLim[l] + RandNum * LimScale[l];
                    v_Pops[i][l] = v_Pops1[i][l] = x_Pops[i][l] * 0.1;
                }
                // the other particles equals to random ones
            else
                for (int l = 0; l < Param.N; l++)
                {

                    x_PBest[i][l] = x_PBest1[i][l] = x_Pops[i][l] = x_Pops1[i][l] = x[l]*(1-0.1*RandNum); //x[l] + (RandNum - 0.5) * LimScale[l];
                    v_Pops[i][l] = v_Pops1[i][l] = x_Pops[i][l] * 0.1;
                }

            // Initialize Fitness of PBest of current swarm and search for GBest
            F_PBest[i] = 0;
            for (int l = 0; l < Param.N; l++)
            {
                F_PBest[i] += x_PBest[i][l] * x_PBest[i][l];
                G_Pops[i][l] = -lambda * x_PBest[i][l];
            }
            F_PBest[i] *= CommonRecomm_NEW.lambda*DataSize;
            double tempB = 0;
            for (int l = CommonRecomm_NEW.featureDimension; l < Param.N; l++)
            {
                tempB += x_PBest[i][l];
            }
            for (int k = 0; k < DataSize; k++)
            {
                double tempXY = 0;
                for (int l = 0; l < CommonRecomm_NEW.featureDimension;  l++)
                {
                    tempXY += x_PBest[i][l] * y[k][l];
                }
                double tempF = Rij[k] - tempXY - tempB;

                F_PBest[i] += tempF * tempF; // calculating the instant error
                for (int l = 0; l < featureDimension; l++)
                    G_Pops[i][l] += tempF * y[k][l];
                for (int l = featureDimension; l < Param.N; l++)
                    G_Pops[i][l] += tempF;
            }
            F_PBest1[i] = F_PBest[i];
            //System.out.println("Init F_PBest[]" + i + " = " + F_PBest[i]);
            if (i == 0 || F_GBest > F_PBest[i])
            {
                F_GBest = F_PBest[i];
                for (int l = 0; l < Param.N; l++)
                    x_GBest[l] = x_PBest[i][l];
            }
        }
        F_GBest1 = F_GBest;
        System.arraycopy(x_GBest, 0, x_GBest1, 0, Param.N);
        // end of the initialization

        // Iterate to search GBest
        double F_Pops = 1;
        int Delay = 0;
        do
        {
            // Serialized PSO
            double F_GBestTmp = F_GBest;
            Param.ErrLim = F_GBest * 0.0001;

            for (int i = 0; i < Param.PopNum; i++)
            {
                // Update VP and P
                double RandNum1 = random.nextDouble();  // the 2 random numbers to calculate v[i][l] should be the SAME for all l
                double RandNum2 = random.nextDouble();
                for (int l = 0; l < Param.N; l++)
                {
                    // update v and x of each dimension
                    switch (Param.PSOApprType)
                    {
                        case BPSO:
                            v_Pops[i][l] = Param.w * v_Pops[i][l] + Param.c1 * RandNum1 * (x_PBest[i][l] - x_Pops[i][l]) +
                                    Param.c2 * RandNum2 * (x_GBest[l] - x_Pops[i][l]);

                            break;
                        case PPSO:
                            v_Pops[i][l] = Param.w * v_Pops[i][l] + Param.c1 * RandNum1 * (x_PBest[i][l] - x_Pops[i][l]) +
                                    Param.c2 * RandNum2 * (x_GBest[l] - x_Pops[i][l]) +
                                    Param.Rao * (Param.c1 * RandNum1 + Param.c2 * RandNum2) * (x_Pops[i][l] - v_Pops[i][l]);
                            break;
							/*case GD_PSO:          comment by CJ on 2021/08/01 for loop all the PSO approach automatically
								double GD_Factor = gen / Param.GDFadingFactor;
								if (GD_Factor > 1)
									GD_Factor = 1;
								//GD_Factor = 1;
								v_Pops[i][l] = GD_Factor * (Param.w * v_Pops[i][l] + Param.c1 * RandNum1 * (x_PBest[i][l] - x_Pops[i][l]) +
			   						    											 Param.c2 * RandNum2 * (x_GBest[l] - x_Pops[i][l])) +
								    		   (1 - GD_Factor) * eta * G_Pops[i][l];
							break;*/
                    }
                    x_Pops[i][l] += v_Pops[i][l];
                }

                // Evaluate fitness of current swarm
                F_Pops = 0;
                for (int l = 0; l < Param.N; l++)
                {
                    F_Pops += x_Pops[i][l] * x_Pops[i][l];
                    G_Pops[i][l] = -lambda * x_Pops[i][l];
                }

                F_Pops *= CommonRecomm_NEW.lambda*DataSize;
                double tempB = 0;
                for (int l = CommonRecomm_NEW.featureDimension; l < Param.N ; l++)
                {
                    tempB += x_Pops[i][l];
                }

                for (int k = 0; k < DataSize; k++)
                {
                    double tempXY = 0;
                    for (int l = 0; l < CommonRecomm_NEW.featureDimension;  l++)
                    {
                        tempXY+= x_Pops[i][l] * y[k][l];
                    }

                    double tempF = Rij[k] - tempXY - tempB;
                    F_Pops += tempF * tempF;
                    for (int l = 0; l < featureDimension; l++)
                        G_Pops[i][l] += tempF * y[k][l];
                    for (int l = featureDimension; l < Param.N; l++)
                        G_Pops[i][l] += tempF;
                }

                // updated pbest
                if (F_Pops < F_PBest[i])
                {
                    F_PBest[i] = F_Pops;
                    for (int l = 0; l < Param.N; l++)
                        x_PBest[i][l] = x_Pops[i][l];
                }

                // update gbest
                if (F_Pops < F_GBest)
                {
                    //Err = F_GBest - F_Pops;		// Fitness improvement
                    F_GBest = F_Pops;
                    for (int l = 0; l < Param.N; l++)
                        x_GBest[l] = x_Pops[i][l];
                }
            }

            //
            for (int i = 0; i < Param.PopNum; i++)
            {
                if (F_PBest[i] < F_GBest)
                {
                    //Err = F_GBest - F_Pops;		// Fitness improvement
                    F_GBest = F_PBest[i];
                    for (int l = 0; l < Param.N; l++)
                        x_GBest[l] = x_PBest[i][l];
                }
            }

            if (DataSize == 0)
            {
                Err1 = 10;
            }
            else {
                Err1 = F_GBestTmp - F_GBest;		// Fitness improvement compared with the last iter
            }

            if (Err1 >= Param.ErrLim)
            {
                Delay = 0;
            }
            else
            {
                // if the fitness improvement compared with last iteration is less than ErrLim for two times, exit the iteration
                Delay = Delay + 1;
            }

            gen++;


        } while (gen < Param.MaxGen && Delay <= 1); // err less the 0.1 for two times || iteration times greater than MaxGen, break the iterations

        Gen[0] = gen;

        for (int l = 0; l < Param.N; l++)
        {
            x[l] = x_GBest[l];
        }

    }


    // Train using Batch-PSO
    public void train_BatchPSO(LossFunSet LossFun, TrainErrorSet TrainError, PSO_Param Param, PrintStream p1) throws IOException
    {

        //
        double[][] tempPu;
        double[][] tempBu;
        double[][] tempQi;
        double[][] tempCi;

        ArrayList<RTuple> Data = new ArrayList<RTuple>();

        this.cacheMinFeatures();

        min_Error_RMSE = this.validCurrentRMSE(LossFun);
        min_Error_MAE = this.validCurrentMAE(LossFun);

        long TimeStart = System.currentTimeMillis();
        for (round = 0; round <= trainingRound; round++)
        {
            switch (LossFun)
            {

                case TwoDimMF:
                    // Use EA method to optimize each Pi
                    Param.N = featureDimension + B_Count;
                    Param.MinLim = new double[Param.N];
                    Param.MaxLim = new double[Param.N];
					/*if (Param.PSOMethodType == PSOApproach.GD_PSO && round <= 5) comment by CJ on 2021/08/01 for loop all the pso approaches autolly
						Param.MaxGen = 1;
					else
						Param.MaxGen = 2;*/
                    Param.MaxGen = 2;

                    // time added for recording calling Optimizer
                    // added for debugging the time, compared with DE, 2021_05_05
                    long STimeDEOptimizer;
                    long ETimeDEOptimizer;

                    STimeDEOptimizer = System.currentTimeMillis(); // START TIME

                    // loop for each i, that means each user
                    for (int i = 1; i <= user_MaxID; i++)
                    {

                        double MinP = Arrays.stream(P[i]).min().getAsDouble();
                        double MaxP = Arrays.stream(P[i]).max().getAsDouble();
                        double MinB = Arrays.stream(B[i]).min().getAsDouble();
                        double MaxB = Arrays.stream(B[i]).max().getAsDouble();
                        double CenterP = (MinP + MaxP) / 2;
                        double RangeP = (MaxP - MinP) * 3;
                        double CenterB = (MinB + MaxB) / 2;
                        double RangeB = (MaxB - MinB) * 3;


                        // each user's f+1 dimension limitations, the first f
                        for (int j = 0; j < featureDimension; j++)
                        {
                            Param.MinLim[j] = CenterP - RangeP/2;
                            Param.MaxLim[j] = CenterP + RangeP/2;
                        }
                        // each user's f+1 dimension limitations, the last f+1
                        for (int j = featureDimension; j < Param.N; j++)
                        {
                            Param.MinLim[j] = CenterB - RangeB/2;
                            Param.MaxLim[j] = CenterB + RangeB/2;
                        }

                        // Pick up {R(i,j)} that j belongs to Omiga_P_Q[i], the already known data
                        Data.clear();
                        tempQi = new double[Omiga_P_Q[i].length][featureDimension];
                        tempCi = new double[Omiga_P_Q[i].length][C_Count];

                        for (int j = 0; j < Omiga_P_Q[i].length; j++)
                        {
                            //
                            Data.add(trainData.get(Omiga_P_R[i][j]));
                            //
                            System.arraycopy(Q[Omiga_P_Q[i][j]], 0, tempQi[j], 0, featureDimension);
                            //
                            System.arraycopy(C[Omiga_P_Q[i][j]], 0, tempCi[j], 0, C_Count);
                        }

                        // input the estimated p,b for optimizing
                        double[] x = new double[Param.N];
                        for (int j = 0; j < featureDimension; j++)
                            x[j] = P[i][j];
                        for (int j = 0; j < B_Count; j++)
                            x[j + featureDimension] = B[i][j];

                        int[] gen = new int[1];

                        /*  Data          the known data for matrix R,
                         *  Param         parameters for PSO
                         *  x             the total |f| p[i] and f+1 dimension b, estimated parameters
                         *  tempQi        fixed estimated paras qi
                         *  tempCi        fixed estimated paras ci
                         *  gen           not care
                         */
                        PSO_Optimizer(Data, Param, x, tempQi, tempCi, gen);

                        // output the optimized paras pu,bu to estimated matrix
                        for (int j = 0; j < featureDimension; j++)
                            P[i][j] = x[j];
                        for (int j = 0; j < B_Count; j++)
                            B[i][j] = x[j + featureDimension];

                    }
                    // added for debugging the time, compared with DE, 2021_05_05
                    System.out.println("user max ID with train_batchPSO" + user_MaxID);
                    ETimeDEOptimizer = System.currentTimeMillis();

                    //System.out.println("All time DE PB: " + (ETimeDEOptimizer - STimeDEOptimizer) / 1000. + "s");


                    // Use EA method to optimize Qi
                    Param.N = featureDimension + C_Count;
                    Param.MinLim = new double[Param.N];
                    Param.MaxLim = new double[Param.N];

                    //MaxGen = 0;
                    for (int j = 1; j <= item_MaxID; j++)
                    {

                        double MinQ = Arrays.stream(Q[j]).min().getAsDouble();
                        double MaxQ = Arrays.stream(Q[j]).max().getAsDouble();
                        double MinC = Arrays.stream(C[j]).min().getAsDouble();
                        double MaxC = Arrays.stream(C[j]).max().getAsDouble();
                        double CenterQ = (MinQ + MaxQ) / 2;
                        double RangeQ = (MaxQ - MinQ) * 2;
                        double CenterC = (MinC + MaxC) / 2;
                        double RangeC = (MaxC - MinC) * 2;
                        for (int i = 0; i < featureDimension; i++)
                        {
                            Param.MinLim[i] = CenterQ - RangeQ/2;
                            Param.MaxLim[i] = CenterQ + RangeQ/2;
                        }
                        for (int i = featureDimension; i < Param.N; i++)
                        {
                            Param.MinLim[i] = CenterC - RangeC/2;
                            Param.MaxLim[i] = CenterC + RangeC/2;
                        }

                        // Pick up {R(i,j)} that i belongs to Omiga_Q_P[j]
                        Data.clear();
                        tempPu = new double[Omiga_Q_P[j].length][featureDimension];
                        tempBu = new double[Omiga_Q_P[j].length][B_Count];

                        for (int i = 0; i < Omiga_Q_P[j].length; i++)
                        {
                            Data.add(trainData.get(Omiga_Q_R[j][i]));
                            System.arraycopy(P[Omiga_Q_P[j][i]], 0, tempPu[i], 0, featureDimension);
                            System.arraycopy(B[Omiga_Q_P[j][i]], 0, tempBu[i], 0, B_Count);
                        }

                        double[] y = new double[Param.N];
                        for (int i = 0; i < featureDimension; i++)
                            y[i] = Q[j][i];
                        for (int i = 0; i < C_Count; i++)
                            y[i + featureDimension] = C[j][i];

                        int[] gen = new int[1];
                        PSO_Optimizer(Data, Param, y, tempPu, tempBu, gen);



                        for (int i = 0; i < featureDimension; i++)
                            Q[j][i] = y[i];
                        for (int i = 0; i < C_Count; i++)
                            C[j][i] = y[i + featureDimension];
                    }


                    break;
                default:
                    break;
            }

            System.out.println("round = " + round + ", Train RMSE = " + this.trainCurrentRMSE(LossFun) + "; Valid RMSE = " + this.validCurrentRMSE(LossFun) + "; test RMSE = " + this.testCurrentRMSE(LossFun));

            // RMSE / MAE calculation
            double curErr_RMSE;
            double curErr_MAE;

            switch (TrainError)
            {
                case RMSE:

                    curErr_RMSE = this.validCurrentRMSE(LossFun);

                    if (min_Error_RMSE - curErr_RMSE >= min_Error_RMSE * iterConvThrd)
                    {
                        System.out.println("round = " + round + ", Previous min RMSE = " + min_Error_RMSE + ", Find Better RMSE = " + curErr_RMSE + ", Delta RMSE = " + (min_Error_RMSE - curErr_RMSE) + ", " + min_Error_RMSE * iterConvThrd);
                        System.out.println("                       test RMSE = " + this.testCurrentRMSE(LossFun));
                        min_Error_RMSE = curErr_RMSE;

                        this.min_Round = round;
                        this.cacheMinFeatures();
                    }
                    else
                    {
                        this.rollBackMinFeatures();
                    }
                    break;

                case MAE:

                    curErr_MAE = this.validCurrentMAE(LossFun);

                    if (min_Error_MAE - curErr_MAE >= min_Error_MAE * iterConvThrd)
                    {
                        System.out.println("PSO round = " + round + ", Previous min MAE = " + min_Error_MAE + ", Find Better MAE = " + curErr_MAE + ", Delta MAE = " + (min_Error_MAE - curErr_MAE) + "," + min_Error_MAE * iterConvThrd);
                        System.out.println("                           test MAE = " + this.testCurrentMAE(LossFun));
                        min_Error_MAE = curErr_MAE;

                        this.min_Round = round;
                        this.cacheMinFeatures();
                    }
                    else
                    {
                        this.rollBackMinFeatures();
                    }
                    break;

                case RMSE_and_MAE:


                    curErr_RMSE = this.validCurrentRMSE(LossFun);


                    boolean IsRollBack = false;
                    System.out.println("PSO round = " + round + " current RMSE = " + curErr_RMSE + "; Test RMSE = " + this.testCurrentRMSE(LossFun));
                    if (min_Error_RMSE - curErr_RMSE >= min_Error_RMSE * iterConvThrd)
                    {
                        System.out.println("PSO round = " + round + ", Previous min RMSE = " + min_Error_RMSE + ", Find Better RMSE = " + curErr_RMSE + ", Delta RMSE = " + (min_Error_RMSE - curErr_RMSE) + ", " + min_Error_RMSE * iterConvThrd);
                        System.out.println("                           test RMSE = " + this.testCurrentRMSE(LossFun));

                        min_Error_RMSE = curErr_RMSE;
                        this.min_Round = round;
                        this.cacheMinFeatures();

                    }
                    else
                    {
                        IsRollBack = true;
                        this.rollBackMinFeatures();
                    }


                    curErr_MAE = this.validCurrentMAE(LossFun);


                    System.out.println("round = " + round + " current MAE = " + curErr_MAE + "; Test MAE = " + this.testCurrentMAE(LossFun));
                    if (min_Error_MAE - curErr_MAE >= min_Error_MAE * iterConvThrd && !IsRollBack)
                    {
                        System.out.println("round = " + round + ", Previous min MAE = " + min_Error_MAE + ", Find Better MAE = " + curErr_MAE + ", Delta MAE = " + (min_Error_MAE - curErr_MAE) + "," + min_Error_MAE * iterConvThrd);
                        System.out.println("                       test MAE = " + this.testCurrentMAE(LossFun));

                        min_Error_MAE = curErr_MAE;
                        this.min_Round = round;
                        this.cacheMinFeatures();
                    }
                    else
                    {
                        IsRollBack = true;
                        this.rollBackMinFeatures();
                    }

                    //if (IsRollBack)
                    //this.rollBackMinFeatures();
                    break;
            }
            long TimeEnd = System.currentTimeMillis();
            System.out.println("Iteration time with Batch_PSO: " + (TimeEnd - TimeStart) / 1000. + "s");

            //System.out.println("round, " + round + ", Train RMSE , " + this.trainCurrentRMSE(LossFun) + ", Train MAE, " + this.trainCurrentMAE(LossFun) +
            //		", Test RMSE, " + this.testCurrentRMSE(LossFun) + ", Test MAE, " + this.testCurrentMAE(LossFun) + ", Iter Time, " + (System.currentTimeMillis() - TimeStart) / 1000);
            p1.println("PSO round, " + round + ", Train RMSE , " + this.trainCurrentRMSE(LossFun) + ", Train MAE, " + this.trainCurrentMAE(LossFun) +
                    ", Test RMSE, " + this.testCurrentRMSE(LossFun) + ", Test MAE, " + this.testCurrentMAE(LossFun) + ", Iter Time, " + (System.currentTimeMillis() - TimeStart) / 1000);

            if ((round - this.min_Round) >= delayCount) {
                break;
            }
        }


        // Calculate error on test data set
        this.rollBackMinFeatures();
        min_Error_MAE = this.testCurrentMAE(LossFun);
        min_Error_RMSE = this.testCurrentRMSE(LossFun);
        double Final_train_RMSE = this.trainCurrentRMSE(LossFun);
        double Final_train_MAE = this.trainCurrentMAE(LossFun);
        System.out.println("Final round = " + round + " Train RMSE = " + Final_train_RMSE + ";Valid RMSE="+this.validCurrentRMSE(LossFun)+ " Test RMSE = " + min_Error_RMSE);
        System.out.println("Final round = " + round + " Train MAE  = " + Final_train_MAE + ";Valid MAE=" +this.validCurrentMAE(LossFun)+ "  Test MAE  = " + min_Error_MAE);

        p1.println();
        p1.println("PSO Final round = " + round + " Train RMSE = " + Final_train_RMSE + "; Valid RMSE = " + this.validCurrentRMSE(LossFun) + "; Test RMSE = " + min_Error_RMSE);
        p1.println("PSO Final round = " + round + " Train MAE  = " + Final_train_MAE + "; Valid MAE = " + this.validCurrentMAE(LossFun) +  ";Test MAE  = " + min_Error_MAE);
        this.TrainRound = round;

        Param.total_round += (round-2.5); //added by CJ, for averaging validation time
    }






    // PLFA model from yuanye
    public void train_pso_from_yuanye(TrainErrorSet TrainError) throws IOException {

        System.out.println("yuanye");
        double[] tempPu = new double[featureDimension];
        double[] tempDeltaPu = new double[featureDimension];
        double[] tempQi = new double[featureDimension];
        double[] tempDeltaQi = new double[featureDimension];
        double[] everyError=new double[trainingRound];
        for (int round = 1; round <= trainingRound; round++) {   //
            double startTime = System.currentTimeMillis();
            for (int k = 0; k < yy_swarmNum; k++) {
                // 娑擄拷鏉烆喚娈慣ain鏉╁洨鈻�
                for (RTuple tempRating:trainData) {
                    // 濮濄倕顦╁锟芥慨瀣劅娑旓拷
                    double ratingHat = this.getLocPrediction(tempRating.iUserID, tempRating.iItemID);
                    double err = tempRating.dRating - ratingHat;

                    // 閺嶈宓侀弴瀛樻煀鐟欏嫬鍨拋锛勭暬閺囧瓨鏌�:Pu=(1-eta*lambda)Pu+eta*err*Qi;Qi閺囧瓨鏌婄猾璁虫妧
                    vectorMutiply(P[tempRating.iUserID], (1 - particles[k] * lambda), tempPu);
                    vectorMutiply(P[tempRating.iUserID], err * particles[k], tempDeltaPu);

                    vectorMutiply(Q[tempRating.iItemID], (1 - particles[k] * lambda), tempQi);
                    vectorMutiply(Q[tempRating.iItemID], err * particles[k], tempDeltaQi);

                    vectorAdd(tempPu, tempDeltaQi, P[tempRating.iUserID]);
                    vectorAdd(tempQi, tempDeltaPu, Q[tempRating.iItemID]);

                    // Update Bu: Bu' = (1-lambda*eta)bu+eta*err
                    for (int j = 0; j < B_Count; j++) {
                        B[tempRating.iUserID][j] = (1 - lambda * particles[k])
                                * B[tempRating.iUserID][j] + particles[k] * err;
                    }

                    // 閺囧瓨鏌婃い鍦窗閸嬪繐妯奲i = (1-lambda*eta)bi+eta*err
                    for (int j = 0; j < C_Count; j++) {
                        C[tempRating.iItemID][j] = (1 - lambda * particles[k])
                                * C[tempRating.iItemID][j] + particles[k] * err;
                    }

                }
                // 鐠侊紕鐣婚張顒冪枂鐠侇厾绮岀紒鎾存将閸氬函绱濋崷銊︾ゴ鐠囨洟娉︽稉濠勬畱鐠囶垰妯�
                double curErr = 0;
                if (TrainError == TrainErrorSet.RMSE ) {

                    curErr = yy_validateCurrentRMSE();  // yy's codes, 7;1:2

                } else {

                    curErr = yy_validateCurrentMAE();        // yy's codes, 7;1:2

                }
                tempRMSE[k] = curErr;

                if (previous_Error > curErr) {
                    previous_Error = curErr;
                    yy_cacheMinFeatures();
                }
            }

            yy_updatebestRMSE(tempRMSE);
            yy_updateParticlesandV();


            double endTime = System.currentTimeMillis();
            this.cacheTotalTime += endTime - startTime;

			/* Comment by CJ On 2021/07/30 for saving time, and there is no need to output file
			// achieve one of thousand of min_Error, then output file, and continue run the lowest accuracy
			// then use 0.01 to run the MPSO
            if (((min_Error - FitnessRMSEgbest) > min_Error * 0.001) & (RecordInitValuetoFile_1 == true)) {

				System.out.println("Output 0.001 initial values to file");
				FileOutputStream InitValueFile = new FileOutputStream(new File(RecordFile_k  +"_0.001_InitValue.txt"));
				PrintStream p2 = new PrintStream(InitValueFile);

			    for (int kk = 1; kk <= user_MaxID; kk++)
			    {
			    	for (int ll = 0; ll < CommonRecomm_NEW.featureDimension; ll++)
			    		p2.println(P[kk][ll]);
			    	for (int ll = 0; ll < B_Count; ll++)
			    		p2.println(B[kk][ll]);
			    }
			    for (int kk = 1; kk <= item_MaxID; kk++)
			    {
			    	for (int ll = 0; ll < CommonRecomm_NEW.featureDimension; ll++)
			    		p2.println(Q[kk][ll]);
			    	for (int ll = 0; ll < C_Count; ll++)
			    		p2.println(C[kk][ll]);
			    }
			    p2.close();
				System.out.println("test RMSE_0.001:"+yy_testCurrentRMSE()+"test MAE_0.001:"+yy_testCurrentMAE());
			    //p.write("test RMSE_0.015:"+yy_testCurrentRMSE()+"test MAE_0.015:"+yy_testCurrentMAE());
				//RecordInitValuetoFile_1 = false;

				this.minTotalTime_cj += this.minTotalTime+this.cacheTotalTime;
				System.out.println("Min total training time above 0.001:\t\t" + this.minTotalTime/1000);

			}   Comment by CJ On 2021/07/30 for saving time, and there is no need to output file   */


            // if ((min_Error - FitnessRMSEgbest) > min_Error * 0.001) { // commented for running yy coding // this is wrong, can not use
            // run to the convergence conditions, the above modification is wrong
            if (min_Error > FitnessRMSEgbest) {
                min_Error = FitnessRMSEgbest;
                min_Round = round;
                this.minTotalTime += this.cacheTotalTime;
                this.cacheTotalTime = 0;
            } else if ((round - min_Round) >delayCount) {

                break;
            }

            System.out.println(FitnessRMSEgbest);

            everyError[round-1]=min_Error;
        }


        this.yy_rollBackMinFeatures();

        if (TrainError == TrainErrorSet.RMSE) {
            min_Error_RMSE  = yy_testCurrentRMSE();
            min_Error_MAE = yy_testCurrentMAE();
        } else {
            min_Error_RMSE = yy_testCurrentRMSE();
            min_Error_MAE  = yy_testCurrentMAE();
        }

        System.out.println("test RMSE :"+min_Error_RMSE);
        System.out.println("test MAE :"+min_Error_MAE);
        System.out.println(min_Round);
        this.round=min_Round; // added by CJ
        System.out.println("Min total training time:\t\t" + this.minTotalTime/1000);
        System.out.println("Min average training time:\t\t" + this.minTotalTime/ this.min_Round/1000);

    }


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //      The above codes are for PSO optimization
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //      The below codes are for DE optimization, added by CJ, 2021.0405
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////


    // Train using DE

    public void train_DE(LossFunSet LossFun, TrainErrorSet TrainError, DE_Param DEParam, PrintStream p1) throws IOException
    {

        long TimeStart = System.currentTimeMillis();
        long finalRoundTimeEnd = System.currentTimeMillis();

        double[][] tempPu;
        double[][] tempBu;
        double[][] tempQi;
        double[][] tempCi;

        ArrayList<RTuple> Data = new ArrayList<RTuple>();

        this.cacheMinFeatures();

        min_Error_RMSE = this.validCurrentRMSE(LossFun);
        min_Error_MAE = this.validCurrentMAE(LossFun);


        // Use EA method to optimize each Pi, move from the training round iterations
        DEParam.DeDim = featureDimension + B_Count;
        DEParam.MinLim = new double[DEParam.DeDim];
        DEParam.MaxLim = new double[DEParam.DeDim];      //int MaxGen = 0;			DELETE
        //DEParam.LimScale = new double[DEParam.DeDim];

        for (round = 0; round <= trainingRound; round++) // use commandNew.traininground = 100
        {


			/*
			 *   comment on 2021/07/14, no need for debugging and saving time // time added for recording calling DE_Optimizer
			System.out.println("trainingRound="+ round);

            long STimeDEOptimizer;
            long ETimeDEOptimizer;

            STimeDEOptimizer = System.currentTimeMillis(); // START TIME
            */
            //System.out.println(" Train RMSE BEFORE DE_OPTIMIZER = " + this.trainCurrentRMSE(LossFunSet.TwoDimMF) + ";Valid RMSE="+this.validCurrentRMSE(LossFunSet.TwoDimMF)+ " Test RMSE = " + this.testCurrentRMSE(LossFunSet.TwoDimMF));


            // loop for each i, that means each user
            for (int i = 1; i <= user_MaxID; i++)      // System.out.println("i="+i);  DELETE
            {

                double MinP = Arrays.stream(P[i]).min().getAsDouble();
                double MaxP = Arrays.stream(P[i]).max().getAsDouble();
                double MinB = Arrays.stream(B[i]).min().getAsDouble();
                double MaxB = Arrays.stream(B[i]).max().getAsDouble();
                double CenterP = (MinP + MaxP) / 2;
                double RangeP = (MaxP - MinP) * 3;
                double CenterB = (MinB + MaxB) / 2;
                double RangeB = (MaxB - MinB) * 3;

                // each user's f+1 dimension limitations, the first f
                for (int j = 0; j < featureDimension; j++)
                {
                    DEParam.MinLim[j] = CenterP - RangeP/2;
                    DEParam.MaxLim[j] = CenterP + RangeP/2;
                    //DEParam.LimScale[j] = DEParam.MaxLim[j] - DEParam.MinLim[j];
                }
                // each user's f+1 dimension limitations, the last f+1
                for (int j = featureDimension; j < DEParam.DeDim; j++)
                {
                    DEParam.MinLim[j] = CenterB - RangeB/2;
                    DEParam.MaxLim[j] = CenterB + RangeB/2;
                    //DEParam.LimScale[j] = DEParam.MaxLim[j] - DEParam.MinLim[j];
                }

                // Pick up {R(i,j)} that j belongs to Omiga_P_Q[i], the already known data
                Data.clear();
                tempQi = new double[Omiga_P_Q[i].length][featureDimension];
                tempCi = new double[Omiga_P_Q[i].length][C_Count];

                for (int j = 0; j < Omiga_P_Q[i].length; j++)
                {
                    Data.add(trainData.get(Omiga_P_R[i][j]));
                    System.arraycopy(Q[Omiga_P_Q[i][j]], 0, tempQi[j], 0, featureDimension);
                    System.arraycopy(C[Omiga_P_Q[i][j]], 0, tempCi[j], 0, C_Count);
                }

                // input the estimated p,b for optimizing
                double[] x = new double[DEParam.DeDim];
                for (int j = 0; j < featureDimension; j++)
                    x[j] = P[i][j];
                for (int j = 0; j < B_Count; j++)
                    x[j + featureDimension] = B[i][j];

                int[] gen = new int[1];

                //if (i% 100 == 0)
                //System.out.println(" Train RMSE BEFORE PB DE_OPTIMIZER = " + this.trainCurrentRMSE(LossFunSet.TwoDimMF) + ";Valid RMSE="+this.validCurrentRMSE(LossFunSet.TwoDimMF)+ " Test RMSE = " + this.testCurrentRMSE(LossFunSet.TwoDimMF));
                //}

                /*  Data     the known data for matrix R,  Param     parameters for PSO,  x    the total |f| p[i] and f+1 dimension b, estimated parameters
                 *  tempQi   fixed estimated paras qi,     tempCi    fixed estimated paras ci 	*/
                //System.out.println("i="+i);
                DE_Optimizer(Data, DEParam, x, tempQi, tempCi, i);

                // output the optimized paras pu,bu to estimated matrix
                for (int j = 0; j < featureDimension; j++)
                    P[i][j] = x[j];
                for (int j = 0; j < B_Count; j++)
                    B[i][j] = x[j + featureDimension];

                //comment on 2021/7/14, for saving time and no need for debugging
				/*if (i == user_MaxID)
					System.out.println(" Train RMSE AFTER PB DE_OPTIMIZER = " + this.trainCurrentRMSE(LossFunSet.TwoDimMF) + ";Valid RMSE="+this.validCurrentRMSE(LossFunSet.TwoDimMF)+ " Test RMSE = " + this.testCurrentRMSE(LossFunSet.TwoDimMF));
			    */
            }

			/*
			 *   comment on 2021/07/14, no need for debugging and saving time
			ETimeDEOptimizer = System.currentTimeMillis();
			long time1 = ETimeDEOptimizer - TimeStart;
			System.out.println("user max ID " + user_MaxID);
			System.out.println("All time DE PB: " + (ETimeDEOptimizer - STimeDEOptimizer) / 1000. + "s");
			 *   comment on 2021/07/14, no need for debugging and saving time

			System.out.println(" Train RMSE = " + this.trainCurrentRMSE(LossFun) + ";Valid RMSE="+this.validCurrentRMSE(LossFun)+ " Test RMSE = " + this.testCurrentRMSE(LossFun));
			System.out.println(" Train MAE  = " + this.trainCurrentMAE(LossFun) + ";Valid MAE=" +this.validCurrentMAE(LossFun)+ "  Test MAE  = " + this.testCurrentMAE(LossFun));

			//MaxGen = 0;
			// time added for recording calling DE_Optimizer
			long STimeDEOp2;
			long ETimeDEOp2;

			STimeDEOp2 = System.currentTimeMillis();*/





            for (int j = 1; j <= item_MaxID; j++)
            {

                // System.out.println("j="+j); delete
                double MinQ = Arrays.stream(Q[j]).min().getAsDouble();
                double MaxQ = Arrays.stream(Q[j]).max().getAsDouble();
                double MinC = Arrays.stream(C[j]).min().getAsDouble();
                double MaxC = Arrays.stream(C[j]).max().getAsDouble();
                double CenterQ = (MinQ + MaxQ) / 2;
                double RangeQ = (MaxQ - MinQ) * 2;
                double CenterC = (MinC + MaxC) / 2;
                double RangeC = (MaxC - MinC) * 2;

                //if (j% 100 == 0)
                //System.out.println(j +" Train RMSE BEFORE QC DE_OPTIMIZER 1= " + this.trainCurrentRMSE(LossFunSet.TwoDimMF) + ";Valid RMSE="+this.validCurrentRMSE(LossFunSet.TwoDimMF)+ " Test RMSE = " + this.testCurrentRMSE(LossFunSet.TwoDimMF));

                for (int i = 0; i < featureDimension; i++)
                {
                    DEParam.MinLim[i] = CenterQ - RangeQ/2;
                    DEParam.MaxLim[i] = CenterQ + RangeQ/2;
                    //DEParam.LimScale[i] = DEParam.MaxLim[i] - DEParam.MinLim[i];
                }

                for (int i = featureDimension; i < DEParam.DeDim; i++)
                {
                    DEParam.MinLim[i] = CenterC - RangeC/2;
                    DEParam.MaxLim[i] = CenterC + RangeC/2;
                    //DEParam.LimScale[i] = DEParam.MaxLim[i] - DEParam.MinLim[i];
                }

                // Pick up {R(i,j)} that i belongs to Omiga_Q_P[j]
                Data.clear();
                tempPu = new double[Omiga_Q_P[j].length][featureDimension];
                tempBu = new double[Omiga_Q_P[j].length][B_Count];

                for (int i = 0; i < Omiga_Q_P[j].length; i++)
                {
                    Data.add(trainData.get(Omiga_Q_R[j][i]));
                    System.arraycopy(P[Omiga_Q_P[j][i]], 0, tempPu[i], 0, featureDimension);
                    System.arraycopy(B[Omiga_Q_P[j][i]], 0, tempBu[i], 0, B_Count);
                }

                double[] y = new double[DEParam.DeDim];
                for (int i = 0; i < featureDimension; i++)
                    y[i] = Q[j][i];
                for (int i = 0; i < C_Count; i++)
                    y[i + featureDimension] = C[j][i];

                //if (j% 100 == 0)
                //System.out.println(j + " Train RMSE BEFORE QC DE_OPTIMIZER 2= " + this.trainCurrentRMSE(LossFunSet.TwoDimMF) + ";Valid RMSE="+this.validCurrentRMSE(LossFunSet.TwoDimMF)+ " Test RMSE = " + this.testCurrentRMSE(LossFunSet.TwoDimMF));


                DE_Optimizer(Data, DEParam, y, tempPu, tempBu,j);

                //if (j% 100 == 0) //j == item_MaxID)
                //System.out.println(j + " Train RMSE AFTER QC DE_OPTIMIZER1 = " + this.trainCurrentRMSE(LossFunSet.TwoDimMF) + ";Valid RMSE="+this.validCurrentRMSE(LossFunSet.TwoDimMF)+ " Test RMSE = " + this.testCurrentRMSE(LossFunSet.TwoDimMF));
                //p.println("j = " + j + ", " + gen[0]);

                for (int i = 0; i < featureDimension; i++)
                    Q[j][i] = y[i];
                for (int i = 0; i < C_Count; i++)
                    C[j][i] = y[i + featureDimension];

                //comment on 2021/7/14, for saving time and no need for debugging
				/*if (j == item_MaxID)
					System.out.println(j +" Train RMSE AFTER QC DE_OPTIMIZER2 = " + this.trainCurrentRMSE(LossFunSet.TwoDimMF) + ";Valid RMSE="+this.validCurrentRMSE(LossFunSet.TwoDimMF)+ " Test RMSE = " + this.testCurrentRMSE(LossFunSet.TwoDimMF));
				*/
            }


            // RMSE / MAE calculation
            double curErr_RMSE;
            double curErr_MAE;

            switch (TrainError)
            {
                case RMSE:

                    curErr_RMSE = this.validCurrentRMSE(LossFun);

                    // modify by CJ ON 2021/06
                    if (min_Error_RMSE - curErr_RMSE >= min_Error_RMSE * iterConvThrd)
                    {
                        System.out.println("round = " + round + ", Previous min RMSE = " + min_Error_RMSE + ", Find Better RMSE, current validation RMSE = " + curErr_RMSE + ", Delta RMSE = " + (min_Error_RMSE - curErr_RMSE) + ", " + min_Error_RMSE * iterConvThrd);

                        this.min_Round = round;

                        // record the final round time without the delay rounds, add by CJ on 2021/07/14
                        finalRoundTimeEnd = System.currentTimeMillis();
                        min_Error_RMSE = curErr_RMSE;

                        this.cacheMinFeatures();
                    }
			    	/*else if (min_Error_RMSE - curErr_RMSE > 0)
			    	{

			    		min_Error_RMSE = curErr_RMSE;

			    		// record the final round time without the delay rounds, add by CJ on 2021/07/14
			    		//long finalRoundTimeEnd = System.currentTimeMillis();

			    		this.cacheMinFeatures();
			    	}*/
                    else {
                        this.rollBackMinFeatures();
                    }
                    break;

                case MAE:

                    curErr_MAE = this.validCurrentMAE(LossFun);

                    if (min_Error_MAE - curErr_MAE >= min_Error_MAE * iterConvThrd)
                    {
                        System.out.println("round = " + round + ", Previous min MAE = " + min_Error_MAE + ", Find Better MAE = " + curErr_MAE + ", Delta MAE = " + (min_Error_MAE - curErr_MAE) + "," + min_Error_MAE * iterConvThrd);

                        this.min_Round = round;
                        min_Error_MAE = curErr_MAE;
                        // record the final round time without the delay rounds, add by CJ on 2021/07/14
                        finalRoundTimeEnd = System.currentTimeMillis();

                        this.cacheMinFeatures();
                    }
                    else
                    {
                        this.rollBackMinFeatures();
                    }
                    break;

                case RMSE_and_MAE:


                    curErr_RMSE = this.validCurrentRMSE(LossFun);


                    boolean IsRollBack = false;
                    System.out.println("round = " + round + " current RMSE = " + curErr_RMSE + "; Test RMSE = " + this.testCurrentRMSE(LossFun));
                    if (min_Error_RMSE - curErr_RMSE >= 0)
                    {
                        System.out.println("round = " + round + ", Previous min RMSE = " + min_Error_RMSE + ", Find Better RMSE = " + curErr_RMSE + ", Delta RMSE = " + (min_Error_RMSE - curErr_RMSE) + ", " + min_Error_RMSE * iterConvThrd);

                        if (min_Error_RMSE - curErr_RMSE >= min_Error_RMSE * iterConvThrd)
                            this.min_Round = round;
                        min_Error_RMSE = curErr_RMSE;
                        this.cacheMinFeatures();
                        // record the final round time without the delay rounds, add by CJ on 2021/07/14
                        finalRoundTimeEnd = System.currentTimeMillis();


                    }
                    else
                    {
                        IsRollBack = true;
                        this.rollBackMinFeatures();
                    }


                    curErr_MAE = this.validCurrentMAE(LossFun);


                    System.out.println("round = " + round + " current MAE = " + curErr_MAE + "; Test MAE = " + this.testCurrentMAE(LossFun));
                    // the errors between two iters more than iterConvThrd*last_mae_error
                    if (min_Error_MAE - curErr_MAE >= 0 && !IsRollBack)
                    {
                        System.out.println("round = " + round + ", Previous min MAE = " + min_Error_MAE + ", Find Better MAE = " + curErr_MAE + ", Delta MAE = " + (min_Error_MAE - curErr_MAE) + "," + min_Error_MAE * iterConvThrd);

                        if (min_Error_MAE - curErr_MAE >= min_Error_MAE * iterConvThrd && !IsRollBack)
                            this.min_Round = round;
                        min_Error_MAE = curErr_MAE;
                        this.cacheMinFeatures();
                    }
                    else
                    {
                        IsRollBack = true;
                        this.rollBackMinFeatures();
                    }


                    break;
            }


            long TimeEnd = System.currentTimeMillis();
            System.out.println("Train DE Iteration time in DE function: " + (TimeEnd - TimeStart) / 1000. + "s");
            System.out.println("Train DE Iteration time in DE function: " + (finalRoundTimeEnd - TimeStart) / 1000. + "s");

            //System.out.println("round, " + round + ", Train RMSE , " + this.trainCurrentRMSE(LossFun) + ", Train MAE, " + this.trainCurrentMAE(LossFun) +
            //		", Test RMSE, " + this.testCurrentRMSE(LossFun) + ", Test MAE, " + this.testCurrentMAE(LossFun) + ", Iter Time, " + (System.currentTimeMillis() - TimeStart) / 1000);
            p1.println("round, " + round + ", Train RMSE , " + this.trainCurrentRMSE(LossFun) + ", Train MAE, " + this.trainCurrentMAE(LossFun) +
                    ", Test RMSE, " + this.testCurrentRMSE(LossFun) + ", Test MAE, " + this.testCurrentMAE(LossFun) + ", Iter Time, " + (System.currentTimeMillis() - TimeStart) / 1000);

            if ((round - this.min_Round) >= delayCount) {
                break;
            }

            System.out.println("This round = " + round + " Train RMSE = " + this.trainCurrentRMSE(LossFun) + ";Valid RMSE="+this.validCurrentRMSE(LossFun)+ " Test RMSE = " + this.testCurrentRMSE(LossFun));
            System.out.println("This round = " + round + " Train MAE  = " + this.trainCurrentMAE(LossFun) + ";Valid MAE=" +this.validCurrentMAE(LossFun)+ "  Test MAE  = " + this.testCurrentMAE(LossFun));

        }


        // Calculate error on test data set
        this.rollBackMinFeatures();
        min_Error_MAE = this.testCurrentMAE(LossFun);
        min_Error_RMSE = this.testCurrentRMSE(LossFun);
        double Final_train_RMSE = this.trainCurrentRMSE(LossFun);
        double Final_train_MAE = this.trainCurrentMAE(LossFun);
        System.out.println("Final round = " + (round + 1 - delayCount) + " Train RMSE = " + Final_train_RMSE + ";Valid RMSE="+this.validCurrentRMSE(LossFun)+ " Test RMSE = " + min_Error_RMSE + "; final time =" + (finalRoundTimeEnd - TimeStart) / 1000.);
        System.out.println("Final round = " + (round + 1 - delayCount) + " Train MAE  = " + Final_train_MAE + ";Valid MAE=" +this.validCurrentMAE(LossFun)+ "  Test MAE  = " + min_Error_MAE+ "; final time =" + (finalRoundTimeEnd - TimeStart) / 1000.);

        p1.println();
        p1.println("Final round = " + (round + 1 - delayCount) + " Train RMSE = " + Final_train_RMSE + "; Valid RMSE = " + this.validCurrentRMSE(LossFun) + "; Test RMSE = " + min_Error_RMSE+ "; final time =" + (finalRoundTimeEnd - TimeStart) / 1000.);
        p1.println("Final round = " + (round + 1 - delayCount) + " Train MAE  = " + Final_train_MAE + "; Valid MAE = " + this.validCurrentMAE(LossFun) +  ";Test MAE  = " + min_Error_MAE+ "; final time =" + (finalRoundTimeEnd - TimeStart) / 1000.);

        this.TrainRound = round + 1 - delayCount;

        // DEParam.total_round += (round-2.5); //added by CJ, for averaging validation time  // remove on 2021/07/14, because i think (round + 1 - delayCount) is the total rounds

        //System.out.println("end of train DE");
    }


    // DE_optimizer() is for optimizing one pu,bu or qi,ci via extending the input as K thoughts
    public void DE_Optimizer(ArrayList<RTuple> TrainData, DE_Param DeParams, double[] x, double[][] y, double[][] c, int userID) throws IOException
    {

        //System.out.println("DE: Differential Evolution");

        ///////////////////////////
        // Initial configuration //
        ///////////////////////////


        //int trial=5;		// the number of trials with different random initial

        double Errlim = 0.01; // the error limit predefined


        // calculate the limit scale of initialization of DE individuals, added by CJ, 2021.0405
        double[] MaxLim = new double[DeParams.DeDim];
        double[] MinLim = new double[DeParams.DeDim];
        double[] LimScale = new double[DeParams.DeDim];

        for (int l = 0; l< DeParams.DeDim; l++)
        {
            MaxLim[l] = DeParams.MaxLim[l];
            MinLim[l] = DeParams.MinLim[l];
            LimScale[l] = MaxLim[l] - MinLim[l];
        }



        //System.out.println(" Train RMSE AFTER DE_OPTIMIZER = " + this.trainCurrentRMSE(LossFunSet.TwoDimMF) + ";Valid RMSE="+this.validCurrentRMSE(LossFunSet.TwoDimMF)+ " Test RMSE = " + this.testCurrentRMSE(LossFunSet.TwoDimMF));


        // experiment for several times to avoid the floating
        //for(int l=0;l<trial;l++){

        // useful or not?
        //double[][] hb = new double[DeParams.DeNum][DeParams.DeDim];   // the historical best vector for personal best

        double[] Gb = new double[DeParams.DeDim];		// the global best vector
        double obGF = 1000;     // the  objective function for the global best, paired with Gb

        // for  aggregating the X and XC, obF is the fitness function with X, X is the current individuals consists of a swarm
        //double[][] X = new double [(int)(DeParams.DeNum*(1+DeParams.de_cp))][DeParams.DeDim];	// a swarm consists of DeParams.DeNum*(1+DeParams.de_cp) individuals
        double[][] X = new double [DeParams.DeNum][DeParams.DeDim];
        double[] obF = new double [DeParams.DeNum];                    // objective(fitness) function for DeParams.DeNum*(1+DeParams.de_cp) individuals

        // variables for mutation
        int r1, r2, r3, r4;     // the random chosen individuals for mutation, define here or outside better?
        double[][] Xm = new double[DeParams.DeNum][DeParams.DeDim];   // output of the mutation, the mutated X
        double[] obF_Xm = new double[DeParams.DeNum];

        int indR1, indR2, indR3, indR4;   // index for selected numbers in the above list

        // variables for crossover
        int crStop = 1; // the variable for stopping select the cross-over individuals
        int crRNum, crRDim;       // the random subscribe for selecting the individuals, and the random dimension
        //int[] indZero = new int[DeParams.DeNum];        //  0-1 index for crossover
        double rc;       // random num for comparing the cross or not

        // functions and variables for sorting the crossoverred X
        CommonRecomm_NEW.IndexSort Isort = new IndexSort();    // the index sort defined in DE_Param.java, used for sorting the X
        Integer [] anum = new Integer [DeParams.DeNum];  // the ascend index sorted with Isort

        //


        // variables for calculating the fitness function
        double tempB = 0;
        double fitValue = 0;      // fitness value for each individual
        int dataSize = TrainData.size();
        double[] tempC = new double[dataSize];
        double Err1 = 0;         // the fitness improvement compared with last itr
        double[] Rij = new double[dataSize];

        int gen = 0;       // the iteration of the swarm has been updated

        Random random = new Random(System.currentTimeMillis());

        //	System.out.println(DeParams.DeNum+"\n dim"+ DeParams.DeDim);

        //System.out.println(" Train RMSE before init = " + this.trainCurrentRMSE(LossFunSet.TwoDimMF) + ";Valid RMSE="+this.validCurrentRMSE(LossFunSet.TwoDimMF)+ " Test RMSE = " + this.testCurrentRMSE(LossFunSet.TwoDimMF));

        // output the optimized paras pu,bu to estimated matrix
			/*for (int j = 0; j < CommonRecomm_NEW.featureDimension; j++)
				P[userID][j] = x[j];
			for (int j = 0; j < B_Count; j++)
				B[userID][j] = x[j + CommonRecomm_NEW.featureDimension];

			System.out.println(" Train RMSE before init = " + this.trainCurrentRMSE(LossFunSet.TwoDimMF) + ";Valid RMSE="+this.validCurrentRMSE(LossFunSet.TwoDimMF)+ " Test RMSE = " + this.testCurrentRMSE(LossFunSet.TwoDimMF));*/

        //////////////////////////////////////////////////////////////////
        // Initialize the evolutionary individuals
        //////////////////////////////////////////////////////////////////
        double InitFitValue = 0.0;
        for(int i=0;i<DeParams.DeNum;i++) {   // extend the one x to a swarm consists of DeParams.DeNum individuals

            //double RandNum = random.nextDouble(); //range  change by CJ on 2021/08/14
            double RandNum = random.nextDouble()*2.0-1.0;

            // the 1st individual equals to the original parameters

            if (i!=0)
            {
                for (int j = 0; j < DeParams.DeDim; j++)
                {
                    //X[i][j] =  hb[i][j] = x[j] + (RandNum - 0.5)* LimScale[l];
                    //X[i][j] = x[j] + (RandNum - 0.5)* LimScale[j];

                    X[i][j] = x[j]*(1-0.1*RandNum);
                }


            }
            // the 2nd to all the evolutionary individuals equals to random values in a scale
            else
            {
                for (int j = 0; j < DeParams.DeDim; j++)
                {
                    //X[i][j] =  hb[i][j] = x[j];
                    X[i][j] =   x[j];
                }
            }

            // Initialize fitness of each individual and the whole community
            // the fitness function ----- sub-group (pu,bu) or (qi,ci) for mini-bach function

            // define variables for fitness function calculation
            fitValue = tempB = 0;      // fitness value for each individual

            // add our fitness function, calculating the learning objective function
            for (int k = 0; k < dataSize; k++)
            {
                tempC[k]= 0;
                for (int z = 0; z < c[k].length; z++)
                {
                    tempC[k] +=c[k][z];
                }

                Rij[k] = TrainData.get(k).dRating - tempC[k];     // the already known rating r - c
            }

            for (int z = 0; z < DeParams.DeDim; z++)
            {
                fitValue += X[i][z]*X[i][z];
            }

            fitValue *= DeParams.de_lambda*dataSize;         // lambda*(pu2+b2), regulation item

            for (int z = CommonRecomm_NEW.featureDimension; z < DeParams.DeDim; z++)
            {
                tempB += X[i][z];                           // bu
            }

            for (int k = 0; k < dataSize; k++)
            {
                double tempXY = 0;
                for (int z = 0; z < CommonRecomm_NEW.featureDimension; z++)
                {
                    tempXY += X[i][z]*y[k][z];
                }
                fitValue += Math.abs(Rij[k] - tempXY - tempB);   // ((r-c)-r^-regulation item- b)2.0          //fitness function, and learning objective function

            }

            obF[i] =  fitValue;      // the i-th individual's objective function


            // find out the global best fitness value
            if (i == 0 || obGF > obF[i])  // find out the global best fitness value
            {
                obGF = obF[i];         // update the global fitness function
                for (int z = 0; z < DeParams.DeDim; z++)
                {
                    Gb[z] = X[i][z];    // update the global best vector
                }
            }

            if(i==0)
                InitFitValue = obGF;
            // end the fitness function, the personal best and global best calculation.

        }
        // end the initialization of the evolutionary individuals
        //System.out.println("end of inti de Evolution");
        //System.out.println("Initial fitness value: "+InitFitValue+"global fitness value after initialization:"+ obGF+"obF:"+ obF[1]);

			/*if (dataSize != 0) {
				for (int z = 0; z < DeParams.DeDim; z++)
				{
					x[z] = X[0][z];
				}
			}

			for (int j = 0; j < CommonRecomm_NEW.featureDimension; j++)
				P[userID][j] = x[j];
			for (int j = 0; j < B_Count; j++)
				B[userID][j] = x[j + CommonRecomm_NEW.featureDimension];

			System.out.println(" Train RMSE after init = " + this.trainCurrentRMSE(LossFunSet.TwoDimMF) + ";Valid RMSE="+this.validCurrentRMSE(LossFunSet.TwoDimMF)+ " Test RMSE = " + this.testCurrentRMSE(LossFunSet.TwoDimMF));*/

        //////////////////////////////////////////////////////////////////

        // Iterate to search gbest, update the individuals each iteration, calculate the fitness function and search the whole swarm
        double F_pops = 1; // temp variable
        int delay = 0; // calculating the error compared times less than limitation
        int[] listRandom = new int[DeParams.DeNum];        //  random list for all the possible selected numbers, 20210506

        do
        {
            double obGF_tmp = obGF;        //  tmp records the global best obtained last itr
            Errlim = obGF * 0.0001;         // the error limit predefined, also mentioned in paper


            /////////////////////////////////////////////////////////////////////////
            // The mutation process Xm = r1 + dw*(r2-r3)



            // add the selected number to array, and delete the already selected ones from arrays
            for (int i = 0; i< DeParams.DeNum; i++) {
                // choosing the r1, r2, r3 for mutation

                // improving the mutation, adding all the subscribes into array, 20210506
                for(int p = 0; p < DeParams.DeNum; p++)
                {
                    listRandom[p] = p;
                }

                listRandom[i] = listRandom[DeParams.DeNum-1];   // for random value, not choose i
                indR1 = random.nextInt(DeParams.DeNum - 1);     // random choose the 1st one
                r1 = listRandom[indR1];

                listRandom[indR1] = listRandom[DeParams.DeNum-2]; // Using indR1 to substitute DeNum-2
                indR2 =  random.nextInt(DeParams.DeNum - 2);
                r2 = listRandom[indR2];


                // add for DE/best/2/bin
					/*listRandom[indR2] = listRandom[DeParams.DeNum-3];
					indR3 =  random.nextInt(DeParams.DeNum - 3);
					r3 = listRandom[indR3];

					listRandom[indR3] = listRandom[DeParams.DeNum-4];
					indR4 =  random.nextInt(DeParams.DeNum - 3);
					r4 = listRandom[indR4];*/

                // code by CJ on 2021/10/5 after deleting the crossover, for sorting selection
                int de_n_i = i+DeParams.DeNum;
                // mutation the i-th individual with each dimension
                for (int j = 0; j < DeParams.DeDim; j++){

                    // DE/rand/1/bin, the accuracy is lower than best, not used
                    //Xm[i][j] = X[r1][j] + DeParams.de_dw * (X[r2][j] - X[r3][j]);

                    // DE/best/1/bin
                    //System.out.println("de_dw "+ DeParams.de_dw);  for debugging
                    Xm[i][j] = Gb[j] + DeParams.de_dw * (X[r1][j] - X[r2][j]);

                    //Xm[i][j] = X[i][j] + 0.5*DeParams.de_dw * (X[r3][j] - X[r4][j]) + 0.5 * DeParams.de_dw*(X[r1][j] - X[r2][j]);

                    // DE/best/2/bin, the accuracy is a little worse, while the time cost is much higher
                    //Xm[i][j] = Gb[j] +0.05* DeParams.de_dw * (X[r1][j] - X[r2][j]) + 0.05*DeParams.de_dw * (X[r3][j] - X[r4][j]);

                    // DE/current-to-best/1/bin, the accuracy is worse than the DE/best/2/bin, while the time cost is higher than DE/best/2/bin.
                    //Xm[i][j] = X[i][j] + 0.5*DeParams.de_dw * (Gb[j] - X[i][j]) + 0.5*DeParams.de_dw * (X[r1][j] - X[r2][j]);
                    //Xm[i][j] = X[i][j] + 0.5*DeParams.de_dw * (Gb[j] - X[i][j]) + 0.0005 * (X[r1][j] - X[r2][j]); // accuracy is 0.7033,lambda=0.25


                    // DE/current-to-best/1/bin on 2021/10/07
                    //Xm[i][j] = Gb[j] + 0.5*DeParams.de_dw * (Gb[j] - X[i][j]) + 0.5*DeParams.de_dw * (X[r1][j] - X[r2][j]); // accuracy is 0.7031,lambda=0.25, DE_dw=0.01; DE_cr=0.1; DE_cp=0.1

                    //Xm[i][j] = Gb[j] + DeParams.de_dw * (Gb[j] - X[i][j]);
                }

                //indZero[i] = 0; // re-initialize the indZero for the following mark 1


                fitValue = 0;  // clear the fitness value, the position is right? why need double type?

                for (int z= 0; z < DeParams.DeDim; z++)
                {
                    fitValue += Xm[i][z]*Xm[i][z];
                }

                fitValue *= DeParams.de_lambda*dataSize;         // lambda*(pu2+b2), regulation item

                for (int z = CommonRecomm_NEW.featureDimension; z < DeParams.DeDim; z++)
                {
                    tempB += Xm[i][z];                       // bu
                }

                for (int k = 0; k < dataSize; k++)
                {
                    double tempXY = 0;
                    for (int z = 0; z < CommonRecomm_NEW.featureDimension; z++)
                    {
                        tempXY += Xm[i][z]*y[k][z];          //?? seems right?? ask, and confirm!!!
                    }

                    fitValue += Math.abs(Rij[k] - tempXY - tempB);   // ((r-c)-r^-regulation item- b)2.0          //fitness function, and learning objective function

                }


                // comment by CJ on 2021/9/27 for changing to the standard DE
                // if the fitness value of the crossover value is smaller than the original one, change it
                if (fitValue < obF[i])
                {
                    obF[i] = fitValue;
                    for (int j= 0; j < DeParams.DeDim; j++) {
                        X[i][j] = Xm[i][j];
                    }

                }

                // code by CJ on 2021/10/5 after deleting the crossover, for sorting selection
                //obF[de_n_i] = fitValue;


            }//end the mutation
            //System.out.println("end of de mutation");
            //System.out.println(" Train RMSE after mutation = " + this.trainCurrentRMSE(LossFunSet.TwoDimMF) + ";Valid RMSE="+this.validCurrentRMSE(LossFunSet.TwoDimMF)+ " Test RMSE = " + this.testCurrentRMSE(LossFunSet.TwoDimMF));
				/*if (dataSize != 0) {
					for (int z = 0; z < DeParams.DeDim; z++)
					{
						x[z] = X[0][z];
					}
				}

				for (int j = 0; j < CommonRecomm_NEW.featureDimension; j++)
					P[userID][j] = x[j];
				for (int j = 0; j < B_Count; j++)
					B[userID][j] = x[j + CommonRecomm_NEW.featureDimension];

				System.out.println(" Train RMSE after mutation = " + this.trainCurrentRMSE(LossFunSet.TwoDimMF) + ";Valid RMSE="+this.validCurrentRMSE(LossFunSet.TwoDimMF)+ " Test RMSE = " + this.testCurrentRMSE(LossFunSet.TwoDimMF));*/

            /////////////////////////////////////////////////////////////////////////

            /////////////////////////////////////////////////////////////////////////
            // The crossover function: choose some individuals to update,

            // tag the selected crossover individual
            //int xp = (int) (DeParams.DeNum - DeParams.DeNum * DeParams.de_cp); only for debugging
            // comment for all the individuals should be crossover on 2021/10/3 by CJ
				/*for (int crSum = 0;crSum <  DeParams.DeNum  - (int) (DeParams.DeNum * DeParams.de_cp); crSum++ ) {

					do {
						crRNum = random.nextInt(DeParams.DeNum);
					}while (indZero[crRNum] == 1);

				    indZero[crRNum] = 1;

				}*/

            // crossover the individuals
				/*int n = DeParams.DeNum; // the subscribe for Xc, which is after X in the aggregation array. DeParams.DeDim is the first one6r
				//int sx = (int) (DeParams.DeNum*DeParams.de_cp);
				//System.out.println("sx=" + sx);

				//for (int i = 0; i < (int) (DeParams.DeNum*DeParams.de_cp); i++){      // wrong codes
				for (int i = 0; i < DeParams.DeNum; i++) {

					// crossover the individuals and calculate the fitness function for crossed one
					// crossover only when the indzero is 0,  total nums = DeParams.de_cp

					//if (indZero[i] == 0){     // comment for all the individuals should be crossover on 2021/10/3 by CJ
						crRDim = random.nextInt(DeParams.DeDim);

						for (int j = 0; j < DeParams.DeDim; j++){
							rc = random.nextDouble();
							if ((j == crRDim) || (rc <= DeParams.de_cr)){
								//Xc[n][j] = Xm[i][j];    // get the mutation individuals, delete after aggregation
								X[n][j] =  Xm[i][j];    // get the mutation individuals
							}else if((j != crRDim) || (rc > DeParams.de_cr)){
								X[n][j] = X[i][j];     // get the original individuals
							}
						}


						// Calculate the fitness function for the cross individuals (Xc after X)
						// the fitness value of X[n]， the k,n,l right or not ????
						fitValue = 0;  // clear the fitness value, the position is right? why need double type?

						for (int z= 0; z < DeParams.DeDim; z++)
						{
							fitValue += X[n][z]*X[n][z];
						}

						fitValue *= DeParams.de_lambda*dataSize;         // lambda*(pu2+b2), regulation item

						for (int z = CommonRecomm_NEW.featureDimension; z < DeParams.DeDim; z++)
						{
							tempB += X[n][z];                       // bu
						}

						for (int k = 0; k < dataSize; k++)
						{
							double tempXY = 0;
							for (int z = 0; z < CommonRecomm_NEW.featureDimension; z++)
							{
								tempXY += X[n][z]*y[k][z];          //?? seems right?? ask, and confirm!!!
							}

							fitValue += Math.pow(Rij[k] - tempXY - tempB, 2.0);   // ((r-c)-r^-regulation item- b)2.0          //fitness function, and learning objective function

						}


						obF[n] =  fitValue;      // the i-th individual's objective function
						// comment by CJ on 2021/9/27 for changing to the standard DE
						// if the fitness value of the crossover value is smaller than the original one, change it
						if (obF[n] < obF[i])
						{
							obF[i] = obF[n];
							for (int j= 0; j < DeParams.DeDim; j++) {
								X[i][j] = X[n][j];
							}

						}

						n++;     // In design, the max n=(1+cp)*DeParams.DeDim-1
					//} // comment for all the individuals should be crossover on 2021/10/3 by CJ
				}
				// end crossover the individuals */
            //System.out.println("end of crossover de");
            //System.out.println(" Train RMSE AFTER cross = " + this.trainCurrentRMSE(LossFunSet.TwoDimMF) + ";Valid RMSE="+this.validCurrentRMSE(LossFunSet.TwoDimMF)+ " Test RMSE = " + this.testCurrentRMSE(LossFunSet.TwoDimMF));

            /////////////////////////////////////////////////////////////////////////


            /////////////////////////////////////////////////////////////////////////
            // Selection: select the DeParams.DeNum individuals after crossover with small fitness values
            // sort obF or only return the index????

            anum = Isort.Ind(obF);  // sort by ascent order and get the index

            // copy to the temp array with sorted X
            // there is no need to sort the X or Xm, DELETE the following codes on 2021/06/17
				/*for (int i = 0; i < DeParams.DeNum; i++){
						for (int j = 0; j < DeParams.DeDim; j++){
							Xm[i][j] = X[anum[i]][j];
						}

					// is there any necessary to the copy to obF_Xm
					obF_Xm[i] = obF[anum[i]];  // copy to the temp array
				}

				// copy back to the formal array
				for (int i = 0; i < DeParams.DeNum; i++){
					for (int j = 0; j < DeParams.DeDim; j++){
						X[i][j] = Xm[i][j];			// re-copy the sorted X back to X
					}

					// is there any necessary to the copy to obF_Xm
					obF[i] = obF_Xm[i];		// re-copy the sorted obF back to obF
				}
				// end the the sorted copy */



            obGF = obF[anum[0]]; // obF[0];   // the smallest fitness value
            //System.out.println("i = " + userID + ";fitnessValue =" + obGF);
            //System.out.println(" Train RMSE AFTER SORTING = " + this.trainCurrentRMSE(LossFunSet.TwoDimMF) + ";Valid RMSE="+this.validCurrentRMSE(LossFunSet.TwoDimMF)+ " Test RMSE = " + this.testCurrentRMSE(LossFunSet.TwoDimMF));

            // add for DE/best/1 on 2021/06/10
            for (int j = 0; j < DeParams.DeDim; j++){
                Gb[j] = X[anum[0]][j]; //Xm[0][j];			// re-copy the sorted X[0] global best to Gb[]
            }
            // end add for DE/best/1 on 2021/06/10



            /////////////////////////////////////////////////////////////////////////

            // calculate the fitness error improvement
            if (dataSize == 0){
                Err1 = 0;
            }
            else {
                Err1 = obGF_tmp - obGF; // the fitness error improvement compared with two itrs
            }

            // accumulate the times of the error less than ErrLim
            if (Err1 >= Errlim) {
                delay = 0;
            }
            else {
                delay = delay + 1;    //System.out.println("iteration times: "+gen + "\n delay:" + delay);
            }

            gen++;    // the iteration time + 1

            //System.out.println("global fitness value:"+ obGF +" gen:"+gen +" delay:" +delay+" DeParams.MaxGen:"+DeParams.MaxGen);
            //System.out.println("Err1:"+Err1+" Errlim:"+Errlim);
				/*if (dataSize != 0) {
					for (int z = 0; z < DeParams.DeDim; z++)
					{
						x[z] = Gb[z];
					}
				}

				for (int j = 0; j < CommonRecomm_NEW.featureDimension; j++)
					P[userID][j] = x[j];
				for (int j = 0; j < B_Count; j++)
					B[userID][j] = x[j + CommonRecomm_NEW.featureDimension];*/

            //System.out.println(" Train RMSE after selection = " + this.trainCurrentRMSE(LossFunSet.TwoDimMF) + ";Valid RMSE="+this.validCurrentRMSE(LossFunSet.TwoDimMF)+ " Test RMSE = " + this.testCurrentRMSE(LossFunSet.TwoDimMF));

        }while((gen < DeParams.MaxGen) & (delay <= 1));
        // err less the predefined for two times || iteration times greater than MaxGen, break the iterations


        // copy the global best one to x
        if (dataSize != 0) {
            for (int z = 0; z < DeParams.DeDim; z++)
            {
                x[z] = Gb[z];
            }
        }

        //}// end several times trials
		/*	for (int j = 0; j < CommonRecomm_NEW.featureDimension; j++)
				P[userID][j] = x[j];
			for (int j = 0; j < B_Count; j++)
				B[userID][j] = x[j + CommonRecomm_NEW.featureDimension];

			System.out.println(" Train RMSE after iteration and re-value P,B: " + this.trainCurrentRMSE(LossFunSet.TwoDimMF) + ";Valid RMSE="+this.validCurrentRMSE(LossFunSet.TwoDimMF)+ " Test RMSE = " + this.testCurrentRMSE(LossFunSet.TwoDimMF));*/

        //System.out.println("end DE: end DE optimizer");
    }// end DE_Optimizer

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //      The above codes are for DE optimization algorithms
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //      The below codes are for BAS optimization algorithms
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Train using BAS optimization

    public void train_BAS(LossFunSet LossFun, TrainErrorSet TrainError, BAS_Param BASParam, PrintStream p1) throws IOException
    {

        long TimeStart = System.currentTimeMillis();
        long finalRoundTimeEnd = System.currentTimeMillis();

        double[][] tempPu;
        double[][] tempBu;
        double[][] tempQi;
        double[][] tempCi;

        ArrayList<RTuple> Data = new ArrayList<RTuple>();

        this.cacheMinFeatures();

        min_Error_RMSE = Double.MAX_VALUE;
        min_Error_MAE = Double.MAX_VALUE;


        // Use BAS method to optimize each Pi, move from the training round iterations
        BASParam.BASDim = featureDimension + B_Count;

        for (round = 0; round <= trainingRound; round++) // use commandNew.traininground = 100
        {


			/*
			 *   comment on 2021/07/14, no need for debugging and saving time // time added for recording calling DE_Optimizer
			 *   Copy it from DE_optimizer
			System.out.println("trainingRound="+ round);

            long STimeDEOptimizer;
            long ETimeDEOptimizer;

            STimeDEOptimizer = System.currentTimeMillis(); // START TIME
            */
            //System.out.println(" Train RMSE BEFORE BAS_OPTIMIZER = " + this.trainCurrentRMSE(LossFunSet.TwoDimMF) + ";Valid RMSE="+this.validCurrentRMSE(LossFunSet.TwoDimMF)+ " Test RMSE = " + this.testCurrentRMSE(LossFunSet.TwoDimMF));


            // loop for each i, that means each user
            for (int i = 1; i <= user_MaxID; i++)      // System.out.println("i="+i);  DELETE
            {


                // Pick up {R(i,j)} that j belongs to Omiga_P_Q[i], the already known data
                Data.clear();
                tempQi = new double[Omiga_P_Q[i].length][featureDimension];
                tempCi = new double[Omiga_P_Q[i].length][C_Count];

                for (int j = 0; j < Omiga_P_Q[i].length; j++)
                {
                    Data.add(trainData.get(Omiga_P_R[i][j]));
                    System.arraycopy(Q[Omiga_P_Q[i][j]], 0, tempQi[j], 0, featureDimension);
                    System.arraycopy(C[Omiga_P_Q[i][j]], 0, tempCi[j], 0, C_Count);
                }

                // input the estimated p,b for optimizing
                double[] x = new double[BASParam.BASDim];
                for (int j = 0; j < featureDimension; j++)
                    x[j] = P[i][j];
                for (int j = 0; j < B_Count; j++)
                    x[j + featureDimension] = B[i][j];

                int[] gen = new int[1];

                if (i == 1) {   //% 100
                    System.out.println(" Train RMSE BEFORE BAS_OPTIMIZER = " + this.trainCurrentRMSE(LossFunSet.TwoDimMF) + ";Valid RMSE="+this.validCurrentRMSE(LossFunSet.TwoDimMF)+ " Test RMSE = " + this.testCurrentRMSE(LossFunSet.TwoDimMF));
                }

                /*  Data     the known data for matrix R,  Param     parameters for BAS,  x    the total |f| p[i] and f+1 dimension b, estimated parameters
                 *  tempQi   fixed estimated paras qi,     tempCi    fixed estimated paras ci 	*/
                //System.out.println("i="+i);

                BAS_Optimizer(Data, BASParam, x, tempQi, tempCi, i, TrainError);

                // output the optimized paras pu,bu to estimated matrix
                for (int j = 0; j < featureDimension; j++)
                    P[i][j] = x[j];
                for (int j = 0; j < B_Count; j++)
                    B[i][j] = x[j + featureDimension];

                // comment on 2021/7/14, for saving time and no need for debugging
                //if (i% 100 == 0) {
                if (i == user_MaxID) {
                    System.out.println(" Train RMSE AFTER BAS_OPTIMIZER = " + this.trainCurrentRMSE(LossFunSet.TwoDimMF) + ";Valid RMSE="+this.validCurrentRMSE(LossFunSet.TwoDimMF)+ " Test RMSE = " + this.testCurrentRMSE(LossFunSet.TwoDimMF));
                }
            }

			/*
			 *   comment on 2021/07/14, no need for debugging and saving time
			ETimeDEOptimizer = System.currentTimeMillis();
			long time1 = ETimeDEOptimizer - TimeStart;
			System.out.println("user max ID " + user_MaxID);
			System.out.println("All time DE PB: " + (ETimeDEOptimizer - STimeDEOptimizer) / 1000. + "s");
            */

            // Use EA method to optimize Qi,
            // because of b_count equals to c_count, delete the below code
			/*DEParam.DeDim = featureDimension + C_Count;
			DEParam.MinLim = new double[DEParam.DeDim];
			DEParam.MaxLim = new double[DEParam.DeDim];*/




			/*
			 *   comment on 2021/07/14, no need for debugging and saving time

			System.out.println(" Train RMSE = " + this.trainCurrentRMSE(LossFun) + ";Valid RMSE="+this.validCurrentRMSE(LossFun)+ " Test RMSE = " + this.testCurrentRMSE(LossFun));
			System.out.println(" Train MAE  = " + this.trainCurrentMAE(LossFun) + ";Valid MAE=" +this.validCurrentMAE(LossFun)+ "  Test MAE  = " + this.testCurrentMAE(LossFun));

			//MaxGen = 0;
			// time added for recording calling DE_Optimizer
			long STimeDEOp2;
			long ETimeDEOp2;

			STimeDEOp2 = System.currentTimeMillis();
            */

            for (int j = 1; j <= item_MaxID; j++)
            {

                // System.out.println("j="+j); delete

                // Pick up {R(i,j)} that i belongs to Omiga_Q_P[j]
                Data.clear();
                tempPu = new double[Omiga_Q_P[j].length][featureDimension];
                tempBu = new double[Omiga_Q_P[j].length][B_Count];

                for (int i = 0; i < Omiga_Q_P[j].length; i++)
                {
                    Data.add(trainData.get(Omiga_Q_R[j][i]));
                    System.arraycopy(P[Omiga_Q_P[j][i]], 0, tempPu[i], 0, featureDimension);
                    System.arraycopy(B[Omiga_Q_P[j][i]], 0, tempBu[i], 0, B_Count);
                }

                double[] y = new double[BASParam.BASDim];
                for (int i = 0; i < featureDimension; i++)
                    y[i] = Q[j][i];
                for (int i = 0; i < C_Count; i++)
                    y[i + featureDimension] = C[j][i];


                if (j == 1) {   //% 100
                    System.out.println(" Train item RMSE BEFORE BAS_OPTIMIZER = " + this.trainCurrentRMSE(LossFunSet.TwoDimMF) + ";Valid RMSE="+this.validCurrentRMSE(LossFunSet.TwoDimMF)+ " Test RMSE = " + this.testCurrentRMSE(LossFunSet.TwoDimMF));
                }

                BAS_Optimizer(Data, BASParam, y, tempPu, tempBu,j, TrainError);

                //p.println("j = " + j + ", " + gen[0]);

                for (int i = 0; i < featureDimension; i++)
                    Q[j][i] = y[i];
                for (int i = 0; i < C_Count; i++)
                    C[j][i] = y[i + featureDimension];

                if (j == item_MaxID) {
                    System.out.println(" Train item RMSE AFTER BAS_OPTIMIZER = " + this.trainCurrentRMSE(LossFunSet.TwoDimMF) + ";Valid RMSE="+this.validCurrentRMSE(LossFunSet.TwoDimMF)+ " Test RMSE = " + this.testCurrentRMSE(LossFunSet.TwoDimMF));
                }
            }

			/*
			 *   comment on 2021/07/14, no need for debugging and saving time

			System.out.println("item max ID " + item_MaxID);

			System.out.println("Iteration time DE QC: " + (ETimeDEOp2 - STimeDEOp2) / 1000. + "s");
			System.out.println(" Train RMSE = " + this.trainCurrentRMSE(LossFun) + ";Valid RMSE="+this.validCurrentRMSE(LossFun)+ " Test RMSE = " + this.testCurrentRMSE(LossFun));
			System.out.println(" Train MAE  = " + this.trainCurrentMAE(LossFun) + ";Valid MAE=" +this.validCurrentMAE(LossFun)+ "  Test MAE  = " + this.testCurrentMAE(LossFun));

			 */
            // RMSE / MAE calculation
            double curErr_RMSE;
            double curErr_MAE;

            switch (TrainError)
            {
                case RMSE:

                    curErr_RMSE = this.validCurrentRMSE(LossFun);

                    // modify by CJ ON 2021/06
                    if (min_Error_RMSE - curErr_RMSE >= min_Error_RMSE * iterConvThrd)
                    {
                        System.out.println("round = " + round + ", Previous min RMSE = " + min_Error_RMSE + ", Find Better RMSE, current validation RMSE = " + curErr_RMSE + ", Delta RMSE = " + (min_Error_RMSE - curErr_RMSE) + ", " + min_Error_RMSE * iterConvThrd);

                        this.min_Round = round;

                        // record the final round time without the delay rounds, add by CJ on 2021/07/14
                        finalRoundTimeEnd = System.currentTimeMillis();
                        min_Error_RMSE = curErr_RMSE;

                        this.cacheMinFeatures();

                        System.out.println("This round is useful for BAS = " + round + " Train RMSE = " + this.trainCurrentRMSE(LossFun) + ";Valid RMSE="+this.validCurrentRMSE(LossFun)+ " Test RMSE = " + this.testCurrentRMSE(LossFun));
                        System.out.println("This round is useful for BAS = " + round + " Train MAE = " + this.trainCurrentMAE(LossFun) + ";Valid MAE="+this.validCurrentMAE(LossFun)+ " Test MAE = " + this.testCurrentMAE(LossFun));

                    }
			    	/*else if (min_Error_RMSE - curErr_RMSE > 0)
			    	{

			    		min_Error_RMSE = curErr_RMSE;

			    		// record the final round time without the delay rounds, add by CJ on 2021/07/14
			    		//long finalRoundTimeEnd = System.currentTimeMillis();

			    		this.cacheMinFeatures();
			    	}*/
                    else {
                        this.rollBackMinFeatures();
                    }
                    break;

                case MAE:

                    curErr_MAE = this.validCurrentMAE(LossFun);

                    if (min_Error_MAE - curErr_MAE >= min_Error_MAE * iterConvThrd)
                    {
                        System.out.println("round = " + round + ", Previous min MAE = " + min_Error_MAE + ", Find Better MAE = " + curErr_MAE + ", Delta MAE = " + (min_Error_MAE - curErr_MAE) + "," + min_Error_MAE * iterConvThrd);

                        this.min_Round = round;
                        min_Error_MAE = curErr_MAE;
                        // record the final round time without the delay rounds, add by CJ on 2021/07/14
                        finalRoundTimeEnd = System.currentTimeMillis();

                        this.cacheMinFeatures();
                    }
                    else
                    {
                        this.rollBackMinFeatures();
                    }
                    break;

                case RMSE_and_MAE:


                    curErr_RMSE = this.validCurrentRMSE(LossFun);


                    boolean IsRollBack = false;
                    System.out.println("round = " + round + " current RMSE = " + curErr_RMSE + "; Test RMSE = " + this.testCurrentRMSE(LossFun));
                    if (min_Error_RMSE - curErr_RMSE >= 0)
                    {
                        System.out.println("round = " + round + ", Previous min RMSE = " + min_Error_RMSE + ", Find Better RMSE = " + curErr_RMSE + ", Delta RMSE = " + (min_Error_RMSE - curErr_RMSE) + ", " + min_Error_RMSE * iterConvThrd);

                        if (min_Error_RMSE - curErr_RMSE >= min_Error_RMSE * iterConvThrd)
                            this.min_Round = round;
                        min_Error_RMSE = curErr_RMSE;
                        this.cacheMinFeatures();
                        // record the final round time without the delay rounds, add by CJ on 2021/07/14
                        finalRoundTimeEnd = System.currentTimeMillis();


                    }
                    else
                    {
                        IsRollBack = true;
                        this.rollBackMinFeatures();
                    }


                    curErr_MAE = this.validCurrentMAE(LossFun);


                    System.out.println("round = " + round + " current MAE = " + curErr_MAE + "; Test MAE = " + this.testCurrentMAE(LossFun));
                    // the errors between two iters more than iterConvThrd*last_mae_error
                    if (min_Error_MAE - curErr_MAE >= 0 && !IsRollBack)
                    {
                        System.out.println("round = " + round + ", Previous min MAE = " + min_Error_MAE + ", Find Better MAE = " + curErr_MAE + ", Delta MAE = " + (min_Error_MAE - curErr_MAE) + "," + min_Error_MAE * iterConvThrd);

                        if (min_Error_MAE - curErr_MAE >= min_Error_MAE * iterConvThrd && !IsRollBack)
                            this.min_Round = round;
                        min_Error_MAE = curErr_MAE;
                        this.cacheMinFeatures();
                    }
                    else
                    {
                        IsRollBack = true;
                        this.rollBackMinFeatures();
                    }


                    break;
            }


            long TimeEnd = System.currentTimeMillis();
            System.out.println("Train BAS Iteration time in BAS function: " + (TimeEnd - TimeStart) / 1000. + "s");
            System.out.println("Train BAS Iteration time in BAS function: " + (finalRoundTimeEnd - TimeStart) / 1000. + "s");

            //System.out.println("round, " + round + ", Train RMSE , " + this.trainCurrentRMSE(LossFun) + ", Train MAE, " + this.trainCurrentMAE(LossFun) +
            //		", Test RMSE, " + this.testCurrentRMSE(LossFun) + ", Test MAE, " + this.testCurrentMAE(LossFun) + ", Iter Time, " + (System.currentTimeMillis() - TimeStart) / 1000);
            p1.println("round, " + round + ", Train RMSE , " + this.trainCurrentRMSE(LossFun) + ", Train MAE, " + this.trainCurrentMAE(LossFun) +
                    ", Test RMSE, " + this.testCurrentRMSE(LossFun) + ", Test MAE, " + this.testCurrentMAE(LossFun) + ", Iter Time, " + (System.currentTimeMillis() - TimeStart) / 1000);

            if ((round - this.min_Round) >= delayCount) {
                break;
            }

            System.out.println("This round for BAS = " + round + " Train RMSE = " + this.trainCurrentRMSE(LossFun) + ";Valid RMSE="+this.validCurrentRMSE(LossFun)+ " Test RMSE = " + this.testCurrentRMSE(LossFun));
            System.out.println("This round for BAS = " + round + " Train MAE  = " + this.trainCurrentMAE(LossFun) + ";Valid MAE=" +this.validCurrentMAE(LossFun)+ "  Test MAE  = " + this.testCurrentMAE(LossFun));

        }


        // Calculate error on test data set
        this.rollBackMinFeatures();
        min_Error_MAE = this.testCurrentMAE(LossFun);
        min_Error_RMSE = this.testCurrentRMSE(LossFun);
        double Final_train_RMSE = this.trainCurrentRMSE(LossFun);
        double Final_train_MAE = this.trainCurrentMAE(LossFun);
        System.out.println("Final round = " + (round + 1 - delayCount) + " Train RMSE = " + Final_train_RMSE + ";Valid RMSE="+this.validCurrentRMSE(LossFun)+ " Test RMSE = " + min_Error_RMSE + "; final time =" + (finalRoundTimeEnd - TimeStart) / 1000.);
        System.out.println("Final round = " + (round + 1 - delayCount) + " Train MAE  = " + Final_train_MAE + ";Valid MAE=" +this.validCurrentMAE(LossFun)+ "  Test MAE  = " + min_Error_MAE+ "; final time =" + (finalRoundTimeEnd - TimeStart) / 1000.);

        p1.println();
        p1.println("Final round = " + (round + 1 - delayCount) + " Train RMSE = " + Final_train_RMSE + "; Valid RMSE = " + this.validCurrentRMSE(LossFun) + "; Test RMSE = " + min_Error_RMSE+ "; final time =" + (finalRoundTimeEnd - TimeStart) / 1000.);
        p1.println("Final round = " + (round + 1 - delayCount) + " Train MAE  = " + Final_train_MAE + "; Valid MAE = " + this.validCurrentMAE(LossFun) +  ";Test MAE  = " + min_Error_MAE+ "; final time =" + (finalRoundTimeEnd - TimeStart) / 1000.);
        this.TrainRound = round;

        // DEParam.total_round += (round-2.5); //added by CJ, for averaging validation time  // remove on 2021/07/14, because i think (round + 1 - delayCount) is the total rounds

        //System.out.println("end of train DE");
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    // change to bas_group_optimizer, and abandon it, because result is not good and the time-cost is huge

    // BAS_Optimizer() is to optimize one pu, bu or qi ci via using one beetle searching path process
    public void BAS_Optimizer(ArrayList<RTuple> TrainData, BAS_Param BASParams, double[] x, double[][] y, double[][] c, int userID, TrainErrorSet TrainError) throws IOException
    {

        //System.out.println("BAS: ");

        ///////////////////////////
        // Initial configuration //
        ///////////////////////////
        int dataSize = TrainData.size();

        if (dataSize == 0) {
            return;
        }

        double Errlim = 0.0001; // the error limit predefined

        //System.out.println(" Train RMSE BEFOR BAS_OPTIMIZER = " + this.trainCurrentRMSE(LossFunSet.TwoDimMF) + ";Valid RMSE="+this.validCurrentRMSE(LossFunSet.TwoDimMF)+ " Test RMSE = " + this.testCurrentRMSE(LossFunSet.TwoDimMF));

        // change the single direction to two-dimensional arrays, there are K-directions for K-pair beetle antennae
        double[] directions = new double[BASParams.BASDim]; // direction of beetle searching at the each step
        double direction_norm;    // the initial normalized direction and xleft, xright

        double[] xleft = new double[BASParams.BASDim]; // the left-hand side
        double[] xright = new double[BASParams.BASDim]; // the right-hand side

        double[] xtemp = new double[BASParams.BASDim]; // the temp x value to record the position updated with xleft and xright
        double[] xtempOpp = new double[BASParams.BASDim]; // the temp opposite x value for the next loop, if the global one is not need to be updated


        double ant_length = BASParams.bas_antlength0; // initialize the antennae length d
        double xt_delta = BASParams.bas_delta0;  // initialize the xt step delta


        // variables for calculating the fitness function
        ////////////////////////////////////////////////
        double tempLeftB;
        double tempRightB;
        double tempCenterB;

        // fitness value for antennae and x temp
        double fitLeftValue;
        double fitRightValue;
        double fitCenterValue;

        double eachNumGbest = 0;	// the global best fitness value
        double PrefitValue = 1000;  // the variable recording previous fitness value
        double[] xGbest = new double[BASParams.BASDim]; //the global best x vector


        double[] tempC = new double[dataSize];
        double bas_lambda = BASParams.regular_lambda;  // get the lambda from bas_param, which should be initialized before calling bas
        double Err1 = 0;         // the fitness improvement compared with last itr
        double[] Rij = new double[dataSize];
        ////////////////
        // end the variables for fitness function

        int basDelay = 0;
        int gen = 0;       // the iteration of the swarm has been updated

        Random random = new Random(System.currentTimeMillis());

        // functions and variables for sorting the fitness function
        CommonRecomm_NEW.IndexSort Isort = new IndexSort();    // the index sort defined in DE_Param.java, used for sorting the X
        Integer anum = 0;  // the ascend index sorted with Isort

        //	System.out.println(DeParams.DeNum+"\n dim"+ DeParams.DeDim);

        //System.out.println(" Train RMSE before init = " + this.trainCurrentRMSE(LossFunSet.TwoDimMF) + ";Valid RMSE="+this.validCurrentRMSE(LossFunSet.TwoDimMF)+ " Test RMSE = " + this.testCurrentRMSE(LossFunSet.TwoDimMF));

        // output the optimized paras pu,bu to estimated matrix
				/*for (int j = 0; j < CommonRecomm_NEW.featureDimension; j++)
					P[userID][j] = x[j];
				for (int j = 0; j < B_Count; j++)
					B[userID][j] = x[j + CommonRecomm_NEW.featureDimension];

				System.out.println(" Train RMSE before init = " + this.trainCurrentRMSE(LossFunSet.TwoDimMF) + ";Valid RMSE="+this.validCurrentRMSE(LossFunSet.TwoDimMF)+ " Test RMSE = " + this.testCurrentRMSE(LossFunSet.TwoDimMF));*/

        //////////////////////////////////////////////////////////////////
        // Initialize the evolutionary individuals
        //////////////////////////////////////////////////////////////////

        // calculating the common parts of already known rates - c
        for (int j = 0; j < dataSize; j++)
        {
            tempC[j]= 0;
            for (int z = 0; z < c[j].length; z++)
            {
                tempC[j] +=c[j][z];
            }

            Rij[j] = TrainData.get(j).dRating - tempC[j];     // the already known rating r - c
        }

        for (int j = 0; j < BASParams.BASDim; j++ )
        {
            // initial xgbest
            xGbest[j] = x[j];
        }

        // calculate the fitness value for RMSE and MAE, for x0, respectively
        if (TrainError == TrainErrorSet.RMSE) // for RMSE, fitness value equals to pu2+b2
        {

            for (int j = 0; j < BASParams.BASDim; j++ )
            {
                eachNumGbest += x[j] * x[j];
            }

        }  else {
            // for MAE, fitness value equals to pu+b
            for (int j = 0; j < BASParams.BASDim; j++ )
            {
                eachNumGbest += Math.abs(x[j]);
            }
        }

        eachNumGbest *= bas_lambda*dataSize;         // lambda* datasize *(pu2+b2), regulation item

        tempCenterB = x[BASParams.BASDim-1];                           // bu

        if (TrainError == TrainErrorSet.RMSE)
        {
            for (int k = 0; k < dataSize; k++)
            {
                double tempCenterXY = 0;

                for (int z = 0; z < CommonRecomm_NEW.featureDimension; z++)
                {
                    tempCenterXY += x[z]*y[k][z];
                }

                eachNumGbest += Math.pow(Rij[k] - tempCenterXY - tempCenterB, 2.0);   // ((r-c-b-pq)2.0+regulation item           //fitness function, and learning objective function
            }
        }else {    // for MAE

            for (int k = 0; k < dataSize; k++)
            {
                double tempCenterXY = 0;

                for (int z = 0; z < CommonRecomm_NEW.featureDimension; z++)
                {
                    tempCenterXY += x[z]*y[k][z];
                }

                eachNumGbest += Math.abs(Rij[k] - tempCenterXY - tempCenterB);   // ((r-c)-r^-regulation item- b)
            }
        }
        // end of calculating fitness function for x0
        // for the initial one, gbest is the center fitness value
        PrefitValue = eachNumGbest;

        double tempLen;

        // the while loop for ADAM optimization
        ////////////////////////////////////////
        do{

            // re-zero the normalized direction, give their directions and calculate their norms

            direction_norm = 0;

            //////////////////////////////
            for (int j = 0; j < BASParams.BASDim; j++ )
            {
                directions[j] = random.nextDouble() - 0.5; //[-0.5, 0.5] the direction
                direction_norm += Math.abs(directions[j]); // the L1 norm
            }

            // calculate xleft, xright, and direction respectively
            // re-zero the k fitness functions
            fitLeftValue =  0;      // fitness value for each individual
            fitRightValue =  0;
            fitCenterValue =  0;

            for (int j = 0; j < BASParams.BASDim; j++ )
            {
                directions[j] = directions[j] / (0.01 + direction_norm); // normalized, the eps for BAS is 0.01

                // the left and right position of the center x
                tempLen = ant_length * directions[j];
                xright[j] = xGbest[j] + tempLen;
                xleft[j] = xGbest[j] - tempLen;
            }

            // calculate the fitness value for RMSE and MAE, for K-pair xleft and xright, respectively
            ///////////////////////////////////////////////////////////////////////////////////
            if (TrainError == TrainErrorSet.RMSE) // for RMSE, fitness value equals to pu2+b2
            {
                for (int j = 0; j < BASParams.BASDim; j++ )
                {
                    // for RMSE, fitness value
                    fitLeftValue += xleft[j] * xleft[j];
                    fitRightValue += xright[j] * xright[j];
                }

            }  else {
                // for MAE, fitness value equals to pu+b
                for (int j = 0; j < BASParams.BASDim; j++ )
                {
                    fitLeftValue += Math.abs(xleft[j]);
                    fitRightValue += Math.abs(xright[j]);
                }
            }

            fitLeftValue *= bas_lambda*dataSize;         // lambda*(pu2+b2), regulation item
            fitRightValue *= bas_lambda*dataSize;         // lambda*(pu2+b2), regulation item

            tempLeftB = xleft[BASParams.BASDim-1];                           // bu
            tempRightB = xright[BASParams.BASDim-1];                         // bu

            //double[] tempLeftXY = new double[K];
            //double[] tempRightXY = new double[K];

            if (TrainError == TrainErrorSet.RMSE)
            {
                for (int z = 0; z < dataSize; z++)
                {
                    double tempLeftXY = 0;
                    double tempRightXY = 0;

                    for (int j = 0; j < CommonRecomm_NEW.featureDimension; j++)
                    {
                        tempLeftXY += xleft[j]*y[z][j];
                        tempRightXY += xright[j]*y[z][j];
                    }

                    fitLeftValue += Math.pow(Rij[z] - tempLeftXY - tempLeftB, 2.0);   // ((r-c-b-pq)2.0+regulation item              //fitness function, and learning objective function
                    fitRightValue += Math.pow(Rij[z] - tempRightXY - tempRightB, 2.0);   // ((r-c-b-pq)2.0+regulation item           //fitness function, and learning objective function
                }
            }else {    // for MAE
                for (int z = 0; z < dataSize; z++)
                {
                    double tempLeftXY = 0;
                    double tempRightXY = 0;

                    for (int j = 0; j < CommonRecomm_NEW.featureDimension; j++)
                    {
                        tempLeftXY += xleft[j]*y[z][j];
                        tempRightXY += xright[j]*y[z][j];
                    }

                    fitLeftValue += Math.abs(Rij[z] - tempLeftXY - tempLeftB);   // ((r-c)-r^-regulation item- b)          //fitness function, and learning objective function
                    fitRightValue += Math.abs(Rij[z] - tempLeftXY - tempLeftB);   // ((r-c)-r^-regulation item- b)         //fitness function, and learning objective function
                }
            } // end of calculating K-pair fitness function of xr and xl

		           	/*if ((fitLeftValue < eachNumGbest) || (fitRightValue < eachNumGbest))
		           	{
						System.out.println("fitLeftValue: " + fitLeftValue  + "fitRightValue: " + fitRightValue + "eachNumGbest: " + eachNumGbest);
		           	}   // for observing the left or right step whether better than the original one      */

            // calculate the xtemp from x, xleft and xright, and the fitness function of xtemp
            // also caculate the opposite direction of xtemp, for the case: xtemp is not the gbest, still update the x in the next loop
            /////////////////////////////////////////////////


            //double RandNum = random.nextDouble()*2.0 - 1.0;

            for (int j = 0; j < BASParams.BASDim; j++)
            {	// x(t) = x(t) + step* direction * sign(f(xr)-f(xl))
                //xtemp[j] = xGbest[j] - (1 - 0.1 * RandNum)* xt_delta * directions[k][j] * Math.signum(fitRightValue[k] - fitLeftValue[k]);
                //xtempOpp[j] = x[j] - xt_delta * directions[j] * Math.signum(fitRightValue - fitLeftValue);
                xtemp[j] = xGbest[j] - xt_delta * directions[j] * Math.signum(fitRightValue - fitLeftValue);
            }


            // calculate the fitness value of xtemp
            if (TrainError == TrainErrorSet.RMSE) // for RMSE, fitness value equals to pu2+b2
            {
                for (int j = 0; j < BASParams.BASDim; j++ )
                {
                    fitCenterValue += xtemp[j] * xtemp[j];
                }

                fitCenterValue *= bas_lambda*dataSize;         // lambda*(pu2+b2), regulation item
                tempCenterB = xtemp[BASParams.BASDim-1];                           // bu

                for (int j = 0; j < dataSize; j++)
                {
                    double tempCenterXY = 0;

                    for (int z = 0; z < CommonRecomm_NEW.featureDimension; z++)
                    {
                        tempCenterXY += xtemp[z]*y[j][z];
                    }

                    fitCenterValue += Math.pow(Rij[j] - tempCenterXY - tempCenterB, 2.0);   // ((r-c-b-pq)2.0+regulation item           //fitness function, and learning objective function
                }


            }  else {
                // for MAE, fitness value equals to pu+b
                for (int j = 0; j < BASParams.BASDim; j++ )
                {
                    fitCenterValue += xtemp[j];
                }

                fitCenterValue *= bas_lambda*dataSize;         // lambda*(pu2+b2), regulation item
                tempCenterB = xtemp[BASParams.BASDim-1];                           // bu

                for (int j = 0; j < dataSize; j++)
                {
                    double tempCenterXY = 0;

                    for (int z = 0; z < CommonRecomm_NEW.featureDimension; z++)
                    {
                        tempCenterXY += xtemp[z]*y[j][z];
                    }

                    fitCenterValue += Math.abs(Rij[j] - tempCenterXY - tempCenterB);   // ((r-c)-r^-regulation item- b)          //fitness function, and learning objective function
                }
            }   // end of calculating fitness function of xtemp

            // if xtemp archives smaller fitness value than global best, substitute the global with xtemp
            // else update the x in the next loop without substituting the global best
            ////////////////////////////////////////////////////////////////////////////////////////



            if(fitCenterValue < eachNumGbest)
            {
                //System.out.println("fitCenterValue: " + fitCenterValue+" eachNumGbest: " + eachNumGbest);
                //System.out.println("eachNumGbest: "+ eachNumGbest + "fitCenterValue[anum[0]]: " + fitCenterValue[anum[0]]);
                eachNumGbest = fitCenterValue;
                for (int j = 0; j < BASParams.BASDim; j++ )
                {
                    xGbest[j] = x[j] = xtemp[j];  //update the global and the next loop
                }
                //System.out.println("fitCenterValue: " + fitCenterValue);

                // updated the step length only when update the global best one, right or improve???
                //////////////////////////////////////////////////
                // d(t) = d(t-1)*0.95 + 0.01
                ant_length = 0.95 * ant_length + 0.01;
                // delta(t) = delta(t-1)*0.95
                xt_delta = 0.95 * xt_delta; //(1 - 0.1 * RandNum) * xt_delta;    // 0.95  * xt_delta
            } // end of updating the global values
	       			/*else if (random.nextDouble() <= 0.5){  // update x in the next loop without substituting the global best

	       				// not clear for the signal, and the signal of xleft and xright is opposite of the paper with the github code
	       				for (int j = 0; j < BASParams.BASDim; j++ )
		           		{
		       			   // x(t) = x(t-1) + step* direction * sign(f(xr)-f(xl))
				           	x[j] = xtemp[j];  //update the next loop
		           		}
	       			}else
	       			{
	       				// not clear for the signal, and the signal of xleft and xright is opposite of the paper with the github code
	       				for (int j = 0; j < BASParams.BASDim; j++ )
		       			{
		  				    // x(t) = x(t-1) + step* direction * sign(f(xr)-f(xl))
     		           		x[j] = xtempOpp[j];  //update the next loop with the opposite position
		       			}
	       			}*/

            ////////////////////////////////////////////////////
            Err1 =  PrefitValue - fitCenterValue; // the fitness error improvement compared with two itrs
            PrefitValue = fitCenterValue;

            // accumulate the times of the error less than ErrLim
            if (Err1 >= Errlim) {
                basDelay = 0;

                //System.out.println("Err1: "+ Err1 + "Errlim: "+ Errlim + "gen: "+ gen + "basDelay: "+ basDelay);
            }
            else {
                basDelay = basDelay + 1;    //System.out.println("iteration times: "+gen + "\n delay:" + delay);
            }

            gen++;    // the iteration time + 1

            //System.out.println("global fitness value:"+ obGF +" gen:"+gen +" delay:" +delay+" DeParams.MaxGen:"+DeParams.MaxGen);
            //System.out.println("Err1:"+Err1+" Errlim:"+Errlim);
					/*if (dataSize != 0) {
						for (int z = 0; z < DeParams.DeDim; z++)
						{
							x[z] = Gb[z];
						}
					}

					for (int j = 0; j < CommonRecomm_NEW.featureDimension; j++)
						P[userID][j] = x[j];
					for (int j = 0; j < B_Count; j++)
						B[userID][j] = x[j + CommonRecomm_NEW.featureDimension];*/

            //System.out.println(" Train RMSE after selection = " + this.trainCurrentRMSE(LossFunSet.TwoDimMF) + ";Valid RMSE="+this.validCurrentRMSE(LossFunSet.TwoDimMF)+ " Test RMSE = " + this.testCurrentRMSE(LossFunSet.TwoDimMF));

        }while((gen < BASParams.MaxGen) && (basDelay <= 4));
        // err less the predefined for two times || iteration times greater than MaxGen, break the iterations


        // copy the global best one to x

        for (int z = 0; z < BASParams.BASDim; z++)
        {
            x[z] = xGbest[z];
        }




			/*for (int j = 0; j < CommonRecomm_NEW.featureDimension; j++)
					P[userID][j] = x[j];
				for (int j = 0; j < B_Count; j++)
					B[userID][j] = x[j + CommonRecomm_NEW.featureDimension];

				System.out.println(" Train RMSE after iteration and re-value P,B for 1: " + this.trainCurrentRMSE(LossFunSet.TwoDimMF) + ";Valid RMSE="+this.validCurrentRMSE(LossFunSet.TwoDimMF)+ " Test RMSE = " + this.testCurrentRMSE(LossFunSet.TwoDimMF));
	     	*/
        //System.out.println("end DE: end BAS optimizer");
    }// end BAS_Optimizer

    // BAS_Group_optimizer() is to optimize one pu,bu or qi,ci via using one beetle searching path process
    public void BAS_Group_Optimizer(ArrayList<RTuple> TrainData, BAS_Param BASParams, double[] x, double[][] y, double[][] c, int userID, TrainErrorSet TrainError) throws IOException
    {

        //System.out.println("BAS: ");

        ///////////////////////////
        // Initial configuration //
        ///////////////////////////
        int dataSize = TrainData.size();

        if (dataSize == 0) {
            return;
        }

        double Errlim = 0.0001; // the error limit predefined

        //System.out.println(" Train RMSE BEFOR BAS_OPTIMIZER = " + this.trainCurrentRMSE(LossFunSet.TwoDimMF) + ";Valid RMSE="+this.validCurrentRMSE(LossFunSet.TwoDimMF)+ " Test RMSE = " + this.testCurrentRMSE(LossFunSet.TwoDimMF));

        // change the single direction to two-dimensional arrays, there are K-directions for K-pair beetle antennae
        int K = BASParams.BASPopNum; // the number of bas
        double[][] directions = new double[K][BASParams.BASDim]; // direction of beetle searching at the each step
        double[] direction_norm = new double[K];    // the initial normalized direction and xleft, xright

        double[][] xleft = new double[K][BASParams.BASDim]; // the left-hand side
        double[][] xright = new double[K][BASParams.BASDim]; // the right-hand side

        double[][] xtemp = new double[K][BASParams.BASDim]; // the temp x value to record the position updated with xleft and xright
        double[][] xtempOpp = new double[K][BASParams.BASDim]; // the temp opposite x value for the next loop, if the global one is not need to be updated


        double ant_length = BASParams.bas_antlength0; // initialize the antennae length d
        double xt_delta = BASParams.bas_delta0;  // initialize the xt step delta


        // variables for calculating the fitness function
        ////////////////////////////////////////////////
        double[] tempLeftB = new double[K];
        double[] tempRightB = new double[K];
        double[] tempCenterB = new double[K];

        // fitness value for K-pair antennae
        double[] fitLeftValue = new double[K];
        double[] fitRightValue = new double[K];
        double[] fitCenterValue = new double[K];

        double eachNumGbest = 0;	// the global best fitness value
        double PrefitValue = 1000;  // the variable recording previous fitness value
        double[] xGbest = new double[BASParams.BASDim]; //the global best x vector


        double[] tempC = new double[dataSize];
        double bas_lambda = BASParams.regular_lambda;  // get the lambda from bas_param, which should be initialized before calling bas
        double Err1 = 0;         // the fitness improvement compared with last itr
        double[] Rij = new double[dataSize];
        ////////////////
        // end the variables for fitness function

        int basDelay = 0;
        int gen = 0;       // the iteration of the swarm has been updated

        Random random = new Random(System.currentTimeMillis());

        // functions and variables for sorting the fitness function
        CommonRecomm_NEW.IndexSort Isort = new IndexSort();    // the index sort defined in DE_Param.java, used for sorting the X
        Integer [] anum = new Integer [K];  // the ascend index sorted with Isort

        //	System.out.println(DeParams.DeNum+"\n dim"+ DeParams.DeDim);

        //System.out.println(" Train RMSE before init = " + this.trainCurrentRMSE(LossFunSet.TwoDimMF) + ";Valid RMSE="+this.validCurrentRMSE(LossFunSet.TwoDimMF)+ " Test RMSE = " + this.testCurrentRMSE(LossFunSet.TwoDimMF));

        // output the optimized paras pu,bu to estimated matrix
			/*for (int j = 0; j < CommonRecomm_NEW.featureDimension; j++)
				P[userID][j] = x[j];
			for (int j = 0; j < B_Count; j++)
				B[userID][j] = x[j + CommonRecomm_NEW.featureDimension];

			System.out.println(" Train RMSE before init = " + this.trainCurrentRMSE(LossFunSet.TwoDimMF) + ";Valid RMSE="+this.validCurrentRMSE(LossFunSet.TwoDimMF)+ " Test RMSE = " + this.testCurrentRMSE(LossFunSet.TwoDimMF));*/

        //////////////////////////////////////////////////////////////////
        // Initialize the evolutionary individuals
        //////////////////////////////////////////////////////////////////

        // calculating the common parts of already known rates - c
        for (int j = 0; j < dataSize; j++)
        {
            tempC[j]= 0;
            for (int z = 0; z < c[j].length; z++)
            {
                tempC[j] +=c[j][z];
            }

            Rij[j] = TrainData.get(j).dRating - tempC[j];     // the already known rating r - c
        }

        for (int j = 0; j < BASParams.BASDim; j++ )
        {
            // initial xgbest
            xGbest[j] = x[j];
        }

        // calculate the fitness value for RMSE and MAE, for x0, respectively
        if (TrainError == TrainErrorSet.RMSE) // for RMSE, fitness value equals to pu2+b2
        {

            for (int j = 0; j < BASParams.BASDim; j++ )
            {
                eachNumGbest += x[j] * x[j];
            }

        }  else {
            // for MAE, fitness value equals to pu+b
            for (int j = 0; j < BASParams.BASDim; j++ )
            {
                eachNumGbest += Math.abs(x[j]);
            }
        }

        eachNumGbest *= bas_lambda*dataSize;         // lambda* datasize *(pu2+b2), regulation item

        tempCenterB[0] = x[BASParams.BASDim-1];                           // bu

        if (TrainError == TrainErrorSet.RMSE)
        {
            for (int k = 0; k < dataSize; k++)
            {
                double tempCenterXY = 0;

                for (int z = 0; z < CommonRecomm_NEW.featureDimension; z++)
                {
                    tempCenterXY += x[z]*y[k][z];
                }

                eachNumGbest += Math.pow(Rij[k] - tempCenterXY - tempCenterB[0], 2.0);   // ((r-c-b-pq)2.0+regulation item           //fitness function, and learning objective function
            }
        }else {    // for MAE

            for (int k = 0; k < dataSize; k++)
            {
                double tempCenterXY = 0;

                for (int z = 0; z < CommonRecomm_NEW.featureDimension; z++)
                {
                    tempCenterXY += x[z]*y[k][z];
                }

                eachNumGbest += Math.abs(Rij[k] - tempCenterXY - tempCenterB[0]);   // ((r-c)-r^-regulation item- b)
            }
        }
        // end of calculating fitness function for x0
        // for the initial one, gbest is the center fitness value
        PrefitValue = eachNumGbest;

        double tempLen;

        // the while loop for ADAM optimization
        ////////////////////////////////////////
        do{

            // re-zero the K-normalized direction, give their directions and calculate their norms
            for ( int k = 0; k < K; k++ )
            {
                direction_norm[k] = 0;

                //////////////////////////////
                for (int j = 0; j < BASParams.BASDim; j++ )
                {
                    directions[k][j] = random.nextDouble() - 0.5; //[-0.5, 0.5] the direction
                    direction_norm[k] += Math.abs(directions[k][j]); // the L1 norm
                }
            }

            // calculate xleft, xright, and direction respectively
            for ( int k = 0; k < K; k++ )
            {
                // re-zero the k fitness functions
                fitLeftValue[k] =  0;      // fitness value for each individual
                fitRightValue[k] =  0;
                fitCenterValue[k] =  0;

                for (int j = 0; j < BASParams.BASDim; j++ )
                {
                    directions[k][j] = directions[k][j] / (0.01 + direction_norm[k]); // normalized, the eps for BAS is 0.01

                    // the left and right position of the center x
                    tempLen = ant_length * directions[k][j];
                    xright[k][j] = xGbest[j] + tempLen;
                    xleft[k][j] = xGbest[j] - tempLen;
                }
            }

            // calculate the fitness value for RMSE and MAE, for K-pair xleft and xright, respectively
            ///////////////////////////////////////////////////////////////////////////////////
            if (TrainError == TrainErrorSet.RMSE) // for RMSE, fitness value equals to pu2+b2
            {
                for ( int k = 0; k < K; k++ )
                {
                    for (int j = 0; j < BASParams.BASDim; j++ )
                    {
                        // for RMSE, fitness value
                        fitLeftValue[k] += xleft[k][j] * xleft[k][j];
                        fitRightValue[k] += xright[k][j] * xright[k][j];
                    }
                }

            }  else {

                for ( int k = 0; k < K; k++ )
                {
                    // for MAE, fitness value equals to pu+b
                    for (int j = 0; j < BASParams.BASDim; j++ )
                    {
                        fitLeftValue[k] += Math.abs(xleft[k][j]);
                        fitRightValue[k] += Math.abs(xright[k][j]);
                    }
                }
            }

            for ( int k = 0; k < K; k++ )
            {
                fitLeftValue[k] *= bas_lambda*dataSize;         // lambda*(pu2+b2), regulation item
                fitRightValue[k] *= bas_lambda*dataSize;         // lambda*(pu2+b2), regulation item

                tempLeftB[k] = xleft[k][BASParams.BASDim-1];                           // bu
                tempRightB[k] = xright[k][BASParams.BASDim-1];                         // bu
            }

            //double[] tempLeftXY = new double[K];
            //double[] tempRightXY = new double[K];

            if (TrainError == TrainErrorSet.RMSE)
            {
                for (int k = 0; k < K; k++)
                {

                    for (int z = 0; z < dataSize; z++)
                    {
                        double tempLeftXY = 0;
                        double tempRightXY = 0;

                        for (int j = 0; j < CommonRecomm_NEW.featureDimension; j++)
                        {
                            tempLeftXY += xleft[k][j]*y[z][j];
                            tempRightXY += xright[k][j]*y[z][j];
                        }

                        fitLeftValue[k] += Math.pow(Rij[z] - tempLeftXY - tempLeftB[k], 2.0);   // ((r-c-b-pq)2.0+regulation item              //fitness function, and learning objective function
                        fitRightValue[k] += Math.pow(Rij[z] - tempRightXY - tempRightB[k], 2.0);   // ((r-c-b-pq)2.0+regulation item           //fitness function, and learning objective function
                    }
                }
            }else {    // for MAE
                for (int k = 0; k < K; k++)
                {
                    for (int z = 0; z < dataSize; z++)
                    {
                        double tempLeftXY = 0;
                        double tempRightXY = 0;

                        for (int j = 0; j < CommonRecomm_NEW.featureDimension; j++)
                        {
                            tempLeftXY += xleft[k][j]*y[z][j];
                            tempRightXY += xright[k][j]*y[z][j];
                        }

                        fitLeftValue[k] += Math.abs(Rij[z] - tempLeftXY - tempLeftB[k]);   // ((r-c)-r^-regulation item- b)          //fitness function, and learning objective function
                        fitRightValue[k] += Math.abs(Rij[z] - tempLeftXY - tempLeftB[k]);   // ((r-c)-r^-regulation item- b)         //fitness function, and learning objective function
                    }
                }
            } // end of calculating K-pair fitness function of xr and xl

	           	/*if ((fitLeftValue < eachNumGbest) || (fitRightValue < eachNumGbest))
	           	{
					System.out.println("fitLeftValue: " + fitLeftValue  + "fitRightValue: " + fitRightValue + "eachNumGbest: " + eachNumGbest);
	           	}   // for observing the left or right step whether better than the original one      */

            // calculate the xtemp from x, xleft and xright, and the fitness function of xtemp
            // also caculate the opposite direction of xtemp, for the case: xtemp is not the gbest, still update the x in the next loop
            /////////////////////////////////////////////////


            double RandNum = random.nextDouble()*2.0 - 1.0;

            for (int k = 0; k < K; k++)
            {
                for (int j = 0; j < BASParams.BASDim; j++)
                {	// x(t) = x(t) + step* direction * sign(f(xr)-f(xl))
                    xtemp[k][j] = xGbest[j] - (1 - 0.1 * RandNum)* xt_delta * directions[k][j] * Math.signum(fitRightValue[k] - fitLeftValue[k]);
                    //xtempOpp[j] = x[j] - xt_delta * directions[j] * Math.signum(fitRightValue - fitLeftValue);
                }
            }


            // calculate the fitness value of xtemp
            if (TrainError == TrainErrorSet.RMSE) // for RMSE, fitness value equals to pu2+b2
            {
                for (int k = 0; k < K; k++)
                {

                    for (int j = 0; j < BASParams.BASDim; j++ )
                    {
                        fitCenterValue[k] += xtemp[k][j] * xtemp[k][j];
                    }

                    fitCenterValue[k] *= bas_lambda*dataSize;         // lambda*(pu2+b2), regulation item
                    tempCenterB[k] = xtemp[k][BASParams.BASDim-1];                           // bu

                    for (int j = 0; j < dataSize; j++)
                    {
                        double tempCenterXY = 0;

                        for (int z = 0; z < CommonRecomm_NEW.featureDimension; z++)
                        {
                            tempCenterXY += xtemp[k][z]*y[j][z];
                        }

                        fitCenterValue[k] += Math.pow(Rij[j] - tempCenterXY - tempCenterB[k], 2.0);   // ((r-c-b-pq)2.0+regulation item           //fitness function, and learning objective function
                    }

                }

            }  else {
                // for MAE, fitness value equals to pu+b
                for (int k = 0; k < K; k++)
                {
                    for (int j = 0; j < BASParams.BASDim; j++ )
                    {
                        fitCenterValue[k] += xtemp[k][j];
                    }

                    fitCenterValue[k] *= bas_lambda*dataSize;         // lambda*(pu2+b2), regulation item
                    tempCenterB[k] = xtemp[k][BASParams.BASDim-1];                           // bu

                    for (int j = 0; j < dataSize; j++)
                    {
                        double tempCenterXY = 0;

                        for (int z = 0; z < CommonRecomm_NEW.featureDimension; z++)
                        {
                            tempCenterXY += xtemp[k][z]*y[j][z];
                        }

                        fitCenterValue[k] += Math.abs(Rij[j] - tempCenterXY - tempCenterB[k]);   // ((r-c)-r^-regulation item- b)          //fitness function, and learning objective function
                    }
                }
            }   // end of calculating fitness function of xtemp

            // if xtemp archives smaller fitness value than global best, substitute the global with xtemp
            // else update the x in the next loop without substituting the global best
            ////////////////////////////////////////////////////////////////////////////////////////
            anum = Isort.Ind(fitCenterValue);  // sort by ascent order and get the index


            if(fitCenterValue[anum[0]] < eachNumGbest)
            {
                //System.out.println("fitCenterValue: " + fitCenterValue+" eachNumGbest: " + eachNumGbest);
                //System.out.println("eachNumGbest: "+ eachNumGbest + "fitCenterValue[anum[0]]: " + fitCenterValue[anum[0]]);
                eachNumGbest = fitCenterValue[anum[0]];
                for (int j = 0; j < BASParams.BASDim; j++ )
                {
                    xGbest[j] = x[j] = xtemp[anum[0]][j];  //update the global and the next loop
                }
                //System.out.println("fitCenterValue: " + fitCenterValue);

                // updated the step length only when update the global best one, right or improve???
                //////////////////////////////////////////////////
                // d(t) = d(t-1)*0.95 + 0.01
                ant_length = 0.95 * ant_length + 0.01;
                // delta(t) = delta(t-1)*0.95
                xt_delta = 0.95 * xt_delta; //(1 - 0.1 * RandNum) * xt_delta;    // 0.95  * xt_delta
            } // end of updating the global values
       			/*else if (random.nextDouble() <= 0.5){  // update x in the next loop without substituting the global best

       				// not clear for the signal, and the signal of xleft and xright is opposite of the paper with the github code
       				for (int j = 0; j < BASParams.BASDim; j++ )
	           		{
	       			   // x(t) = x(t-1) + step* direction * sign(f(xr)-f(xl))
			           	x[j] = xtemp[j];  //update the next loop
	           		}
       			}else
       			{
       				// not clear for the signal, and the signal of xleft and xright is opposite of the paper with the github code
       				for (int j = 0; j < BASParams.BASDim; j++ )
	       			{
	  				    // x(t) = x(t-1) + step* direction * sign(f(xr)-f(xl))
 		           		x[j] = xtempOpp[j];  //update the next loop with the opposite position
	       			}
       			}*/

            ////////////////////////////////////////////////////
            Err1 =  PrefitValue - fitCenterValue[anum[0]]; // the fitness error improvement compared with two itrs
            PrefitValue = fitCenterValue[anum[0]];

            // accumulate the times of the error less than ErrLim
            if (Err1 >= Errlim) {
                basDelay = 0;

                //System.out.println("Err1: "+ Err1 + "Errlim: "+ Errlim + "gen: "+ gen + "basDelay: "+ basDelay);
            }
            else {
                basDelay = basDelay + 1;    //System.out.println("iteration times: "+gen + "\n delay:" + delay);
            }

            gen++;    // the iteration time + 1

            //System.out.println("global fitness value:"+ obGF +" gen:"+gen +" delay:" +delay+" DeParams.MaxGen:"+DeParams.MaxGen);
            //System.out.println("Err1:"+Err1+" Errlim:"+Errlim);
				/*if (dataSize != 0) {
					for (int z = 0; z < DeParams.DeDim; z++)
					{
						x[z] = Gb[z];
					}
				}

				for (int j = 0; j < CommonRecomm_NEW.featureDimension; j++)
					P[userID][j] = x[j];
				for (int j = 0; j < B_Count; j++)
					B[userID][j] = x[j + CommonRecomm_NEW.featureDimension];*/

            //System.out.println(" Train RMSE after selection = " + this.trainCurrentRMSE(LossFunSet.TwoDimMF) + ";Valid RMSE="+this.validCurrentRMSE(LossFunSet.TwoDimMF)+ " Test RMSE = " + this.testCurrentRMSE(LossFunSet.TwoDimMF));

        }while((gen < BASParams.MaxGen) && (basDelay <= 4));
        // err less the predefined for two times || iteration times greater than MaxGen, break the iterations


        // copy the global best one to x

        for (int z = 0; z < BASParams.BASDim; z++)
        {
            x[z] = xGbest[z];
        }


		/*for (int j = 0; j < CommonRecomm_NEW.featureDimension; j++)
				P[userID][j] = x[j];
			for (int j = 0; j < B_Count; j++)
				B[userID][j] = x[j + CommonRecomm_NEW.featureDimension];

			System.out.println(" Train RMSE after iteration and re-value P,B for 1: " + this.trainCurrentRMSE(LossFunSet.TwoDimMF) + ";Valid RMSE="+this.validCurrentRMSE(LossFunSet.TwoDimMF)+ " Test RMSE = " + this.testCurrentRMSE(LossFunSet.TwoDimMF));
     	*/
        //System.out.println("end DE: end BAS optimizer");
    }// end BAS_Optimizer

////////////////////////////////////////////////////////////////////////////////////////////////////////////////





    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //      The above codes are for optimization algorithms
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //      The below codes are main function
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////


    public static void main(String[] argv) throws NumberFormatException,
            IOException, InterruptedException {


        LossFunSet[] LossFun = new LossFunSet[] {LossFunSet.TwoDimMF};


        int ValidationTime = 20;  // times of validation set by authors
        int[][] Min_RMSE_Position = new int[ValidationTime][2];
        int[][] Min_MAE_Position = new int[ValidationTime][2];
        double[] Min_RMSE = new double[ValidationTime];
        double[] Min_MAE = new double[ValidationTime];
        double[] Est_Time_RMSE = new double[ValidationTime];
        double[] Est_Time_MAE = new double[ValidationTime];
        double[] Each_round = new double[ValidationTime];  // add by CJ on 2021/09/15

        double Avg_RMSE;
        double Avg_MAE;
        double Avg_Est_Time_RMSE;
        double Avg_Est_Time_MAE;
        double Avg_each_round; // add by CJ on 2021/09/15

        //                        0          1         2          3          4           5           6        7            8              9         10        11         12         13        14          15        16       17       18      19    20
        String[] folderNames = {"Jester3","Epinion","DatingT","WSD_3_RT", "WSD_3_TP","tag_genome","douban", "ML20M", "ExtEpinions2D", "EpinionNew","ML1M","EachMovie","WSD_2_RT","WSD_2_TP","Jester","artificial","ML10M","flixter","Yahoo", "rt", "tp"};
        double[] eta_set =      {0.01,      0.01,     0.001,     0.01,       0.0005,    0.001,    0.05,      0.01,    0.1,             0.1,        0.1,   0.015,      0.1,       0.1,       0.008,   0.001,      0.03,   0.015,     0.01,     0.03,  0.01};
        double[] lambda_set =   {0.01,      0.03,      0.1,      0.01,       0.02,      0.01,     0.02,      0.03,    0.01,            0.01,       0.01,   0.01,      0.01,      0.02,      0.01,    0.01,       0.03,    0.02,      0.03,   0.1, 8};

        // DE lambda sets, which is different in various data sets, the default value is 0.01
        // double[] de_lda_set =   {0.01,      0.01,      0.01,      0.01,       0.01,      0.01,     0.1,      0.3,    0.01,            0.01,      0.01,    0.5,      0.01,      0.01,      0.01,     0.01,      0.01,     1.3,     0.01};

        int[] ProcessingDatasetNo = {17};//{0,2,3,4,5,8,10,12,13};		//Selecting Processing datasets, choose multiple datasets
        if (argv.length > 0)
            ProcessingDatasetNo = new int[] {Integer.valueOf(argv[0])};

        // especially for PSO, no use for DE
        PSO_Param PSO_Params = new PSO_Param();

        // especially for DE, no use for PSO
        DE_Param DE_Params = new DE_Param();

        // especially for BAS
        BAS_Param BAS_Params = new BAS_Param();

        int PSODelayCount = 5;		//PSO iteration
        int DEDelayCount = 5; // DE iterations
        int BASDelayCount = 5; // bas iterations

        // add by CJ on 2021/07/31 for looping all the optimize algorithms for each data set
        OptimizeAlgorithmSet[] opt_alg_set = new OptimizeAlgorithmSet[] {OptimizeAlgorithmSet.BAS}; // OptimizeAlgorithmSet.BPSO,,OptimizeAlgorithmSet.PPSO


        //PSO_Params.PSOMethodType = PSOApproach.PSO;  //PSO, PPSO, GD_PSO
        // end the add for looping all the optimize algorithms for each data set

        PSO_Params.GDFadingFactor = 2;

        CommonRecomm_NEW.iterConvThrd = 0.0001;

       /*comment on 2021/07/30, because these parameters are fixed, there is no need for loop to search the best values
	   // PSO parameters set in main function
	   double[] PSO_w_Set = new double[] {0.3};//, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1}
	   double[] PSO_c_Set = new double[] {1.0};	//Setting PSO parameter of c */



        double[] DE_dw_Set = new double[] {0.1};
        double[] DE_cr_Set = new double[] {0.1};
        //double[] DE_cp_Set = new double[] {1}; COMMNET ON 2021/10/3 FOR ALL INDIVIDUALS SHALL BE CROSSOVER

        int[] DE_Num_Set = new int[] {4,6,8,10,12,14};//4,6,8,10,12,15,20,25,30,35,40,50,60,70,100	 ,35,40,45//Setting de parameter of number of particles
        double[] MOME_rao_Set = new double [] {0.0,0.001,0.005,0.01,0.1}; //1.0 is worse than 1.3 on douban //
        double[] de_lda_Set = new double [] {0.01};  //,0.03,0.05,0.1,0.3,0.5,1,1.5

	   /* comment by CJ on 2021/07/31, for removing the original code that assign the same type method for one dataset
 	   // Selecting Algorithm used for LFA model, added by CJ 2021.4.14
	   int algm = 1; // 0 is for SGD, 1 is for PSO, 2 is for DE
	   OptimizeAlgorithmSet[] OptSet = new OptimizeAlgorithmSet[19];      // initialize algorithm sets, need to update with adding the datasets

	   if (algm == 1) {     //  PSO

		  OptSet = new OptimizeAlgorithmSet[] {OptimizeAlgorithmSet[] OptSet = new OptimizeAlgorithmSet[19];  OptimizeAlgorithmSet.BPSO, OptimizeAlgorithmSet.BPSO
	    		   , OptimizeAlgorithmSet.BPSO, OptimizeAlgorithmSet.BPSO, OptimizeAlgorithmSet.BPSO, OptimizeAlgorithmSet.BPSO,
	    		   OptimizeAlgorithmSet.BPSO, OptimizeAlgorithmSet.BPSO, OptimizeAlgorithmSet.BPSO
	    		   , OptimizeAlgorithmSet.BPSO, OptimizeAlgorithmSet.BPSO, OptimizeAlgorithmSet .BPSO, OptimizeAlgorithmSet.BPSO
	    		   ,OptimizeAlgorithmSet.BPSO,OptimizeAlgorithmSet.BPSO, OptimizeAlgorithmSet.BPSO,OptimizeAlgorithmSet.BPSO,OptimizeAlgorithmSet.BPSO,OptimizeAlgorithmSet.BPSO};


	   } else if (algm == 2) {    // DE

		OptSet = new OptimizeAlgorithmSet[] {OptimizeAlgorithmSet.DE, OptimizeAlgorithmSet.DE, OptimizeAlgorithmSet.DE
	    		   , OptimizeAlgorithmSet.DE, OptimizeAlgorithmSet.DE, OptimizeAlgorithmSet.DE, OptimizeAlgorithmSet.DE,
	    		   OptimizeAlgorithmSet.DE, OptimizeAlgorithmSet.DE, OptimizeAlgorithmSet.DE
	    		   , OptimizeAlgorithmSet.DE, OptimizeAlgorithmSet.DE, OptimizeAlgorithmSet.DE, OptimizeAlgorithmSet.DE
	    		   ,OptimizeAlgorithmSet.DE,OptimizeAlgorithmSet.DE, OptimizeAlgorithmSet.DE,OptimizeAlgorithmSet.DE,OptimizeAlgorithmSet.DE,OptimizeAlgorithmSet.DE};

	   }// for 0 is SGD, not write here
	   comment by CJ on 2021/07/31, for removing the original code that assign the same type method for one dataset */

        Date dt = new Date(System.currentTimeMillis());
        DateFormat df = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss");
        String DateStr = df.format(dt);

        /////////////////////////////  参数调整    ////////////////////////////////////////////////////////////////////////////
        String InputDataDir = "D:\\postGraduateMaterial\\Ensemble_SI\\src\\resources\\dataset\\";		//修改输入数据路径
        String OutputDataDir = "D:\\postGraduateMaterial\\Ensemble_SI\\src\\resources\\output\\";		//修改输出数据路径
        int TestResultNo = 1;				//测试结果数据编号
        TrainErrorSet TrainError = TrainErrorSet.RMSE;		// 训练迭代收敛指标，RMSE：只用RMSE做为收敛指标；MAE：只用MAE作为收敛指标；RMSE_and_MAE：用RMSE和MAE做为收敛指标

        FileOutputStream fs_overall = new FileOutputStream(new File(OutputDataDir + "Train_Result_" + LossFun[0] + "_InitMax=" + init_Max + "_" + TestResultNo + ".csv"));
        PrintStream p_overall = new PrintStream(fs_overall);
        String Overall_Results = "";

        //comment on 2021/07/30, for saving time, there is no need to output or input file
        ReadInitValuefromFile = true; // for pso set true;    //是否从文件读取P，Q，B，C的初始值，
        RecordInitValuetoFile = false;    //是否往文件写入P，Q，B，C的初始值，
        //RecordInitValuetoFile_5 = true;    //是否往文件写入P，Q，B，C的初始值，
        RecordInitValuetoFile_1 = false;//     for pso set true;    //是否往文件写入P，Q，B，C的初始值，





        /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        for(int k : ProcessingDatasetNo)
        {
            CommonRecomm_NEW.eta = eta_set[k]; // especially for wsd_3_tp 0.0008; wsd_2_tp 0.0005 //initial for all 0.01
            //CommonRecomm_NEW.lambda = lambda_set[k]; // especially for wsd_3_tp & wsd_2_tp 0.2;  //initial for all 0.01
            CommonRecomm_NEW.trainingRound = 1000;

            String DataFilename_Behaviour;
            String DataFilename_Valid;
            String DataFilename_Profile;

            // added for output files
            CommonRecomm_NEW.RecordFile_k = OutputDataDir+folderNames[k]+"_";

            // create initializeRatings two parameters, 3 data files, training set, validation set and test set;
//			DataFilename_Behaviour = InputDataDir + folderNames[k] + "\\trainNew.txt"; //"\\trainNew.txt"; 4_1_95;\\8_2_90
//			DataFilename_Valid = InputDataDir + folderNames[k] + "\\valid.txt";		//"\\valid.txt";
//			DataFilename_Profile = InputDataDir + folderNames[k] + "\\test.txt";	 //"\\test.txt";

            // added for debugging
//			System.out.println(DataFilename_Behaviour);
//			System.out.println(DataFilename_Valid);
//			System.out.println(DataFilename_Profile);

            //7：1：2 validation initialization, 3 data files, training set, validation set and test set;
//			CommonRecomm_NEW.initializeRatings(DataFilename_Behaviour, DataFilename_Valid, DataFilename_Profile, "::");

            FileOutputStream fs;
            fs = new FileOutputStream(new File(OutputDataDir + "Test_Results_TrainValidTest_" + folderNames[k] +   "_" + TrainError + "_New_" + DateStr +"_" + eta_set[k]+".csv"));
            PrintStream p = new PrintStream(fs);

            //for (double lambda : lambda_set)
            {
                CommonRecomm_NEW.lambda = lambda_set[k];

                //CommonRecomm_NEW.featureDimension = f = f_set[0];

                Overall_Results = folderNames[k] + ", Lambda=" + CommonRecomm_NEW.lambda + ", eta=" + CommonRecomm_NEW.eta + ",f="  + CommonRecomm_NEW.featureDimension + "\n" + ", RMSE, Training Time, Loop#, MAE, Training Time, Loop#\n";

                init_Scale = init_Max = 0.004;

                long SGDTime1;
                long SGDTime2;
                long SGDTime;

                // initialize yuanye's code
                CommonRecomm_NEW.yy_initiStaticArrays();
                CommonRecomm_NEW.yy_initialX();
                CommonRecomm_NEW.yy_initialV();
                CommonRecomm_NEW.yy_initialfitness();
                CommonRecomm_NEW.yy_initBiasSettings(true, true, 1, 1);

                System.out.println("#################################################");
                // end initializing yuanye's code

                Ensemble_SI SI_New_2 = new Ensemble_SI();
                //SI_New_2.err_type = Err_Type;

				/* comment by CJ on 2021/07/30 for no need to case on PSO/DE for PLFA training
				switch (OptSet[k])
				{
					case BPSO:
					case DE:*/


//						System.out.println("PLFA training ");
//						CommonRecomm_NEW.trainingRound = 1000;    // training round is set 1000
//						CommonRecomm_NEW.delayCount = 5;  // for PLFA delay count, while PSO and DE have their own delay count value
//
//						SGDTime1 = System.currentTimeMillis();
//
//						// train with pso using Yuanye's codes
//						// PLFA model before MPSO
//						// if (algm != 1 || PSO_Params.PSOMethodType != PSOApproach.GD_PSO)  // modify by cj on 2021/07/30 for fixing  the DE training no use PLFA bug
//						//if (PSO_Params.PSOMethodType != PSOApproach.GD_PSO) remove by CJ on  2021/07/31 for loop w/o GD_PSO
//						//{
//						SI_New_2.train_pso_from_yuanye(TrainError);
//						//}  remove by CJ on  2021/07/31 for loop w/o GD_PSO
//						SGDTime2 = System.currentTimeMillis();
//						SGDTime = (SGDTime2 - SGDTime1) / 1000;
//
//						p.println("SGD_RMSE=" + min_Error_RMSE + ", SGD_MAE=" + min_Error_MAE + "Iter. round = " + round+ ". SGD_Time=" + SGDTime + "s");
//
//
//						//SI_New_2.testCurrentRMSE(LossFun[0]);
//
//						System.out.println("PLFA_Train_RMSE=" + SI_New_2.trainCurrentRMSE(LossFun[0]) +"PLFA_valid_RMSE=" + SI_New_2.validCurrentRMSE(LossFun[0])  +"PLFA_Test_RMSE=" + SI_New_2.testCurrentRMSE(LossFun[0]) + ", PLFA_MAE=" + SI_New_2.testCurrentMAE(LossFun[0]) + "Iter. round = " + round+ ". SGD_Time=" + SGDTime + "s");
//						//added by cj, for validating yy's P,Q,B,C files ,but seems wrong
//						CommonRecomm_NEW.cacheArrays();
//						/* comment by CJ on 2021/07/30 for no need to case on PSO/DE for PLFA training
//						break;
//
//				}*/

                if (RecordInitValuetoFile)
                {
                    System.out.println("Output initial values to file");
                    FileOutputStream InitValueFile = new FileOutputStream(new File(RecordFile_k + "_DE_InitValue.txt"));
                    PrintStream p2 = new PrintStream(InitValueFile);

                    for (int kk = 1; kk <= user_MaxID; kk++)
                    {
                        for (int ll = 0; ll < CommonRecomm_NEW.featureDimension; ll++)
                            p2.println(P[kk][ll]);
                        for (int ll = 0; ll < B_Count; ll++)
                            p2.println(B[kk][ll]);
                    }
                    for (int kk = 1; kk <= item_MaxID; kk++)
                    {
                        for (int ll = 0; ll < CommonRecomm_NEW.featureDimension; ll++)
                            p2.println(Q[kk][ll]);
                        for (int ll = 0; ll < C_Count; ll++)
                            p2.println(C[kk][ll]);
                    }
                    p2.close();
                }


                if (ReadInitValuefromFile)
                {
                    System.out.println("Read existing initial values from file");
                    BufferedReader br = new BufferedReader(new FileReader(new File(RecordFile_k + "_Standard_InitValue.txt")));;
                    for (int kk = 1; kk <= user_MaxID; kk++)
                    {
                        for (int ll = 0; ll < CommonRecomm_NEW.featureDimension; ll++)
                            P[kk][ll] = Double.parseDouble(br.readLine());
                        for (int ll = 0; ll < B_Count; ll++)
                            B[kk][ll] = Double.parseDouble(br.readLine());
                    }
                    for (int kk = 1; kk <= item_MaxID; kk++)
                    {
                        for (int ll = 0; ll < CommonRecomm_NEW.featureDimension; ll++)
                            Q[kk][ll] = Double.parseDouble(br.readLine());
                        for (int ll = 0; ll < C_Count; ll++)
                            C[kk][ll] = Double.parseDouble(br.readLine());
                    }
                    br.close();
                }


//						System.out.println("Read_Train_RMSE=" + SI_New_2.trainCurrentRMSE(LossFun[0]) +"Read_valid_RMSE=" + SI_New_2.validCurrentRMSE(LossFun[0])  +"Read_Test_RMSE=" + SI_New_2.testCurrentRMSE(LossFun[0]) + ", PLFA_MAE=" + SI_New_2.testCurrentMAE(LossFun[0]) + "Iter. round = " + round+ ". SGD_Time=" + SGDTime + "s");
                //added by cj, for validating yy's P,Q,B,C files ,but seems wrong
                CommonRecomm_NEW.cacheArrays();


                for (int DeNum : DE_Num_Set)
                {
					/*comment on 2021/07/30, because these parameters are fixed, there is no need for loop to search the best values
					   // PSO parameters set in main function
					   double[] PSO_w_Set
					//  Loop for multiple PSO parameters' set
					for (double c : PSO_c_Set)
					{
						for (double w : PSO_w_Set)
			       	   	{

							{end tuning the paras  */
                    for (double dw : DE_dw_Set)
                    {
                        for (double cr : DE_cr_Set)
                        {
                            //for (double cp: DE_cp_Set)  COMMNET ON 2021/10/3 FOR ALL INDIVIDUALS SHALL BE CROSSOVER
                            //{
                            for (double de_lda:de_lda_Set)
                            {
                                for ( OptimizeAlgorithmSet optAlg: opt_alg_set)
                                {
					/*for(double rao: MOME_rao_Set)
					{


										System.out.println("PSO_w="+ w + "; PSO_c=" + c);
										p.println("PSO_w=" + w);
										p.println("PSO_c=" + c);

										System.out.println("PSO_Lambda="+ lambda);
										p.println("PSO_Lambda="+ lambda);*/

//										System.out.println("DE_dw=" + dw + "; DE_cr=" + cr + "; DE_lambda=" + de_lda+ "; DE_num=" + DeNum);
//										p.println("DE_dw=" + dw + "; DE_cr=" + cr +  "; DE_lambda=" + de_lda+ "; DE_num=" + DeNum);

                                    // output the optimize type by CJ on 2021/07/31

                                    // System.out.println("Optimize type is " + optAlg+"; RAO is " + rao);
                                    // p.println("Optimize type is " + optAlg+"; RAO is " + rao);

                                    //   System.out.println("DE Swarm Num="+PopNum);
                                    //	p.println("DE Swarm Num="+PopNum);
                                    // end the output, by CJ on 2021/07/31


                                    Avg_RMSE = Avg_MAE = 0;
                                    Avg_Est_Time_RMSE = Avg_Est_Time_MAE = 0;
                                    Avg_each_round = 0;

                                    PSO_Params.total_round = 0; // added by CJ, for averaging validation round

                                    for (int t = 0; t < ValidationTime; t++)
                                    {
                                        System.out.println("t="+ t);
                                        p.println("t="+ t);

                                        //init_Scale = init_Max = 0.004;
											/*comment by CJ on 2021/07/30 for saving time
											System.out.println("Init_Scale="+ init_Scale);
											p.println("Init_Scale="+ init_Scale);

											System.out.println("eta=" + CommonRecomm_NEW.eta + "; Lamda=" + CommonRecomm_NEW.lambda);
											p.println("eta=" + CommonRecomm_NEW.eta + "; Lamda=" + CommonRecomm_NEW.lambda);
											System.out.println("feature dimension=" +  CommonRecomm_NEW.featureDimension);
											p.println("feature dimension=" + CommonRecomm_NEW.featureDimension);*/

                                        //------------------------------- Train for min RMSE & MAE-------------------------------------//
                                        long StartTime;
                                        long EndTime;

                                        StartTime = System.currentTimeMillis();

                                        switch (optAlg)
                                        {

                                            case BPSO:
                                            case PPSO:
                                                CommonRecomm_NEW.recallArrays();
                                                CommonRecomm_NEW.trainingRound = 100; // reset for PSO
                                                CommonRecomm_NEW.delayCount = PSODelayCount;
                                                CommonRecomm_NEW.yy_rho = 1.0;
                                                PSO_Params.PopNum = 5; // change to fixed value on 2021/07/30,  //PopNum;
                                                PSO_Params.w = 0.3; // change to fixed value on 2021/07/30,  //w;
                                                PSO_Params.c1 = 1.0; // change to fixed value on 2021/07/30,  //c;
                                                PSO_Params.c2 =1.0; // change to fixed value on 2021/07/30,  //c;
                                                //PSO_Params.Rao = rao; // add on 2021/08/02
                                                PSO_Params.Opt_Type = 1;
                                                PSO_Params.MaxGen = 200;
                                                PSO_Params.ErrLim = 1e-1;

                                                // add by CJ on 2021/08/01 for introducing the pso type into pso_optimizer
                                                PSO_Params.PSOApprType = optAlg;

                                                System.out.println("PSO training " + optAlg);
                                                // MPSO Main algorithm
                                                SI_New_2.train_BatchPSO(LossFun[0], TrainError, PSO_Params, p);
                                                break;

                                            case DE:        // DE operation added by CJ, 2021.04.19

                                                // should be updated according to the real needs
                                                CommonRecomm_NEW.recallArrays();
                                                CommonRecomm_NEW.trainingRound = 100;
                                                CommonRecomm_NEW.delayCount = DEDelayCount; // DEDelayCount
                                                DE_Params.DeNum = DeNum;
                                                DE_Params.DeDim = 21;

                                                // changing the parameters for better results
                                                DE_Params.de_dw = dw; // change to fixed value on 2021/07/30,  //dw;
                                                DE_Params.de_cr = cr; // change to fixed value on 2021/07/30,  //cr;
                                                //COMMNET ON 2021/10/3 FOR ALL INDIVIDUALS SHALL BE CROSSOVER
                                                //DE_Params.de_cp = cp; // change to fixed value on 2021/07/30,  //cp;
                                                DE_Params.de_lambda = de_lda;        //de_lda_set[k];
                                                DE_Params.MaxGen = 50; // following the PSO algorithm 200, change to smaller by CJ


                                                System.out.println("DE training ");
                                                // MPSO Main algorithm
                                                SI_New_2.train_DE(LossFun[0], TrainError, DE_Params, p);
                                                System.out.println("DE training ends with "+t+" times");

                                                break;


                                            case BAS:
                                                // should be updated according to the real needs
                                                CommonRecomm_NEW.recallArrays();
                                                CommonRecomm_NEW.trainingRound = 100;
                                                CommonRecomm_NEW.delayCount = BASDelayCount; // BASDelayCount

                                                BAS_Params.BASPopNum = 5;	// need to pre-setting
                                                BAS_Params.BASDim = 21;		// the common settings
                                                // changing the parameters for better results
                                                BAS_Params.regular_lambda = 0.0001;
                                                // douban 11/18 0.1, 0.7076; 0.3,0.7076;0.03,
                                                //FLIXTER 11/19 0.03,0.8860;0.3, 1.032;0.1,
                                                // ep 0.01, 0.5454; 0.1,fly,0.03, 0.5504
                                                BAS_Params.MaxGen = 70; // following the PSO algorithm 200, change to smaller by CJ

                                                System.out.println("BAS training ");
                                                // MPSO Main algorithm
                                                SI_New_2.train_BAS(LossFun[0], TrainError, BAS_Params, p);
                                                System.out.println("BAS training ends with "+t+" times");



                                            default:
                                                break;
                                        }  	// end switch (OptSet[k])

                                        EndTime = System.currentTimeMillis();
                                        System.out.println("Search time (s):" + (EndTime - StartTime) / 1000.);
                                        p.println("Search time (s):" + (EndTime - StartTime) / 1000.);
                                        System.out.println("");
                                        p.println("");

                                        Avg_RMSE += min_Error_RMSE;
                                        Avg_Est_Time_RMSE += (EndTime - StartTime) / 1000.;
                                        Overall_Results += ", " + min_Error_RMSE + ", " + (EndTime - StartTime) / 1000. + ", " + TrainRound;

                                        Avg_MAE += min_Error_MAE;
                                        Avg_Est_Time_MAE += (EndTime - StartTime) / 1000.;
                                        Overall_Results += ", " + min_Error_MAE + ", " + (EndTime - StartTime) / 1000. + ", " + TrainRound;

                                        Min_RMSE[t] = min_Error_RMSE;
                                        Min_MAE[t] = min_Error_MAE;
                                        Est_Time_MAE[t] = Est_Time_RMSE[t] = (EndTime - StartTime) / 1000.;
                                        Min_RMSE_Position[t][0] = Min_RMSE_Position[t][1] = 1;
                                        Min_MAE_Position[t][0] = Min_MAE_Position[t][1] = 1;

                                        Each_round[t] = TrainRound;
                                        Avg_each_round += TrainRound;


                                    } // end loop t

                                    Avg_RMSE /= ValidationTime;
                                    Avg_MAE /= ValidationTime;
                                    Avg_Est_Time_RMSE /= ValidationTime;
                                    Avg_Est_Time_MAE /= ValidationTime;
                                    Avg_each_round /= ValidationTime;
                                    Overall_Results += "," + Avg_RMSE + "," + Avg_Est_Time_RMSE + ", ," + Avg_MAE + "," + Avg_Est_Time_MAE;
                                    float avg_round = (float) PSO_Params.total_round/ValidationTime;
                                    p_overall.println(Overall_Results);

                                    System.out.println("Result Summary:   ");
                                    System.out.println("yy_w="+ yy_w +" yy_r1=" + yy_r1 + " yy_r2=" + yy_r2);

                                    p.println("Result Summary:");

                                    System.out.println("DE_lambda=" + de_lda + "; DE_cr=" + cr ); //+ "; DE_cp=" + cp);
                                    p.println("DE_lambda=" + de_lda + "; DE_cr=" + cr ); //+ "; DE_cp=" + cp);

                                    System.out.println("t      Min RMSE     (x,y)minRMSE     RMSE Est Time(s)        Min MAE      (x,y)minMAE       MAE ESt Time(s)");
                                    p.println("t      Min RMSE     (x,y)minRMSE     RMSE Est Time(s)        Min MAE      (x,y)minMAE       MAE ESt Time(s)");
                                    for (int t = 0; t < ValidationTime; t++)
                                    {
                                        System.out.println("" + t + "\t  " + Min_RMSE[t] + " \t (" + Min_RMSE_Position[t][0] + "," + Min_RMSE_Position[t][1] + ")\t" + Est_Time_RMSE[t] +
                                                "\t " + Min_MAE[t] + "  \t    (" + Min_MAE_Position[t][0] + "," + Min_MAE_Position[t][1] + ")\t" + Est_Time_MAE[t] + "\t  " + Each_round[t] );
                                        p.println("" + t + "\t  " + Min_RMSE[t] + " \t (" + Min_RMSE_Position[t][0] + "," + Min_RMSE_Position[t][1] + ")\t" + Est_Time_RMSE[t] +
                                                "\t " + Min_MAE[t] + "  \t    (" + Min_MAE_Position[t][0] + "," + Min_MAE_Position[t][1] + ")\t" + Est_Time_MAE[t]+ "\t  " + Each_round[t] );

                                    }
                                    System.out.println("Avg RMSE = " + Avg_RMSE + "; Avg Est Time RMSE = " + Avg_Est_Time_RMSE + "s; Avg MAE = " + Avg_MAE + "; Avg Est Time MAE = " + Avg_Est_Time_MAE + "s;Average round="+ Avg_each_round);
                                    p.println("Avg RMSE = " + Avg_RMSE + "; Avg Est Time RMSE = " + Avg_Est_Time_RMSE + "s; Avg MAE = " + Avg_MAE + "; Avg Est Time MAE = " + Avg_Est_Time_MAE + "s;Average round="+ Avg_each_round);

                                    p.println("");
                                    p.println("");

                                    //}// rao set
                                } //  end for different algorithm type loop
                            }// end loop de_lda


			/*comment on 2021/07/30, because these parameters are fixed, there is no need for loop to search the best values
									   // PSO parameters set in main function
									   double[] PSO_w_Set*/
                            //} //end loop cp   // COMMNET ON 2021/10/3 FOR ALL INDIVIDUALS SHALL BE CROSSOVER
                        } // end loop cr
                    }	// end loop dw
					/*  	} // end loop w
					} // end loop c*/
                } // end loop PopNum

                SI_New_2 = null;
                System.gc();
            } // end loop f


            p.close();

            //CommonRecomm_NEW.clearRatings();
        }  // end loop k
        p_overall.close();

    }

    @Override
    public void train() throws IOException {
        // TODO Auto-generated method stub

    }

}
