package Ensemble_SI;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;
import java.util.StringTokenizer;

import Ensemble_SI.RTuple;

public abstract class CommonRecomm_NEW {
		

	public double minTotalTime = 0;
	
	public double minTotalTime_cj = 0;

	public double cacheTotalTime = 0;

	public static double min_Error = 1e10;

	public double previous_Error = 1e10;

	public  int min_Round = 0;

	public static int delayCount = 20;

	public static double iterConvThrd = 0.001;		// 濞寸媴绲奸悳顖炲礄閼恒儲娈堕弶鈺婂幒閸烆剟寮ㄩ懜鍨異闂傚�?�鍔戝锟�


	public static double min_Error_RMSE = 1e10;
	public static double min_Error_MAE = 1e10;
	
	public static int TrainRound;
	
	// from yuanye
	public static double[] tempRMSE;
	public static double[] FitnessRMSEpbest;
	public static double FitnessRMSE;
	public static double FitnessRMSEgbest;
	public static double[] particles;
	public static double[] pbest;
	public static double gbest;
	public static double[] V;
	public static int yy_swarmNum = 10;   // add for PLFA
	public static int yy_bestSwarm = 0;
	public static int yy_dimension = 1;
	public static double yy_rho;

	
	public static double Vmax =1;
	public static double Vmin =-1;
	public static int etaMax = 12, etaMin = 8;    // 12, 8 are setted by YY, changed by CJ
	
	public static double yy_r1 ;             // add for PLFA
	public static double yy_r2 ;            // add for PLFA
	public static double yy_c1 = 2;         // add for PLFA
	public static double yy_c2 = 2;         // add for PLFA

	public static ArrayList<RTuple> validationData = null;
	public static ArrayList[] user_Rating_Arrays = null;
	public static ArrayList[] item_Rating_Arrays = null;	
	
	public static double[] user_FW_Array, item_FW_Array;

	public static double yy_w=0.729;      //0.729;	    // add for PLFA
	
	public static boolean ReadInitValuefromFile = false;    //閺勵垰鎯佹禒搴㈡瀮娴犳儼顕伴崣鏈犻敍瀛敍�?�婇敍�?�嬮惃鍕灥婵锟界�?�绱�?
	public static boolean RecordInitValuetoFile = true;    //閺勵垰鎯佸锟介弬鍥︽閸愭瑥鍙哖閿涘閿涘瓓閿涘瓔閻ㄥ嫬鍨垫慨瀣拷纭风礉	
	public static boolean RecordInitValuetoFile_5 = true;    //閺勵垰鎯佸锟介弬鍥︽閸愭瑥鍙哖閿涘閿涘瓓閿涘瓔閻ㄥ嫬鍨垫慨瀣拷纭风礉	
	public static boolean RecordInitValuetoFile_1 = true;    //閺勵垰鎯佸锟介弬鍥︽閸愭瑥鍙哖閿涘閿涘瓓閿涘瓔閻ㄥ嫬鍨垫慨瀣拷纭风礉	
	public static String RecordFile_k=null;
	
	// 闂佽崵鍋炵粙鎴︽儔閸忚偐鐝堕柟鐑樺焾閸ゆ鏌ㄩ悤鍌涘�?
	public static int B_Count = 1;

	//闂備胶鍎甸崑鎾诲礉鐎ｎ偆顩查柟鐑橆殔缁�鍡樼�?闂堟稒锛嶆繛鍏碱殜閺屾盯寮介妸锕�顩梻浣圭湽閸斿瞼锟芥凹鍓熼幃銏ゆ晸娴犲鐓涘ù锝囨�?椤ュ绱掗鑲┬ч柡浣哥Ф娴狅箓鎳栭埡鍐惧晬闂佽崵濮村ú锝囩博缁�?ount濠电偞鍨堕幐鎼佹晝閵堝懐鏆︾�癸拷閸曨偆鐫勯柣鐘辩濠�杈╃矆婢跺ň妲堥柟鎯х－瀛濋梺璇″枟瑜扮釜unt闂佽娴烽弫鎼佸箠閹炬儼濮抽柡灞诲劜閸庡秹鏌涢弴銊ヤ簼闁稿﹤宕埥澶愬箼閸曨剙顏跺┑鐐村灦閹尖晜绂嶅鍫熷仾闁告洦鍓氭刊鎾煙閹规劕鐓愰柣锝変憾閺屾稑顭ㄩ崘顏嗕紘濠电偞娼欓ˇ杈╁垝閻㈠憡鏅搁柨鐕傛嫹
	public static double[] B_Base;

	public static double[][] B;

	public static double[][] min_B, B_cache, B_delta, B_tmp;

	public static double[][] P;

	public static double[][] P_tmp;

	public static double[][] min_P, P_cache, P_delta;

	public static int[][] Omiga_P_Q;
	
	public static int[][] Omiga_P_T;
	
	public static int[][] Omiga_P_R;
	
	// 闂備礁鎲＄敮妤呫�冮崼鐔虹彾闁圭儤鍩堥崵妤呮煥閻曞�?��?��
	public static int C_Count = 1;

	//闂備胶鍎甸崑鎾诲礉鐎ｎ偆顩查柟鐑橆殔缁�鍡樼�?闂堟稒锛嶆繛鍏碱殜閺屾盯寮介妸锔剧窗闂備焦鐪归崝�?�锟芥凹鍓熼幃銏ゆ晸娴犲鐓涘ù锝囨嚀椤ュ绱掗鑲┬ч柡浣哥Ф娴狅箓鎳栭埡鍐惧晬闂佽崵濮村ú锝囩博缁挮ount濠电偞鍨堕幐鎼佹晝閵堝懐鏆︾�癸拷閸曨偆鐫勯柣鐘辩濠�杈╃矆婢跺ň妲堥柟鎯х－瀛濋梺璇″枟瑜扮釜unt闂佽娴烽弫鎼佸箠閹炬儼濮抽柡灞诲劜閸庡秹鏌涢弴銊ヤ簼闁稿﹤宕埥澶愬箼閸曨剙顏跺┑鐐村灦閹尖晜绂嶅鍫熷仾闁告洦鍓氭刊鎾煙閹规劕鐓愰柣锝変憾閺屾稑顭ㄩ崘顏嗕紘濠电偞娼欓ˇ杈╁垝閻㈠憡鏅搁柨鐕傛嫹
	public static double[]  C_Base;
	
	public static double[][] C;
	
	public static boolean flag_B = true;

	public static boolean flag_C = true;

	public static double[][] min_C, C_cache, C_delta, C_tmp;

	public static double[][] Q;

	public static double[][] min_Q, Q_cache, Q_delta, Q_tmp;
	
	public static int[][] Omiga_Q_P;
	
	public static int[][] Omiga_Q_T;	
	
	public static int[][] Omiga_Q_R;
	
	
	// 濠电偞鍨堕幖顐﹀箯閻戣姤鈷掗柛鎰靛幖閻撴劖銇勯顐＄凹闁诡垱妫冮弫鎾绘晸閿燂�?
	public static double[][] B_gradient, C_gradient, P_gradient, Q_gradient;

	// 闂備胶鍎甸崑鎾诲礉�?�ュ拋鐒介柛顭戝亝娴溿�?�绻涢幋鐐电煠闁猴拷娴犲鍋ｅ�?�姘功绾惧潡鏌ｉ敐鍜佺吋鐎规洘顨嗗蹇涱敃閿濆洦楗繝娈垮枛椤剟宕归幍顕嗘嫹鐟欏嫬鈻曢柟顖氬暣�?�曠喖顢�?悩宸П闂佽瀛╃粙鎺椻�﹂崶顒佸亗闁冲搫鎳忛埛鎾绘煥閻曞倹�?��
	public static double[][] B_r, B_p, B_r_prime, C_r, C_p, C_r_prime, P_r,
			P_p, P_r_prime, Q_r, Q_r_prime, Q_p;

	// 闂備胶鍎甸崑鎾诲礉�?�ュ拋鐒介柣銏犵仛婵ジ鏌℃径搴㈢�?�缂佸绔積ssian闂備焦妞块崢鎼佸疾閻樺弬鐔兼晸閽樺妲堥柟鐐�?▕椤庢鏌熼鐣�?煉闁哄苯鐭佺粻娑欐償閿濆骸鏅紓鍌欑椤︿粙宕归崘娴嬫灁闁瑰濮风壕浠嬫煙閹咃紞闁圭晫濞�閺岋綁顢樺☉娆愮彋濠碘槅鍨介幏锟�?
	public static double[][] B_hp, C_hp, P_hp, Q_hp;

	// 闂佽崵濮抽悞锕�顭垮Ο鑲╃鐎广儱顦伴崑銊╂煏婵犲繒鐣遍柣鎾存礋閺屾稑顭ㄩ崘顏嗗姱闂侀潧妫楃粔褰掑箚閸曨垱鍋╅柛婵堫仱ing闂備浇妗ㄥ鎺楀�?閹惰棄闂繛宸簼閸庡秹鏌涢弴銊ュ婵炲牆澧庣槐鎾寸瑹婵犲啫顏�
	public static double[] user_Rating_count, item_Rating_count, time_Rating_count;

	// 闂備胶绮�?�鍫ュ箠閹捐埖顐芥繛鎴炴皑绾惧ジ鎮楀☉娅虫垵鈻撻敓锟�?
	public static int featureDimension = 20;

	// 闂佽崵濮崇欢锟犲储閸撗冨灊閻忕偟鍋撴慨婊勩亜閺冨洤袚婵炲牞鎷�?
	public static int trainingRound = 2000;
	
	public static int round = 0;

	// 闂備胶绮�?�鍫ュ箠閹捐埖顐芥繛鎴欏灩缁�鍡樼節闂堟稒锛嶆繛鍏碱殜閺屾稒绻濊箛鏂款伓
	public static double AvgRating = 1;
	
	public static double init_Max = 0.004;

	public static double init_Scale = 0.004;

	public static int mapping_Scale = 1000;

	// 闂備胶顢婇惌鍥礃閵娧冨箑闂備礁鎲￠悷銉╁磹瑜版帒姹查柨鐕傛�?
	public static double eta = 0.001;

	public static double lambda = 0.005;

	public static double gama = 0.01;

	public static double tau = 0.001;

	public static double epsilon = 0.001;

	public static ArrayList<RTuple> trainData = null;
	
	public static ArrayList<RTuple> validData = null;
	
	public static ArrayList<RTuple> virtualTrainData = null;   //闂佹儳绻戠喊宥囷拷姘懅閹峰顢�?悩鐢电厐闂佽桨鑳舵晶妤�鐣垫笟锟藉鍧楁晸閿燂拷
	
	public static ArrayList<RTuple> virtualTestData = null;    //闂佹儳绻戠喊宥囷拷姘懃闇夐悗锝庡幘濡叉悂鏌℃担鍝勵暭鐎规挷绶氬鍧楁晸閿燂�?

	public static ArrayList<RTuple> testData = null;
	
	public static int item_MaxID = 0, user_MaxID = 0;

	public static int Index1_Max = 0, Index2_Max = 0, Index3_Max = 0, Index4_Max = 0;
	
	public abstract void train() throws IOException;
	
	

	//public PSO_Param PSO_Params;
	
	public CommonRecomm_NEW() throws NumberFormatException, IOException {
		initInstanceFeatures();
	}

	public enum LossFunSet
	{
		TwoDimMF,
	}
	
	public enum OptimizeAlgorithmSet          
	{
		//SGD,		// Stochastic Gradient Descent
		BPSO,		// Batch Particle Swarm Optimization
		PPSO,
		DE,		// Differential Evolution
		BAS
		//GD_PSO
	}
	
	public enum TrainErrorSet
	{
		RMSE,
		MAE,
		RMSE_and_MAE
	}
	public enum TrainApproach
	{
		TrainTest,
		TrainValidTest				
	}
	
	/*public enum PSOApproach    //comment by CJ on 2021/07/31 for removing psoApproach, combine them to OptimizeAlgorithmSet
	{
		PSO,
		PPSO,
		GD_PSO
	}*/
	
	public static void initBiasSettings(boolean ifB, boolean ifC, int B_C,
			int C_C) {
		flag_B = ifB;
		flag_C = ifC;
		B_Count = B_C;
		C_Count = C_C;
		
		B_tmp = new double[user_MaxID + 1][B_Count];
		B_cache = new double[user_MaxID + 1][B_Count];
		B_delta = new double[user_MaxID + 1][B_Count];
		B_r = new double[user_MaxID + 1][B_Count];
		B_r_prime = new double[user_MaxID + 1][B_Count];
		B_p = new double[user_MaxID + 1][B_Count];
		B_hp = new double[user_MaxID + 1][B_Count];
		B_gradient = new double[user_MaxID + 1][B_Count];

		C_tmp = new double[item_MaxID + 1][C_Count];
		C_cache = new double[item_MaxID + 1][C_Count];
		C_delta = new double[item_MaxID + 1][C_Count];
		C_r = new double[item_MaxID + 1][C_Count];
		C_r_prime = new double[item_MaxID + 1][C_Count];
		C_p = new double[item_MaxID + 1][C_Count];
		C_hp = new double[item_MaxID + 1][C_Count];
		C_gradient = new double[item_MaxID + 1][C_Count];
		min_B = new double[user_MaxID + 1][B_Count];
		min_C = new double[item_MaxID + 1][C_Count];
		
		
		System.gc();

		if(B_Count!=0){
			for (int i = 1; i <= user_MaxID; i++) {
				double tempUB = B_Base[i]/B_Count;
				for (int j = 0; j < B_Count; j++) {
					B_cache[i][j] = tempUB;
					min_B[i][j] = B_cache[i][j];
				}
			}
		}

		if(C_Count!=0){
			for (int i = 1; i <= item_MaxID; i++) {
				double tempIB = C_Base[i]/C_Count;
				for (int j = 0; j < C_Count; j++) {
					C_cache[i][j] = tempIB;
					min_C[i][j] = C_cache[i][j];
				}
			}
		}
		

	}

	public static void initiStaticArrays() {
		// 闂備礁鎲″Λ鎴�?箯閿燂拷1闂備礁鎼�氱兘宕归婊勫闁绘柨鎽滈�?�鏌ユ煕閳╁喚娈曢棅顒夊墴楠炴牠寮堕幋鐘殿唶閻熸粎澧楃�笛呯矙婢跺绡�濞达�?娅ｉ幉绺凞濠电儑绲藉ú锔炬崲閸愵亷鎷烽崹顐ｂ拹缂佸顦甸弫鎾寸鐎ｎ偄娈滈梺璺ㄥ櫐閹凤�?
		user_Rating_count = new double[user_MaxID + 1];
		item_Rating_count = new double[item_MaxID + 1];
		
		P_tmp = new double[user_MaxID + 1][featureDimension];
		P_cache = new double[user_MaxID + 1][featureDimension];
		P_delta = new double[user_MaxID + 1][featureDimension];
		P_r = new double[user_MaxID + 1][featureDimension];
		P_r_prime = new double[user_MaxID + 1][featureDimension];
		P_p = new double[user_MaxID + 1][featureDimension];
		P_hp = new double[user_MaxID + 1][featureDimension];
		P_gradient = new double[user_MaxID + 1][featureDimension];

		Q_tmp = new double[item_MaxID + 1][featureDimension];
		Q_cache = new double[item_MaxID + 1][featureDimension];
		Q_delta = new double[item_MaxID + 1][featureDimension];
		Q_r = new double[item_MaxID + 1][featureDimension];
		Q_r_prime = new double[item_MaxID + 1][featureDimension];
		Q_p = new double[item_MaxID + 1][featureDimension];
		Q_hp = new double[item_MaxID + 1][featureDimension];
		Q_gradient = new double[item_MaxID + 1][featureDimension];
		min_P = new double[user_MaxID + 1][featureDimension];
		min_Q = new double[item_MaxID + 1][featureDimension];
		
		
		
		B_Base = new double[user_MaxID + 1];
		C_Base = new double[item_MaxID + 1];


		// 闂備礁鎲＄敮妤冩崲閸�?儑缍栭柟鐗堟緲缁�宀勬煛瀹ュ骸浜滄い銉︻殘閿熺晫鏁婚。锕傛嚄閸洘鍋傞柍鍝勬噺閳锋捇鏌ㄩ悤鍌涘,闂傚倷鐒﹁ぐ鍐洪妸鈺佹瀬闁靛牆顦伴埛鏇㈢叓閸ャ劍鈷愮紒銊﹀哺閺屾稒绻濊箛鏂款伓,濠电偛顕慨鏉戭潩閿曞�?�鏅搁柦妯侯槼鐎氫即鏌ら梹鎰�?祮妤犵偛顑夐獮鍥敂閸♀晙绱樺┑鐐村灦閹稿墽锟芥氨鍏樺濠氬礃閵娧冪厽闂佽法鍠庨敓鐣屽枎椤ュ秶绱掓导娆愬�?
		Random random = new Random(System.currentTimeMillis());
		for (int i = 0; i <= user_MaxID; i++) {
			user_Rating_count[i] = 0;
			int tempBB = random.nextInt(mapping_Scale);
			B_Base[i] = init_Max - init_Scale * tempBB / mapping_Scale;
			for (int j = 0; j < featureDimension; j++) {
				int temp = random.nextInt(mapping_Scale);
				P_cache[i][j] = init_Max - init_Scale * temp / mapping_Scale;
				min_P[i][j] = P_cache[i][j];
			}
		}
		for (int i = 0; i <= item_MaxID; i++) {
			item_Rating_count[i] = 0;
			int tempCB = random.nextInt(mapping_Scale);
			C_Base[i] = init_Max - init_Scale * tempCB / mapping_Scale;
			for (int j = 0; j < featureDimension; j++) {
				int temp = random.nextInt(mapping_Scale);
				Q_cache[i][j] = init_Max - init_Scale * temp / mapping_Scale;
				min_Q[i][j] = Q_cache[i][j];

			}
		}


		for (RTuple tempRating : trainData) {
			user_Rating_count[tempRating.iUserID] += 1;
			item_Rating_count[tempRating.iItemID] += 1;
			time_Rating_count[tempRating.iTimeSlotID] += 1;

		}
	}

	public static void cacheArrays()
	{
		for (int i = 0; i <= user_MaxID; i++) {
			for (int j = 0; j < featureDimension; j++)
				P_tmp[i][j] = P[i][j];   
			for (int j = 0; j < B_Count; j++)
				B_tmp[i][j] = B[i][j];
		}
		for (int i = 0; i <= item_MaxID; i++) {
			for (int j = 0; j < featureDimension; j++)
				Q_tmp[i][j] = Q[i][j];
			for (int j = 0; j < C_Count; j++)
				C_tmp[i][j] = C[i][j];
		}
	}
	
	public static void recallArrays()
	{
		for (int i = 0; i <= user_MaxID; i++) {
			for (int j = 0; j < featureDimension; j++)
				P[i][j] = P_tmp[i][j];
			for (int j = 0; j < B_Count; j++)
				B[i][j] = B_tmp[i][j];
		}
		for (int i = 0; i <= item_MaxID; i++) {
			for (int j = 0; j < featureDimension; j++)
				Q[i][j] = Q_tmp[i][j];
			for (int j = 0; j < C_Count; j++)
				C[i][j] = C_tmp[i][j];
		}
	}
	public static void resetOneDimensionArray(double[] destArray) {
		for (int i = 0; i < destArray.length; i++) {
			destArray[i] = 0;
		}
	}

	public static void copyOneDimensionArray(double[] originArray,
			double[] destArray) {
		for (int i = 0; i < destArray.length; i++) {
			destArray[i] = originArray[i];
		}
	}

	public static void resetTwoDimenionArray(double[][] destArray) {
		for (int i = 0; i < destArray.length; i++) {
			for (int j = 0; j < destArray[i].length; j++) {
				destArray[i][j] = 0;
			}
		}
	}

	public void computeRatingHat() {
		for (RTuple tempRating : trainData) {
			double ratingHat = getLocPrediction(tempRating.iUserID,
					tempRating.iItemID);
			tempRating.dRatingHat = ratingHat;
		}
	}




	public void initInstanceFeatures() {
		B = new double[user_MaxID + 1][B_Count];
		C = new double[item_MaxID + 1][C_Count];
		P = new double[user_MaxID + 1][featureDimension];
		Q = new double[item_MaxID + 1][featureDimension];
		
		for (int i = 0; i <= user_MaxID; i++) {
			for (int j = 0; j < B_Count; j++) {
				B[i][j] = B_cache[i][j];
			}
			for (int j = 0; j < featureDimension; j++) {
				P[i][j] = P_cache[i][j];
			}
		}
		for (int i = 0; i <= item_MaxID; i++) {
			for (int j = 0; j < C_Count; j++) {
				C[i][j] = C_cache[i][j];
			}
			for (int j = 0; j < featureDimension; j++) {
				Q[i][j] = Q_cache[i][j];
			}
		}

	}

	public void cacheMinFeatures() {
		for (int i = 1; i <= user_MaxID; i++) {
			for (int j = 0; j < B_Count; j++) {
				min_B[i][j] = B[i][j];
			}
			for (int j = 0; j < featureDimension; j++) {
				min_P[i][j] = P[i][j];
			}
		}
		for (int i = 1; i <= item_MaxID; i++) {
			for (int j = 0; j < C_Count; j++) {
				min_C[i][j] = C[i][j];
			}
			for (int j = 0; j < featureDimension; j++) {
				min_Q[i][j] = Q[i][j];
			}
		}

	}
	
	
	//from yuanye
	public void yy_cacheMinFeatures() {
		for (int i = 1; i <= user_MaxID; i++) {
			for (int j = 0; j < B_Count; j++) {
				min_B[i][j] = B[i][j];
			}
			for (int j = 0; j < featureDimension; j++) {
				min_P[i][j] = P[i][j];
			}
		}
		for (int i = 1; i <= item_MaxID; i++) {
			for (int j = 0; j < C_Count; j++) {
				min_C[i][j] = C[i][j];
			}
			for (int j = 0; j < featureDimension; j++) {
				min_Q[i][j] = Q[i][j];
			}
		}
	}
	

	public void rollBackMinFeatures() {
		for (int i = 1; i <= user_MaxID; i++) {
			for (int j = 0; j < B_Count; j++) {
				B[i][j] = min_B[i][j];
			}
			for (int j = 0; j < featureDimension; j++) {
				P[i][j] = min_P[i][j];
			}
		}
		for (int i = 1; i <= item_MaxID; i++) {
			for (int j = 0; j < C_Count; j++) {
				C[i][j] = min_C[i][j];
			}
			for (int j = 0; j < featureDimension; j++) {
				Q[i][j] = min_Q[i][j];
			}
		}

	}
	public void yy_rollBackMinFeatures() {
		for (int i = 1; i <= user_MaxID; i++) {
			for (int j = 0; j < B_Count; j++) {
				B[i][j] = min_B[i][j];
			}
			for (int j = 0; j < featureDimension; j++) {
				P[i][j] = min_P[i][j];
			}
		}
		for (int i = 1; i <= item_MaxID; i++) {
			for (int j = 0; j < C_Count; j++) {
				C[i][j] = min_C[i][j];
			}
			for (int j = 0; j < featureDimension; j++) {
				Q[i][j] = min_Q[i][j];
			}
		}
	}
	
	// Read training data and test data
	public static void initializeRatings(String trainFileName,
			String testFileName, String separator)
			throws NumberFormatException, IOException {
		// 闂備礁鎲″缁樻叏閹绢喖鐭楅柛鈩冪懕閹风兘鎮介悽鍨問atingMap闂備焦鐪归崝�?�锟芥凹鍓熷畷褰掑垂椤愶絽寮块柣搴秵娴滄粓顢氶敓锟�?
		initTestData(testFileName, separator);
		initTrainData(trainFileName, separator);
	}
	
	// Read training data, validation data and test data
	public static void initializeRatings(String trainFileName, String validFilename,
			String testFileName, String separator)
			throws NumberFormatException, IOException {
		// 闂備礁鎲″缁樻叏閹绢喖鐭楅柛鈩冪懕閹风兘鎮介悽鍨問atingMap闂備焦鐪归崝�?�锟芥凹鍓熷畷褰掑垂椤愶絽寮块柣搴秵娴滄粓顢氶敓锟�?
		initTestData(testFileName, separator);
		initValidData(validFilename, separator);
		initTrainData(trainFileName, separator);
	}
	
	public static void initTrainData(String fileName, String separator)
			throws NumberFormatException, IOException {
		trainData = new ArrayList<RTuple>();
		File personSource = new File(fileName);
		BufferedReader in = new BufferedReader(new FileReader(personSource));

		int dataCount = 0;
		String tempVoting;
		AvgRating = 0;		
		while (((tempVoting = in.readLine()) != null)) {
			StringTokenizer st = new StringTokenizer(tempVoting, separator);
			String personID = null;
			if (st.hasMoreTokens())
				personID = st.nextToken();
			String movieID = null;
			if (st.hasMoreTokens())
				movieID = st.nextToken();
			String personRating = null;
			if (st.hasMoreTokens())
				personRating = st.nextToken();
			int iUserID = Integer.valueOf(personID);
			int iItemID = Integer.valueOf(movieID);

			// 闂佽崵濮抽悞锕�顭垮Ο鑲╃鐎广儱妫涢埢鏃堟�?�閿濆骸浜濆ù鐘趁�?�鍧�?醇閸℃鍣洪柣锝囨箞temid闂備礁鎲＄划�?勬儊閸х櫝rid闂備焦�?�х粙鎺旂矙閹存緷鍝勵煥閸曨厸鏋栫紓渚婄磿閼垫亰mid闂備礁鎲＄划�?勬儊閸х櫝rid闂備礁鎼�氱兘宕归悽鍨床婵炴垶姘ㄧ壕鍏笺亜閹捐泛校闁伙綁浜堕弻銊モ槈濡粯鎷辨繝鈷�鍐�?�ｇ紒鍌涘浮婵＄兘鏁傞幋鎺斿笡濠电姰鍨归悥銏ゅ川椤旂偓娈竔temid闂備礁鎲＄划�?勬儊閸х櫝rid濠电偞鍨跺濠氬窗�?�ュ鑸规い鎺戝�归崑姗�鏌曟繛鍨拷妤呭疮婵犲洦鐓曟慨姗嗗墮椤忣剟鏌涢妶鍡欑煉闁诡垰鍟村畷鐔碱敆娴ｈ鍟�闂備胶鍎甸弲顏堝箯閿燂拷
			user_MaxID = (user_MaxID > iUserID) ? user_MaxID : iUserID;
			item_MaxID = (item_MaxID > iItemID) ? item_MaxID : iItemID;
			double dRating = Double.valueOf(personRating);

			RTuple temp = new RTuple();
			temp.iUserID = iUserID;
			temp.iItemID = iItemID;
			temp.iIndex = dataCount;
			temp.dRating = dRating;
			AvgRating += dRating;
			trainData.add(temp);
			dataCount++;
		}
		in.close();
		AvgRating /= dataCount;
		
		// Record Index of Omiga_P, Omiga_Q, Omiga_T
		ArrayList<ArrayList<Integer>> ListOmiga_P_Q = new ArrayList<ArrayList<Integer>>();
		ArrayList<ArrayList<Integer>> ListOmiga_Q_P = new ArrayList<ArrayList<Integer>>();
		
		ArrayList<ArrayList<Integer>> ListOmiga_P_R = new ArrayList<ArrayList<Integer>>();
		ArrayList<ArrayList<Integer>> ListOmiga_Q_R = new ArrayList<ArrayList<Integer>>();
		
		
		Omiga_P_Q = new int[user_MaxID + 1][];
		Omiga_P_R = new int[user_MaxID + 1][];
		Omiga_Q_P = new int[item_MaxID + 1][];
		Omiga_Q_R = new int[item_MaxID + 1][];
		
		for (int i = 0; i <= user_MaxID; i++)
		{
			ListOmiga_P_Q.add(new ArrayList<Integer>());
			ListOmiga_P_R.add(new ArrayList<Integer>());
		}
		
		for (RTuple temp : trainData)
		{
			ListOmiga_P_Q.get(temp.iUserID).add(temp.iItemID);
			ListOmiga_P_R.get(temp.iUserID).add(temp.iIndex);
		}
		
		for (int i = 0; i <= user_MaxID; i++)
		{
			Omiga_P_Q[i] = new int[ListOmiga_P_Q.get(i).size()];
			Omiga_P_R[i] = new int[ListOmiga_P_R.get(i).size()];
			for (int j = 0; j < Omiga_P_Q[i].length; j++)
			{
				Omiga_P_Q[i][j] = ListOmiga_P_Q.get(i).get(j);
				Omiga_P_R[i][j] = ListOmiga_P_R.get(i).get(j);
			}
		}
		
		for (int i = 0; i <= item_MaxID; i++)
		{
			ListOmiga_Q_P.add(new ArrayList<Integer>());
			ListOmiga_Q_R.add(new ArrayList<Integer>());
		}
		
		for (RTuple temp : trainData)
		{
			ListOmiga_Q_P.get(temp.iItemID).add(temp.iUserID);
			ListOmiga_Q_R.get(temp.iItemID).add(temp.iIndex);
		}
		
		for (int i = 0; i <= item_MaxID; i++)
		{
			Omiga_Q_P[i] = new int[ListOmiga_Q_P.get(i).size()];
			Omiga_Q_R[i] = new int[ListOmiga_Q_R.get(i).size()];
			for (int j = 0; j < Omiga_Q_P[i].length; j++)
			{
				Omiga_Q_P[i][j] = ListOmiga_Q_P.get(i).get(j);
				Omiga_Q_R[i][j] = ListOmiga_Q_R.get(i).get(j);
			}
		}
			
	}
	
	
	
	// from yuanye
	public static void yy_initTrainData(String fileName, String separator)
			throws NumberFormatException, IOException {
		int Total_Instance_Count = 0;
		trainData = new ArrayList<RTuple>();
		File personSource = new File(fileName);
		BufferedReader in = new BufferedReader(new FileReader(personSource));

		String tempVoting;
		while (((tempVoting = in.readLine()) != null)) {
			StringTokenizer st = new StringTokenizer(tempVoting, separator);
			String personID = null;
			if (st.hasMoreTokens())
				personID = st.nextToken();
			String movieID = null;
			if (st.hasMoreTokens())
				movieID = st.nextToken();
			String personRating = null;
			if (st.hasMoreTokens())
				personRating = st.nextToken();
			int iUserID = Integer.valueOf(personID);
			int iItemID = Integer.valueOf(movieID);

			// 闁荤姳鐒�?妯肩礊瀹ュ棛鈻旈悗锝庡亝娴犳ê顭块崼鍡楁噺閻ｇ湈temid闂佸憡绮岄惁鍧癳rid闂佹寧绋掔粙鎴澝哄鍕枖缂侊紕鑵恊mid闂佸憡绮岄惁鍧癳rid闂佸搫�?�烽崹鐢垫崲濞戞氨纾兼い鎾跺У閻ｉ亶鏌ㄥ☉妯绘拱濠�?冪У缁傛帡濡烽敂鎴掔帛婵犮垹鐖㈤崨顔炬殸itemid闂佸憡绮岄惁鍧癳rid婵炴垶姊婚崰宥夊船椤掑�?�鍋�?柕濞垮�楅崯濠囨煕濮橆剙顏柛銈嗙矒閹啴宕熼浣诡�?闂佺儵鏅幏锟�
			user_MaxID = (user_MaxID > iUserID) ? user_MaxID : iUserID;
			item_MaxID = (item_MaxID > iItemID) ? item_MaxID : iItemID;
			double dRating = Double.valueOf(personRating);

			RTuple temp = new RTuple();
			temp.iUserID = iUserID;
			temp.iItemID = iItemID;
			temp.dRating = dRating;
			trainData.add(temp);
			Total_Instance_Count++;
		}
//		//******************************
//		// For Sym Matrices
//		if(user_MaxID<item_MaxID)
//			user_MaxID = item_MaxID;
//		else
//			item_MaxID = user_MaxID;
		
		in.close();
		
		// Record Index of Omiga_P, Omiga_Q, Omiga_T
		ArrayList<ArrayList<Integer>> ListOmiga_P_Q = new ArrayList<ArrayList<Integer>>();
		ArrayList<ArrayList<Integer>> ListOmiga_Q_P = new ArrayList<ArrayList<Integer>>();
		
		ArrayList<ArrayList<Integer>> ListOmiga_P_R = new ArrayList<ArrayList<Integer>>();
		ArrayList<ArrayList<Integer>> ListOmiga_Q_R = new ArrayList<ArrayList<Integer>>();
		
		
		Omiga_P_Q = new int[user_MaxID + 1][];
		Omiga_P_R = new int[user_MaxID + 1][];
		Omiga_Q_P = new int[item_MaxID + 1][];
		Omiga_Q_R = new int[item_MaxID + 1][];
		
		for (int i = 0; i <= user_MaxID; i++)
		{
			ListOmiga_P_Q.add(new ArrayList<Integer>());
			ListOmiga_P_R.add(new ArrayList<Integer>());
		}
		
		for (RTuple temp : trainData)
		{
			ListOmiga_P_Q.get(temp.iUserID).add(temp.iItemID);
			ListOmiga_P_R.get(temp.iUserID).add(temp.iIndex);
		}
		
		for (int i = 0; i <= user_MaxID; i++)
		{
			Omiga_P_Q[i] = new int[ListOmiga_P_Q.get(i).size()];
			Omiga_P_R[i] = new int[ListOmiga_P_R.get(i).size()];
			for (int j = 0; j < Omiga_P_Q[i].length; j++)
			{
				Omiga_P_Q[i][j] = ListOmiga_P_Q.get(i).get(j);
				Omiga_P_R[i][j] = ListOmiga_P_R.get(i).get(j);
			}
		}
		
		for (int i = 0; i <= item_MaxID; i++)
		{
			ListOmiga_Q_P.add(new ArrayList<Integer>());
			ListOmiga_Q_R.add(new ArrayList<Integer>());
		}
		
		for (RTuple temp : trainData)
		{
			ListOmiga_Q_P.get(temp.iItemID).add(temp.iUserID);
			ListOmiga_Q_R.get(temp.iItemID).add(temp.iIndex);
		}
		
		for (int i = 0; i <= item_MaxID; i++)
		{
			Omiga_Q_P[i] = new int[ListOmiga_Q_P.get(i).size()];
			Omiga_Q_R[i] = new int[ListOmiga_Q_R.get(i).size()];
			for (int j = 0; j < Omiga_Q_P[i].length; j++)
			{
				Omiga_Q_P[i][j] = ListOmiga_Q_P.get(i).get(j);
				Omiga_Q_R[i][j] = ListOmiga_Q_R.get(i).get(j);
			}
		}
	}

	
	//from yuanye
	public static void yy_initValidationData(String fileName, String separator) throws NumberFormatException, IOException {
		validationData = new ArrayList<RTuple>();
		File personSource = new File(fileName);
		BufferedReader in = new BufferedReader(new FileReader(personSource));

		String tempVoting;
		while (((tempVoting = in.readLine()) != null)) {
			StringTokenizer st = new StringTokenizer(tempVoting, separator);
			String personID = null;
			if (st.hasMoreTokens())
				personID = st.nextToken();
			String movieID = null;
			if (st.hasMoreTokens())
				movieID = st.nextToken();
			String personRating = null;
			if (st.hasMoreTokens())
				personRating = st.nextToken();
			int iUserID = Integer.valueOf(personID);
			int iItemID = Integer.valueOf(movieID);
			
			user_MaxID = (user_MaxID > iUserID) ? user_MaxID : iUserID;
			item_MaxID = (item_MaxID > iItemID) ? item_MaxID : iItemID;
			double dRating = Double.valueOf(personRating);

			RTuple temp = new RTuple();
			temp.iUserID = iUserID;
			temp.iItemID = iItemID;
			temp.dRating = dRating;
			validationData.add(temp);
		}
		in.close();
	
	}
	
	//from yuanye
	public static void yy_initTestData(String fileName, String separator)
			throws NumberFormatException, IOException {
		testData = new ArrayList<RTuple>();
		File personSource = new File(fileName);
		BufferedReader in = new BufferedReader(new FileReader(personSource));

		String tempVoting;
		while (((tempVoting = in.readLine()) != null)) {
			StringTokenizer st = new StringTokenizer(tempVoting, separator);
			String personID = null;
			if (st.hasMoreTokens())
				personID = st.nextToken();
			String movieID = null;
			if (st.hasMoreTokens())
				movieID = st.nextToken();
			String personRating = null;
			if (st.hasMoreTokens())
				personRating = st.nextToken();
			int iUserID = Integer.valueOf(personID);
			int iItemID = Integer.valueOf(movieID);

			// 闁荤姳鐒�?妯肩礊瀹ュ棛鈻旈悗锝庡亝娴犳ê顭块崼鍡楁噺閻ｇ湈temid闂佸憡绮岄惁鍧癳rid闂佹寧绋掔粙鎴澝哄鍕枖缂侊紕鑵恊mid闂佸憡绮岄惁鍧癳rid闂佸搫�?�烽崹鐢垫崲濞戞氨纾兼い鎾跺У閻ｉ亶鏌ㄥ☉妯绘拱濠�?冪У缁傛帡濡烽敂鎴掔帛婵犮垹鐖㈤崨顔炬殸itemid闂佸憡绮岄惁鍧癳rid婵炴垶姊婚崰宥夊船椤掑�?�鍋�?柕濞垮�楅崯濠囨煕濮橆剙顏柛銈嗙矒閹啴宕熼浣诡�?闂佺儵鏅幏锟�
			user_MaxID = (user_MaxID > iUserID) ? user_MaxID : iUserID;
			item_MaxID = (item_MaxID > iItemID) ? item_MaxID : iItemID;

			double dRating = Double.valueOf(personRating);

			RTuple temp = new RTuple();
			temp.iUserID = iUserID;
			temp.iItemID = iItemID;
			temp.dRating = dRating;
			testData.add(temp);
		}
		in.close();
		// 闂佸憡甯楃换鍌烇綖閹版澘�?岄柡宓啰顦版繛鎴炴惄娴滄粓鎮块崟顒佸�?�闁规崘鍩栭悾閬嶆煛娴ｅ搫顣肩�规挷绶氬顐�?箥椤斿墽鐓�
		user_Rating_Arrays = new ArrayList[user_MaxID + 1];
		for (int userID = 1; userID <= user_MaxID; userID++) {
			user_Rating_Arrays[userID] = new ArrayList<RTuple>();
		}
		item_Rating_Arrays = new ArrayList[item_MaxID + 1];
		for (int itemID = 1; itemID <= item_MaxID; itemID++) {
			item_Rating_Arrays[itemID] = new ArrayList<RTuple>();
		}
		for (RTuple tempRating : trainData) {
			item_Rating_Arrays[tempRating.iItemID].add(tempRating);
			user_Rating_Arrays[tempRating.iUserID].add(tempRating);
		}
	}
	
	

	public static void initValidData(String fileName, String separator)
			throws NumberFormatException, IOException {
		validationData = new ArrayList<RTuple>();
		File personSource = new File(fileName);
		BufferedReader in = new BufferedReader(new FileReader(personSource));

		String tempVoting;
		while (((tempVoting = in.readLine()) != null)) {
			StringTokenizer st = new StringTokenizer(tempVoting, separator);
			String personID = null;
			if (st.hasMoreTokens())
				personID = st.nextToken();
			String movieID = null;
			if (st.hasMoreTokens())
				movieID = st.nextToken();
			String personRating = null;
			if (st.hasMoreTokens())	
				personRating = st.nextToken();
			int iUserID = Integer.valueOf(personID);
			int iItemID = Integer.valueOf(movieID);

			// 闂佽崵濮抽悞锕�顭垮Ο鑲╃鐎广儱妫涢埢鏃堟�?�閿濆骸浜濆ù鐘趁�?�鍧�?醇閸℃鍣洪柣锝囨箞temid闂備礁鎲＄划�?勬儊閸х櫝rid闂備焦�?�х粙鎺旂矙閹存緷鍝勵煥閸曨厸鏋栫紓渚婄磿閼垫亰mid闂備礁鎲＄划�?勬儊閸х櫝rid闂備礁鎼�氱兘宕归悽鍨床婵炴垶姘ㄧ壕鍏笺亜閹捐泛校闁伙綁浜堕弻銊モ槈濡粯鎷辨繝鈷�鍐�?�ｇ紒鍌涘浮婵＄兘鏁傞幋鎺斿笡濠电姰鍨归悥銏ゅ川椤旂偓娈竔temid闂備礁鎲＄划�?勬儊閸х櫝rid濠电偞鍨跺濠氬窗�?�ュ鑸规い鎺戝�归崑姗�鏌曟繛鍨拷妤呭疮婵犲洦鐓曟慨姗嗗墮椤忣剟鏌涢妶鍡欑煉闁诡垰鍟村畷鐔碱敆娴ｈ鍟�闂備胶鍎甸弲顏堝箯閿燂拷
			//user_MaxID = (user_MaxID > iUserID) ? user_MaxID : iUserID;
			//item_MaxID = (item_MaxID > iItemID) ? item_MaxID : iItemID;

			double dRating = Double.valueOf(personRating);

			RTuple temp = new RTuple();
			temp.iUserID = iUserID;
			temp.iItemID = iItemID;
			temp.dRating = dRating;
			validationData.add(temp);
		}
		in.close();
	}
	public static void initTestData(String fileName, String separator)
			throws NumberFormatException, IOException {
		testData = new ArrayList<RTuple>();
		File personSource = new File(fileName);
		BufferedReader in = new BufferedReader(new FileReader(personSource));

		String tempVoting;
		while (((tempVoting = in.readLine()) != null)) {
			StringTokenizer st = new StringTokenizer(tempVoting, separator);
			String personID = null;
			if (st.hasMoreTokens())
				personID = st.nextToken();
			String movieID = null;
			if (st.hasMoreTokens())
				movieID = st.nextToken();
			String personRating = null;
			if (st.hasMoreTokens())
				personRating = st.nextToken();
			int iUserID = Integer.valueOf(personID);
			int iItemID = Integer.valueOf(movieID);

			// 闂佽崵濮抽悞锕�顭垮Ο鑲╃鐎广儱妫涢埢鏃堟�?�閿濆骸浜濆ù鐘趁�?�鍧�?醇閸℃鍣洪柣锝囨箞temid闂備礁鎲＄划�?勬儊閸х櫝rid闂備焦�?�х粙鎺旂矙閹存緷鍝勵煥閸曨厸鏋栫紓渚婄磿閼垫亰mid闂備礁鎲＄划�?勬儊閸х櫝rid闂備礁鎼�氱兘宕归悽鍨床婵炴垶姘ㄧ壕鍏笺亜閹捐泛校闁伙綁浜堕弻銊モ槈濡粯鎷辨繝鈷�鍐�?�ｇ紒鍌涘浮婵＄兘鏁傞幋鎺斿笡濠电姰鍨归悥銏ゅ川椤旂偓娈竔temid闂備礁鎲＄划�?勬儊閸х櫝rid濠电偞鍨跺濠氬窗�?�ュ鑸规い鎺戝�归崑姗�鏌曟繛鍨拷妤呭疮婵犲洦鐓曟慨姗嗗墮椤忣剟鏌涢妶鍡欑煉闁诡垰鍟村畷鐔碱敆娴ｈ鍟�闂備胶鍎甸弲顏堝箯閿燂拷
			//user_MaxID = (user_MaxID > iUserID) ? user_MaxID : iUserID;
			//item_MaxID = (item_MaxID > iItemID) ? item_MaxID : iItemID;

			double dRating = Double.valueOf(personRating);

			RTuple temp = new RTuple();
			temp.iUserID = iUserID;
			temp.iItemID = iItemID;
			temp.dRating = dRating;
			testData.add(temp);
		}
		in.close();
	}



	// 闂備礁鎲＄喊宥夊垂閸ф闂繛宸簼閸婄兘姊洪锝囥�掓い鏃�甯￠弻娑㈠箣閿濆憛鎾绘煛閸☆厽�?��
	public static double dotMultiply(double[] x, double[] y) {
		double sum = 0;
		for (int i = 0; i < x.length; i++) {
			sum += x[i] * y[i];
		}
		return sum;
	}

	public static double dotMultiply(double[] x, double[] y, double[] z) {
		double sum = 0;
		for (int i = 0; i < x.length; i++) {
			sum += x[i] * y[i] * z[i];
		}
		return sum;
	}
	
	public static double dotMultiply(double[] x, double[] y, double[] z, double[] t) {
		double sum = 0;
		for (int i = 0; i < x.length; i++) {
			sum += x[i] * y[i] * z[i] * t[i];
		}
		return sum;
	}
	
	// 闂備礁鎲＄喊宥夊垂閸ф闂繛宸簻缁�澶愭煥閻曞倹�?��
	public static void vectorAdd(double[] first, double[] second,
			double[] result) {
		for (int i = 0; i < first.length; i++) {
			result[i] = first[i] + second[i];
		}
	}

	public static void vectorAdd(double[] first, double[] second) {
		for (int i = 0; i < first.length; i++) {
			first[i] = first[i] + second[i];
		}
	}



	// 闂備礁鎲＄喊宥夊垂閸ф闂柣鎴烆焽閳绘洟鏌ㄩ悤鍌涘
	public static void vectorMutiply(double[] vector, double time,
			double[] result) {
		for (int i = 0; i < vector.length; i++) {
			result[i] = vector[i] * time;
		}
	}

	public static double[] vectorMutiply(double[] vector, double time) {
		double[] result = new double[vector.length];
		for (int i = 0; i < vector.length; i++) {
			result[i] = vector[i] * time;
		}
		return result;
	}

	public static double[] vectorDotMultiply(double[] vector1, double[] vector2)
	{
		double[] result = new double[vector1.length];
		
		for (int i = 0; i < vector1.length; i++)
			result[i] = vector1[i] * vector2[i];
		
		return result;
	}
	
	public static double[] vectorDotMultiply(double[] vector1, double[] vector2, double[] vector3)
	{
		double[] result = new double[vector1.length];
		
		for (int i = 0; i < vector1.length; i++)
			result[i] = vector1[i] * vector2[i] * vector3[i];
		
		return result;
	}
	

	public double getMinPrediction(int userID, int itemID) {
		double ratingHat = 0;
		ratingHat += dotMultiply(min_P[userID], min_Q[itemID]);
		if (flag_B) {
			for (int j = 0; j < B_Count; j++) {
				ratingHat += min_B[userID][j];
			}
		}
		if (flag_C) {
			for (int j = 0; j < C_Count; j++) {
				ratingHat += min_C[itemID][j];
			}
		}
		return ratingHat;
	}

	public double getLocPrediction(int userID, int itemID) {
		double ratingHat = 0;
		ratingHat += dotMultiply(P[userID], Q[itemID]);
		//if (Double.isNaN(ratingHat))
		//	System.out.println("rating (" + userID + "," + itemID + ") is NaN");
		if (flag_B) {
			for (int j = 0; j < B_Count; j++) {
				ratingHat += B[userID][j];
			}
		}
		if (flag_C) {
			for (int j = 0; j < C_Count; j++) {
				ratingHat += C[itemID][j];
			}
		}
		return ratingHat;
	}

	
	public int getMaxIndex(double[] array) {
		int temp = -1;
		double max = -1000;
		for (int i = 0; i < array.length - 1; i++) {
			if (max < array[i]) {
				max = array[i];
				temp = i;
			}
		}
		return temp;
	}

	//from yuanye
	public double yy_trainCurrentRMSE() {
		// 闂佽崵濮崇欢銈囨閺囥垺鍋╁┑鐘宠壘閹瑰爼鏌曟繝蹇擃洭闁靛牊鎸抽幃褰掑炊閵婏妇顦板┑鐐差檧闂勫嫮绮欐径�?�瘈闁告洦鍓涢弳鐕E
		double sumRMSE = 0, sumCount = 0;
		for (RTuple tempTestRating : trainData) {
			double actualRating = tempTestRating.dRating;
			double ratinghat = this.getLocPrediction(tempTestRating.iUserID,
						tempTestRating.iItemID);

			sumRMSE += Math.pow((actualRating - ratinghat), 2);
			sumCount++;
		}
		double RMSE = Math.sqrt(sumRMSE / sumCount);
		return RMSE;
	}
		
		//from yuanye
	public double yy_trainCurrentMAE() {
		// 闂佽崵濮崇欢銈囨閺囥垺鍋╁┑鐘宠壘閹瑰爼鏌曟繝蹇擃洭闁靛牊鎸抽幃褰掑炊閵婏妇顦板┑鐐差檧闂勫嫮绮欐径�?�瘈闁告洦鍓涢弳鐕E
		double sumMAE = 0, sumCount = 0;
		for (RTuple tempTestRating : trainData) {
			double actualRating = tempTestRating.dRating;
			double ratinghat = this.getLocPrediction(tempTestRating.iUserID,
					tempTestRating.iItemID);

			sumMAE += Math.abs(actualRating - ratinghat);
			sumCount++;
		}
		double MAE = sumMAE / sumCount;
		return MAE;
	}
	
	//from yuanye
	public double yy_testCurrentMAE() {
		// 闂佽崵濮崇欢銈囨閺囥垺鍋╁┑鐘宠壘閹瑰爼鏌曟繝蹇擃洭闁靛牊鎸抽幃褰掑炊閵婏妇顦板┑鐐差檧闂勫嫮绮欐径�?�瘈闁告洦鍓涢弳鐕E
		double sumMAE = 0, sumCount = 0;
		for (RTuple tempTestRating : testData) {
			double actualRating = tempTestRating.dRating;
			double ratinghat = this.getLocPrediction(tempTestRating.iUserID,
					tempTestRating.iItemID);

			sumMAE += Math.abs(actualRating - ratinghat);
			sumCount++;
		}
		double MAE = sumMAE / sumCount;
		return MAE;
	}
	
	// public double staticBias = 0;
	// from yuanye, for 7:1:2
	public  double yy_validateCurrentRMSE() {
		// 闁荤姳绶ょ槐鏇㈡偩婵犳艾鎹堕柕濠忓閵堟挳鎮归崶銊︾婵炲闄勭粙澶嬬節閸曨剛鏆燫MSE
		double sumRMSE = 0, sumCount = 0;
		for (RTuple tempTestRating : validationData) {
			double actualRating = tempTestRating.dRating;
			double ratinghat =  this.getLocPrediction(tempTestRating.iUserID, tempTestRating.iItemID);
			sumRMSE += Math.pow((actualRating - ratinghat), 2);
			sumCount++;
		}
		double RMSE = Math.sqrt(sumRMSE / sumCount);
		return RMSE;
	}
	
	// from yuanye,  for 7:1:2
	public  double yy_validateCurrentMAE() {
		// 闁荤姳绶ょ槐鏇㈡偩婵犳艾鎹堕柕濠忓閵堟挳鎮归崶銊︾婵炲闄勭粙澶嬬節閸曨剛鏆燤AE
		double sumMAE = 0, sumCount = 0;
		for (RTuple tempTestRating : validationData) {
			double actualRating = tempTestRating.dRating;
			double ratinghat =  this.getLocPrediction(tempTestRating.iUserID, tempTestRating.iItemID);

			sumMAE += Math.abs(actualRating - ratinghat);
			sumCount++;
		}
		double MAE = sumMAE / sumCount;
		return MAE;
	}

	
	
	//from yuanye
	public static void yy_initialX() 
	{	
		particles = new double[yy_swarmNum];	
		pbest = new double[yy_swarmNum];
		gbest = 100;
		Random random = new Random(System.currentTimeMillis());  		
        for (int j = 0; j < yy_swarmNum; j++) 
    	{	//闁荤姳绀佹晶浠嬫偪閸℃鈻旈柨鐕傛嫹8-12闂佹眹鍔�?�氭澘鈻撻姀銈呭珘闁告繂瀚搁幏鐑芥晸閿燂�?
        	int intNum = etaMin + random.nextInt(etaMax - etaMin+1);
    		particles[j] = Math.pow(2, -etaMax)*2;//Math.pow(2, -intNum);
    		pbest[j] = particles[j];   			
    	}        	      	      	
    }
	
	
	
	//from yuanye
	public static void yy_initialV() 
	{
		
		V = new double[yy_swarmNum];	
//		Vmax=(Math.pow(2, -etaMin)-Math.pow(2, -etaMax));
		Vmin=-Vmax;				
		Random random = new Random(System.currentTimeMillis());
		//dimension=1闂佹寧绋戦鎶碼rmNum=10
		for(int i =0; i < yy_dimension ; i++)
        {    		
        	for (int j = 0; j < yy_swarmNum; j++) 
    		{
        		V[j] = random.nextDouble()*Vmax;
    		}        	
        } 							
    }
	
	//from yuanye
	public static void yy_initialfitness() 
	{
		FitnessRMSE = 100;
		FitnessRMSEgbest = 100;
		tempRMSE = new double[yy_swarmNum];
		FitnessRMSEpbest = new double[yy_swarmNum];
    	for (int j = 0; j < yy_swarmNum; j++) 
		{
    		FitnessRMSEpbest[j] = 100;
    		tempRMSE[j] = 0;
		} 
    }
	
	//from yuanye
	public static void yy_initiStaticArrays() {
		// 闂佸憡妫戦幏锟�1闂佸搫�?�烽崹顖滄嫻閻斿摜顩查柛鈩冾殕闊剟骞栭弶鎴犵鐟滅増�?�х粙澶嬬�?娴ｈ櫣鎲縄D婵烇絽娲︾换鍐拷鍨⒐缁嬪鏁撴禒瀣殜闁跨噦鎷�
		user_Rating_count = new double[user_MaxID + 1];
		item_Rating_count = new double[item_MaxID + 1];
		user_FW_Array = new double[user_MaxID + 1];
		item_FW_Array = new double[item_MaxID + 1];
		
		P_tmp = new double[user_MaxID + 1][featureDimension];   // Add from our codes
		P_cache = new double[user_MaxID + 1][featureDimension];
		P_delta = new double[user_MaxID + 1][featureDimension];
		P_r = new double[user_MaxID + 1][featureDimension];
		P_r_prime = new double[user_MaxID + 1][featureDimension];
		P_p = new double[user_MaxID + 1][featureDimension];
		P_hp = new double[user_MaxID + 1][featureDimension];
		P_gradient = new double[user_MaxID + 1][featureDimension];

		Q_tmp = new double[item_MaxID + 1][featureDimension];     // Add from our codes
		Q_cache = new double[item_MaxID + 1][featureDimension];
		Q_delta = new double[item_MaxID + 1][featureDimension];
		Q_r = new double[item_MaxID + 1][featureDimension];
		Q_r_prime = new double[item_MaxID + 1][featureDimension];
		Q_p = new double[item_MaxID + 1][featureDimension];
		Q_hp = new double[item_MaxID + 1][featureDimension];
		Q_gradient = new double[item_MaxID + 1][featureDimension];
		min_P = new double[user_MaxID + 1][featureDimension];
		min_Q = new double[item_MaxID + 1][featureDimension];

		B_Base = new double[user_MaxID + 1];
		C_Base = new double[item_MaxID + 1];

		// 闂佸憡甯楃换鍌烇綖閹版澘�?岄柡宥庡亜椤ユ锟界敻顣﹂懗鍫曟偂閳哄懏鈷撻柨鐕傛嫹,闂備焦褰冨ú銊╁极閵堝鈷曢煫鍥ㄦ⒐缁ㄦ岸鏌涙繝蹇斿,婵炲濮村锕傛晸閽樺瀚伴柤闀愬嵆楠炲骞囬鍡╀紘婵炴垶鎸剧�氱兘姊婚崘銊у煟闁跨喎锟界喎顥嶇紒浼欐嫹
		Random random = new Random(System.currentTimeMillis());
		for (int i = 0; i <= user_MaxID; i++) {
			user_Rating_count[i] = 0;
			int tempBB = random.nextInt(mapping_Scale);
			B_Base[i] = init_Max - init_Scale * tempBB / mapping_Scale;
			for (int j = 0; j < featureDimension; j++) {
				int temp = random.nextInt(mapping_Scale);
				P_cache[i][j] = init_Max - init_Scale * temp / mapping_Scale;
				min_P[i][j] = P_cache[i][j];
			}
		}
		for (int i = 0; i <= item_MaxID; i++) {
			item_Rating_count[i] = 0;
			int tempCB = random.nextInt(mapping_Scale);
			C_Base[i] = init_Max - init_Scale * tempCB / mapping_Scale;
			for (int j = 0; j < featureDimension; j++) {
				int temp = random.nextInt(mapping_Scale);
				Q_cache[i][j] = init_Max - init_Scale * temp / mapping_Scale;
				min_Q[i][j] = Q_cache[i][j];

			}
		}	
	}
		
	// from yuanye, for 7:1:2
	public static void yy_initializeRatings1(String trainFileName, String validationFileName, String testFileName, String separator)
				throws NumberFormatException, IOException {
			user_MaxID = 0;
			item_MaxID = 0;
			yy_initTrainData(trainFileName, separator);
			yy_initValidationData(validationFileName, separator);
			yy_initTestData(testFileName, separator);
	}
	
	
	// from yuanye, changing the above function for 8:2 
	public static void yy_initializeRatings1(String trainFileName, String testFileName, String separator)
				throws NumberFormatException, IOException {
			user_MaxID = 0;
			item_MaxID = 0;
			yy_initTrainData(trainFileName, separator);
			//yy_initValidationData(validationFileName, separator);
			yy_initTestData(testFileName, separator);
	}	
	
	
	// from yuanye
	public static void yy_initBiasSettings(boolean ifB, boolean ifC, int B_C,
			int C_C) {
		flag_B = ifB;
		flag_C = ifC;
		B_Count = B_C;
		C_Count = C_C;

		B_tmp = new double[user_MaxID + 1][B_Count];       // add from our codes
		B_cache = new double[user_MaxID + 1][B_Count];
		B_delta = new double[user_MaxID + 1][B_Count];
		B_r = new double[user_MaxID + 1][B_Count];
		B_r_prime = new double[user_MaxID + 1][B_Count];
		B_p = new double[user_MaxID + 1][B_Count];
		B_hp = new double[user_MaxID + 1][B_Count];
		B_gradient = new double[user_MaxID + 1][B_Count];

		C_tmp = new double[item_MaxID + 1][C_Count];         // add from our codes
		C_cache = new double[item_MaxID + 1][C_Count];
		C_delta = new double[item_MaxID + 1][C_Count];
		C_r = new double[item_MaxID + 1][C_Count];
		C_r_prime = new double[item_MaxID + 1][C_Count];
		C_p = new double[item_MaxID + 1][C_Count];
		C_hp = new double[item_MaxID + 1][C_Count];
		C_gradient = new double[item_MaxID + 1][C_Count];
		min_B = new double[user_MaxID + 1][B_Count];
		min_C = new double[item_MaxID + 1][C_Count];

		System.gc();

		if (B_Count != 0) {
			for (int i = 1; i <= user_MaxID; i++) {
				double tempUB = B_Base[i] / B_Count;
				for (int j = 0; j < B_Count; j++) {
					B_cache[i][j] = tempUB;
					min_B[i][j] = B_cache[i][j];
				}
			}
		}

		if (C_Count != 0) {
			for (int i = 1; i <= item_MaxID; i++) {
				double tempIB = C_Base[i] / C_Count;
				for (int j = 0; j < C_Count; j++) {
					C_cache[i][j] = tempIB;
					min_C[i][j] = C_cache[i][j];
				}
			}
		}
	}	
	
	// from yuanye
	public static void yy_updateParticlesandV() 
	{
		Random random = new Random(System.currentTimeMillis());
		yy_r1 = random.nextDouble();
		yy_r2 = random.nextDouble();
		//yy_w = random.nextDouble(); commented after adjusting with 0.729 by WZ
//		double g=random.nextGuassion();
		for (int k = 0; k < yy_swarmNum; k++) 
		{	//闂佸搫娲ら悺銊╁蓟婵犲洦鐒婚柣鏂垮槻椤旓�?
		   V[k] = yy_w*V[k] + yy_c1*yy_r1*(pbest[k] - (1-yy_rho) * particles[k] - yy_rho * V[k])    //濞ｈ濮濸LFA閻╃鍙ч惃鍕��?
				            + yy_c2*yy_r2*(gbest - (1-yy_rho) * particles[k] - yy_rho *V[k]);
		   //婵犵锟藉啿锟藉綊鎮樻繅姝攌]
		   if(V[k] > Vmax)
		   {
			 V[k]= Vmax;
		   }
		   else if(V[k] < Vmin)
		   {
			 V[k] = Vmin;
		   }
		   //闂佸搫娲ら悺銊╁蓟婵犲啯濯寸�广儱娲ㄩ弸锟�
		   particles[k] = particles[k] + V[k];
		   //婵犵锟藉啿锟藉綊鎮樻繅绔宺ticles[k]婵炶揪绲界粔鍫曟偪閸℃ê绶炵憸�?�鑺遍鍕珘闁跨喕濮ゅ鍕檪缂佽鲸绻堝畷姘枎閹寸姵鐎梺绋跨箺濞夋盯鏁撴禒瀣紵閻犳劗鍠栧鐢告晸閼恒儱绶炵憸宥夋晸閻ｅ被锟藉妲愬┑鍥︾箚闁稿本绋撴禍鐥痑rticles[k]婵炶揪绲界粔鍫曟偪閸℃瑤鐒婇煫鍥ф捣閼归箖鏌￠崼顒佸闁诲繐绻愭蹇曟濠靛牏纾炬い鏃傚帶瀵版捇鏌涙繝鍥紵閻犳劗鍠栧鐢告晸閻ｅ奔鐒婇煫鍥ㄦ惈閹风兘鏁撻敓锟�?
		   if(particles[k] > Math.pow(2, -etaMin))
		   {
			   particles[k] = Math.pow(2, -etaMin);
		   }
		   if(particles[k] < Math.pow(2, -etaMax))
		  {
			   particles[k] = Math.pow(2, -etaMax);
		  }
	   }
		
	}	
	
	// from yuanye
	public static void yy_updatebestRMSE(double [] tempPbestRMSE) {
		for (int k = 0; k < yy_swarmNum; k++) {
			if (tempPbestRMSE[k] < FitnessRMSEpbest[k]) {
					FitnessRMSEpbest[k] = tempPbestRMSE[k];
					pbest[k] = particles[k];
				}

				if (tempPbestRMSE[k] < FitnessRMSEgbest) {
					FitnessRMSEgbest = tempPbestRMSE[k];
					yy_bestSwarm = k;
					gbest = particles[k];
				}
			}
//		
	}
	
	
	
	
	
	public double trainCurrentMAE(LossFunSet LossFun) {
		// 闂佽崵濮崇欢銈囨閺囥垺鍋╁┑鐘宠壘閹瑰爼鏌曟繝蹇擃洭闁靛牊鎸抽幃褰掑炊閵婏妇顦板┑鐐差檧闂勫嫮绮欐径�?�瘈闁告洦鍓涢弳鐕E
		double sumMAE = 0, sumCount = 0;
		for (RTuple tempTestRating : trainData) {
			double actualRating = tempTestRating.dRating;
			double ratinghat = 0;
			switch (LossFun)
			{
				case TwoDimMF:
					ratinghat = this.getLocPrediction(tempTestRating.iUserID,	tempTestRating.iItemID);
					break;
				default:
					break;
			}

			sumMAE += Math.abs(actualRating - ratinghat);
			sumCount++;
		}
		double MAE = sumMAE / sumCount;
		return MAE;
	}
	
	public double testCurrentMAE(LossFunSet LossFun) {
		// 闂佽崵濮崇欢銈囨閺囥垺鍋╁┑鐘宠壘閹瑰爼鏌曟繝蹇擃洭闁靛牊鎸抽幃褰掑炊閵婏妇顦板┑鐐差檧闂勫嫮绮欐径�?�瘈闁告洦鍓涢弳鐕E
		double sumMAE = 0, sumCount = 0;
		for (RTuple tempTestRating : testData) {
			double actualRating = tempTestRating.dRating;
			double ratinghat = 0;
			switch (LossFun)
			{
				case TwoDimMF:
					ratinghat = this.getLocPrediction(tempTestRating.iUserID,	tempTestRating.iItemID);
					break;
	
				default:
					break;
			}

			sumMAE += Math.abs(actualRating - ratinghat);
			sumCount++;
		}
		double MAE = sumMAE / sumCount;
		return MAE;
	}
	
	public double validCurrentMAE(LossFunSet LossFun) {
		// 闂佽崵濮崇欢銈囨閺囥垺鍋╁┑鐘宠壘閹瑰爼鏌曟繝蹇擃洭闁靛牊鎸抽幃褰掑炊閵婏妇顦板┑鐐差檧闂勫嫮绮欐径�?�瘈闁告洦鍓涢弳鐕E
		double sumMAE = 0, sumCount = 0;
		for (RTuple tempTestRating : validationData) {
			double actualRating = tempTestRating.dRating;
			double ratinghat = 0;
			switch (LossFun)
			{
				case TwoDimMF:
					ratinghat = this.getLocPrediction(tempTestRating.iUserID,	tempTestRating.iItemID);
					break;
	
				default:
					break;
			}

			sumMAE += Math.abs(actualRating - ratinghat);
			sumCount++;
		}
		double MAE = sumMAE / sumCount;
		return MAE;
	}
	public double testCurrentVirtualMAE(int trainFlag) {
		// 闂佽崵濮崇欢銈囨閺囥垺鍋╁┑鐘宠壘閹瑰爼鏌曟繝蹇擃洭闁靛牊鎸抽幃褰掑炊閵婏妇顦板┑鐐差檧闂勫嫮绮欐径�?�瘈闁告洦鍓涢弳鐕E
		double sumMAE = 0, sumCount = 0;
		if (trainFlag == 0)
		{
			for (RTuple tempTestRating : virtualTestData) {
				double actualRating = tempTestRating.dRating;
				double ratinghat = this.getLocPrediction(tempTestRating.iUserID,
						tempTestRating.iItemID);
	
				sumMAE += Math.abs(actualRating - ratinghat);
				sumCount++;
			}
		}
		else
		{
			for (RTuple tempTestRating : testData) {
				double actualRating = tempTestRating.dRating;
				double ratinghat = this.getLocPrediction(tempTestRating.iUserID,
						tempTestRating.iItemID);

				sumMAE += Math.abs(actualRating - ratinghat);
				sumCount++;
			}
		}
		double MAE = sumMAE / sumCount;
		return MAE;
	}

	public double trainCurrentRMSE() {
		// 闂佽崵濮崇欢銈囨閺囥垺鍋╁┑鐘宠壘閹瑰爼鏌曟繝蹇擃洭闁靛牊鎸抽幃褰掑炊閵婏妇顦板┑鐐差檧闂勫嫮绮欐径�?�瘈闁告洦鍓涢弳鐕�?SE
		double sumRMSE = 0, sumCount = 0;
		for (RTuple tempTestRating : trainData) {
			double actualRating = tempTestRating.dRating;
			double ratinghat = this.getLocPrediction(tempTestRating.iUserID,
					tempTestRating.iItemID);

			sumRMSE += Math.pow((actualRating - ratinghat), 2);
			sumCount++;
		}
		double RMSE = Math.sqrt(sumRMSE / sumCount);
		return RMSE;
	}
	
	//from yuanye
	public double yy_testCurrentRMSE() {
		// 闂佽崵濮崇欢銈囨閺囥垺鍋╁┑鐘宠壘閹瑰爼鏌曟繝蹇擃洭闁靛牊鎸抽幃褰掑炊閵婏妇顦板┑鐐差檧闂勫嫮绮欐径�?�瘈闁告洦鍓涢弳鐕�?SE
		double sumRMSE = 0, sumCount = 0;
		for (RTuple tempTestRating : testData) {
			double actualRating = tempTestRating.dRating;
			double ratinghat = this.getLocPrediction(tempTestRating.iUserID,
					tempTestRating.iItemID);

			sumRMSE += Math.pow((actualRating - ratinghat), 2);
			sumCount++;
		}
		double RMSE = Math.sqrt(sumRMSE / sumCount);
		return RMSE;
	}
	
	public double trainCurrentRMSE(LossFunSet LossFun) {
		// 闂佽崵濮崇欢銈囨閺囥垺鍋╁┑鐘宠壘閹瑰爼鏌曟繝蹇擃洭闁靛牊鎸抽幃褰掑炊閵婏妇顦板┑鐐差檧闂勫嫮绮欐径�?�瘈闁告洦鍓涢弳鐕�?SE
		double sumRMSE = 0, sumCount = 0;
		for (RTuple tempTestRating : trainData) {
			double actualRating = tempTestRating.dRating;
			double ratinghat = 0;
			switch (LossFun)
			{
				case TwoDimMF:
					ratinghat = this.getLocPrediction(tempTestRating.iUserID,	tempTestRating.iItemID);
					break;
				default:
					break;
			}

			sumRMSE += Math.pow((actualRating - ratinghat), 2);
			sumCount++;
		}
		double RMSE = Math.sqrt(sumRMSE / sumCount);
		return RMSE;
	}

	public double validCurrentRMSE(LossFunSet LossFun) {
		// 闂佽崵濮崇欢銈囨閺囥垺鍋╁┑鐘宠壘閹瑰爼鏌曟繝蹇擃洭闁靛牊鎸抽幃褰掑炊閵婏妇顦板┑鐐差檧闂勫嫮绮欐径�?�瘈闁告洦鍓涢弳鐕�?SE
		double sumRMSE = 0, sumCount = 0;
		for (RTuple tempTestRating : validationData) {
			double actualRating = tempTestRating.dRating;
			double ratinghat = 0;
			switch (LossFun)
			{
				case TwoDimMF:
					ratinghat = this.getLocPrediction(tempTestRating.iUserID,	tempTestRating.iItemID);
					break;
				default:
					break;
			}

			sumRMSE += Math.pow((actualRating - ratinghat), 2);
			sumCount++;
		}
		double RMSE = Math.sqrt(sumRMSE / sumCount);
		return RMSE;
	}
	
	public double testCurrentRMSE(LossFunSet LossFun) {
		// 闂佽崵濮崇欢銈囨閺囥垺鍋╁┑鐘宠壘閹瑰爼鏌曟繝蹇擃洭闁靛牊鎸抽幃褰掑炊閵婏妇顦板┑鐐差檧闂勫嫮绮欐径�?�瘈闁告洦鍓涢弳鐕�?SE
		double sumRMSE = 0, sumCount = 0;
		for (RTuple tempTestRating : testData) {
			double actualRating = tempTestRating.dRating;
			double ratinghat = 0;
			switch (LossFun)
			{
				case TwoDimMF:
					ratinghat = this.getLocPrediction(tempTestRating.iUserID,	tempTestRating.iItemID);
					break;
				default:
					break;
			}

			sumRMSE += Math.pow((actualRating - ratinghat), 2);
			sumCount++;
		}
		double RMSE = Math.sqrt(sumRMSE / sumCount);
		return RMSE;
	}
		
	public double testCurrentVirtualRMSE(int trainFlag) {
		double sumRMSE = 0, sumCount = 0;
		
		if (trainFlag == 0)
		{
			for (RTuple tempTestRating : virtualTestData) {
				double actualRating = tempTestRating.dRating;
				double ratinghat = this.getLocPrediction(tempTestRating.iUserID,
					tempTestRating.iItemID);

				sumRMSE += Math.pow((actualRating - ratinghat), 2);
				sumCount++;
			}
		}
		else
		{
			for (RTuple tempTestRating : testData) {
				double actualRating = tempTestRating.dRating;
				double ratinghat = this.getLocPrediction(tempTestRating.iUserID,
						tempTestRating.iItemID);

				sumRMSE += Math.pow((actualRating - ratinghat), 2);
				sumCount++;
			}
		}
		
		double RMSE = Math.sqrt(sumRMSE / sumCount);
		return RMSE;
	}
	
	
	// copy the indexSort function from DE code obtained from github, the source file is IndexSort.java in DE package
	public class IndexSort {
	
        public Integer[] Ind(double[] a){	
            Integer[] b = new Integer[a.length];
            for (int i = 0 ; i < b.length ; i++){
                b[i] = i; 
            }
            Arrays.sort(b, new Comparator<Integer>() {
                public int compare(Integer o1, Integer o2) {
                 return Double.valueOf(a[o1]).compareTo(Double.valueOf(a[o2]));
                 }
            });
            
            return b;
        }
    
    }
}
