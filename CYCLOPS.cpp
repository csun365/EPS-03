// note that the code below uses "Eigen"; linked via make include-options
#include <iostream>
#include <stdio.h>

#include <fstream>
#include <string>
#include <omp.h>
#include <string.h>
#include <vector>
#include <math.h>
#include <Eigen/Dense>
#include <nr3/nr3.h>
#include <nr3/mins.h>
#include <nr3/mins_ndim.h>

using namespace Eigen;
using namespace std;


typedef Matrix<double, 18, 18, 0, 18, 18> Matrix18d;
typedef Array<double, 18, 18, 0, 18, 18> Array18d;
typedef Matrix<double, 10, 8, 0, 10, 8> Matrix108f;
typedef Matrix<double, 18, 8, 0, 18, 8> Matrix188d;

typedef Matrix<double, 8, 1, 0, 8, 1> Vector8f;
typedef Matrix<double, 10, 1, 0, 10, 1> Vector10f;
typedef Matrix<double, 18, 1, 0, 18, 1> Vector18f;
typedef Matrix<double, 20, 1, 0, 20, 1> Vector20f;
typedef Array<double, 8, 1, 0, 8, 1> ArVector8f;
typedef Array<double, 4, 1, 0, 4, 1> ArVector4f;
typedef Array<double, 18, 1, 0, 18, 1> ArVector18f;
typedef Array<double, 20, 1, 0, 20, 1> ArVector20f;
typedef Array<double, 136,1, 0, 136,1> OutArVectorf;
typedef Array<double, 18, 1, 0, 18, 1> ArVector18d;
typedef Matrix<double, 18, 1, 0, 18, 1> Vector18d;

//typedef Array<double, 290,1, 0, 290,1> OUTROWArray;
//typedef Array<double, 729,290, 0, 729,290> OUTArray;
typedef Array<double,8,1,0,8,1>	AllExArray;
struct binS {ArVector18f bin0; ArVector18f bin1; ArVector18f bin2;} bin;
struct flag {bool NotUsed;} flag;
struct tracerList {Vector18f P; Vector18f Preg; Vector18f C; Vector18f dc13; Vector18f Alk;Vector18f Alkreg; Vector18f N;Vector18f Sal; Vector18f Temp; Vector18f Si; Vector18f dc30; Vector18f dc14;} tracer;
struct venttracerList {Matrix188d vent; Vector18f trueage; Vector18f pref14Cage;} venttracer;
struct rainList {Vector8f P; Vector8f Ca; Vector8f Si; Vector8f d13Corg; Vector8f d13Ccc; Vector8f d30Si; Vector8f d14Corg; Vector8f d14Ccc;} rain;
struct basin {ArVector18f FArea, FCa; ArVector20f K1, K2, Kb, Kw, Ks, Hsitu, CO3situ, omega;} NCW, DSO, Atl, Ind, SPac, NPac;
struct Cchem {ArVector18f H; ArVector18f H2CO3; ArVector18f HCO3; ArVector18f CO3; ArVector18f pCO2; ArVector18f BOH4; ArVector18f beta; ArVector18f omega; ArVector4f omegasitu; ArVector4f LysDepth;} Csolved;
struct seafloor {basin NCW, DSO, Atl, Ind, SPac, NPac; ArVector18f SFdepth;Vector4f Fdiss;}SF;
struct boxList {Vector18f vol; Vector18f vol_inv; Vector8f setP; Vector8f setSi; ArVector8f ORGe; Vector8f CaRatio; Vector18f top; Vector18f bottom; Vector18f NtoC; Vector18f CtoN; ArVector8f Area; Cchem Csolved;} box;
struct Ktable {ArVector18f K0,K1,K2,Kb,Ks;} Ksurf;
struct paramQ14 {Array<double, 501, Dynamic, 0, 501, Dynamic> OUT2; Array<double, 367, 4, 0, 367, 4> Q14Cforcing; double Qnode; double Qnextnode; double DQ; double Dt; double yrstep; double prod; int ExNo; int row; int init_true; Array<double, 501, Dynamic, 0, 501, Dynamic> OUT; int OUTrow;} Q14;
struct DGLforcing {Array<double, 12, 1, 0, 12, 1> Forcing; Array<int, 12, 1, 0, 12, 1> ForceTime; double node; double nextnode; double D; int Dt; int yrstep; double value; int row; int init_true;};
struct DGLforcingCIRC {Array<double, 16, 1, 0, 16, 1> Forcing; Array<int, 16, 1, 0, 16, 1> ForceTime; double node; double nextnode; double D; int Dt; int yrstep; double value; int row; int init_true;};
struct DGLF {DGLforcingCIRC F1; DGLforcing F2; DGLforcing F3; DGLforcing F4; int init_true; int trigerID;};
struct ALKaddForcing {Array<double, 381, 1, 0, 381, 1> ALKaddForcing; Array<double, 381, 1, 0, 381, 1> ALKaddForceTime; Array<double, 121, 1, 0, 121, 1> RCPCO2Forcing; Array<double, 121, 1, 0, 121, 1> RCPCO2ForceTime;int init_true; int trigerID;};
struct ALKadd {int flag; double Crate_PgCperYear;double CumCarbonFlux; double totalALK; int tau; double annualALK; Array<double, Dynamic, Dynamic, 0, Dynamic, Dynamic> OUTPUT; int outrow;};
struct ALKaddOUT {int flag;};
struct VolChangeSchemes{Matrix18d DOtoMD; Matrix18d MDtoDO;  Matrix18d DOtoMDviaAApLL;  Matrix18d MDtoDOviaNA; Matrix18d DOtoUO; Matrix18d UOtoDO; Matrix18d NO;} Schemes;
struct VolChange{float scale; Vector18f newvol; Vector18f reducedvol; int DoIt; tracerList newtracer; venttracerList newventtracer; Matrix18d VolOp;} VolParams;
struct oceanS {Matrix18d circulationM; Matrix108f RainOrg,RainCC, RainSi; tracerList tracer; rainList rain; boxList box; Ktable Ksurf; Ktable Kdeep; seafloor SF; venttracerList venttracer; binS bin; VolChangeSchemes Schemes; VolChange VolParams;} ocean;
struct atmosphereS {double surfT; double prevppm; double ppm; double dn13; double dn14; ArVector8f oldH;} ;
struct geosphereS {double ppmCorg; double flux; double dn14; double dn13; double d14Corg; bool NotUsed;};
struct parametersS {ALKadd ALK;ALKaddOUT ALKout; Array<double, 401, 2, 0, 401, 2> ExOUT; int Ex1, Ex2, Exflag; double TxS; double VolcX; double WeathX;double CaX; double RivX; double SetCO2;double InitCO2;double ORGe; int flag; double Spike;double SpikeDelta;double DissolveX; double alphaSi; double scalelength; double PAZiceX; double PAZsv; double PAZarea; double MixFrac; int year; double C14X; paramQ14 Q14; DGLF DGLFall;ALKaddForcing ALKF;} ;
struct experiment {oceanS ocean; atmosphereS atmosphere; geosphereS geosphere; parametersS param;};

/////////////////////////////////////Minimization Functor//////////////////////////////////////////////////////
struct MinimizationFunctor {
	// CONSTRUCTOR takes InitialState as input
	MinimizationFunctor (experiment InitialState, int StartYearIN) : Init( InitialState ), StartYear( StartYearIN ) {}

	// STORAGE of InitialState
	experiment Init, Run;
	int StartYear;

	// MEMBER FUNCTION PROTOTYPE
	double RunExForFunctor (VecDoub P, oceanS &ocean, atmosphereS &atmosphere, geosphereS &geosphere, parametersS &param, int Nyears);

	// TIMESTEPPING for minimization
	double operator() (VecDoub_I &P)
	{
		Run = Init;

		double RMS;
		RMS = RunExForFunctor( P, Run.ocean, Run.atmosphere, Run.geosphere, Run.param, Run.param.ALK.tau); // this is where he is calling the equation

		return RMS;
	}

	void df(VecDoub_I &P, VecDoub_O &deriv) // partial derivative for my RMS equation
	{
		VecDoub_I P0 = P;
		VecDoub_O Pt;
		double fup, fdown, f0;

		Run = Init;
		f0 = RunExForFunctor( P0, Run.ocean, Run.atmosphere, Run.geosphere, Run.param, Run.param.ALK.tau);

		// 1st dim
		Run = Init;
		Pt = P0;
		Pt[0] = Pt[0]*1.00001; // slight increase of parameter
		fup = RunExForFunctor( Pt, Run.ocean, Run.atmosphere, Run.geosphere, Run.param, Run.param.ALK.tau);
		//cout<<"fup is "<<fup<<"\n";
		Run = Init;
		Pt = P0;
		Pt[0] = Pt[0]*0.99999; // slight decrease of parameter
		fdown = RunExForFunctor( Pt, Run.ocean, Run.atmosphere, Run.geosphere, Run.param, Run.param.ALK.tau);
		//cout<<"fdown is "<<fdown<<"\n";
		if ((fup>f0) && (fdown>f0))
		{
			deriv[0] = 0.0;
		}
		else
		{
			deriv[0] = (fup-fdown)/(0.00002*P0[0]);
		}
		//std::cout<<"Deriv:  "<<deriv[0]<<" \n";
	}
};
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

void LoadDGLforcing(DGLF& F)
{
	FILE * pFile;

    pFile = fopen("FORCING/EPSL2014_FORCE/FORCE_DGL_NAcirc.txt","r");
    //pFile = fopen ("DGL_FORCING_B_DT_win.txt","r");
    for(int row=0;row<16;row++)
    {
         //printf("initQ: row=%d",row);
         fscanf(pFile, "%i", &F.F1.ForceTime(row,0));
         fscanf(pFile, "%lf", &F.F1.Forcing(row,0));
         //printf("  init: year=%d, val=%f\n", F.F1.ForceTime(row,0),F.F1.Forcing(row,0));
    }
    fclose (pFile);

    pFile = fopen ("FORCING/EPSL2014_FORCE/FORCE_DGL_SOc.txt","r");
    //pFile = fopen ("DGL_FORCING_B_DT_win.txt","r");
    for(int row=0;row<12;row++)
    {
         fscanf(pFile, "%i", &F.F2.ForceTime(row,0));
         fscanf(pFile, "%lf", &F.F2.Forcing(row,0));
         fscanf(pFile, "%i", &F.F3.ForceTime(row,0));
         fscanf(pFile, "%lf", &F.F3.Forcing(row,0));
         //printf("  init: year=%d, val=%f\n", F.F1.ForceTime(row,0),F.F1.Forcing(row,0));
    }
    fclose (pFile);

    pFile = fopen ("FORCING/EPSL2014_FORCE/FORCE_HOLO_Vol.txt","r");
    //pFile = fopen ("DGL_FORCING_B_DT_win.txt","r");
    for(int row=0;row<12;row++)
    {
         fscanf(pFile, "%i", &F.F4.ForceTime(row,0));
         fscanf(pFile, "%lf", &F.F4.Forcing(row,0));
         //printf("  init: year=%d, val=%f\n", F.F1.ForceTime(row,0),F.F1.Forcing(row,0));
    }
    fclose (pFile);
}

void LoadALKForcing(ALKaddForcing& ALKF)
{
	FILE * pFile;

    pFile = fopen("FORCING/Project3/AlkalinityScenarioChris.txt","r");
    for(int row=0;row<380;row++) //271 for 1750 historicaldata
    {
         fscanf(pFile, "%lf", &ALKF.ALKaddForceTime(row,0));
         fscanf(pFile, "%lf", &ALKF.ALKaddForcing(row,0));
    }
    fclose (pFile);

    // for(int row=0;row<80;row++)
 //    {
 //         fscanf(pFile, "%lf", &ALKF.ALKaddForceTime(row,0));
 //         fscanf(pFile, "%lf", &ALKF.ALKaddForcing(row,0));
 //    }
 //    fclose (pFile);


}

void LoadRCPForcing(ALKaddForcing& ALKF)
{
	FILE * pFile;

    pFile = fopen("FORCING/Project3/RCP85_emissions.txt","r");
    for(int row=0;row<121;row++)
    {
         fscanf(pFile, "%lf", &ALKF.RCPCO2ForceTime(row,0));
         fscanf(pFile, "%lf", &ALKF.RCPCO2Forcing(row,0));
    }
    fclose (pFile);


}

void InitALKforcing(ALKaddForcing& ALKF) //do I need this or is this where I should put the algorithm?
{
	LoadALKForcing(ALKF);
	LoadRCPForcing(ALKF);

}

void Input(oceanS& ocean, atmosphereS& atmosphere, geosphereS& geosphere)
{
	FILE * pFile;
	//char  g [80];

	for (int box=0;box<18; box++) ocean.tracer.Sal(box)=34.7;
	for (int box=0;box<18; box++) ocean.tracer.Temp(box)=5;
	for (int box=0;box<18; box++) ocean.tracer.Si(box)=92;
	for (int box=0;box<18; box++) ocean.tracer.dc30(box)=0;

printf( " Initializing model CYCLOPS++ V1 ...\n");
printf( " Testing system ...\n");
printf( " %d CPU cores detected.\n",omp_get_max_threads());

pFile = fopen ("GITCY/CYCLOPSpp_INPUTunicode_silica.txt","r");

	for(int row=0;row<18;row++){ for(int col=0;col<18;col++){fscanf (pFile, "%lf",&ocean.circulationM(row,col));}}
	for(int row=0;row<10;row++){ for(int col=0;col<8;col++){fscanf (pFile, "%lf", &ocean.RainOrg(row,col));}}
	for(int row=0;row<10;row++){ for(int col=0;col<8;col++){fscanf (pFile, "%lf", &ocean.RainCC(row,col));}}
	ocean.RainSi = ocean.RainCC;
	for(int col=0;col<18;col++){fscanf (pFile, "%lf", &ocean.box.vol(col));}
	ocean.box.vol	=	ocean.box.vol*1.35E+18*0.01;
	ocean.box.CtoN	=	ocean.box.vol*1024;
	ocean.box.NtoC	=	ocean.box.vol.array().inverse()/1024;
	for(int col=0;col<8;col++){fscanf (pFile, "%lf", &ocean.box.Area(col));}
	ocean.box.Area	=	ocean.box.Area/100 * 3.265E+14;
	for(int col=0;col<18;col++){fscanf (pFile, "%lf", &ocean.tracer.P(col));ocean.tracer.N(col)=16*ocean.tracer.P(col);}
	for(int col=0;col<8;col++){fscanf (pFile, "%lf", &ocean.box.setP(col));}
    for(int col=0;col<8;col++){fscanf (pFile, "%lf", &ocean.box.setSi(col));}
	for(int col=0;col<8;col++){fscanf (pFile, "%lf", &ocean.box.CaRatio(col));}
	for(int col=0;col<8;col++){fscanf (pFile, "%lf", &ocean.tracer.Sal(col));}
	for(int col=0;col<8;col++){fscanf (pFile, "%lf", &ocean.tracer.Temp(col));}
	for(int col=0;col<18;col++){fscanf (pFile, "%lf", &ocean.tracer.C(col));}
	for(int col=0;col<18;col++){fscanf (pFile, "%lf", &ocean.tracer.Alk(col));}

	for (int level=0; level<18; level++) //initializes the lysoclines ocean.SF.xxx
	{
		fscanf (pFile,"%lf", &ocean.SF.SFdepth(level));
		fscanf (pFile,"%lf", &ocean.SF.NCW.FArea(level));
		fscanf (pFile,"%lf", &ocean.SF.NCW.FCa(level));
		fscanf (pFile,"%lf", &ocean.SF.DSO.FArea(level));
		fscanf (pFile,"%lf", &ocean.SF.DSO.FCa(level));
		fscanf (pFile,"%lf", &ocean.SF.Atl.FArea(level));
		fscanf (pFile,"%lf", &ocean.SF.Atl.FCa(level));
		fscanf (pFile,"%lf", &ocean.SF.Ind.FArea(level));
		fscanf (pFile,"%lf", &ocean.SF.Ind.FCa(level));
		fscanf (pFile,"%lf", &ocean.SF.SPac.FArea(level));
		fscanf (pFile,"%lf", &ocean.SF.SPac.FCa(level));
		fscanf (pFile,"%lf", &ocean.SF.NPac.FArea(level));
		fscanf (pFile,"%lf", &ocean.SF.NPac.FCa(level));

		ocean.SF.NCW.Hsitu(level) = 1e-2;
		ocean.SF.DSO.Hsitu(level) = 1e-2;
		ocean.SF.Atl.Hsitu(level) = 1e-2;
		ocean.SF.Ind.Hsitu(level) = 1e-2;
		ocean.SF.SPac.Hsitu(level) = 1e-2;
		ocean.SF.NPac.Hsitu(level) = 1e-2;
	}

    geosphere.ppmCorg = 2500 * 1e15 / 12 / 1.773e14 ; // 2500PgC in atmpspheric-ppm units
    geosphere.flux = 0.01*geosphere.ppmCorg;
    geosphere.dn14 = 0;
    geosphere.d14Corg = 0;
    geosphere.dn13 = -30*geosphere.ppmCorg;

    //ocean.tracer.C	= ocean.tracer.C.cwiseProduct(Vector18f::Constant(0.995));
	atmosphere.surfT = 0;
	atmosphere.ppm = 300;
	atmosphere.dn13 = 0;
	atmosphere.dn14 = 1000*atmosphere.ppm;
	atmosphere.oldH = ArVector8f::Constant(1e-2);
    ocean.box.ORGe = ArVector8f::Constant(20);
	ocean.tracer.dc13	= Vector18f::Constant(0);
	ocean.tracer.Preg= Vector18f::Constant(0);
	ocean.tracer.Alkreg= Vector18f::Constant(0);
	ocean.tracer.dc14	= ocean.tracer.C.cwiseProduct(Vector18f::Constant(850));



	ocean.box.Csolved.H		= ArVector18f::Constant(-99);
	ocean.box.Csolved.H2CO3	= ArVector18f::Constant(-99);
	ocean.box.Csolved.HCO3	= ArVector18f::Constant(-99);
	ocean.box.Csolved.CO3	= ArVector18f::Constant(-99);
	ocean.box.Csolved.pCO2	= ArVector18f::Constant(-99);
	ocean.box.Csolved.BOH4	= ArVector18f::Constant(-99);
	ocean.box.Csolved.beta	= ArVector18f::Constant(-99);

    ocean.venttracer.trueage= Vector18f::Constant(0);
    ocean.venttracer.pref14Cage= Vector18f::Constant(0);
    ocean.venttracer.vent = Matrix188d::Constant(0);
    for (int box=0;box<8;box++) {ocean.venttracer.vent(box,box)=1;}

	fclose (pFile);
}

void LoadQ14C(paramQ14& Q14)
{
	FILE * pFile;
	char filename[128] = "empty";

    if (Q14.ExNo==0)
    {
        pFile = fopen ("14CPROD/QrecFILE/Qrec_GLOPIS.txt","r");
        for(int row=0;row<367;row++)
        {
             //printf("initQ: row=%d",row);
             fscanf(pFile, "%lf", &Q14.Q14Cforcing(row,0));
             fscanf(pFile, "%lf", &Q14.Q14Cforcing(row,1));
             fscanf(pFile, "%lf", &Q14.Q14Cforcing(row,2));
             fscanf(pFile, "%lf", &Q14.Q14Cforcing(row,3));
             //printf("  initQ: year=%f, Q=%f\n", Q14.Q14Cforcing(row,0),Q14.Q14Cforcing(row,2));
        }
        fclose (pFile);
    }

    if (Q14.ExNo>0)
    {
        sprintf(filename,"14CPROD/QrecFILE/Qrec_GLOPIS_SCENARIO_%d.txt",Q14.ExNo);
        pFile = fopen (filename,"r");
        for(int row=0;row<367;row++)
        {
             //printf("initQ: row=%d",row);
             fscanf(pFile, "%lf", &Q14.Q14Cforcing(row,0));
             fscanf(pFile, "%lf", &Q14.Q14Cforcing(row,1));
             fscanf(pFile, "%lf", &Q14.Q14Cforcing(row,2));
             //printf("  initQ: year=%f, Q=%f\n", Q14.Q14Cforcing(row,0),Q14.Q14Cforcing(row,2));
        }
        fclose (pFile);
    }
}

void InitDGLforcing(DGLF& F, int year)
{
    LoadDGLforcing(F);

    F.F1.row = 15;
    F.F1.node = F.F1.Forcing(F.F1.row,0);
    F.F1.nextnode = F.F1.Forcing(F.F1.row-1,0);
    F.F1.D = F.F1.nextnode - F.F1.node;
    F.F1.Dt = -F.F1.ForceTime(F.F1.row-1,0) + F.F1.ForceTime(F.F1.row,0);
    F.F1.yrstep = F.F1.ForceTime(F.F1.row,0) - year;
    F.F1.init_true = 1;

    F.F2.row = 11;
    F.F2.node = F.F2.Forcing(F.F2.row,0);
    F.F2.nextnode = F.F2.Forcing(F.F2.row-1,0);
    F.F2.D = F.F2.nextnode - F.F2.node;
    F.F2.Dt = -F.F2.ForceTime(F.F2.row-1,0) + F.F2.ForceTime(F.F2.row,0);
    F.F2.yrstep = F.F2.ForceTime(F.F2.row,0) - year;
    F.F2.init_true = 1;

    F.F3.row = 11;
    F.F3.node = F.F3.Forcing(F.F3.row,0);
    F.F3.nextnode = F.F3.Forcing(F.F3.row-1,0);
    F.F3.D = F.F3.nextnode - F.F3.node;
    F.F3.Dt = -F.F3.ForceTime(F.F3.row-1,0) + F.F3.ForceTime(F.F3.row,0);
    F.F3.yrstep = F.F3.ForceTime(F.F3.row,0) - year;
    F.F3.init_true = 1;

    F.F4.row = 11;
    F.F4.node = F.F4.Forcing(F.F3.row,0);
    F.F4.nextnode = F.F4.Forcing(F.F4.row-1,0);
    F.F4.D = F.F4.nextnode - F.F4.node;
    F.F4.Dt = -F.F4.ForceTime(F.F4.row-1,0) + F.F4.ForceTime(F.F4.row,0);
    F.F4.yrstep = F.F4.ForceTime(F.F4.row,0) - year;
    F.F4.init_true = 1;

    F.init_true = 1;
}

void UpdateDGLforcing(DGLF& F, int year)
{
   double  Frac = -77;
    // FORCING 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (F.F1.yrstep == F.F1.Dt)
    {
        F.F1.row = F.F1.row -1;
        if (F.F1.row>=1)
        {
            F.F1.node = F.F1.nextnode;
            F.F1.nextnode = F.F1.Forcing(F.F1.row-1,0);
            F.F1.D = F.F1.nextnode - F.F1.node;
            F.F1.Dt = -F.F1.ForceTime(F.F1.row-1,0) + F.F1.ForceTime(F.F1.row,0);
            F.F1.yrstep = 0;
        }
        else if (F.F1.row==0)
        {
            F.F1.node = F.F1.nextnode;
            F.F1.D = 0;
            F.F1.Dt = -1;
            F.F1.yrstep = 0;
        }
    }
    F.F1.value = -77;
    if (F.F1.yrstep == 0)
    {
        F.F1.value = F.F1.node;// + F.F1.D*(F.F1.yrstep/F.F1.Dt);
    }
    F.F1.yrstep = F.F1.yrstep +1;


    // FORCING 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (F.F2.yrstep == F.F2.Dt)
    {
        F.F2.row = F.F2.row -1;
        if (F.F2.row>=1)
        {
            F.F2.node = F.F2.nextnode;
            F.F2.nextnode = F.F2.Forcing(F.F2.row-1,0);
            F.F2.D = F.F2.nextnode - F.F2.node;
            F.F2.Dt = -F.F2.ForceTime(F.F2.row-1,0) + F.F2.ForceTime(F.F2.row,0);
            F.F2.yrstep = 0;
        }
        else if (F.F2.row==0)
        {
            F.F2.node = F.F2.nextnode;
            F.F2.D = 0;
            F.F2.Dt = -1;
            F.F2.yrstep = 0;
        }
    }
    Frac = static_cast<double>(F.F2.yrstep)/static_cast<double>(F.F2.Dt);
    F.F2.value = F.F2.node + F.F2.D*(Frac);
    //if ((year<=19000)&& (year % 50 == 0)) {printf("oooo %d  -- %f - %f - %f - %d - %d - %f - %d\n ",year, F.F2.node, F.F2.value, F.F2.D, F.F2.yrstep, F.F2.Dt,Frac, F.F2.row);}
    F.F2.yrstep = F.F2.yrstep +1;

    // FORCING 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (F.F3.yrstep == F.F3.Dt)
    {
        F.F3.row = F.F3.row -1;
        if (F.F3.row>=1)
        {
            F.F3.node = F.F3.nextnode;
            F.F3.nextnode = F.F3.Forcing(F.F3.row-1,0);
            F.F3.D = F.F3.nextnode - F.F3.node;
            F.F3.Dt = -F.F3.ForceTime(F.F3.row-1,0) + F.F3.ForceTime(F.F3.row,0);
            F.F3.yrstep = 0;
        }
        else if (F.F3.row==0)
        {
            F.F3.node = F.F3.nextnode;
            F.F3.D = 0;
            F.F3.Dt = -1;
            F.F3.yrstep = 0;
        }
    }
    Frac = static_cast<double>(F.F3.yrstep)/static_cast<double>(F.F3.Dt);
    F.F3.value = F.F3.node + F.F3.D*(Frac);
    F.F3.yrstep = F.F3.yrstep +1;

    // FORCING 4 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (F.F4.yrstep == F.F4.Dt)
    {
        F.F4.row = F.F4.row -1;
        if (F.F4.row>=1)
        {
            F.F4.node = F.F4.nextnode;
            F.F4.nextnode = F.F4.Forcing(F.F4.row-1,0);
            F.F4.D = F.F4.nextnode - F.F4.node;
            F.F4.Dt = -F.F4.ForceTime(F.F4.row-1,0) + F.F4.ForceTime(F.F4.row,0);
            F.F4.yrstep = 0;
        }
        else if (F.F4.row==0)
        {
            F.F4.node = F.F4.nextnode;
            F.F4.D = 0;
            F.F4.Dt = -1;
            F.F4.yrstep = 0;
        }
    }

    if (F.F4.yrstep == 0)
    {
        F.F4.value = F.F4.node;// + F.F1.D*(F.F1.yrstep/F.F1.Dt);
        //printf("doneUpdate year=%d; value=%f; row=%d\n",year, F.F4.value, F.F4.row);
    }
    F.F4.yrstep = F.F4.yrstep +1;

}

void ChangeCirc(Matrix18d& CirculationM, int circID, int ex5, int ex6, int ex7)
{
	FILE * pFile;
	char circname[128] = "empty";
	//char  g [80];

	switch (circID)
	{
	    case 0: /*printf("using NADW\n");*/  pFile = fopen ("CIRCULATIONS/NADW_HainGBC2010_MYNADW2.txt","r"); break;
	    case 1: /*printf("using GNAIW\n");*/ pFile = fopen ("CIRCULATIONS/GNAIW_HainGBC2010_MYGNAIW5normIDmix.txt","r"); break;
	    case 98: /*printf("using HScirc\n");*/ sprintf(circname,"CIRCULATIONS/GNAIWslow/GNAIWslow_%d0.txt",ex5); pFile = fopen (circname,"r"); break;
	    case 99: sprintf(circname,"CIRCULATIONS/GNAIWmod/GNAIWmod_%d_%d_%d.txt",ex5,ex6,ex7); /*printf("using custom [%s]\n",circname);*/ pFile = fopen (circname,"r"); break;
	    default: printf("XXX CIRC-WARNING: ILLEGITIMATE CHOICE - default to NADW\n"); pFile = fopen ("CIRCULATIONS/NADW_HainGBC2010_MYNADW2.txt","r"); break;
	}

	for(int row=0;row<18;row++){ for(int col=0;col<18;col++){fscanf (pFile, "%lf",&CirculationM(row,col));}}
	fclose (pFile);
}

void InitVolSchemes(Matrix18d& MDtoDO, Matrix18d& DOtoMD, Matrix18d& MDtoDOviaNA, Matrix18d& DOtoMDviaAApLL, Matrix18d& UOtoDO, Matrix18d& DOtoUO, Matrix18d& NO)
{
	MDtoDO = Matrix18d::Zero();
	MDtoDO(14,8) = 0.2567;
	MDtoDO(15,9) = 0.206;
	MDtoDO(16,10) = 0.2388;
	MDtoDO(17,11) = 0.2985;
	MDtoDO = MDtoDO*1E6*(365*24*60*60);
	//printf("  init: valUP=%f  valDOWN=%f\n", MDtoDO(11,17),MDtoDO(17,11));

	DOtoMD = Matrix18d::Zero();
	DOtoMD(8,14) = 0.2567;
	DOtoMD(9,15) = 0.206;
	DOtoMD(10,16) = 0.2388;
	DOtoMD(11,17) = 0.2985;
	DOtoMD = DOtoMD*1E6*(365*24*60*60);
	//printf("  init: valUP=%f  valDOWN=%f\n", DOtoMD(11,17),DOtoMD(17,11));

    DOtoMDviaAApLL = Matrix18d::Zero();
	DOtoMDviaAApLL(13,14) = 0.2567;
	DOtoMDviaAApLL(13,15) = 0.206;
	DOtoMDviaAApLL(13,16) = 0.2388 + 0.2985;
	DOtoMDviaAApLL(16,17) = 0.2985;
	DOtoMDviaAApLL(5,13) = 1;
	DOtoMDviaAApLL(6,5) = 1;
	DOtoMDviaAApLL(8,6) = 0.2567;
	DOtoMDviaAApLL(9,6) = 0.206;
	DOtoMDviaAApLL(10,6) = 0.2388 + 0.2985;
	DOtoMDviaAApLL(11,10) = 0.2985;
	DOtoMDviaAApLL(8,0) = 0.2567;
	DOtoMDviaAApLL(9,1) = 0.206;
	DOtoMDviaAApLL(10,2) = 0.2388 ;
	DOtoMDviaAApLL(11,3) = 0.2985;
	DOtoMDviaAApLL(0,8) = 0.2567;
	DOtoMDviaAApLL(1,9) = 0.206;
	DOtoMDviaAApLL(2,10) = 0.2388 ;
	DOtoMDviaAApLL(3,11) = 0.2985;
	DOtoMDviaAApLL = DOtoMDviaAApLL*1E6*(365*24*60*60);

	MDtoDOviaNA = Matrix18d::Zero();
	MDtoDOviaNA(8,9) = 0.206;
	MDtoDOviaNA(8,10) = 0.2388 + 0.2985;
	MDtoDOviaNA(10,11) = 0.2985;
	MDtoDOviaNA(12,8) = 1;
	MDtoDOviaNA(14,12) = 1;
	MDtoDOviaNA(15,14) = 0.2388 + 0.2985 + 0.206;
	MDtoDOviaNA(16,15) = 0.2388 + 0.2985;
	MDtoDOviaNA(17,16) = 0.2985;
	MDtoDOviaNA = MDtoDOviaNA*1E6*(365*24*60*60);

	UOtoDO = Matrix18d::Zero();
	UOtoDO(14,8) = 0.2567;
	UOtoDO(8,0) = 0.2567/2;
	UOtoDO(15,9) = 0.206;
	UOtoDO(9,1) = 0.206/2;
	UOtoDO(16,10) = 0.2388;
	UOtoDO(10,2) = 0.2388/2;
	UOtoDO(17,11) = 0.2985;
	UOtoDO(11,3) = 0.2985/2;
	UOtoDO = UOtoDO*1E6*(365*24*60*60);
	//printf("  init: valUP=%f  valDOWN=%f\n", MDtoDO(11,17),MDtoDO(17,11));

	DOtoUO = Matrix18d::Zero();
	DOtoUO(8,14) = 0.2567;
	DOtoUO(0,8) = 0.2567/2;
	DOtoUO(9,15) = 0.206;
	DOtoUO(1,9) = 0.206/2;
	DOtoUO(10,16) = 0.2388;
	DOtoUO(2,10) = 0.2388/2;
	DOtoUO(11,17) = 0.2985;
	DOtoUO(3,11) = 0.2985/2;
	DOtoUO = DOtoUO*1E6*(365*24*60*60);
	//printf("  init: valUP=%f  valDOWN=%f\n", DOtoMD(11,17),DOtoMD(17,11));

	NO = Matrix18d::Zero();
}

double OcAvConc(Vector18f& tracer, Vector18f& vol, int top, int N)
{
   return (tracer.segment(top,N).cwiseProduct(vol.segment(top,N))).sum()/vol.segment(top,N).sum();
}

double OcAvDelta(Vector18f& delta, Vector18f& tracer, Vector18f& vol, int top, int N)
{
   return delta.segment(top,N).cwiseProduct(tracer.segment(top,N).cwiseProduct(vol.segment(top,N))).sum()/tracer.segment(top,N).cwiseProduct(vol.segment(top,N)).sum();
}

double OcAvDC(Vector18f& delta, Vector18f& tracer, Vector18f& vol, int top, int N)
{
   return delta.segment(top,N).cwiseProduct(vol.segment(top,N)).sum()/tracer.segment(top,N).cwiseProduct(vol.segment(top,N)).sum();
}

void ChangeVol(Matrix18d& SchemeUsed, Matrix18d& CircOp,tracerList& tracer, venttracerList& venttracer, Vector18f& vol, Vector18f& vol_inv, Vector18f& CtoN, Vector18f& NtoC, VolChange& params)
{
    //Vector18f test=Vector18f::Ones();
    //test(11)= 1;
    //test(17)= 1;
    //printf("%f  before: %f ---",params.scale, OcAvConc(test, vol, 0, 18));
    params.reducedvol = vol;
    for (int from=0;from<18;from++){for (int to=0;to<18;to++)  params.reducedvol(from) -= params.scale*SchemeUsed(to,from);}

    params.VolOp = params.scale*SchemeUsed;
    params.VolOp.diagonal()=params.reducedvol;
    params.newvol = params.VolOp * Vector18f::Ones();
    for (int row=0;row<18;row++){params.VolOp.row(row) = params.VolOp.row(row)/params.newvol(row);}

    for (int row=0;row<18;row++){CircOp.row(row) = CircOp.row(row)*vol(row);}
    CircOp.diagonal() = CircOp.diagonal() + (params.newvol-vol);
    for (int row=0;row<18;row++){CircOp.row(row) = CircOp.row(row)/params.newvol(row);}
    vol = params.newvol;
    //printf("%f  volTESTS %f ---",params.scale,vol.sum()-params.newvol.sum());
    vol_inv = vol.array().inverse();
    CtoN	=	vol*1024;
	NtoC	=	vol.array().inverse()/1024;

    tracer.P = params.VolOp*tracer.P;
    tracer.Preg = params.VolOp*tracer.Preg;
    tracer.C = params.VolOp*tracer.C;
    tracer.dc13 = params.VolOp*tracer.dc13;
    tracer.Alk = params.VolOp*tracer.Alk;
    tracer.Alkreg = params.VolOp*tracer.Alkreg;
    tracer.N = params.VolOp*tracer.N;
    tracer.Sal = params.VolOp*tracer.Sal;
    tracer.Temp = params.VolOp*tracer.Temp;
    tracer.Si = params.VolOp*tracer.Si;
    tracer.dc30 = params.VolOp*tracer.dc30;
    tracer.dc14 = params.VolOp*tracer.dc14;

    venttracer.vent.bottomRows(10)=params.VolOp.bottomRows(10)*venttracer.vent;
	venttracer.trueage.tail(10)=params.VolOp.bottomRows(10)*venttracer.trueage;
    venttracer.pref14Cage.tail(10)=params.VolOp.bottomRows(10)*venttracer.pref14Cage;
    //printf(" after: %f\n",OcAvConc(test, vol, 0, 18));

}

void MixCirc(Matrix18d& CirculationM, int circID, int ex5, int ex6, int ex7, double MixFrac)
{
	FILE * pFile1;
	FILE * pFile2;
	char circname[128] = "empty";
	double val1, val2;
	//char  g [80];

	switch (circID)
	{
	    case 10: sprintf(circname,"C:/Users/Mathis Hain/Desktop/CYCLOPS++_V1_silica/CYCLOPS++_V1_silica/GNAIWmod/GNAIWmod_%d_%d_%d.txt",ex5,ex6,ex7); pFile1 = fopen (circname,"r");pFile2 = fopen ("NADW_HainGBC2010_MYNADW2.txt","r"); break;
	    default: printf("XXX CIRC-WARNING: ILLEGITIMATE CHOICE - default to NADW \n"); pFile1 = fopen ("NADW_HainGBC2010_MYNADW2.txt","r");pFile2 = fopen ("NADW_HainGBC2010_MYNADW2.txt","r"); break;
	}

	for(int row=0;row<18;row++){ for(int col=0;col<18;col++){fscanf(pFile1, "%lf",&val1); fscanf(pFile2, "%lf",&val2); CirculationM(row,col) = MixFrac*val2 + (1-MixFrac)*val1;}}
	fclose (pFile1);fclose (pFile2);
}

void InitCirc(Matrix18d& circulationM, Vector18f& vol, Vector18f& vol_inv)
{
	Matrix18d temp;
	Vector18d self;
	//vol=vol*1.35E+18*0.01;					// PERCENT VOLUME TIMES TOTAL VOLUME IN METERS-CUBED
	circulationM=circulationM*1E6*(365*24*60*60);

	for (int i=0;i<18;i++)self(i)=vol(i);
	for (int i=0;i<18;i++){for (int j=0;j<18;j++)temp(i,j)=circulationM(i,j);}

	for (int row=0;row<18;row++){for (int col=0;col<18;col++)self(row)-=temp(row,col);}
	for (int row=0;row<18;row++)temp(row,row)=self(row);


	for (int row=0;row<18;row++)
	{
		for (int col=0; col<18; col++){circulationM(row,col)=temp(row,col)/vol(row);}
	}

	for (int row=0;row<18;row++)
	{
		vol_inv(row)=1/vol(row);
	}


	//cout << "this is the calculated circulation matrix:	\n"<<circulationM<<"\n";
}

void PAZmix(Matrix18d& circ, Vector18f& vol, double SVmix)
{
	SVmix = SVmix*1E6*(365*24*60*60);
    circ(7,7) += circ(7,13) - SVmix/vol(7);   circ(13,13) += circ(13,7) - SVmix/vol(13);
    circ(7,13) = SVmix/vol(7); circ(13,7) = SVmix/vol(13);
}

void PFZmix(Matrix18d& circ, Vector18f& vol, double SVmix)
{
	SVmix = SVmix*1E6*(365*24*60*60);
    circ(8,8) += circ(8,13) - SVmix/vol(8);
    circ(9,9) += circ(9,13) - SVmix/vol(9);
    circ(10,10) += circ(10,13) - SVmix/vol(10);
    circ(13,13) += circ(13,8)+circ(13,9)+circ(13,10) - 3*SVmix/vol(13);
    circ(8,13) = SVmix/vol(8); circ(13,8) = SVmix/vol(13);
    circ(9,13) = SVmix/vol(9); circ(13,9) = SVmix/vol(13);
    circ(10,13) = SVmix/vol(10); circ(13,10) = SVmix/vol(13);
}

void oAAmix(Matrix18d& circ, Vector18f& vol, double SVmix)
{
	SVmix = SVmix*1E6*(365*24*60*60);
    circ(5,5) += circ(5,13) - SVmix/vol(5);   circ(13,13) += circ(13,5) - SVmix/vol(13);
    circ(5,13) = SVmix/vol(5); circ(13,5) = SVmix/vol(13);
}

double D14Ccalc(double C, double dc13, double dc14)
{
   return 1000*((1+(dc14/C-1000)/1000)* 0.950625/(1+(dc13/C/1000))/(1+(dc13/C/1000)) -1);
}

void Circ(Matrix18d& circ, tracerList& tracer)
{
	tracer.P=circ*tracer.P;
	tracer.C=circ*tracer.C;
	tracer.dc13=circ*tracer.dc13;
	tracer.Alk=circ*tracer.Alk;
	tracer.N=circ*tracer.N;
	tracer.Si=circ*tracer.Si;
	tracer.dc30=circ*tracer.dc30;
	tracer.dc14=circ*tracer.dc14;

	//tracer.Temp=circ*tracer.Temp;
	//tracer.Sal=circ*tracer.Sal;
	tracer.Temp.tail(10)=circ.bottomRows(10)*tracer.Temp;
	tracer.Sal.tail(10)=circ.bottomRows(10)*tracer.Sal;
	tracer.Preg.tail(10)=circ.bottomRows(10)*tracer.Preg;
	tracer.Alkreg.tail(10)=circ.bottomRows(10)*tracer.Alkreg;


}

void VentTrack(Matrix18d& circ, venttracerList& venttracer, atmosphereS& atm, Vector8f C, Vector8f dc13, Vector8f dc14)
{
	venttracer.vent.bottomRows(10)=circ.bottomRows(10)*venttracer.vent;
	venttracer.trueage.tail(10)=circ.bottomRows(10)*venttracer.trueage;
	venttracer.trueage.tail(10)=venttracer.trueage.tail(10).array()+1;

    double atmD14C = D14Ccalc(atm.ppm, atm.dn13, atm.dn14);
    //   ((1000+1000*((1+(dc14.array()/C.array()-1000)/1000)* 0.950625/(1+(dc13.array()/C.array()/1000))/(1+(dc13.array()/C.array()/1000)) -1)).array()/(1000+atmD14C));
    venttracer.pref14Cage.head(8) = -5730* 0.69314718056 *  ((1000+1000*((1+(dc14.array()/C.array()-1000)/1000)* 0.950625/(1+(dc13.array()/C.array()/1000))/(1+(dc13.array()/C.array()/1000)) -1)).array()/(1000+atmD14C)).log();
    venttracer.pref14Cage.tail(10)=circ.bottomRows(10)*venttracer.pref14Cage;

}

void Prod(tracerList& tracer, Vector8f setP, Vector8f setSi, Vector8f CaRatio, rainList& rain, double alphaSi, Vector18f CtoN, ArVector8f const &ORGe)
{
	rain.d13Ccc	= tracer.dc13.head(8).array()/tracer.C.head(8).array();
	//d13Corg = d13Ccc - Vector8f::Constant(ORGe);      // WARNING: should be standard 20!!!!
	rain.d13Corg = rain.d13Ccc + ORGe.matrix();

    rain.d14Ccc	= tracer.dc14.head(8).array()/tracer.C.head(8).array();
	//d13Corg = d13Ccc - Vector8f::Constant(ORGe);      // WARNING: should be standard 20!!!!
	rain.d14Corg = rain.d14Ccc + 2*ORGe.matrix();


	//d14Ccc	= tracer.dc14.head(8).array()/tracer.C.head(8).array();
	//d14Corg = d14Ccc + 2*ORGe.matrix();

	for (int SurfBox=0;SurfBox<8;SurfBox++)
	{
		if (setP(SurfBox)<tracer.P(SurfBox)){
			rain.P(SurfBox)=(tracer.P(SurfBox)-setP(SurfBox));
			tracer.P(SurfBox)=setP(SurfBox);
		}
		else {
			rain.P(SurfBox)=0;
		}
	}

	for (int SurfBox=0;SurfBox<8;SurfBox++){
		tracer.C(SurfBox)	-=	106*rain.P(SurfBox);
		tracer.Alk(SurfBox)	-=	-16*rain.P(SurfBox);
		tracer.N(SurfBox)	-=	16*rain.P(SurfBox);
		tracer.dc13(SurfBox)	-=	106*rain.P(SurfBox)*rain.d13Corg(SurfBox);
		tracer.dc14(SurfBox)	-=	106*rain.P(SurfBox)*rain.d14Corg(SurfBox);
	}

//	rainCa=rainP*106*CaRatio;
	rain.Ca=106*rain.P.cwiseProduct(CaRatio);

	for (int SurfBox=0;SurfBox<8;SurfBox++){
		tracer.dc13(SurfBox)	-=	rain.Ca(SurfBox)*rain.d13Ccc(SurfBox);
		tracer.dc14(SurfBox)	-=	rain.Ca(SurfBox)*rain.d14Ccc(SurfBox);
		tracer.C(SurfBox)	-=	rain.Ca(SurfBox);
		tracer.Alk(SurfBox)	-=	2*rain.Ca(SurfBox);
	}
	rain.P	=	rain.P.cwiseProduct(CtoN.head(8));
	rain.Ca	=	rain.Ca.cwiseProduct(CtoN.head(8));


    double newdelta;
	for (int SurfBox=0;SurfBox<8;SurfBox++)
	{
		if (setSi(SurfBox)<tracer.Si(SurfBox)){
			//d30Si(SurfBox) = (tracer.dc30(SurfBox)/tracer.Si(SurfBox)+1000)  *  ((1-pow(setSi(SurfBox)/tracer.Si(SurfBox),alphaSi-1))/(1-(setSi(SurfBox)/tracer.Si(SurfBox)))) -1000;
			//newdelta = (tracer.dc30(SurfBox)/tracer.Si(SurfBox)+1000) * pow(setSi(SurfBox)/tracer.Si(SurfBox),alphaSi-1) -1000;
			newdelta = (tracer.dc30(SurfBox)/tracer.Si(SurfBox)) + 1000*(alphaSi-1)*log(setSi(SurfBox)/tracer.Si(SurfBox));
			if (setSi(SurfBox)<0.0001) {newdelta=0;}
			rain.d30Si(SurfBox) = (tracer.dc30(SurfBox) - newdelta*setSi(SurfBox))/(tracer.Si(SurfBox)-setSi(SurfBox));
			tracer.dc30(SurfBox)	=	newdelta*setSi(SurfBox);
			rain.Si(SurfBox)=(tracer.Si(SurfBox)-setSi(SurfBox));
			tracer.Si(SurfBox)=setSi(SurfBox);
		}
		else {
			rain.Si(SurfBox)=0;
		}
	}
    rain.Si	=	rain.Si.cwiseProduct(CtoN.head(8));
}

void Remin(tracerList& tracer, Matrix108f RainOrg, Matrix108f RainSi, rainList rain, Vector18f NtoC)
{
	Vector10f addOrg;
	addOrg=(RainOrg*rain.P);

	for (int Box=8;Box<18;Box++){tracer.P(Box)+=	addOrg(Box-8)		*	NtoC(Box);}		// the 976-factor converts m^3 to kg and mol to µM
	for (int Box=8;Box<18;Box++){tracer.Preg(Box)+=	addOrg(Box-8)		*	NtoC(Box);}
	for (int Box=8;Box<18;Box++){tracer.C(Box)+=	106*addOrg(Box-8)	*	NtoC(Box);}
	for (int Box=8;Box<18;Box++){tracer.Alk(Box)+=	-16*addOrg(Box-8)	*	NtoC(Box);}
	for (int Box=8;Box<18;Box++){tracer.N(Box)+=	16*addOrg(Box-8)	*	NtoC(Box);}

	addOrg=RainOrg*(rain.P.cwiseProduct(rain.d13Corg));
	for (int Box=8;Box<18;Box++){tracer.dc13(Box)+=	106*addOrg(Box-8)	*	NtoC(Box);}

	addOrg=RainOrg*(rain.P.cwiseProduct(rain.d14Corg));
	for (int Box=8;Box<18;Box++){tracer.dc14(Box)+=	106*addOrg(Box-8)	*	NtoC(Box);}

	addOrg=(RainSi*rain.Si);
	for (int Box=8;Box<18;Box++){tracer.Si(Box)+=	addOrg(Box-8)		*	NtoC(Box);}

	addOrg=RainSi*(rain.Si.cwiseProduct(rain.d30Si));
	for (int Box=8;Box<18;Box++){tracer.dc30(Box)+=	addOrg(Box-8)	*	NtoC(Box);}
}

void River(Vector18f& C, Vector18f& dc13, Vector18f& dc14, Vector18f& A,Vector4d NtoC, double& Appm, double& dn13, double& dn14, double WeathX, double RivX, double const SetCO2)
{
	double CCflux	=	1.6 * 1e19 *RivX;                                 //0.875*0.65e+19;
	double SWflux	=	0.2 * 1e19 *(Appm/250)*WeathX;              //0.5*1.67e+19*(Appm/280)/WeathX;

	dn13	        =	dn13 / (Appm) * (Appm-(2*SWflux)/1.773e+20);
	dn14	        =	dn14 / (Appm) * (Appm-(2*SWflux)/1.773e+20);
	Appm            -= (2*SWflux)/1.773e+20;

	C.head(4)		+= 0.25* (CCflux+2*SWflux)*NtoC;
	A.head(4)		+= 0.25* 2*(CCflux+SWflux)*NtoC;
	dc13.head(4)	+= 0.25* dn13/Appm * (2*SWflux)*NtoC;
	dc14.head(4)	+= 0.25* dn14/Appm * (2*SWflux)*NtoC;
	//dc13.head(4)	+= 0.25* (0) * (CCflux)*NtoC; // not needed for dc13 because d13C of dissolving CaCO3 is assumed to equal zero
	//dc14.head(4)	+= 0.25* (0) * (CCflux)*NtoC; // not needed for dc13 because d14C of dissolving CaCO3 is assumed to equal zero
	dc13.head(4)	+= 0.25* (3) * (CCflux)*NtoC; // pre-PETM ocean d13C deep=+1 LLsurf=+3, hence input d13C=3
}

void Dissolve(tracerList& tracer, Matrix108f RainCC, Vector8f& rainCa, Vector8f d13Ccc, Vector8f d14Ccc, Vector18f& NtoC, Vector4f Fdiss)
{
	Vector10f addCC;

	addCC=(RainCC*rainCa);
	addCC(6) = addCC(6)*Fdiss(0);	addCC(7) = addCC(7)*Fdiss(1);	addCC(8) = addCC(8)*Fdiss(2); addCC(9) = addCC(9)*Fdiss(3);
	for (int box=8; box<18; box++) {	tracer.C(box)+=	addCC(box-8)	*	NtoC(box);	tracer.Alk(box)+=	2*addCC(box-8)	*	NtoC(box); tracer.Alkreg(box)+=	2*addCC(box-8)	*	NtoC(box);}

	addCC=RainCC*(rainCa.cwiseProduct(d13Ccc));
	addCC(6) = addCC(6)*Fdiss(0);	addCC(7) = addCC(7)*Fdiss(1);	addCC(8) = addCC(8)*Fdiss(2); addCC(9) = addCC(9)*Fdiss(3);
	for (int box=8; box<18; box++) {	tracer.dc13(box)+=	addCC(box-8)	*	NtoC(box); }

	addCC=RainCC*(rainCa.cwiseProduct(d14Ccc));
	addCC(6) = addCC(6)*Fdiss(0);	addCC(7) = addCC(7)*Fdiss(1);	addCC(8) = addCC(8)*Fdiss(2); addCC(9) = addCC(9)*Fdiss(3);
	for (int box=8; box<18; box++) {	tracer.dc14(box)+=	addCC(box-8)	*	NtoC(box); }
}

void Kcalc(Ktable& Ksurf, Vector18f TempV,Vector18f Sal,Vector18f top,Vector18f bot, seafloor& SF)
{
	ArVector18f T=TempV.array()+273.15;
	ArVector18f Temp=TempV.array();
	ArVector18f Tinv=T.inverse();
	ArVector18f S=Sal.array();
	ArVector18f Srt=S.sqrt();
	ArVector18f  dV,dk;

	// note that K's below are calculated for all 18 boxes per ARRAY instruction syntax
	Ksurf.K0 =	exp(-60.2409 + 9345.17*Tinv + 23.3585*log(T/100) + S*( 0.023517 - 2.3656e-4*T + 4.7036e-7*T*T)) ;
	Ksurf.K1 =	1e6 * exp( (62.008 - 3670.7*Tinv - 9.7944*log(T) + 0.0118*S - 1.16e-4*S*S)*log(10));
	Ksurf.K2 =	1e6 * exp((-4.777 -1394.7*Tinv + 0.0184*S - 1.18e-4*S*S)*log(10));
	Ksurf.Kb =	1e6 * exp(Tinv*(-8966.9 - 2890.53*Srt - 77.942*S + 1.728*S*Srt - 0.0996*S*S)  + 148.0248 +  137.1942*Srt + 1.62142*S + 0.053105*T*Srt + log(T)*(-24.4344 - 25.085*Srt - 0.2474*S));
	Ksurf.Ks =	1e12 * exp(-395.8293 + 6537.773*Tinv + 71.595*T.log() - 0.17959*T + Srt*(-1.78938 + 410.64*Tinv + 0.0065453*T) -0.17755*S + 0.0094979*S*Srt);

	SF.NCW.K1.matrix().head(18) = Ksurf.K1(12)* exp(((+2420 - 8500 * Temp(12)) * SF.SFdepth) / (166286 * T(12)));		// note that SFdepth is in [km], not [m], such that 8.5 becomes 8500
	SF.DSO.K1.matrix().head(18)  = Ksurf.K1(13)* exp(((+2420 - 8500 * Temp(13)) * SF.SFdepth) / (166286 * T(13)));		// note that (2 * 1e+3 * 83.143)=const.=166286
	SF.Atl.K1.matrix().head(18)  = Ksurf.K1(14)* exp(((+2420 - 8500 * Temp(14)) * SF.SFdepth) / (166286 * T(14)));
	SF.Ind.K1.matrix().head(18)  = Ksurf.K1(15)* exp(((+2420 - 8500 * Temp(15)) * SF.SFdepth) / (166286 * T(15)));
	SF.SPac.K1.matrix().head(18)  = Ksurf.K1(16)* exp(((+2420 - 8500 * Temp(16)) * SF.SFdepth) / (166286 * T(16)));
	SF.NPac.K1.matrix().head(18)  = Ksurf.K1(17)* exp(((+2420 - 8500 * Temp(17)) * SF.SFdepth) / (166286 * T(17)));

	SF.NCW.K2.matrix().head(18)  = Ksurf.K2(12)* exp(((1640 - 4000 * Temp(12)) * SF.SFdepth) / (166286 * T(12)));		// note that SFdepth is in [km], not [m], such that 4 becomes 4000
	SF.DSO.K2.matrix().head(18)  = Ksurf.K2(13)* exp(((1640 - 4000 * Temp(13)) * SF.SFdepth) / (166286 * T(13)));		// note that (2 * 1e+3 * 83.143)=const.=166286
	SF.Atl.K2.matrix().head(18)  = Ksurf.K2(14)* exp(((1640 - 4000 * Temp(14)) * SF.SFdepth) / (166286 * T(14)));
	SF.Ind.K2.matrix().head(18)  = Ksurf.K2(15)* exp(((1640 - 4000 * Temp(15)) * SF.SFdepth) / (166286 * T(15)));
	SF.SPac.K2.matrix().head(18)  = Ksurf.K2(16)* exp(((1640 - 4000 * Temp(16)) * SF.SFdepth) / (166286 * T(16)));
	SF.NPac.K2.matrix().head(18)  = Ksurf.K2(17)* exp(((1640 - 4000 * Temp(17)) * SF.SFdepth) / (166286 * T(17)));

	SF.NCW.Kb.matrix().head(18)  = Ksurf.Kb(12)* exp(((2750 - 9500 * Temp(12)) * SF.SFdepth) / (166286 * T(12)));		// note that SFdepth is in [km], not [m], such that 9.5 becomes 9500
	SF.DSO.Kb.matrix().head(18)  = Ksurf.Kb(13)* exp(((2750 - 9500 * Temp(13)) * SF.SFdepth) / (166286 * T(13)));		// note that (2 * 1e+3 * 83.143)=const.=166286
	SF.Atl.Kb.matrix().head(18)  = Ksurf.Kb(14)* exp(((2750 - 9500 * Temp(14)) * SF.SFdepth) / (166286 * T(14)));
	SF.Ind.Kb.matrix().head(18)  = Ksurf.Kb(15)* exp(((2750 - 9500 * Temp(15)) * SF.SFdepth) / (166286 * T(15)));
	SF.SPac.Kb.matrix().head(18)  = Ksurf.Kb(16)* exp(((2750 - 9500 * Temp(16)) * SF.SFdepth) / (166286 * T(16)));
	SF.NPac.Kb.matrix().head(18)  = Ksurf.Kb(17)* exp(((2750 - 9500 * Temp(17)) * SF.SFdepth) / (166286 * T(17)));

	dV= -65.28 - 0.397*Temp - 5.155e-3*Temp.pow(2) + (19.816 - 4.41e-2*Temp - 1.7e-4*Temp.pow(2))*(S/35).sqrt();
	dk = 1.847e-2 + 1.956e-4*Temp - 2.212e-6*Temp.pow(2) + (-3.217e-2 - 7.11e-5*Temp + 2.212e-6*Temp.pow(2))*(S/35).sqrt();

	//Ks = Ks + 0.0077*(Layer[j].LDepth*1.0E+3) + 1.524e-6*(sqr(Layer[j].LDepth*1.0E+3);


	SF.NCW.Ks.matrix().head(18) 	= Ksurf.Ks(12)* exp(-dV*1.202747e-2*Tinv(12)*(1+(SF.SFdepth*100)) + 0.5*dk*1.202747e-2*Tinv(12)*(1+(SF.SFdepth*100))*(1+(SF.SFdepth*100)));		// note that SFdepth is in [km], not [m], such that 9.5 becomes 9500
	SF.DSO.Ks.matrix().head(18) 	= Ksurf.Ks(13)* exp(-dV*1.202747e-2*Tinv(13)*(1+(SF.SFdepth*100)) + 0.5*dk*1.202747e-2*Tinv(13)*(1+(SF.SFdepth*100))*(1+(SF.SFdepth*100)));
	SF.Atl.Ks.matrix().head(18) 	= Ksurf.Ks(14)* exp(-dV*1.202747e-2*Tinv(14)*(1+(SF.SFdepth*100)) + 0.5*dk*1.202747e-2*Tinv(14)*(1+(SF.SFdepth*100))*(1+(SF.SFdepth*100)));
	SF.Ind.Ks.matrix().head(18) 	= Ksurf.Ks(15)* exp(-dV*1.202747e-2*Tinv(15)*(1+(SF.SFdepth*100)) + 0.5*dk*1.202747e-2*Tinv(15)*(1+(SF.SFdepth*100))*(1+(SF.SFdepth*100)));
	SF.SPac.Ks.matrix().head(18) 	= Ksurf.Ks(16)* exp(-dV*1.202747e-2*Tinv(16)*(1+(SF.SFdepth*100)) + 0.5*dk*1.202747e-2*Tinv(16)*(1+(SF.SFdepth*100))*(1+(SF.SFdepth*100)));
	SF.NPac.Ks.matrix().head(18) 	= Ksurf.Ks(17)* exp(-dV*1.202747e-2*Tinv(17)*(1+(SF.SFdepth*100)) + 0.5*dk*1.202747e-2*Tinv(17)*(1+(SF.SFdepth*100))*(1+(SF.SFdepth*100)));

	//cout << Ksurf.Ks(12)<<" this & above is the KsSurf of NCW\n";
	//cout << -dV*1.202747e-2*Tinv(12)*(1+(SF.SFdepth*100)) <<" this  is the dV-Ks-factor of NCW\n";
	//cout << 0.5*dk*1.202747e-2*Tinv(12)*(1+(SF.SFdepth*100))*(1+(SF.SFdepth*100)) <<" this  is the dk-Ks-factor of NCW\n";
	//cout << SF.NCW.Ks<<" this  is the Ks of NCW\n";
	//cout << (1+(SF.SFdepth*100))<<" this & above is the Ks of NCW\n";
}

void SFpHcalcKernel(double Sal,double C, double A, double Ca, ArVector20f const &K1,ArVector20f const &K2, ArVector20f const &Kb,ArVector20f const &Ks, ArVector20f &Hs, ArVector20f &CO3situ, ArVector20f &omega)
{

	ArVector20f Hx		= Hs*1.01;
	ArVector20f K1inv	= K1.inverse();

	ArVector20f tmpA1	= C*(Hs*K1inv + 1 + K2*Hs.inverse()).inverse()  +  2*K2*C*(Hs*Hs*K1inv + Hs + K2).inverse() + Kb*(12.12255*Sal)*(Hs+Kb).inverse() - Hs + 1e-2*Hs.inverse();
	ArVector20f tmpA2	= C*(Hx*K1inv + 1 + K2*Hx.inverse()).inverse()  +  2*K2*C*(Hx*Hx*K1inv + Hx + K2).inverse() + Kb*(12.12255*Sal)*(Hx+Kb).inverse() - Hx + 1e-2*Hx.inverse();

	Hs		=	Hs*exp(0.01*(tmpA1-A)/(tmpA1-tmpA2));
	CO3situ	=	K2*C*(Hs*Hs*K1inv + Hs + K2).inverse();
	omega	=	Ca*CO3situ/Ks;

}

void mySQRT(ArVector18f const &X, ArVector18f &Y)
{
    Y = (X+1)*0.5;
    Y = (Y+X/Y)*0.5;
    Y = (Y+X/Y)*0.5;
    Y = (Y+X/Y)*0.5;
}

void SFpHcalc(Vector18f const &S, Vector18f const &C, Vector18f const &A, seafloor& SF, double const CaX, double const DissolveX)
{
	double Ca; Ca = CaX*10600;
	SFpHcalcKernel(S(12), C(12), A(12), Ca, SF.NCW.K1, SF.NCW.K2, SF.NCW.Kb, SF.NCW.Ks, SF.NCW.Hsitu, SF.NCW.CO3situ, SF.NCW.omega);
	//SFpHcalcKernel(S(13), C(13), A(13), Ca, SF.DSO.K1, SF.DSO.K2, SF.DSO.Kb, SF.DSO.Ks, SF.DSO.Hsitu, SF.DSO.CO3situ, SF.DSO.omega);
	SFpHcalcKernel(S(14), C(14), A(14), Ca, SF.Atl.K1, SF.Atl.K2, SF.Atl.Kb, SF.Atl.Ks, SF.Atl.Hsitu, SF.Atl.CO3situ, SF.Atl.omega);
	SFpHcalcKernel(S(15), C(15), A(15), Ca, SF.Ind.K1, SF.Ind.K2, SF.Ind.Kb, SF.Ind.Ks, SF.Ind.Hsitu, SF.Ind.CO3situ, SF.Ind.omega);
	SFpHcalcKernel(S(16), C(16), A(16), Ca, SF.SPac.K1, SF.SPac.K2, SF.SPac.Kb, SF.SPac.Ks, SF.SPac.Hsitu, SF.SPac.CO3situ, SF.SPac.omega);
	SFpHcalcKernel(S(17), C(17), A(17), Ca, SF.NPac.K1, SF.NPac.K2, SF.NPac.Kb, SF.NPac.Ks, SF.NPac.Hsitu, SF.NPac.CO3situ, SF.NPac.omega);

	ArVector18f Fdiss;

	if (false)
	{
		for (int ly = 0; ly<18; ly++)
		{
			if (SF.Atl.omega(ly)>1) {SF.Atl.omega(ly)=1;}
			if (SF.Ind.omega(ly)>1) {SF.Ind.omega(ly)=1;}
			if (SF.SPac.omega(ly)>1) {SF.SPac.omega(ly)=1;}
			if (SF.NPac.omega(ly)>1) {SF.NPac.omega(ly)=1;}
		}

		Fdiss				=	20*SF.Atl.FCa*(1-SF.Atl.omega.matrix().head(18).array());
		if (1<Fdiss.maxCoeff()) {for (int ly = 0; ly<18; ly++){if (Fdiss(ly)>1) {Fdiss(ly)=1;};};}
		SF.Fdiss(0)			=	(SF.Atl.FArea*Fdiss).sum();
		SF.Atl.FCa			=	0.99*(SF.Atl.FCa) + 0.01*( 0.9*(1-Fdiss) / (1 - 0.9*Fdiss) );


		Fdiss				=	20*SF.Ind.FCa*(1-SF.Ind.omega.matrix().head(18).array());
		if (1<Fdiss.maxCoeff()) {for (int ly = 0; ly<18; ly++){if (Fdiss(ly)>1) {Fdiss(ly)=1;};};}
		SF.Fdiss(1)			=	(SF.Ind.FArea*Fdiss).sum();
		SF.Ind.FCa			=	0.99*(SF.Ind.FCa) + 0.01*( 0.9*(1-Fdiss) / (1 - 0.9*Fdiss) );

		Fdiss				=	20*SF.SPac.FCa*(1-SF.SPac.omega.matrix().head(18).array());
		if (1<Fdiss.maxCoeff()) {for (int ly = 0; ly<18; ly++){if (Fdiss(ly)>1) {Fdiss(ly)=1;};};}
		SF.Fdiss(2)			=	(SF.SPac.FArea*Fdiss).sum();
		SF.SPac.FCa			=	0.99*(SF.SPac.FCa) + 0.01*( 0.9*(1-Fdiss) / (1 - 0.9*Fdiss) );

		Fdiss				=	20*SF.NPac.FCa*(1-SF.NPac.omega.matrix().head(18).array());
		if (1<Fdiss.maxCoeff()) {for (int ly = 0; ly<18; ly++){if (Fdiss(ly)>1) {Fdiss(ly)=1;};};}
		SF.Fdiss(3)			=	(SF.NPac.FArea*Fdiss).sum();
		SF.NPac.FCa			=	0.99*(SF.NPac.FCa) + 0.01*( 0.9*(1-Fdiss) / (1 - 0.9*Fdiss) );
	}

	if (true)
	{
	    ArVector18f rootFCa;
	    //ArVector18f testIN,testOUT;

		for (int ly = 0; ly<18; ly++)
		{
			if (SF.Atl.omega(ly)>1) {SF.Atl.omega(ly)=1;}
			if (SF.Ind.omega(ly)>1) {SF.Ind.omega(ly)=1;}
			if (SF.SPac.omega(ly)>1) {SF.SPac.omega(ly)=1;}
			if (SF.NPac.omega(ly)>1) {SF.NPac.omega(ly)=1;}
		}

        //testOUT =   (((SF.Atl.FCa+1)/2)+ SF.Atl.FCa/((SF.Atl.FCa+1)/2))/2;

        mySQRT(SF.Ind.FCa,rootFCa);
		Fdiss				=	300*DissolveX*rootFCa*(1-SF.Atl.omega.matrix().head(18).array());
		if (1<Fdiss.maxCoeff()) {for (int ly = 0; ly<18; ly++){if (Fdiss(ly)>1) {Fdiss(ly)=1;};};}
		SF.Fdiss(0)			=	(SF.Atl.FArea*Fdiss).sum();
		SF.Atl.FCa			=	0.999*(SF.Atl.FCa) + 0.001*( 0.9*(1-Fdiss) / (1 - 0.9*Fdiss) );


		mySQRT(SF.SPac.FCa,rootFCa);
		Fdiss				=	300*DissolveX*rootFCa*(1-SF.Ind.omega.matrix().head(18).array());
		if (1<Fdiss.maxCoeff()) {for (int ly = 0; ly<18; ly++){if (Fdiss(ly)>1) {Fdiss(ly)=1;};};}
		SF.Fdiss(1)			=	(SF.Ind.FArea*Fdiss).sum();
		SF.Ind.FCa			=	0.999*(SF.Ind.FCa) + 0.001*( 0.9*(1-Fdiss) / (1 - 0.9*Fdiss) );

		mySQRT(SF.NPac.FCa,rootFCa);
		Fdiss				=	300*DissolveX*rootFCa*(1-SF.SPac.omega.matrix().head(18).array());
		if (1<Fdiss.maxCoeff()) {for (int ly = 0; ly<18; ly++){if (Fdiss(ly)>1) {Fdiss(ly)=1;};};}
		SF.Fdiss(2)			=	(SF.SPac.FArea*Fdiss).sum();
		SF.SPac.FCa			=	0.999*(SF.SPac.FCa) + 0.001*( 0.9*(1-Fdiss) / (1 - 0.9*Fdiss) );

		mySQRT(SF.Atl.FCa,rootFCa);
		Fdiss				=	300*DissolveX*rootFCa*(1-SF.NPac.omega.matrix().head(18).array());
		if (1<Fdiss.maxCoeff()) {for (int ly = 0; ly<18; ly++){if (Fdiss(ly)>1) {Fdiss(ly)=1;};};}
		SF.Fdiss(3)			=	(SF.NPac.FArea*Fdiss).sum();
		SF.NPac.FCa			=	0.999*(SF.NPac.FCa) + 0.001*( 0.9*(1-Fdiss) / (1 - 0.9*Fdiss) );
	}

	if (false)
	{
		double Kdis=7.5;
		for (int ly = 0; ly<18; ly++)
		{
			if (SF.Atl.omega(ly)>1) {SF.Atl.omega(ly)=1;}	if (SF.Atl.omega(ly)<(Kdis-1)/Kdis) {SF.Atl.omega(ly)=(Kdis-1)/Kdis;}
			if (SF.Ind.omega(ly)>1) {SF.Ind.omega(ly)=1;}	if (SF.Ind.omega(ly)<(Kdis-1)/Kdis) {SF.Ind.omega(ly)=(Kdis-1)/Kdis;}
			if (SF.SPac.omega(ly)>1) {SF.SPac.omega(ly)=1;}	if (SF.SPac.omega(ly)<(Kdis-1)/Kdis) {SF.SPac.omega(ly)=(Kdis-1)/Kdis;}
			if (SF.NPac.omega(ly)>1) {SF.NPac.omega(ly)=1;}	if (SF.NPac.omega(ly)<(Kdis-1)/Kdis) {SF.NPac.omega(ly)=(Kdis-1)/Kdis;}
		}

		Fdiss				=	SF.Atl.FArea*7.5*(1-SF.Atl.omega.matrix().head(18).array());
		SF.Fdiss(0)			=	Fdiss.sum();
		SF.Atl.FCa			=	0.99*(SF.Atl.FCa) + 0.01*( 0.9*(1-Fdiss) / (1 - 0.9*Fdiss) );

		Fdiss				=	SF.Ind.FArea*7.5*(1-SF.Ind.omega.matrix().head(18).array());
		SF.Fdiss(1)			=	Fdiss.sum();
		SF.Ind.FCa			=	0.99*(SF.Ind.FCa) + 0.01*( 0.9*(1-Fdiss) / (1 - 0.9*Fdiss) );

		Fdiss				=	SF.SPac.FArea*7.5*(1-SF.SPac.omega.matrix().head(18).array());
		SF.Fdiss(2)			=	Fdiss.sum();
		SF.SPac.FCa			=	0.99*(SF.SPac.FCa) + 0.01*( 0.9*(1-Fdiss) / (1 - 0.9*Fdiss) );

		Fdiss				=	SF.NPac.FArea*7.5*(1-SF.NPac.omega.matrix().head(18).array());
		SF.Fdiss(3)			=	Fdiss.sum();
		SF.NPac.FCa			=	0.99*(SF.NPac.FCa) + 0.01*( 0.9*(1-Fdiss) / (1 - 0.9*Fdiss) );
	}


	//cout<<"omega: "<<SF.NPac.omega(15)<<" Fdiss: "<<Fdiss(15)<<" "<<SF.Fdiss(0)<<" "<<SF.Fdiss(1)<<" "<<SF.Fdiss(2)<<" "<<SF.Fdiss(3)<<endl;
}

void CO2find(ArVector8f const &C, ArVector8f const &A, ArVector8f const &Sal, ArVector8f const &K0, ArVector8f const &K1, ArVector8f const &K2, ArVector8f const &Kb, ArVector8f& Hs, ArVector8f& H2CO3)
{
	ArVector8f Hx = Hs*1.02;
	ArVector8f K1inv	= K1.inverse();

	//cout << Hs(1)<<" Hatl in ;; Alk = "<<A(1)<<" \n";

	ArVector8f tmpA1	= C*(Hs*K1inv + 1 + K2*Hs.inverse()).inverse()  +  2*K2*C*(Hs*Hs*K1inv + Hs + K2).inverse() + Kb*(12.12255*Sal)*(Hs+Kb).inverse() - Hs + 1e-2*Hs.inverse();
	ArVector8f tmpA2	= C*(Hx*K1inv + 1 + K2*Hx.inverse()).inverse()  +  2*K2*C*(Hx*Hx*K1inv + Hx + K2).inverse() + Kb*(12.12255*Sal)*(Hx+Kb).inverse() - Hx + 1e-2*Hx.inverse();

	Hs		=	Hs*exp(0.02*(tmpA1-A)/(tmpA1-tmpA2));
	//tmpA1	= C*(Hs*K1inv + 1 + K2*Hs.inverse()).inverse()  +  2*K2*C*(Hs*Hs*K1inv + Hs + K2).inverse() + Kb*(12.12255*Sal)*(Hs+Kb).inverse() - Hs + 1e-2*Hs.inverse();   //not needed

	H2CO3		=	C* Hs*Hs/(Hs*Hs + K1*Hs + K1*K2);
	//cout << Hs(1)<<" Hatl in ;; AlkOUT = "<<tmpA1(1)<<"  ;; H2CO3= "<<H2CO3(1)/K0(1)<<" \n\n";
}

void finalCsolve(ArVector18f const &C, ArVector18f const &A, ArVector18f const &Sal, ArVector18f const &K0, ArVector18f const &K1, ArVector18f const &K2, ArVector18f const &Kb, ArVector18f const &Ks, double const &CaX, Cchem& Csolved)
{
	double Ca; Ca = CaX*10600;
	ArVector18f Hx,Hs,tmpA1, tmpA2, K1inv;
	K1inv	=	K1.inverse();
	Hs		=	ArVector18f::Constant(0.005);

	Hx		=	Hs*1.3;
	tmpA1	=	C*(Hs*K1inv + 1 + K2*Hs.inverse()).inverse()  +  2*K2*C*(Hs*Hs*K1inv + Hs + K2).inverse() + Kb*(12.12255*Sal)*(Hs+Kb).inverse() - Hs + 1e-2*Hs.inverse();
	tmpA2	=	C*(Hx*K1inv + 1 + K2*Hx.inverse()).inverse()  +  2*K2*C*(Hx*Hx*K1inv + Hx + K2).inverse() + Kb*(12.12255*Sal)*(Hx+Kb).inverse() - Hx + 1e-2*Hx.inverse();
	Hs		=	Hs*exp(0.3*(tmpA1-A)/(tmpA1-tmpA2));

	Hx		=	Hs*1.1;
	tmpA1	=	C*(Hs*K1inv + 1 + K2*Hs.inverse()).inverse()  +  2*K2*C*(Hs*Hs*K1inv + Hs + K2).inverse() + Kb*(12.12255*Sal)*(Hs+Kb).inverse() - Hs + 1e-2*Hs.inverse();
	tmpA2	=	C*(Hx*K1inv + 1 + K2*Hx.inverse()).inverse()  +  2*K2*C*(Hx*Hx*K1inv + Hx + K2).inverse() + Kb*(12.12255*Sal)*(Hx+Kb).inverse() - Hx + 1e-2*Hx.inverse();
	Hs		=	Hs*exp(0.1*(tmpA1-A)/(tmpA1-tmpA2));

	Hx		=	Hs*1.03;
	tmpA1	=	C*(Hs*K1inv + 1 + K2*Hs.inverse()).inverse()  +  2*K2*C*(Hs*Hs*K1inv + Hs + K2).inverse() + Kb*(12.12255*Sal)*(Hs+Kb).inverse() - Hs + 1e-2*Hs.inverse();
	tmpA2	=	C*(Hx*K1inv + 1 + K2*Hx.inverse()).inverse()  +  2*K2*C*(Hx*Hx*K1inv + Hx + K2).inverse() + Kb*(12.12255*Sal)*(Hx+Kb).inverse() - Hx + 1e-2*Hx.inverse();
	Hs		=	Hs*exp(0.03*(tmpA1-A)/(tmpA1-tmpA2));

	Hx		=	Hs*1.01;
	tmpA1	=	C*(Hs*K1inv + 1 + K2*Hs.inverse()).inverse()  +  2*K2*C*(Hs*Hs*K1inv + Hs + K2).inverse() + Kb*(12.12255*Sal)*(Hs+Kb).inverse() - Hs + 1e-2*Hs.inverse();
	tmpA2	=	C*(Hx*K1inv + 1 + K2*Hx.inverse()).inverse()  +  2*K2*C*(Hx*Hx*K1inv + Hx + K2).inverse() + Kb*(12.12255*Sal)*(Hx+Kb).inverse() - Hx + 1e-2*Hx.inverse();
	Hs		=	Hs*exp(0.01*(tmpA1-A)/(tmpA1-tmpA2));

	////////////////////////////////////////////////////
	Csolved.H		=	Hs;
	Csolved.H2CO3	=	C* Hs*Hs/(Hs*Hs + K1*Hs + K1*K2);
	Csolved.pCO2	=	Csolved.H2CO3/K0;
	Csolved.HCO3	=	C/(Hs*K1inv + 1 + K2/Hs);
	Csolved.CO3		=	C/(Hs*Hs*K1inv/K2 + Hs/K2 + 1);
	Csolved.omega	=	Csolved.CO3*Ca/Ks;
	Csolved.BOH4	=	Kb*(12.12255*Sal)*(Hs+Kb).inverse();
	////////////////////////////////////////////////////

	Hx		=	Hs*1.02;
	tmpA1	=	C*(Hs*K1inv + 1 + K2*Hs.inverse()).inverse()  +  2*K2*C*(Hs*Hs*K1inv + Hs + K2).inverse() + Kb*(12.12255*Sal)*(Hs+Kb).inverse() - Hs + 1e-2*Hs.inverse();
	tmpA2	=	C*(Hx*K1inv + 1 + K2*Hx.inverse()).inverse()  +  2*K2*C*(Hx*Hx*K1inv + Hx + K2).inverse() + Kb*(12.12255*Sal)*(Hx+Kb).inverse() - Hx + 1e-2*Hx.inverse();
	Hs		=	Hs*exp(0.02*(tmpA1-(A-1))/(tmpA1-tmpA2));

	Hx		=	Hs*1.01;
	tmpA1	=	C*(Hs*K1inv + 1 + K2*Hs.inverse()).inverse()  +  2*K2*C*(Hs*Hs*K1inv + Hs + K2).inverse() + Kb*(12.12255*Sal)*(Hs+Kb).inverse() - Hs + 1e-2*Hs.inverse();
	tmpA2	=	C*(Hx*K1inv + 1 + K2*Hx.inverse()).inverse()  +  2*K2*C*(Hx*Hx*K1inv + Hx + K2).inverse() + Kb*(12.12255*Sal)*(Hx+Kb).inverse() - Hx + 1e-2*Hx.inverse();
	Hs		=	Hs*exp(0.01*(tmpA1-(A-1))/(tmpA1-tmpA2));

	//////////////////////////////////////
	Csolved.beta	=	Hs - Csolved.H;		// beta = d(H)/d(Alk) .... = d(H)/(-1mmM)
	//////////////////////////////////////

}

void GasEx(atmosphereS& atm, ArVector8f const &S, ArVector8f const &T, ArVector8f &ORGe, Vector18f& DIC, Vector18f& dc13, Vector18f& dc14, ArVector8f const &A, ArVector8f const &NtoC, Ktable const &Ksurf, ArVector8f const &Area)
{
	ArVector8f H2CO3,C,AirtoSea, SeatoAir, d13Fsa, d13Fas, d14Fsa, d14Fas;
	C=DIC.head(8);

    CO2find(C, A, S, Ksurf.K0.matrix().head(8), Ksurf.K1.matrix().head(8), Ksurf.K2.matrix().head(8), Ksurf.Kb.matrix().head(8), atm.oldH, H2CO3);
    CO2find(C, A, S, Ksurf.K0.matrix().head(8), Ksurf.K1.matrix().head(8), Ksurf.K2.matrix().head(8), Ksurf.Kb.matrix().head(8), atm.oldH, H2CO3);

    int N =15;
	for (int dt=1;dt<(N+1);dt++)
	{
		CO2find(C, A, S, Ksurf.K0.matrix().head(8), Ksurf.K1.matrix().head(8), Ksurf.K2.matrix().head(8), Ksurf.Kb.matrix().head(8), atm.oldH, H2CO3);

        AirtoSea = Ksurf.K0.head(8) * atm.ppm * Area * (1536000/((N+1)/2*N)) *dt;	//* (1500* 1/45) * 1024;			// reconsider *1024 ... µM= 1e-3mol/m3 ... nothing about kg ...
		SeatoAir = H2CO3 * Area * (1536000/((N+1)/2*N)) *dt;					    //* (1500* 1/45) * 1024;

		d13Fsa	=	((dc13.head(8).array()/C) + (0.107*T - 10.53 - 0.875))*SeatoAir;
		d13Fas	=	(atm.dn13/atm.ppm - 0.875)*AirtoSea;
		d14Fsa	=	((dc14.head(8).array()/C) + 2*(0.107*T - 10.53 - 0.875))*SeatoAir;
		d14Fas	=	(atm.dn14/atm.ppm - 2*0.875)*AirtoSea;

		dc13.head(8)	=	dc13.head(8).array()	+ (d13Fas - d13Fsa)*NtoC;
		atm.dn13			=	atm.dn13			+ (d13Fsa.sum() - d13Fas.sum()) / (1.773E+20);
        dc14.head(8)	=	dc14.head(8).array()	+ (d14Fas - d14Fsa)*NtoC;
		atm.dn14			=	atm.dn14			+ (d14Fsa.sum() - d14Fas.sum()) / (1.773E+20);

		C		+=	(AirtoSea - SeatoAir) * NtoC; // mol / time step
		atm.ppm	-=	(AirtoSea.sum() - SeatoAir.sum()) / (1.773E+20); // 1.773E+20 molecules in atm with 1e6 added in

	}
    CO2find(C, A, S, Ksurf.K0.matrix().head(8), Ksurf.K1.matrix().head(8), Ksurf.K2.matrix().head(8), Ksurf.Kb.matrix().head(8), atm.oldH, H2CO3);
    //ORGe = -12.03*0.4342945 * H2CO3.log() - 1.19 + 1 + (-9866*(T+273.15).inverse())+24.12;      // 12/1.2 is from F&H92 ; 0.434 is ln to log10 ; +1 is heterotroph enrichment ; CO2/DIC epsilon from Mook86
	ORGe = -(25.3 - (182 * H2CO3.inverse() *0.8)) + 1 + ((-9866*(T+273.15).inverse())+24.12);      // 12/1.2 is from F&H92 ; 0.434 is ln to log10 ; +1 is heterotroph enrichment ; CO2/DIC epsilon from Mook86

	DIC.head(8) = C;
}

void Volcano(double& Appm, double& dn13, double VolcX)
{
	Appm	+=	0.2*1e19*VolcX/1.773e+20;               //0.5*VolcX*1.67e+19/1.773e+20;		// note: the 0.5 factor prevents alk flux > alk rain at high VolcX ... consider residence time of Cstar
	dn13	+=	(10.5)*0.2*1e19*VolcX/1.773e+20;          // organic carbon burial raises d13C of C* input
	// no need to calc because new C has delta of zero // dn14	+=	(0)*0.2*1e19*VolcX/1.773e+20;
}

void ResetAtm(double& Appm, double& dn13, double const SetCO2)
{
	    dn13 += (1)*(SetCO2-Appm);
	    Appm =  SetCO2;
}

void handle14C(double& Appm, double& dn14, Vector18f& dc14, parametersS& param, geosphereS& geo)
{
	param.Q14.prod = 1.704; //LGM--2.05;//PI--1.704; //1.785; // how much radiocarbon when system starts
        if (param.year>=0)
	    {
	        if (param.Q14.init_true<1)
	        {
	            LoadQ14C(param.Q14);
	            param.Q14.row = 366;
	            param.Q14.Qnode = param.Q14.Q14Cforcing(param.Q14.row,2);
	            param.Q14.Qnextnode = param.Q14.Q14Cforcing(param.Q14.row-1,2);
	            param.Q14.DQ = param.Q14.Qnextnode - param.Q14.Qnode;
	            //param.Q14.DQ =  param.Q14.Q14Cforcing(param.Q14.row-1,2) - param.Q14.Q14Cforcing(param.Q14.row,2);
	            param.Q14.Dt = -param.Q14.Q14Cforcing(param.Q14.row-1,0) + param.Q14.Q14Cforcing(param.Q14.row,0);
	            param.Q14.yrstep = param.Q14.Q14Cforcing(param.Q14.row,0) - param.year;
	            param.Q14.init_true = 1;
	            //printf("initQ: year=%d, forcyr=%f, yrstep=%f, Dt=%f\n",param.year,  param.Q14.Q14Cforcing(param.Q14.row,0), param.Q14.yrstep, param.Q14.Dt);
	        }

            if (param.Q14.yrstep == param.Q14.Dt)
            {
                param.Q14.row = param.Q14.row -1;
                if (param.Q14.row>=1)
                {
                    param.Q14.Qnode = param.Q14.Qnextnode;
                    param.Q14.Qnextnode = param.Q14.Q14Cforcing(param.Q14.row-1,2);
                    //param.Q14.Qnode = param.Q14.Q14Cforcing(param.Q14.row,2);
                    //param.Q14.DQ =  param.Q14.Q14Cforcing(param.Q14.row-1,2) - param.Q14.Q14Cforcing(param.Q14.row,2);
                    param.Q14.DQ = param.Q14.Qnextnode - param.Q14.Qnode;
                    param.Q14.Dt = -param.Q14.Q14Cforcing(param.Q14.row-1,0) + param.Q14.Q14Cforcing(param.Q14.row,0);
                    param.Q14.yrstep = 0;
                    //printf("initQ: year=%d, forcyr=%f, yrstep=%f, Dt=%f, row=%d\n",param.year,  param.Q14.Q14Cforcing(param.Q14.row,0), param.Q14.yrstep, param.Q14.Dt,param.Q14.row);
                }
                else if (param.Q14.row==0)
                {
                    param.Q14.Qnode = param.Q14.Qnextnode;
                    //param.Q14.Qnode = param.Q14.Q14Cforcing(param.Q14.row,2);
                    param.Q14.DQ = 0;
                    param.Q14.Dt = -1;
                    param.Q14.yrstep = 0;
                    //printf("initQ: year=%d, forcyr=%f, yrstep=%f, Dt=%f, row=%d\n",param.year,  param.Q14.Q14Cforcing(param.Q14.row,0), param.Q14.yrstep, param.Q14.Dt,param.Q14.row);
                }

            }

	        param.Q14.prod = param.Q14.Qnode + param.Q14.DQ*(param.Q14.yrstep/param.Q14.Dt);
	        param.Q14.yrstep = param.Q14.yrstep +1;
	    }


	    dn14 += 1000 * (param.C14X*param.Q14.prod*1e10*510072000*31556736/6.02214179e23) / (1.773e+20*1e-6) / (0.95*1.25E-12); //*add new radiocarbon, 1.88-atoms/km2/s ... 5100-km2 ... 315-s/yr ... 6-atom/mol ... 1.773molatm ...0.95 see Stuiver&Pollach ... 1.25-14C/C(ref;Schneideretal95) // Appm cancels
        dn14 = dn14*0.999879039; // decay/yr ... 0.5^(1/5730) = 0.999879039
	    dc14 = dc14*0.999879039; // tracer in ocean
	    geo.dn14 = geo.dn14*0.999879039; //
	    geo.d14Corg = geo.d14Corg*0.999879039; //organic carbon on land

        //2500PgC = 1175ppm ;  25PgC-flux = 11.75ppm
        //3000PgC = 1410ppm ;  30PgC-flux = 14.1ppm
	    float tmp = geo.d14Corg;
        geo.d14Corg = 0.99*geo.d14Corg + 0.01*(dn14/Appm);
        //dn14 = (11.75)*(tmp) + (Appm-11.75)*(dn14/Appm); //2500PgC
        dn14 = (14.1)*(tmp) + (Appm-14.1)*(dn14/Appm); //3000PgC


}
void modifySurfT(atmosphereS &atmosphere, double initCO2, double afterCO2)
{
	atmosphere.surfT += 1.66 * log(afterCO2/initCO2);
}

void RunEx (oceanS &ocean, atmosphereS &atmosphere, geosphereS &geosphere, parametersS &param, int Nyears)
{

	int ID = 0;
	for (int t=0;t<Nyears;t++)
	{

		/// EXPERIMENTS / SPIN-UP
		if (param.InitCO2 > 0)
			atmosphere.ppm = param.InitCO2;

		if (param.ALK.flag == 1)
		{
			atmosphere.prevppm = atmosphere.ppm;
			atmosphere.ppm += param.ALK.Crate_PgCperYear * 1 * 1e15 / 12 / 1.773e14; // Atmospheric mass or volume is 1.773e+20 with 1e6 added in
			atmosphere.dn13	+=	(0)* param.ALK.Crate_PgCperYear * 1 * 1e15 / 12 / 1.773e14;
			param.ALK.CumCarbonFlux += param.ALK.Crate_PgCperYear * 1;
			param.ALK.totalALK += param.ALK.annualALK * 1 * 1e15;
			ocean.tracer.Alk(2) += param.ALK.annualALK * 1e6 * 1e15 * ocean.box.NtoC(2);
			// for (int box=0;box<=7;box++)
// 			{
// 				ocean.tracer.Alk(box) = (param.ALK.annualALK/1) * 1e6 * ocean.box.NtoC(box);
// 			}

		}


        if (param.DGLFall.init_true == 1) //loads and processes forcing files
        {
            UpdateDGLforcing(param.DGLFall, param.year);
            if (param.DGLFall.trigerID == 1)
            {
                param.DGLFall.F1.value = 0;
                param.DGLFall.F4.value = 0;
                param.DGLFall.trigerID = 0;
            }

            ID = static_cast<int>(param.DGLFall.F1.value);
            if (ID>=0)
            {
                switch (ID)
                {
                    case 0: ChangeCirc(ocean.circulationM, 0,0,0,0); break;
                    case 1: ChangeCirc(ocean.circulationM, 1,0,0,0); break;
                    case 2: ChangeCirc(ocean.circulationM, 98,0,0,0); break;
                    default: printf("XXX CIRC-WARNING: ILLEGITIMATE CHOICE - default to NADW - ID=%d\n",ID); ChangeCirc(ocean.circulationM, 0,0,0,0); break;
                }
                InitCirc(ocean.circulationM,ocean.box.vol,ocean.box.vol_inv);
            }
            PAZmix(ocean.circulationM,ocean.box.vol, (3+ 17*param.DGLFall.F2.value));
            ocean.box.setP(7) = (1+ 1*param.DGLFall.F2.value);
            param.PAZiceX = 0.5 - 0.5*param.DGLFall.F2.value; ocean.box.Area(7) = (1-param.PAZiceX) * param.PAZarea;
            ocean.box.setP(6) = 1.2 - (0.5*(1-param.DGLFall.F3.value)*(1-param.DGLFall.F3.value));

            ID = static_cast<int>(param.DGLFall.F4.value);
            if ((ID!=0) && (false))
            {
                switch (ID)
                {
                    case 0: break;
                    case 1:  ocean.VolParams.scale= 40.0*100/1000*2*param.DGLFall.F4.yrstep/1000; ChangeVol(ocean.Schemes.DOtoMDviaAApLL, ocean.circulationM, ocean.tracer, ocean.venttracer, ocean.box.vol, ocean.box.vol_inv, ocean.box.CtoN, ocean.box.NtoC, ocean.VolParams); break;
                    case 2:  ocean.VolParams.scale= 40.0*100/1400*2*param.DGLFall.F4.yrstep/1400; ChangeVol(ocean.Schemes.DOtoMDviaAApLL, ocean.circulationM, ocean.tracer, ocean.venttracer, ocean.box.vol, ocean.box.vol_inv, ocean.box.CtoN, ocean.box.NtoC, ocean.VolParams); break;
                    case -1: ocean.VolParams.scale=40; ChangeVol(ocean.Schemes.MDtoDOviaNA, ocean.circulationM, ocean.tracer, ocean.venttracer, ocean.box.vol, ocean.box.vol_inv, ocean.box.CtoN, ocean.box.NtoC, ocean.VolParams); break;
                    default: printf("XXX VOL-WARNING: ILLEGITIMATE CHOICE - default to NoChange - ID=%d\n",ID); break;
                }
            }
        }

        //if (param.year == 15212){ cout<<ocean.VolParams.VolOp;}
		if (abs(param.TxS - (ocean.tracer.Temp.sum()*ocean.tracer.Sal.sum())) >= 0.1)
		{
			Kcalc(ocean.Ksurf, ocean.tracer.Temp, ocean.tracer.Sal, ocean.box.top, ocean.box.bottom, ocean.SF);
			param.TxS = (ocean.tracer.Temp.sum()*ocean.tracer.Sal.sum());
		}


		if (t % 1 == 0) { SFpHcalc(ocean.tracer.Sal, ocean.tracer.C, ocean.tracer.Alk, ocean.SF, param.CaX, param.DissolveX);}


		/// MODEL
		Circ(ocean.circulationM,ocean.tracer);
		//ChangeVol(ocean.Schemes.NO, ocean.circulationM, ocean.tracer, ocean.venttracer, ocean.box.vol, ocean.box.vol_inv, ocean.VolParams);
		//VentTrack(ocean.circulationM, ocean.venttracer);

		Prod(ocean.tracer, ocean.box.setP, ocean.box.setSi, ocean.box.CaRatio, ocean.rain, param.alphaSi, ocean.box.CtoN, ocean.box.ORGe);//.P, ocean.rain.Si, ocean.rain.Ca, ocean.rain.d13Corg, ocean.rain.d13Ccc, ocean.rain.d30Si,param.alphaSi, ocean.box.CtoN, ocean.box.ORGe, atmosphere.ppm, atmosphere.dn13);
		Remin(ocean.tracer, ocean.RainOrg, ocean.RainSi, ocean.rain, ocean.box.NtoC);
		Dissolve(ocean.tracer, ocean.RainCC, ocean.rain.Ca, ocean.rain.d13Ccc, ocean.rain.d14Ccc, ocean.box.NtoC, ocean.SF.Fdiss);
		GasEx(atmosphere, ocean.tracer.Sal.head(8), ocean.tracer.Temp.head(8), ocean.box.ORGe, ocean.tracer.C, ocean.tracer.dc13, ocean.tracer.dc14, ocean.tracer.Alk.head(8), ocean.box.NtoC.head(8), ocean.Ksurf, ocean.box.Area);
		handle14C(atmosphere.ppm, atmosphere.dn14, ocean.tracer.dc14, param, geosphere);
		Volcano(atmosphere.ppm, atmosphere.dn13, param.VolcX);
		River(ocean.tracer.C, ocean.tracer.dc13, ocean.tracer.dc14, ocean.tracer.Alk, ocean.box.NtoC.head(4),atmosphere.ppm,atmosphere.dn13,atmosphere.dn14, param.WeathX, param.RivX, param.SetCO2);
        VentTrack(ocean.circulationM,ocean.venttracer, atmosphere, ocean.tracer.C.head(8), ocean.tracer.dc13.head(8), ocean.tracer.dc14.head(8));

		if(param.year < -2020)
		{
			modifySurfT(atmosphere, atmosphere.prevppm, atmosphere.ppm);
		}
		else if(param.year == -2020)
		{
			atmosphere.surfT = 14.88;
		}
		else
		{
			atmosphere.surfT = 0;
		}

		/// OUTPUT

		if (param.ALKout.flag == 2)
		{
			if ((param.year<=-1980) && (param.year>=-2100)) // yearly output from 1750 to 2100
			{
				param.ALK.OUTPUT(param.ALK.outrow, 0) = param.year;
				param.ALK.OUTPUT(param.ALK.outrow, 1) = param.ALK.Crate_PgCperYear;
				param.ALK.OUTPUT(param.ALK.outrow, 2) = atmosphere.ppm;
				//param.ALK.OUTPUT(param.ALK.outrow, 3) = atmosphere.ppm;
				double ALK_surf=OcAvConc(ocean.tracer.Alk,ocean.box.vol, 0, 8); // average TA
				param.ALK.OUTPUT(param.ALK.outrow, 3) = ALK_surf;

				finalCsolve(ocean.tracer.C, ocean.tracer.Alk,ocean.tracer.Sal, ocean.Ksurf.K0, ocean.Ksurf.K1, ocean.Ksurf.K2, ocean.Ksurf.Kb,ocean.Ksurf.Ks,param.CaX, ocean.box.Csolved);

				double H0 = ocean.box.Csolved.H(0);
				double H1 = ocean.box.Csolved.H(1);
				double H2 = ocean.box.Csolved.H(2);
				double H3 = ocean.box.Csolved.H(3);
				double H4 = ocean.box.Csolved.H(4);
				double H5 = ocean.box.Csolved.H(5);
				double H6 = ocean.box.Csolved.H(6);
				double H7 = ocean.box.Csolved.H(7);
				double Htotal = 1e-6*((H0+H1+H2+H3+H4+H5+H6+H7)/8);
				double pH_surf = -1 * log10(Htotal);
				param.ALK.OUTPUT(param.ALK.outrow, 6) = pH_surf;

				//Temporary print statements for debugging//////////////////////////////
				cout<<"pH: "<<pH_surf<<"\n";
				cout<<"CO2: "<<atmosphere.ppm<<"\n";
				cout<<"surfT: "<<atmosphere.surfT<<"\n";
				////////////////////////////////////////////////////////////
				double CO3_0 = ocean.box.Csolved.CO3(0);
				double CO3_1 = ocean.box.Csolved.CO3(1);
				double CO3_2 = ocean.box.Csolved.CO3(2);
				double CO3_3 = ocean.box.Csolved.CO3(3);
				double CO3_4 = ocean.box.Csolved.CO3(4);
				double CO3_5 = ocean.box.Csolved.CO3(5);
				double CO3_6 = ocean.box.Csolved.CO3(6);
				double CO3_7 = ocean.box.Csolved.CO3(7);
				double CO3_surf = (CO3_0+CO3_1+CO3_2+CO3_3+CO3_4+CO3_5+CO3_6+CO3_7)/8;
				param.ALK.OUTPUT(param.ALK.outrow, 4) = CO3_surf;


				double omega0 = ocean.box.Csolved.omega(0);
				double omega1 = ocean.box.Csolved.omega(1);
				double omega2 = ocean.box.Csolved.omega(2);
				double omega3 = ocean.box.Csolved.omega(3);
				double omega4 = ocean.box.Csolved.omega(4);
				double omega5 = ocean.box.Csolved.omega(5);
				double omega6 = ocean.box.Csolved.omega(6);
				double omega7 = ocean.box.Csolved.omega(7);
				double omega_surf = (omega0+omega1+omega2+omega3+omega4+omega5+omega6+omega7)/8;
				param.ALK.OUTPUT(param.ALK.outrow, 5) = omega_surf;

				param.ALK.OUTPUT(param.ALK.outrow, 7) = param.ALK.annualALK;
				param.ALK.OUTPUT(param.ALK.outrow, 8) = param.ALK.totalALK;
				param.ALK.OUTPUT(param.ALK.outrow, 9) = atmosphere.surfT;
				param.ALK.outrow += 1;
			}

		}

        param.year = param.year -1;
    }
}
///////////////////////////////////////////////// added RunExForFunctor ////////////////////////////////////////////////////////////////////
double MinimizationFunctor::RunExForFunctor (VecDoub P, oceanS &ocean, atmosphereS &atmosphere, geosphereS &geosphere, parametersS &param, int Nyears )
{
	int ID = 0;
	for (int t=0;t<Nyears;t++)
	{

		param.ALK.annualALK = P[0];

		/// EXPERIMENTS / SPIN-UP
		if (param.InitCO2 > 0)
			atmosphere.ppm = param.InitCO2;

		if (param.ALK.flag == 1)
		{
			atmosphere.prevppm = atmosphere.ppm;
			atmosphere.ppm += param.ALK.Crate_PgCperYear * 1 * 1e15 / 12 / 1.773e14; // Atmospheric mass or volume is 1.773e+20 with 1e6 added in
			atmosphere.dn13	+=	(0)* param.ALK.Crate_PgCperYear * 1 * 1e15 / 12 / 1.773e14;
			param.ALK.CumCarbonFlux += param.ALK.Crate_PgCperYear * 1;
			param.ALK.totalALK += param.ALK.annualALK * 1 * 1e15;
			ocean.tracer.Alk(2) += param.ALK.annualALK * 1e15 * 1e6 * ocean.box.NtoC(2);
			// for (int box=0;box<=7;box++)
// 			{
// 				ocean.tracer.Alk(box) = (param.ALK.annualALK/1) * 1e6 * ocean.box.NtoC(box);
// 			}

		}

        if (param.DGLFall.init_true == 1) //loads and processes forcing files
        {
            UpdateDGLforcing(param.DGLFall, param.year);
            if (param.DGLFall.trigerID == 1)
            {
                param.DGLFall.F1.value = 0;
                param.DGLFall.F4.value = 0;
                param.DGLFall.trigerID = 0;
            }

            ID = static_cast<int>(param.DGLFall.F1.value);
            if (ID>=0)
            {
                switch (ID)
                {
                    case 0: ChangeCirc(ocean.circulationM, 0,0,0,0); break;
                    case 1: ChangeCirc(ocean.circulationM, 1,0,0,0); break;
                    case 2: ChangeCirc(ocean.circulationM, 98,0,0,0); break;
                    default: printf("XXX CIRC-WARNING: ILLEGITIMATE CHOICE - default to NADW - ID=%d\n",ID); ChangeCirc(ocean.circulationM, 0,0,0,0); break;
                }
                InitCirc(ocean.circulationM,ocean.box.vol,ocean.box.vol_inv);
            }
            PAZmix(ocean.circulationM,ocean.box.vol, (3+ 17*param.DGLFall.F2.value));
            ocean.box.setP(7) = (1+ 1*param.DGLFall.F2.value);
            param.PAZiceX = 0.5 - 0.5*param.DGLFall.F2.value; ocean.box.Area(7) = (1-param.PAZiceX) * param.PAZarea;
            ocean.box.setP(6) = 1.2 - (0.5*(1-param.DGLFall.F3.value)*(1-param.DGLFall.F3.value));

            ID = static_cast<int>(param.DGLFall.F4.value);
            if ((ID!=0) && (false))
            {
                switch (ID)
                {
                    case 0: break;
                    case 1:  ocean.VolParams.scale= 40.0*100/1000*2*param.DGLFall.F4.yrstep/1000; ChangeVol(ocean.Schemes.DOtoMDviaAApLL, ocean.circulationM, ocean.tracer, ocean.venttracer, ocean.box.vol, ocean.box.vol_inv, ocean.box.CtoN, ocean.box.NtoC, ocean.VolParams); break;
                    case 2:  ocean.VolParams.scale= 40.0*100/1400*2*param.DGLFall.F4.yrstep/1400; ChangeVol(ocean.Schemes.DOtoMDviaAApLL, ocean.circulationM, ocean.tracer, ocean.venttracer, ocean.box.vol, ocean.box.vol_inv, ocean.box.CtoN, ocean.box.NtoC, ocean.VolParams); break;
                    case -1: ocean.VolParams.scale=40; ChangeVol(ocean.Schemes.MDtoDOviaNA, ocean.circulationM, ocean.tracer, ocean.venttracer, ocean.box.vol, ocean.box.vol_inv, ocean.box.CtoN, ocean.box.NtoC, ocean.VolParams); break;
                    default: printf("XXX VOL-WARNING: ILLEGITIMATE CHOICE - default to NoChange - ID=%d\n",ID); break;
                }
            }
        }

        //if (param.year == 15212){ cout<<ocean.VolParams.VolOp;}
		if (abs(param.TxS - (ocean.tracer.Temp.sum()*ocean.tracer.Sal.sum())) >= 0.1)
		{
			Kcalc(ocean.Ksurf, ocean.tracer.Temp, ocean.tracer.Sal, ocean.box.top, ocean.box.bottom, ocean.SF);
			param.TxS = (ocean.tracer.Temp.sum()*ocean.tracer.Sal.sum());
		}

		if (t % 1 == 0) { SFpHcalc(ocean.tracer.Sal, ocean.tracer.C, ocean.tracer.Alk, ocean.SF, param.CaX, param.DissolveX);}

		/// MODEL
		Circ(ocean.circulationM,ocean.tracer);
		//ChangeVol(ocean.Schemes.NO, ocean.circulationM, ocean.tracer, ocean.venttracer, ocean.box.vol, ocean.box.vol_inv, ocean.VolParams);
		//VentTrack(ocean.circulationM, ocean.venttracer);

		Prod(ocean.tracer, ocean.box.setP, ocean.box.setSi, ocean.box.CaRatio, ocean.rain, param.alphaSi, ocean.box.CtoN, ocean.box.ORGe);//.P, ocean.rain.Si, ocean.rain.Ca, ocean.rain.d13Corg, ocean.rain.d13Ccc, ocean.rain.d30Si,param.alphaSi, ocean.box.CtoN, ocean.box.ORGe, atmosphere.ppm, atmosphere.dn13);
		Remin(ocean.tracer, ocean.RainOrg, ocean.RainSi, ocean.rain, ocean.box.NtoC);
		Dissolve(ocean.tracer, ocean.RainCC, ocean.rain.Ca, ocean.rain.d13Ccc, ocean.rain.d14Ccc, ocean.box.NtoC, ocean.SF.Fdiss);
		GasEx(atmosphere, ocean.tracer.Sal.head(8), ocean.tracer.Temp.head(8), ocean.box.ORGe, ocean.tracer.C, ocean.tracer.dc13, ocean.tracer.dc14, ocean.tracer.Alk.head(8), ocean.box.NtoC.head(8), ocean.Ksurf, ocean.box.Area);
		handle14C(atmosphere.ppm, atmosphere.dn14, ocean.tracer.dc14, param, geosphere);
		Volcano(atmosphere.ppm, atmosphere.dn13, param.VolcX);
		River(ocean.tracer.C, ocean.tracer.dc13, ocean.tracer.dc14, ocean.tracer.Alk, ocean.box.NtoC.head(4),atmosphere.ppm,atmosphere.dn13,atmosphere.dn14, param.WeathX, param.RivX, param.SetCO2);
        VentTrack(ocean.circulationM,ocean.venttracer, atmosphere, ocean.tracer.C.head(8), ocean.tracer.dc13.head(8), ocean.tracer.dc14.head(8));

    	if(param.year < -2020)
		{
			modifySurfT(atmosphere, atmosphere.prevppm, atmosphere.ppm);
		}
		else if(param.year == -2020)
		{
			atmosphere.surfT = 14.88;
		}
		else
		{
			atmosphere.surfT = 0;
		}

	}

	finalCsolve(ocean.tracer.C, ocean.tracer.Alk, ocean.tracer.Sal, ocean.Ksurf.K0, ocean.Ksurf.K1, ocean.Ksurf.K2, ocean.Ksurf.Kb, ocean.Ksurf.Ks, param.CaX, ocean.box.Csolved);
	double H0 = ocean.box.Csolved.H(0);
	double H1 = ocean.box.Csolved.H(1);
	double H2 = ocean.box.Csolved.H(2);
	double H3 = ocean.box.Csolved.H(3);
	double H4 = ocean.box.Csolved.H(4);
	double H5 = ocean.box.Csolved.H(5);
	double H6 = ocean.box.Csolved.H(6);
	double H7 = ocean.box.Csolved.H(7);
	double Htotal = 1e-6*((H0+H1+H2+H3+H4+H5+H6+H7)/8);
	double pH_surf = -1 * log10(Htotal);

	double RMS;
	double modelCO2 = atmosphere.ppm;
	double temp_surf = atmosphere.surfT;

	// RMS = 10000*pow((8.0 - pH_surf), 2);	 // 2D optimization pH and CO2: RCP 2.6
	// RMS = sqrt(abs(500 - modelCO2))/100 + abs(8 - pH_surf); // 2D optimization pH and CO2: RCP 4.5, 6.0, and 8.5
	// RMS = sqrt((pow((500 - modelCO2), 2) + 80000*pow((8 - pH_surf), 2) + 80000*pow((15 - temp_surf), 2))/3); // Test RMS for 3D optimization
	RMS = pow((500 - modelCO2), 2) + 5*pow((8 - pH_surf), 2) + 5*pow((15 - temp_surf), 2);
	return RMS;

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void OutToFile(oceanS &ocean, atmosphereS &atmosphere, geosphereS &geosphere, parametersS &param, AllExArray &AllEx, Array<double,154,1,0,154,1> &OUTROWex )
{

    OUTROWex(0)		=	param.year;
	OUTROWex(1)		=	AllEx(0);
	OUTROWex(2)		=	AllEx(1);
	OUTROWex(3)		=	AllEx(2);
	OUTROWex(4)		=	AllEx(3);
	OUTROWex(5)		=	AllEx(4);
	OUTROWex(6)		=	AllEx(5);
	OUTROWex(7)		=	AllEx(6);
	OUTROWex(8)		=	AllEx(5);
	OUTROWex(9)		=	AllEx(6);

	OUTROWex(10)		=	atmosphere.ppm;
	OUTROWex(11)		=	atmosphere.dn13/atmosphere.ppm;
	OUTROWex(12)	=	atmosphere.dn14/atmosphere.ppm;
	OUTROWex(13)	=	D14Ccalc(atmosphere.ppm, atmosphere.dn13, atmosphere.dn14);

    for(int box =0; box<18; box++) {OUTROWex(14+box)	=	ocean.tracer.dc13(box)/ocean.tracer.C(box); }
    for(int box =0; box<18; box++) {OUTROWex(32+box)	=	ocean.tracer.dc14(box)/ocean.tracer.C(box); }
    for(int box =0; box<18; box++) {OUTROWex(50+box)	=	D14Ccalc(ocean.tracer.C(box),ocean.tracer.dc13(box),ocean.tracer.dc14(box)); }
    for(int box =0; box<18; box++) {OUTROWex(68+box)	=	ocean.venttracer.trueage(box); }

        double atmD14C = D14Ccalc(atmosphere.ppm, atmosphere.dn13, atmosphere.dn14);
        Vector18d Age14C = -5730* 0.69314718056 *  ((1000+1000*((1+(ocean.tracer.dc14.array()/ocean.tracer.C.array()-1000)/1000)* 0.950625/(1+(ocean.tracer.dc13.array()/ocean.tracer.C.array()/1000))/(1+(ocean.tracer.dc13.array()/ocean.tracer.C.array()/1000)) -1)).array()/(1000+atmD14C)).log();
    for(int box =0; box<18; box++) {OUTROWex(86+box)	=	Age14C(box); }
    for(int box =0; box<18; box++) {OUTROWex(104+box)	=	ocean.venttracer.pref14Cage(box); }

    OUTROWex(122)	=	OcAvConc(ocean.venttracer.trueage,ocean.box.vol, 0, 18);
    OUTROWex(123)	=	OcAvDelta(Age14C,ocean.tracer.C,ocean.box.vol, 0, 18);
    OUTROWex(124)	=	OcAvConc(ocean.venttracer.pref14Cage,ocean.box.vol, 0, 18);
    OUTROWex(125)	=   OcAvDC(ocean.tracer.dc13,ocean.tracer.C,ocean.box.vol,0,18);
    OUTROWex(126)	=   OcAvDC(ocean.tracer.dc14,ocean.tracer.C,ocean.box.vol,0,18);
    OUTROWex(127)	=   D14Ccalc(1,OcAvDC(ocean.tracer.dc13,ocean.tracer.C,ocean.box.vol,0,18),OcAvDC(ocean.tracer.dc14,ocean.tracer.C,ocean.box.vol,0,18));

    OUTROWex(128)	=	OcAvConc(ocean.venttracer.trueage,ocean.box.vol, 12, 6);
    OUTROWex(129)	=	OcAvDelta(Age14C,ocean.tracer.C,ocean.box.vol, 12, 6);
    OUTROWex(130)	=	OcAvConc(ocean.venttracer.pref14Cage,ocean.box.vol, 12, 6);
    OUTROWex(131)	=   OcAvDC(ocean.tracer.dc13,ocean.tracer.C,ocean.box.vol,12,6);
    OUTROWex(132)	=   OcAvDC(ocean.tracer.dc14,ocean.tracer.C,ocean.box.vol,12,6);
    OUTROWex(133)	=   D14Ccalc(1,OcAvDC(ocean.tracer.dc13,ocean.tracer.C,ocean.box.vol,12,6),OcAvDC(ocean.tracer.dc14,ocean.tracer.C,ocean.box.vol,12,6));

    OUTROWex(134)	=	OcAvConc(ocean.venttracer.trueage,ocean.box.vol, 8, 4);
    OUTROWex(135)	=	OcAvDelta(Age14C,ocean.tracer.C,ocean.box.vol, 8, 4);
    OUTROWex(136)	=	OcAvConc(ocean.venttracer.pref14Cage,ocean.box.vol, 8, 4);
    OUTROWex(137)	=   OcAvDC(ocean.tracer.dc13,ocean.tracer.C,ocean.box.vol,8,4);
    OUTROWex(138)	=   OcAvDC(ocean.tracer.dc14,ocean.tracer.C,ocean.box.vol,8,4);
    OUTROWex(139)	=   D14Ccalc(1,OcAvDC(ocean.tracer.dc13,ocean.tracer.C,ocean.box.vol,8,4),OcAvDC(ocean.tracer.dc14,ocean.tracer.C,ocean.box.vol,8,4));

    OUTROWex(140)	=	OcAvConc(ocean.venttracer.trueage,ocean.box.vol, 0, 8);
    OUTROWex(141)	=	OcAvDelta(Age14C,ocean.tracer.C,ocean.box.vol, 0, 8);
    OUTROWex(142)	=	OcAvConc(ocean.venttracer.pref14Cage,ocean.box.vol, 0, 8);
    OUTROWex(143)	=   OcAvDC(ocean.tracer.dc13,ocean.tracer.C,ocean.box.vol,0,8);
    OUTROWex(144)	=   OcAvDC(ocean.tracer.dc14,ocean.tracer.C,ocean.box.vol,0,8);
    OUTROWex(145)	=   D14Ccalc(1,OcAvDC(ocean.tracer.dc13,ocean.tracer.C,ocean.box.vol,0,8),OcAvDC(ocean.tracer.dc14,ocean.tracer.C,ocean.box.vol,0,8));

/*
    OUTROWex(119)	=	OcAvConc(ocean.venttracer.trueage,ocean.box.vol, 0, 18);
    OUTROWex(120)	=	OcAvDelta(Age14C,ocean.tracer.C,ocean.box.vol, 0, 18);
    OUTROWex(121)	=	OcAvConc(ocean.venttracer.pref14Cage,ocean.box.vol, 0, 18);
    OUTROWex(122)	=   OcAvDelta(ocean.tracer.dc13.cwiseQuotient(ocean.tracer.C).matrix(),ocean.tracer.C,ocean.box.vol,0,18);
    OUTROWex(123)	=   OcAvDelta(ocean.tracer.dc14.cwiseQuotient(ocean.tracer.C),ocean.tracer.C,ocean.box.vol,0,18);
    OUTROWex(124)	=   D14Ccalc(1,OcAvDelta(ocean.tracer.dc13.cwiseQuotient(ocean.tracer.C),ocean.tracer.C,ocean.box.vol,0,18),OcAvDelta(ocean.tracer.dc14.cwiseQuotient(ocean.tracer.C),ocean.tracer.C,ocean.box.vol,0,18));

    OUTROWex(125)	=	OcAvConc(ocean.venttracer.trueage,ocean.box.vol, 12, 6);
    OUTROWex(126)	=	OcAvDelta(Age14C,ocean.tracer.C,ocean.box.vol, 12, 6);
    OUTROWex(127)	=	OcAvConc(ocean.venttracer.pref14Cage,ocean.box.vol, 12, 6);
    OUTROWex(128)	=   OcAvDelta(ocean.tracer.dc13.cwiseQuotient(ocean.tracer.C),ocean.tracer.C,ocean.box.vol,12,6);
    OUTROWex(129)	=   OcAvDelta(ocean.tracer.dc14.cwiseQuotient(ocean.tracer.C),ocean.tracer.C,ocean.box.vol,12,6);
    OUTROWex(130)	=   D14Ccalc(1,OcAvDelta(ocean.tracer.dc13.cwiseQuotient(ocean.tracer.C),ocean.tracer.C,ocean.box.vol,12,6),OcAvDelta(ocean.tracer.dc14.cwiseQuotient(ocean.tracer.C),ocean.tracer.C,ocean.box.vol,12,6));

    OUTROWex(131)	=	OcAvConc(ocean.venttracer.trueage,ocean.box.vol, 8, 4);
    OUTROWex(132)	=	OcAvDelta(Age14C,ocean.tracer.C,ocean.box.vol, 8, 4);
    OUTROWex(133)	=	OcAvConc(ocean.venttracer.pref14Cage,ocean.box.vol, 8, 4);
    OUTROWex(134)	=   OcAvDelta(ocean.tracer.dc13.cwiseQuotient(ocean.tracer.C),ocean.tracer.C,ocean.box.vol,8,4);
    OUTROWex(135)	=   OcAvDelta(ocean.tracer.dc14.cwiseQuotient(ocean.tracer.C),ocean.tracer.C,ocean.box.vol,8,4);
    OUTROWex(136)	=   D14Ccalc(1,OcAvDelta(ocean.tracer.dc13.cwiseQuotient(ocean.tracer.C),ocean.tracer.C,ocean.box.vol,8,4),OcAvDelta(ocean.tracer.dc14.cwiseQuotient(ocean.tracer.C),ocean.tracer.C,ocean.box.vol,8,4));

    OUTROWex(137)	=	OcAvConc(ocean.venttracer.trueage,ocean.box.vol, 0, 8);
    OUTROWex(138)	=	OcAvDelta(Age14C,ocean.tracer.C,ocean.box.vol, 0, 8);
    OUTROWex(139)	=	OcAvConc(ocean.venttracer.pref14Cage,ocean.box.vol, 0, 8);
    OUTROWex(140)	=   OcAvDelta(ocean.tracer.dc13.cwiseQuotient(ocean.tracer.C),ocean.tracer.C,ocean.box.vol,0,8);
    OUTROWex(141)	=   OcAvDelta(ocean.tracer.dc14.cwiseQuotient(ocean.tracer.C),ocean.tracer.C,ocean.box.vol,0,8);
    OUTROWex(142)	=   D14Ccalc(1,OcAvDelta(ocean.tracer.dc13.cwiseQuotient(ocean.tracer.C),ocean.tracer.C,ocean.box.vol,0,8),OcAvDelta(ocean.tracer.dc14.cwiseQuotient(ocean.tracer.C),ocean.tracer.C,ocean.box.vol,0,8));
*/


    Vector18f vec;
    vec = ocean.venttracer.vent.block<18,1>(0,0); OUTROWex(146)	=   OcAvConc(vec,ocean.box.vol,0,18);
    vec = ocean.venttracer.vent.block<18,1>(0,1); OUTROWex(147)	=   OcAvConc(vec,ocean.box.vol,0,18);
    vec = ocean.venttracer.vent.block<18,1>(0,2); OUTROWex(148)	=   OcAvConc(vec,ocean.box.vol,0,18);
    vec = ocean.venttracer.vent.block<18,1>(0,3); OUTROWex(149)	=   OcAvConc(vec,ocean.box.vol,0,18);
    vec = ocean.venttracer.vent.block<18,1>(0,4); OUTROWex(150)	=   OcAvConc(vec,ocean.box.vol,0,18);
    vec = ocean.venttracer.vent.block<18,1>(0,5); OUTROWex(151)	=   OcAvConc(vec,ocean.box.vol,0,18);
    vec = ocean.venttracer.vent.block<18,1>(0,6); OUTROWex(152)	=   OcAvConc(vec,ocean.box.vol,0,18);
    vec = ocean.venttracer.vent.block<18,1>(0,7); OUTROWex(153)	=   OcAvConc(vec,ocean.box.vol,0,18);
}

experiment LGMSpinUp(int SpinUpYears)
{
	experiment LGM;

	//LGM.param.Exflag = 0;
	LGM.param.flag = 0;
	LGM.param.TxS = 0; LGM.param.VolcX = 0; LGM.param.WeathX = 0; LGM.param.CaX = 1; LGM.param.RivX = 1; LGM.param.SetCO2 = 280; LGM.param.ORGe = 20; LGM.param.Spike = 0; LGM.param.SpikeDelta = -50; LGM.param.DissolveX = 1;
    LGM.param.alphaSi = 0.9989; LGM.param.scalelength = 2000;
    LGM.param.VolcX=0; LGM.param.WeathX=0;
	//Leave negative number if I want to have constant radiocarbon production. Never gets triggered
	LGM.param.year = -9999999;
// 	LGM.param.Q14.ExNo = 0;
	// this is will turn on transient radiocarbon production

    // LGM.param.Q14.OUTrow = 0;
//     LGM.param.Q14.init_true = 0;
// 	LGM.param.Q14.ExNo = 0;
    LGM.param.C14X = 1; // 1 is the modern radiocarbon spin up, this is the X factor to scale either up or down. This is the number for LGM 11/17/20 number was 1.268 (matchs the obs record at 20kybp) 1.391 matches 30kybp obs. //1.62 is where it matchs GLOPIS starting from 20k //1.288 matchs constant constant at 20k. 1.621 for 20k turned on

    LGM.param.Q14.OUT.resize(501,36);
    LGM.param.Q14.OUT2.resize(501,20);


	Input(LGM.ocean, LGM.atmosphere, LGM.geosphere);
	ChangeCirc(LGM.ocean.circulationM, 1,0,0,0);
	LGM.ocean.circulationM(7,13) = 3; LGM.ocean.circulationM(13,7) = 3;
	LGM.ocean.box.setP(7) = 1; LGM.ocean.box.setP(6) = 0.7;
	LGM.param.PAZarea = LGM.ocean.box.Area(7); LGM.param.PAZiceX = 0.5; LGM.ocean.box.Area(7) = (1-LGM.param.PAZiceX) * LGM.param.PAZarea;
	InitCirc(LGM.ocean.circulationM,LGM.ocean.box.vol,LGM.ocean.box.vol_inv);
	InitVolSchemes(LGM.ocean.Schemes.MDtoDO, LGM.ocean.Schemes.DOtoMD,LGM.ocean.Schemes.MDtoDOviaNA, LGM.ocean.Schemes.DOtoMDviaAApLL, LGM.ocean.Schemes.UOtoDO, LGM.ocean.Schemes.DOtoUO, LGM.ocean.Schemes.NO);
	LGM.ocean.VolParams.scale = 1;
	//ChangeVol(LGM.ocean.Schemes.NO, LGM.ocean.circulationM, LGM.ocean.tracer, LGM.ocean.venttracer, LGM.ocean.box.vol, LGM.ocean.box.vol_inv, LGM.ocean.VolParams);
    LGM.ocean.tracer.Temp = Vector18f::Constant(5);
	printf("Model has been initialized with LGM parameter settings.");


	printf("Model is spinning up for 1,000,000 years ...");
    RunEx(LGM.ocean, LGM.atmosphere, LGM.geosphere, LGM.param, SpinUpYears);
    printf("Initialization complete. Spun-up model resides in 'LGM' container.\n");
    printf("CO2=%f; d14C=%f; D14C=%f; d14Corg=%f\n", LGM.atmosphere.ppm, LGM.atmosphere.dn14/LGM.atmosphere.ppm, D14Ccalc(LGM.atmosphere.ppm, LGM.atmosphere.dn13, LGM.atmosphere.dn14), LGM.geosphere.d14Corg);
    double tmp=OcAvConc(LGM.ocean.tracer.C,LGM.ocean.box.vol, 0, 18);
    printf("Ocean average [DIC]=%f;\n",tmp);

	return LGM;
}

experiment IGSpinUp(int SpinUpYears)
{
	experiment IG;

    IG.param.Exflag = 0;
	IG.param.flag = 0;
	IG.param.TxS = 0; IG.param.VolcX = 0; IG.param.WeathX = 0; IG.param.CaX = 1; IG.param.RivX = 1; IG.param.SetCO2 = 280; IG.param.ORGe = 20; IG.param.Spike = 0; IG.param.SpikeDelta = -50; IG.param.DissolveX = 1;
    IG.param.alphaSi = 0.9989; IG.param.scalelength = 2000;
    IG.param.VolcX=0; IG.param.WeathX=0;
    IG.param.year = -9999999;
    IG.param.C14X = 1;//1.268;
    IG.param.Q14.ExNo = 0;
    IG.param.Q14.OUT.resize(501,36);
    IG.param.Q14.OUT2.resize(501,20);
	Input(IG.ocean, IG.atmosphere, IG.geosphere);
	ChangeCirc(IG.ocean.circulationM, 0,0,0,0);
	IG.ocean.circulationM(7,13) = 20; IG.ocean.circulationM(13,7) = 20;
	IG.ocean.box.setP(7) = 2; IG.ocean.box.setP(6) = 1.2;
	IG.param.PAZarea = IG.ocean.box.Area(7); IG.param.PAZiceX = 0.0; IG.ocean.box.Area(7) = (1-IG.param.PAZiceX) * IG.param.PAZarea;
	InitCirc(IG.ocean.circulationM,IG.ocean.box.vol,IG.ocean.box.vol_inv);
	InitVolSchemes(IG.ocean.Schemes.MDtoDO, IG.ocean.Schemes.DOtoMD,IG.ocean.Schemes.MDtoDOviaNA, IG.ocean.Schemes.DOtoMDviaAApLL, IG.ocean.Schemes.UOtoDO, IG.ocean.Schemes.DOtoUO, IG.ocean.Schemes.NO);
	IG.ocean.VolParams.scale = 1;
	//ChangeVol(IG.ocean.Schemes.NO, IG.ocean.circulationM, IG.ocean.tracer, IG.ocean.venttracer, IG.ocean.box.vol, IG.ocean.box.vol_inv, IG.ocean.VolParams);
    //IG.ocean.tracer.Temp = Vector18f::Constant(5);
	printf("Model has been initialized with IG parameter settings.");


	printf("Model is spinning up for 1,000,000 years ...");
    RunEx(IG.ocean, IG.atmosphere, IG.geosphere, IG.param, SpinUpYears);
    printf("Initialization complete. Spun-up model resides in 'IG' container.\n");
    printf("CO2=%f; d14C=%f; D14C=%f; d14Corg=%f\n", IG.atmosphere.ppm, IG.atmosphere.dn14/IG.atmosphere.ppm, D14Ccalc(IG.atmosphere.ppm, IG.atmosphere.dn13, IG.atmosphere.dn14), IG.geosphere.d14Corg);
    double tmp=OcAvConc(IG.ocean.tracer.C,IG.ocean.box.vol, 0, 18);
    printf("Ocean average [DIC]=%f;\n",tmp);

	return IG;
}

double newCO2(experiment test, double ALKrate)
{
	test.param.ALK.annualALK = ALKrate; // PgC/yr
	RunEx(test.ocean, test.atmosphere,test.geosphere, test.param, test.param.ALK.tau);
return test.atmosphere.ppm;
}

double CO2_bisection(experiment &initial)
{

		experiment step = initial;

		double tolerance = 0.01;
		double highR = 5.0e+17;
		double lowR = 0;
		//double InitialGuessR = -(step.param.Cinc.initialD14C - step.param.Cinc.targetD14C)/(-45);
		double midR = (highR + lowR)/2;

		double final = newCO2(step,midR);
		//cout<<"final is "<<final<<"\n";
		double target = 500;

		if (newCO2(step,0)<target){
			return 0.0;
		}
		else
		{


		//cout<<"0"<<" "<<midR<<" "<<final-target<<"\n";

		int counter = 0;
		while (abs(final-target) > tolerance) {
			counter += 1;
			if ((final-target) > 0) {
				lowR = midR;
				midR = (highR + lowR)/2;
			}
			else {
				highR = midR;
				midR = (highR + lowR)/2;
			}
			final = newCO2(step,midR);
			cout<<counter<<" "<<midR<<" "<<final-target<<"\n";
		}
		//cout<<"TIME STEP DONE"<<"\n";
		return midR;
		}
}

double newpH(experiment test, double ALKrate)
{
	test.param.ALK.annualALK = ALKrate; // PgC/yr
	RunEx(test.ocean, test.atmosphere,test.geosphere, test.param, test.param.ALK.tau);
	finalCsolve(test.ocean.tracer.C, test.ocean.tracer.Alk,test.ocean.tracer.Sal, test.ocean.Ksurf.K0, test.ocean.Ksurf.K1, test.ocean.Ksurf.K2, test.ocean.Ksurf.Kb,test.ocean.Ksurf.Ks,test.param.CaX, test.ocean.box.Csolved);

	double H0 = test.ocean.box.Csolved.H(0);
	double H1 = test.ocean.box.Csolved.H(1);
	double H2 = test.ocean.box.Csolved.H(2);
	double H3 = test.ocean.box.Csolved.H(3);
	double H4 = test.ocean.box.Csolved.H(4);
	double H5 = test.ocean.box.Csolved.H(5);
	double H6 = test.ocean.box.Csolved.H(6);
	double H7 = test.ocean.box.Csolved.H(7);
	double Htotal = 1e-6*((H0+H1+H2+H3+H4+H5+H6+H7)/8);
	double pH_surf = -1 * log10(Htotal);
return pH_surf;
}


double pH_bisection(experiment &initial)
{

		experiment step = initial;

		double tolerance = 0.01;
		double highR = 5.0e+17;
		double lowR = 0;
		//double InitialGuessR = -(step.param.Cinc.initialD14C - step.param.Cinc.targetD14C)/(-45);
		double midR = (highR + lowR)/2;

		double final = newpH(step,midR);
		//cout<<"final is "<<final<<"\n";
		double target = 8;

		if (newpH(step,0)>target){
			return 0.0;
		}
		else
		{


		//cout<<"0"<<" "<<midR<<" "<<final-target<<"\n";

		int counter = 0;
		while (abs(final-target) > tolerance) {
			counter += 1;
			if ((final-target) > 0) {
				highR = midR;
				midR = (highR + lowR)/2;
			}
			else {
				lowR = midR;
				midR = (highR + lowR)/2;
			}
			final = newpH(step,midR);
			cout<<counter<<" "<<midR<<" "<<final-target<<"\n";
		}
		//cout<<"TIME STEP DONE"<<"\n";
		return midR;
		}
}

int main(void)
{
	//SIP test run
	if (false)
	{
		// still need to fix if I want this to be as IG spin up or LGM spin up
		experiment Initial = IGSpinUp(1000000);

		oceanS			ocean		= Initial.ocean;
		atmosphereS		atmosphere	= Initial.atmosphere;
		geosphereS		geosphere	= Initial.geosphere;
		parametersS		param		= Initial.param;

		//output setup
		string LGMfilepath = "/home/christopher/CYCLOPS";
		// input observations
		// historical and RCP
		InitALKforcing(Initial.param.ALKF);

		// CO2 seems to rise faster than real life.
		Initial.param.year = -1980;
		experiment ex = Initial;

		ex.param.ALK.flag = 1;
		ex.param.ALK.OUTPUT.resize(3021,10); // shaping the output array
		ex.param.ALK.tau = 1; // yearly data

		// past
		for (int step=0;step<=40;step++)
		{

			if (ex.param.year > -2021)
			{
				ex.param.ALK.Crate_PgCperYear = ex.param.ALKF.RCPCO2Forcing[step];
				ex.param.ALK.annualALK = 0;
			}

			ex.param.ALKout.flag = 2;
			RunEx( ex.ocean,  ex.atmosphere, ex.geosphere,  ex.param,  ex.param.ALK.tau);

		}
		// future
		for (int step=0;step<=79;step++)
		{
			// future
			if (ex.param.year <= -2021)
			{
				ex.param.ALK.Crate_PgCperYear = ex.param.ALKF.RCPCO2Forcing[41+step];
				ex.param.ALK.annualALK = ex.param.ALKF.ALKaddForcing[step];
				//ex.param.ALK.annualALK = 0;
			}

			ex.param.ALKout.flag = 2;
			RunEx( ex.ocean,  ex.atmosphere, ex.geosphere,  ex.param,  ex.param.ALK.tau);
		}

		//Switch on for Chris's longer experiment
		//for chris' longer experiment I just need to change the output size, th years where they want output, and the size of the forcing file
		if (ex.param.year < -2100)
		{
			for (int step = 0;step<=299;step++)
			{
				ex.param.ALK.Crate_PgCperYear = ex.param.ALKF.RCPCO2Forcing[120];
				//ex.param.ALK.annualALK = ex.param.ALKF.ALKaddForcing[step];
				ex.param.ALK.annualALK = ex.param.ALKF.ALKaddForcing[step+80];
				RunEx( ex.ocean,  ex.atmosphere, ex.geosphere,  ex.param,  ex.param.ALK.tau);
			}

			ex.param.ALK.Crate_PgCperYear = 0;
			ex.param.ALK.annualALK = 0;
			RunEx( ex.ocean,  ex.atmosphere, ex.geosphere,  ex.param,  2600);

		}
		// if (ex.param.year < -2100)
// 			{
// 				ex.param.ALK.Crate_PgCperYear = 0;
// 				ex.param.ALK.annualALK = 0;
// 				RunEx( ex.ocean,  ex.atmosphere, ex.geosphere,  ex.param, 2900);
// 			}

		// output
		// char filename[128];
	    // sprintf(filename,"/test_surfTnoALK.txt");
	    // ofstream fout(LGMfilepath+filename);
	    // fout<< ex.param.ALK.OUTPUT;
	    // fout.close();
	}

	//SIP minimization run
	if (false)
	{
		experiment Initial = IGSpinUp(1000000);

		oceanS			ocean		= Initial.ocean;
		atmosphereS		atmosphere	= Initial.atmosphere;
		geosphereS		geosphere	= Initial.geosphere;
		parametersS		param		= Initial.param;

		//output setup
		string LGMfilepath = "/home/christopher/CYCLOPS";
		// input observations
		// historical and RCP
		InitALKforcing(Initial.param.ALKF);

		// CO2 seems to rise faster than real life.
		Initial.param.year = -1980;
		experiment ex = Initial;

		ex.param.ALK.flag = 1;
		ex.param.ALK.OUTPUT.resize(121,10); // shaping the output array
		ex.param.ALK.tau = 1; // yearly data
		// past
		for (int step=0;step<=40;step++)
		{

			if (ex.param.year > -2021)
			{
				ex.param.ALK.Crate_PgCperYear = ex.param.ALKF.RCPCO2Forcing[step];
				ex.param.ALK.annualALK = 0;
			}

			ex.param.ALKout.flag = 2;
			RunEx( ex.ocean,  ex.atmosphere, ex.geosphere,  ex.param,  ex.param.ALK.tau);

		}
		// future
		for (int step=0;step<=79;step++)
		{
			// future
			if (ex.param.year <= -2021)
			{
				ex.param.ALK.Crate_PgCperYear = ex.param.ALKF.RCPCO2Forcing[41+step];
				ex.param.ALK.annualALK = pH_bisection(ex);
				//ex.param.ALK.annualALK = 0;
			}

			ex.param.ALKout.flag = 2;
			RunEx( ex.ocean,  ex.atmosphere, ex.geosphere,  ex.param,  ex.param.ALK.tau);
		}

		// for Chris's longer experiment
		// for chris' longer experiment I just need to change the output size, th years where they want output, and the size of the forcing file
		if (ex.param.year < -2100)
		{
			for (int step = 0;step<=299;step++)
			{
				ex.param.ALK.Crate_PgCperYear = ex.param.ALKF.RCPCO2Forcing[120];
				//ex.param.ALK.annualALK = ex.param.ALKF.ALKaddForcing[step];
				ex.param.ALK.annualALK = ex.param.ALKF.ALKaddForcing[step+80];
				RunEx( ex.ocean,  ex.atmosphere, ex.geosphere,  ex.param,  ex.param.ALK.tau);
			}

			ex.param.ALK.Crate_PgCperYear = 0;
			ex.param.ALK.annualALK = 0;
			RunEx( ex.ocean,  ex.atmosphere, ex.geosphere,  ex.param,  2600);

		}

	// output
	// char filename[128];
    // sprintf(filename,"/testwithsurfT.txt");
    // ofstream fout(LGMfilepath+filename);
    // fout<< ex.param.ALK.OUTPUT;
    // fout.close();
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Christopher 2D Minimization run
	if (true)
	{
		experiment Initial = IGSpinUp(1000000);

		oceanS			ocean		= Initial.ocean;
		atmosphereS		atmosphere	= Initial.atmosphere;
		geosphereS		geosphere	= Initial.geosphere;
		parametersS		param		= Initial.param;

		string LGMfilepath = "/home/christopher/CYCLOPS";

		InitALKforcing(Initial.param.ALKF);

		Initial.param.year = -1980;
		experiment ex = Initial;

		ex.param.ALK.flag = 1;
		ex.param.ALK.OUTPUT.resize(121,10); // shaping the output array (1980 to 2100) with surfT so 10 columns
		ex.param.ALK.tau = 1; // timestep = 1

		// int counter_step = 0; // will print this out to make sure

		// past
		for (int step=0;step<=40;step++)
		{

			if (ex.param.year > -2021)
			{
				ex.param.ALK.Crate_PgCperYear = ex.param.ALKF.RCPCO2Forcing[step];
				ex.param.ALK.annualALK = 0;
			}

			ex.param.ALKout.flag = 2;
			RunEx( ex.ocean,  ex.atmosphere, ex.geosphere,  ex.param,  ex.param.ALK.tau);

		}

		for (int step=0;step<=79;step++)
		{
			// future
			if (ex.param.year <= -2021)
			{
				ex.param.ALK.Crate_PgCperYear = ex.param.ALKF.RCPCO2Forcing[41+step];

				MinimizationFunctor RunMinimization(ex, ex.param.year); // RunMinimization is an instance of the functor

				// CG MINIMIZATION Polak-Ribiere CALL FUNCTOR OPERATOR
				VecDoub SDP(1);
				SDP[0] = 0.2; // alkalinity addition in MOLES
				Frprmn<MinimizationFunctor> frprmn(RunMinimization);
				SDP = frprmn.minimize(SDP);

				if (SDP[0] < 0)
				{
					SDP[0] = 0;
				}

				ex.param.ALK.annualALK = SDP[0];
				// counter_step += 1;
				cout<<"ALK: "<<SDP[0]<<"\n";
			}

			ex.param.ALKout.flag = 2;
			RunEx( ex.ocean,  ex.atmosphere, ex.geosphere,  ex.param,  ex.param.ALK.tau);
		}

		// outputs
		// char filename[128];
		// sprintf(filename,"/2Dmin85.txt");
		// ofstream fout(LGMfilepath+filename);
	    // fout<< ex.param.ALK.OUTPUT;
	    // fout.close();
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	}
	printf("REACHED END OF CODE; passing exit code 0.\n");
	return 0;
}


// Right now, the optimization is failing because surfT starts out as inf
