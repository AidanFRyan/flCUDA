// FlipFluid Basis Operator

#ifndef _OP_FLIPFLUIDBASIS_H
#define _OP_FLIPFLUIDBASIS_H

#include "Include/CuSolver.h"
#include <Thinking.h>
#include "resource.h"
#include "Operator.h"



#define OP_FLIPFLUIDBASIS_CLASS_ID		Class_ID(0x57000317, 0x18e17e1)

#define OP_FLIPFLUIDBASIS_PARAM_BLOCK 0

namespace Cu {
	template <typename T>
	class CuSolver;
}

struct ParticleData
{
	int    pid;
	Point3 pos;
	Point3 vel;
	float  mass;
	float  dt_factor;
};


struct FlipINodeBoundaryData
{
	//With the function GetMesh() in the TP_MeshNode class you get the mesh from the node 
	TP_MeshNode *meshnode;

	Interval tm_ivalid;
	Matrix3 objToWorld;
	Matrix3 worldToObject;
	Matrix3 objToWorld_dt;
	Matrix3 worldToObject_dt;

	FlipINodeBoundaryData() { /*inode = NULL;*/  meshnode = NULL; }
	~FlipINodeBoundaryData()
	{
		if(meshnode) GetGlobalMeshManager()->Remove(meshnode);
	}
};

struct FlipFluidBasisThreadData
{
	Tab<ParticleData> datas;
	float dt;
};


class OP_FlipFluidBasis_DlgProc;

class OP_FlipFluidBasis : public DynOP
{
private:
	friend class OP_FlipFluidBasis_DlgProc;

	IParamMap2         *pmap;
	IParamBlock2       *pb2;
	OP_FlipFluidBasis_DlgProc *dlgProc;

	PGroup *mPGroup;
	BOOL   *mPGroupSub;
	PGroup *mSPGroup;
	BOOL   *mSPGroupSub;
	int     mType;
	float   mVoxelSize;
	int     mSubSamples;
	Cu::CuSolver<double>* solver;
	Cu::Vec3<double> *prevV, *prevP;
	bool mDoInitial;
	bool mInvalidBoundaryDatas;
	Tab<FlipINodeBoundaryData*> mINodeBoundaryDatas;

	TP_CalculateList *calculateList;


	int Update(InOutNode *inout, DYN &dyn);

	
	//int  ReadBParticleDatas(DYN &dyn, BodyforceThreadDataOVDB *tdata);
	void ReadParticleDatas(InOutNode *inout , DYN &dyn, FlipFluidBasisThreadData *tdata);
	void WriteParticleDatas(InOutNode *inout, DYN &dyn, FlipFluidBasisThreadData *tdata);
	void SetupBoundaryDatas(InOutNode *inout, DYN &dyn, FlipFluidBasisThreadData *tdata);
	void RemoveBoundaryDatas();

public:
	OP_FlipFluidBasis();
	~OP_FlipFluidBasis();


	void SetPGroup(PGroup *group);
	PGroup *GetPGroup();

	void SetSPGroup(PGroup *group);
	PGroup *GetSPGroup();


	//From DynBase
	int Type();
	BOOL IsPShapeOwner() { return FALSE; }
	int Version();
	TCHAR* GetName();
	BOOL InitInOutputs(InOutNode *inout);
	//BOOL InitInOutputsGroup(InOutNode *inout);
	TCHAR *GetInOutputName(InOutNode *inout, int id, BOOL input);
	int Calculate(int id, void *val, InOutNode *inout, DYN &dyn);
	int SetInReCalculate(int id, InOutNode *inout, DYN &dyn);
	DynResult EnumDyn(InOutNode *inout, int message, void *arg);
	//int TP_ParticleDraw(InOutNode *inout, int pid, TP_ParticleDrawInfo &drawinfo);

	//From ReferenceTarget
	int NumRefs();
	RefTargetHandle GetReference(int i);
	void SetReference(int i, RefTargetHandle rtarg);
	RefTargetHandle Clone(RemapDir &remap = DefaultRemapDir());
#if GET_MAX_RELEASE(VERSION_3DSMAX) < 17000
	RefResult       NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget, PartID& partID, RefMessage message);
#else
	RefResult       NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget, PartID& partID, RefMessage message, BOOL propagate);
#endif

	//From Animatable
	Class_ID ClassID();
	void GetClassName(TSTR& s);

	int NumParamBlocks();
	IParamBlock2* GetParamBlock(int i = 0);
	IParamBlock2* GetParamBlockByID(short id);

	int NumSubs();
	Animatable *SubAnim(int i);
	TSTR SubAnimName(int i);
	void BeginEditParams(IObjParam *ip, ULONG flags, Animatable *prev = NULL);
	void EndEditParams(IObjParam *ip, ULONG flags, Animatable *next = NULL);

	//IOResult Save(ISave *isave);
	//IOResult Load(ILoad *iload);

	void DeleteThis();
};




class OP_FlipFluidBasis_DlgProc : public ParamMap2UserDlgProc
{
private:
	OP_FlipFluidBasis *op;
	IParamMap2		  *map;
	HWND			   SBox;
	HWND			   GBox;
	
	DynamicInfo info;

public:
	OP_FlipFluidBasis_DlgProc(OP_FlipFluidBasis *p);

	void SetObject(OP_FlipFluidBasis *p);

	void UpdateGroups();
	
	INT_PTR DlgProc(TimeValue t, IParamMap2 *map, HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
	void DeleteThis();
};



#endif