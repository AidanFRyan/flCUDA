//OP_FlipFluidBasis 
//#include <Thinking.h>
#include "OP_FlipFluidBasis.h"
#include "MatterWaves.h"
#include "Calculate.h"

//class descriptor
class OP_FlipFluidBasis_ClassDesc : public ClassDesc2
{
public:
	int IsPublic() { return FALSE; }
	void *Create(BOOL loading = FALSE) { return new OP_FlipFluidBasis; }
	const TCHAR* ClassName() { return GetString(IDS_OP_FLIPFLUIDBASIS_CLASS, hInstance); }
	SClass_ID SuperClassID() { return REF_TARGET_CLASS_ID; }
	Class_ID ClassID() { return OP_FLIPFLUIDBASIS_CLASS_ID; }
	const TCHAR* Category() { return GetString(IDS_CATEGORY_FLIPFLUID, hInstance); }
	const TCHAR* InternalName() { return _T("OP_FlipFluidBasis"); }
	HINSTANCE HInstance() { return hInstance; }
};

static OP_FlipFluidBasis_ClassDesc	op_flipfluidbasis_Desc;


ClassDesc* GetOP_FlipFluidBasis_Desc() { return &op_flipfluidbasis_Desc; }
//end class descriptor


//Parameter
#define FLIP_SPGROUP			50	
#define FLIP_SPGROUP_SUB		51	
#define FLIP_PGROUP				52	
#define FLIP_PGROUP_SUB			53	
#define FLIP_TYPE				54	
#define FLIP_VOXEL_SIZE			55	
#define FLIP_SUB_SAMPLES		56
#define FLIP_BOUNDARY_INODES	57


#define FLIP_PGROUP_SUB_DEF		FALSE
#define FLIP_SPGROUP_SUB_DEF	FALSE

#define FLIP_TYPE_1		0
#define FLIP_TYPE_2		1
#define FLIP_TYPE_3		2

#define FLIP_TYPE_DEF	0
#define FLIP_TYPE_MIN	0
#define FLIP_TYPE_MAX	2


#define FLIP_VOXEL_SIZE_DEF		0.025f
#define FLIP_VOXEL_SIZE_MIN		0.0
#define FLIP_VOXEL_SIZE_MAX		MAX_FVALUE

#define FLIP_SUB_SAMPLES_DEF	0
#define FLIP_SUB_SAMPLES_MIN	0
#define FLIP_SUB_SAMPLES_MAX	MAX_IVALUE

//Node Input 
#define FLIP_ON_IN				0
#define FLIP_TIME_IN			1
#define FLIP_SPGROUP_IN			2
#define FLIP_SPGROUP_SUB_IN		3
#define FLIP_PGROUP_IN			4
#define FLIP_PGROUP_SUB_IN		5
#define FLIP_TYPE_IN			6
#define FLIP_VOXEL_SIZE_IN		7
#define FLIP_SUB_SAMPLES_IN		8


class BoundaryObjectValidator : public PBValidator
{
public:
	BOOL Validate(PB2Value& v){ return TRUE; }

	BOOL Validate(PB2Value& v, ReferenceMaker* owner, ParamID id, int tabIndex)
	{
		if(id == FLIP_BOUNDARY_INODES)
		{
			if(!v.r) return FALSE;

			int i;

			for(i = ((OP_FlipFluidBasis*)owner)->GetParamBlock()->Count(id) -1; i >= 0; i--)
			{
				if(v.r == ((OP_FlipFluidBasis*)owner)->GetParamBlock()->GetINode(id, 0, i)) return FALSE; 
			}
		}

		return TRUE; 
	}
};

BoundaryObjectValidator object_validator;


static ParamBlockDesc2 op_flipfluidbasis_ParamBlock(
	OP_FLIPFLUIDBASIS_PARAM_BLOCK, _T("Parameters"), 0, &op_flipfluidbasis_Desc, P_AUTO_CONSTRUCT, 0,

	// params
	FLIP_SPGROUP, _T("SourcePGroup"), TYPE_REFTARG, P_NO_REF, IDS_OP_FLIPFLUIDBASIS_SPGROUP,
	p_classID, PGROUP_CLASS_ID,
	p_end,

	FLIP_SPGROUP_SUB, _T("SourcePGroupSub"), TYPE_BOOL, P_ANIMATABLE, IDS_OP_FLIPFLUIDBASIS_SPGROUP_SUB,
	p_default, FLIP_SPGROUP_SUB_DEF,
	p_ui, TYPE_SINGLECHEKBOX, IDC_OP_FLIPFLUIDBASIS_SPGROUP_SUB,
	p_end,



	FLIP_PGROUP, _T("TargetPGroup"), TYPE_REFTARG, P_NO_REF, IDS_OP_FLIPFLUIDBASIS_PGROUP,
	p_classID, PGROUP_CLASS_ID,
	p_end,

	FLIP_PGROUP_SUB, _T("TargetPGroupSub"), TYPE_BOOL, P_ANIMATABLE, IDS_OP_FLIPFLUIDBASIS_PGROUP_SUB,
	p_default, FLIP_PGROUP_SUB_DEF,
	p_ui, TYPE_SINGLECHEKBOX, IDC_OP_FLIPFLUIDBASIS_PGROUP_SUB,
	p_end,


	FLIP_TYPE, _T("Type"), TYPE_INT, 0, IDS_OP_FLIPFLUIDBASIS_TYPE,
	p_default, FLIP_TYPE_DEF,
	p_range, FLIP_TYPE_MIN, FLIP_TYPE_MAX,
	p_ui, TYPE_INTLISTBOX, IDC_OP_FLIPFLUIDBASIS_TYPE, 3,
	IDS_OP_FLIPFLUIDBASIS_TYPE_1,
	IDS_OP_FLIPFLUIDBASIS_TYPE_2,
	IDS_OP_FLIPFLUIDBASIS_TYPE_3,
	p_end,

	FLIP_VOXEL_SIZE, _T("VoxelSize"), TYPE_WORLD, P_ANIMATABLE, IDS_OP_FLIPFLUIDBASIS_VOXEL_SIZE,
	p_default, FLIP_VOXEL_SIZE_DEF,
	p_range, FLIP_VOXEL_SIZE_MIN, FLIP_VOXEL_SIZE_MAX,
	p_ui, TYPE_SPINNER, EDITTYPE_UNIVERSE, IDC_OP_FLIPFLUIDBASIS_VOXEL_SIZE, IDC_OP_FLIPFLUIDBASIS_VOXEL_SIZE_SPIN, 0.01f,
	p_end,

	FLIP_SUB_SAMPLES, _T("SubSamples"), TYPE_INT, P_ANIMATABLE, IDS_OP_FLIPFLUIDBASIS_SUB_SAMPLES,
	p_default, FLIP_SUB_SAMPLES_DEF,
	p_range, FLIP_SUB_SAMPLES_MIN, FLIP_SUB_SAMPLES_MAX,
	p_ui, TYPE_SPINNER, EDITTYPE_INT, IDC_OP_FLIPFLUIDBASIS_SUB_SAMPLES, IDC_OP_FLIPFLUIDBASIS_SUB_SAMPLES_SPIN, 0.01f,
	p_end,

	FLIP_BOUNDARY_INODES, _T("BoundaryObjects"), TYPE_INODE_TAB, 0, P_VARIABLE_SIZE, IDS_OP_FLIPFLUIDBASIS_BOUNDARY_INODES,
	p_validator, &object_validator,
	p_ui, TYPE_NODELISTBOX, IDC_OP_FLIPFLUIDBASIS_INODE_LIST, IDC_OP_FLIPFLUIDBASIS_PICKNODE, 0, IDC_OP_FLIPFLUIDBASIS_REMOVENODE,
	p_end,

	p_end,

	p_end);





int OP_FlipFluidBasis::Update(InOutNode *inout, DYN &dyn)
{
	int ret, changed = 0;
	INOUT_GetInValue gi;

	gi.last_ivalid = ivalid;

	gi.t = inout->GetInTime();
	gi.pb = pb2;



	gi.id = FLIP_SPGROUP;

	if ((ret = inout->GetInValue(gi, FLIP_SPGROUP_IN, &mSPGroup, dyn)) < 0) return -1;
	if (ret > 0)
	{
		changed = 1;
	}


	gi.id = FLIP_SPGROUP_SUB;

	if ((ret = inout->GetInValue(gi, FLIP_SPGROUP_SUB_IN, &mSPGroupSub, dyn)) < 0) return -1;
	if (ret > 0)
	{
		changed = 1;
	}



	gi.id = FLIP_PGROUP;

	if ((ret = inout->GetInValue(gi, FLIP_PGROUP_IN, &mPGroup, dyn)) < 0) return -1;
	if (ret > 0)
	{
		changed = 1;
	}


	gi.id = FLIP_PGROUP_SUB;

	if ((ret = inout->GetInValue(gi, FLIP_PGROUP_SUB_IN, &mPGroupSub, dyn)) < 0) return -1;
	if (ret > 0)
	{
		changed = 1;
	}


	gi.id = FLIP_TYPE;

	if ((ret = inout->GetInValue(gi, FLIP_TYPE_IN, &mType, dyn)) < 0) return -1;
	if (ret > 0)
	{
		ClampMinMax(mType, FLIP_TYPE_MIN, FLIP_TYPE_MAX);

		changed = 1;
	}

	gi.id = FLIP_VOXEL_SIZE;

	if ((ret = inout->GetInValue(gi, FLIP_VOXEL_SIZE_IN, &mVoxelSize, dyn)) < 0) return -1;
	if (ret > 0)
	{
		ClampMinMax(mVoxelSize, FLIP_VOXEL_SIZE_MIN, FLIP_VOXEL_SIZE_MAX);

		changed = 1;
	}

	gi.id = FLIP_SUB_SAMPLES;

	if ((ret = inout->GetInValue(gi, FLIP_SUB_SAMPLES_IN, &mSubSamples, dyn)) < 0) return -1;
	if (ret > 0)
	{
		ClampMinMax(mSubSamples, FLIP_SUB_SAMPLES_MIN, FLIP_SUB_SAMPLES_MAX);

		changed = 1;
	}



	if (!gi.ivalid.Empty()) ivalid = gi.ivalid;


	return changed;
}



int OP_FlipFluidBasis::Calculate(int id, void *val, InOutNode *inout, DYN &dyn)
{
	if (id >= 0)
	{
		return FALSE;
	}

	if (Update(inout, dyn) < 0) return FALSE;

	if (!mPGroup) return FALSE;

	FlipFluidBasisThreadData tdata;

	tdata.dt = float(dyn.global_dt) / float(TIME_TICKSPERSEC); //secound based delta time from tp

	

	//Read the particle data from tp
	ReadParticleDatas(inout, dyn, &tdata);
	
	//Read meshes and transformation matrix from max
	SetupBoundaryDatas(inout, dyn, &tdata);

	cudaSetDevice(0);
	cudaDeviceSynchronize();
	//solver->resizeTime(tdata.dt);
	//improve efficiency on this, cache on GPU with single CuSolver for class, single initialization from first frame particles and keep data on GPU
	//With a class bool variable it is better to control when to make a initialization
	//But keep in mind, the data can't left on the GPU, so when the solver is done and anyone use the gpu, for rendering as sample the data are corrupt   
	if (mDoInitial == true) {
		if (pb2->Count(FLIP_BOUNDARY_INODES)) {
			Point3 min, max;
			Box3 bb;
			bb = mINodeBoundaryDatas[0]->meshnode->GetBoundingBox();
			bb = bb * mINodeBoundaryDatas[0]->objToWorld_dt;
			min = bb.pmin;
			max = bb.pmax;
			for (int i = 1; i < pb2->Count(FLIP_BOUNDARY_INODES); ++i) {
				bb = mINodeBoundaryDatas[i]->meshnode->GetBoundingBox();
				bb = bb * mINodeBoundaryDatas[i]->objToWorld_dt;
				if (bb.pmin.x < min.x)
					min.x = bb.pmin.x;
				if (bb.pmin.y < min.y)
					min.y = bb.pmin.y;
				if (bb.pmin.z < min.z)
					min.z = bb.pmin.z;
				if (bb.pmax.x < max.x)
					max.x = bb.pmax.x;
				if (bb.pmax.y < max.y)
					max.y = bb.pmax.y;
				if (bb.pmax.z < max.z)
					max.z = bb.pmax.z;
			}
			solver->setDimensions(max.x - min.x, max.y - min.y, max.z - min.z);	//placeholder dimensions, need to read in geometry dimensions
			solver->setWorldPosition(min.x, min.y, min.z);
			solver->setdx(mVoxelSize);
		}
		solver->setSubSamples(mSubSamples);

		if (prevV != nullptr) {
			delete[] prevV;
			prevV = nullptr;
		}
		if (prevP != nullptr) {
			delete[] prevP;
			prevP = nullptr;
		}

		mDoInitial = false;
	}
	solver->readParticlesFromTP(&tdata, prevV, prevP);
	cudaDeviceSynchronize();
	if(mType == 0)
		solver->advectSingleFrameCPU();
	else if (mType == 1)
		solver->advectSingleFrameGPU();
	cudaDeviceSynchronize();
	solver->writeParticlesToTP(&tdata, prevV, prevP);
	cudaDeviceSynchronize();
	//write back the changed velocity of the particles to tp
	WriteParticleDatas(inout, dyn, &tdata);

	return TRUE;
}

void OP_FlipFluidBasis::SetupBoundaryDatas(InOutNode *inout, DYN &dyn, FlipFluidBasisThreadData *tdata)
{
int i, count;
FlipINodeBoundaryData *ndata;
INode* inode = NULL;

	//Checked the picked max nodes in the parameter block and load the meshes and make a transformation matrix update
	//mInvalidBoundaryDatas is a bool, at load or create and when the user picked or removed a max node this flag will turned on
	
	count = pb2->Count(FLIP_BOUNDARY_INODES);

	if(mInvalidBoundaryDatas == true)
	{
		RemoveBoundaryDatas();
		
		mINodeBoundaryDatas.SetCount(count);

		for(i = 0; i < count; i++)
		{
			mINodeBoundaryDatas[i] = NULL;
		}

		mInvalidBoundaryDatas = false;
	}

	

	for(i = 0; i < count; i++)
	{
		inode =  pb2->GetINode(FLIP_BOUNDARY_INODES, dyn.global_time, i);
		
		if(!inode)
		{
			if(mINodeBoundaryDatas[i]) delete mINodeBoundaryDatas[i]; 
			
			mINodeBoundaryDatas[i] = NULL;

			continue;
		}
		
		if(!mINodeBoundaryDatas[i])
		{
			ndata = new FlipINodeBoundaryData();
			
			ndata->objToWorld_dt = inode->GetObjTMAfterWSM(dyn.global_time - dyn.global_dt);
			ndata->worldToObject_dt = Inverse(ndata->objToWorld_dt);
			ndata->tm_ivalid.SetEmpty();

			 mINodeBoundaryDatas[i] = ndata;
		}
		else
			ndata = mINodeBoundaryDatas[i];	
	
		if(ndata->meshnode && !ndata->meshnode->ValidInterval().InInterval(dyn.global_time))
		{
			GetGlobalMeshManager()->Remove(ndata->meshnode);
			ndata->meshnode = NULL;
		}

		if(!ndata->meshnode)
		{
			ndata->meshnode = GetGlobalMeshManager()->Create(inode, dyn.global_time);

			//With the function GetMesh() in the TP_MeshNode class you get the max mesh from the node, with all vertex and face information
			//the vertex coordinates are in object space, to get the current world coordinates you must multiply with the objToWorld_dt matrix
		}
		

		if(!ndata->tm_ivalid.InInterval(dyn.global_time))
		{	
			ndata->tm_ivalid.SetInfinite();
			
			ndata->objToWorld    = ndata->objToWorld_dt;
			ndata->worldToObject    = ndata->worldToObject_dt;
			ndata->objToWorld_dt = inode->GetObjTMAfterWSM(dyn.global_time, &ndata->tm_ivalid);
			ndata->worldToObject_dt = Inverse(ndata->objToWorld_dt);

			//objToWorld is the last transformation matrix and objToWorld_dt the current one of the node, if the matrix not animated than both are the same
		}
	}
}

void OP_FlipFluidBasis::RemoveBoundaryDatas()
{
	int count = mINodeBoundaryDatas.Count();
		
	for(int i = 0; i < count; i++)
	{
		if(mINodeBoundaryDatas[i]) delete mINodeBoundaryDatas[i];
	}

	mINodeBoundaryDatas.SetCount(0);
}


void OP_FlipFluidBasis::ReadParticleDatas(InOutNode *inout, DYN &dyn, FlipFluidBasisThreadData *tdata)
{
	Tab<PGroup*> groups;
	int i, pid, pcount, gcount;
	CNode *idnode;

	groups.SetCount(0);

	if (mPGroupSub)
		mPGroup->EnumDyn(NULL, DYN_MSG_GROUP_GETALL, &groups);
	else
		groups.Append(1, &mPGroup, 20);

	gcount = groups.Count();

	pcount = 0;

	for (i = 0; i < gcount; i++)
		pcount += groups[i]->partcount;

	if (!pcount) return;


	tdata->datas.SetCount(pcount);

	pcount = 0;

	for (i = 0; i < gcount; i++)
	{
		idnode = groups[i]->pidlist.GetFirstNode();

		for (; idnode != NULL; idnode = idnode->GetNextNode())
		{
			if (dyn.mastersystem->Alive(((ParticleNode*)idnode)->id))
			{
				pid = ((ParticleNode*)idnode)->id;

				tdata->datas[pcount].pid = pid;

				tdata->datas[pcount].pos = dyn.mastersystem->Position(pid);
				//max have a fixed integer ticks per secound time base, so the velocity must multipled with ticks per secound to get the real speed in secound
				tdata->datas[pcount].vel = dyn.mastersystem->Velocity(pid) * float(TIME_TICKSPERSEC); //max time values are in integer tick
				tdata->datas[pcount].dt_factor = dyn.mastersystem->DTFactor(pid);
				tdata->datas[pcount].mass = dyn.mastersystem->Mass(pid);

				pcount++;
			}
		}
	}

	if (tdata->datas.Count() != pcount) tdata->datas.SetCount(pcount, FALSE);
}

void OP_FlipFluidBasis::WriteParticleDatas(InOutNode *inout, DYN &dyn, FlipFluidBasisThreadData *tdata)
{
	int i, pcount;

	pcount = tdata->datas.Count();


	for (i = 0; i < pcount; i++)
	{
		//max have a fixed integer ticks per secound time base, so the velocity must divide with ticks per secound to get the max ticks speed in secound

		dyn.mastersystem->SetVelocity(tdata->datas[i].pid, tdata->datas[i].vel / float(TIME_TICKSPERSEC));

		dyn.mastersystem->SetPosition(tdata->datas[i].pid, tdata->datas[i].pos);

		/*
		if (tdata->datas[i].flags & 16)
		{
			dyn.mastersystem->Die(tdata->datas[i].pid);
		}
		*/
	}
}




TCHAR *OP_FlipFluidBasis::GetInOutputName(InOutNode *inout, int id, BOOL input)
{
	return NULL;
}

BOOL OP_FlipFluidBasis::InitInOutputs(InOutNode *inout)
{
	inout->AddInput(PORT_TYPE_GROUP, _T("SourcePGroup"), INOUT_FLAG_ENABLE | INOUT_FLAG_INVISIBLE);
	inout->AddInput(PORT_TYPE_BOOL, _T("SourcePGroupSubs"), INOUT_FLAG_ENABLE | INOUT_FLAG_INVISIBLE);
	inout->AddInput(PORT_TYPE_GROUP, _T("TargetPGroup"), INOUT_FLAG_ENABLE | INOUT_FLAG_INVISIBLE);
	inout->AddInput(PORT_TYPE_BOOL, _T("TargetPGroupSubs"), INOUT_FLAG_ENABLE | INOUT_FLAG_INVISIBLE);
	inout->AddInput(PORT_TYPE_INT, _T("Type"), INOUT_FLAG_ENABLE | INOUT_FLAG_INVISIBLE);
	inout->AddInput(PORT_TYPE_FLOAT, _T("VoxelSize"), INOUT_FLAG_ENABLE | INOUT_FLAG_INVISIBLE);
	inout->AddInput(PORT_TYPE_INT, _T("SubSamples"), INOUT_FLAG_ENABLE | INOUT_FLAG_INVISIBLE);

	return TRUE;
}

TCHAR* OP_FlipFluidBasis::GetName()
{
	return GetString(IDS_OP_FLIPFLUIDBASIS_CLASS, hInstance);
}

OP_FlipFluidBasis::OP_FlipFluidBasis()
{
	cudaSetDevice(1);
	cudaDeviceSynchronize();
	pb2 = NULL;
	op_flipfluidbasis_Desc.MakeAutoParamBlocks(this);

	pmap = NULL;
	dlgProc = NULL;
	mPGroup = NULL;
	mSPGroup = NULL;
	solver = new Cu::CuSolver<double>(1, 1, 1, 1, 0.1, 1.0/30);
	mVoxelSize = 1;
	prevV = nullptr;
	prevP = nullptr;
	mDoInitial = true;
	mInvalidBoundaryDatas = true;

	calculateList = NULL;//TP_CreateCalculateList();

	//cudaSetDevice(1);
}

OP_FlipFluidBasis::~OP_FlipFluidBasis()
{
	if (calculateList) calculateList->DeleteThis();
	delete solver;

	RemoveBoundaryDatas();
}


RefTargetHandle OP_FlipFluidBasis::Clone(RemapDir &remap)
{
	OP_FlipFluidBasis *obj = new OP_FlipFluidBasis;
	if (!obj)
		return NULL;

	obj->ReplaceReference(0, remap.CloneRef(pb2));

	return obj;
}

int OP_FlipFluidBasis::Version()
{
	return 0;
}

int OP_FlipFluidBasis::SetInReCalculate(int id, InOutNode *inout, DYN &dyn)
{
	dyn.calculate->AddOperator(inout);

	if (prevV != nullptr) {
		delete[] prevV;
		prevV = nullptr;
	}
	if (prevP != nullptr) {
		delete[] prevP;
		prevP = nullptr;
	}

	return TRUE;//inout->SetOutReCalculate(-1, dyn);
}

static inline void DrawCube(GraphicsWindow *gw, Point3 &pmin, Point3 &pmax)
{
	Point3 pts[9];
	Point3 tmppt[5];

	pts[0] = Point3(pmax[0], pmax[1], pmax[2]);
	pts[1] = Point3(pmax[0], pmax[1], pmin[2]);
	pts[2] = Point3(pmax[0], pmin[1], pmin[2]);
	pts[3] = Point3(pmax[0], pmin[1], pmax[2]);
	pts[4] = Point3(pmin[0], pmax[1], pmax[2]);
	pts[5] = Point3(pmin[0], pmax[1], pmin[2]);
	pts[6] = Point3(pmin[0], pmin[1], pmin[2]);
	pts[7] = Point3(pmin[0], pmin[1], pmax[2]);

	int k;

	gw->polyline(4, pts, NULL, NULL, TRUE, NULL);

	for (k = 0; k < 4; k++) tmppt[k] = pts[4 + k];
	gw->polyline(4, tmppt, NULL, NULL, TRUE, NULL);

	tmppt[0] = pts[0]; tmppt[1] = pts[4];
	gw->polyline(2, tmppt, NULL, NULL, FALSE, NULL);

	tmppt[0] = pts[1]; tmppt[1] = pts[5];
	gw->polyline(2, tmppt, NULL, NULL, FALSE, NULL);

	tmppt[0] = pts[2]; tmppt[1] = pts[6];
	gw->polyline(2, tmppt, NULL, NULL, FALSE, NULL);

	tmppt[0] = pts[3]; tmppt[1] = pts[7];
	gw->polyline(2, tmppt, NULL, NULL, FALSE, NULL);
}


DynResult OP_FlipFluidBasis::EnumDyn(InOutNode *inout, int message, void *arg)
{
	switch (message)
	{
	case DYN_MSG_INOUT_CHANGED:
		ivalid.SetEmpty();
		break;


	case DYN_MSG_DYNSET_USERMSG:
	{

	}
	break;

	case DYN_MSG_DYNSET_REPLACE_GROUP:
	{
		DYN_MSG_DYNSET_Replace_Group *replace = (DYN_MSG_DYNSET_Replace_Group *)arg;

		if (GetPGroup() == replace->group) SetPGroup(replace->to_group);
		if (GetSPGroup() == replace->group) SetSPGroup(replace->to_group);
	}
	break;

	case DYN_MSG_DYNSET_GETALLUSED_GROUPS:
	{
		PGroup *group = GetPGroup();
		if (group) ((Tab<PGroup*>*)arg)->Append(1, &group, 10);

		group = GetSPGroup();
		if (group) ((Tab<PGroup*>*)arg)->Append(1, &group, 10);
	}
	break;


	case DYN_MSG_GROUP_PREREMOVE:
	{
		if (((GroupRemove*)arg)->group->IsSubGroup(GetPGroup()))
			SetPGroup(NULL);

		if (((GroupRemove*)arg)->group->IsSubGroup(GetSPGroup()))
			SetSPGroup(NULL);
	}
	break;

	case DYN_MSG_GROUP_TREECHANGED:
		if (dlgProc)	dlgProc->UpdateGroups();
		break;


	//case DYN_MSG_DISPLAY:
	//{
	//	DynDisplay *data = (DynDisplay*)arg;
	//	DWORD origRndLimits;
	//	//DynDisplay *data;
	//	GraphicsWindow *gw;
	//	Color col(1.0f, 0.0f, 0.0f);
	//	Point3 pos(0.0f, 0.0f, 0.0f);
	//	Point3 p[5];
	//	Point3 pmin, pmax;
	//	bool multipass = false;


	//	gw = data->vpt->getGW();

	//	//set the drawing transformation, "Matrix3(1)" mean Identity, all points to draw are in world space
	//	gw->setTransform(Matrix3(1));

	//	origRndLimits = gw->getRndLimits();

	//	gw->setRndLimits(origRndLimits | GW_WIREFRAME | GW_Z_BUFFER);//make sure wireframe will displayed and the z (depth) buffer will evaluated

	//	if (gw->querySupport(GW_SPT_GEOM_ACCEL))
	//	{
	//		gw->multiplePass(-1, TRUE);
	//		multipass = true;
	//	}



	//	if (multipass) gw->startMarkers();


	//	gw->setColor(LINE_COLOR, col.r, col.g, col.b);

	//	gw->marker(&pos, POINT_MRKR);


	//	if (multipass)  gw->endMarkers();




	//	if (multipass) gw->startSegments();

	//	pmin = Point3(-5.0f, -5.0f, -5.0f);
	//	pmax = Point3(5.0f, 5.0f, 5.0f);

	//	gw->setColor(LINE_COLOR, 1.0f, 1.0f, 0.0f);

	//	DrawCube(gw, pmin, pmax);


	//	p[0] = Point3(0, 0, 0);
	//	p[1] = Point3(10.0f, 0.0f, 0.0f);

	//	gw->setColor(LINE_COLOR, 1.0f, 0.0f, 0.0f);

	//	gw->segment(p, 1);

	//	p[0] = Point3(0, 0, 0);
	//	p[1] = Point3(0.0f, 10.0f, 0.0f);

	//	gw->setColor(LINE_COLOR, 0.0f, 1.0f, 0.0f);

	//	gw->segment(p, 1);

	//	p[0] = Point3(0, 0, 0);
	//	p[1] = Point3(0.0f, 0.0f, 10.0f);

	//	gw->setColor(LINE_COLOR, 0.0f, 0.0f, 1.0f);

	//	gw->segment(p, 1);


	//	if (multipass) gw->endSegments();



	//	if (multipass) gw->multiplePass(-1, FALSE);


	//	gw->setRndLimits(origRndLimits);
	//}
	//break;

	//case DYN_MSG_GETWORLDBOX: //the bounding box in world coordinates, enclosed all points that will displayed in the GraphicsWindow
	//{
	//	DynGetBox *data = (DynGetBox*)arg;
	//	Box3 bbox; //empty box

	//	//add points, the box will enclose these point 
	//	bbox += Point3(-10.0f, -10.0f, -10.0f);
	//	bbox += Point3(10.0f, 10.0f, 10.0f);


	//	if (!bbox.IsEmpty()) data->box += bbox;
	//}
	//break;


	case DYN_MSG_DYNSET_GETSTARTTIME:
	{
		mDoInitial = true;
	}
	break;

	case DYN_MSG_DYNSET_UPDATE:
	{
		UpdateInfo *info = (UpdateInfo*)arg;

		if (info->dyn->calculate)
			info->dyn->calculate->AddOperator(inout);
	}
	break;
	}
	return DYN_SUCCEED;
}



#if GET_MAX_RELEASE(VERSION_3DSMAX) < 17000
RefResult OP_OVDB_Bodyforce::NotifyRefChanged(Interval changeInt, RefTargetHandle hTarget, PartID& partID, RefMessage message)
#else
RefResult OP_FlipFluidBasis::NotifyRefChanged(const Interval &changeInt, RefTargetHandle hTarget, PartID& partID, RefMessage message, BOOL propagate)
#endif
{
	switch (message)
	{
	case REFMSG_CHANGE:
		if (hTarget == pb2)
		{
			int index = -1;
			ivalid.SetEmpty();

			switch (pb2->LastNotifyParamID(index))
			{
			case FLIP_PGROUP:
			case FLIP_SPGROUP:
				if (dlgProc) dlgProc->UpdateGroups();
				break;

			case FLIP_VOXEL_SIZE:
				mDoInitial = true;
				break;

			case FLIP_BOUNDARY_INODES:
			{
				if (index >= 0 && !pb2->GetINode(FLIP_BOUNDARY_INODES, 0, index))
				{
					pb2->EnableNotifications(FALSE);
						pb2->Delete(FLIP_BOUNDARY_INODES, index, 1);
					pb2->EnableNotifications(TRUE);
				}

				if(pb2->Count(FLIP_BOUNDARY_INODES) != mINodeBoundaryDatas.Count())
					mInvalidBoundaryDatas = true;
			}
			break;
			}
		}
		break;
	}
	return REF_SUCCEED;
}

int OP_FlipFluidBasis::NumRefs()
{
	return 1;
}

RefTargetHandle OP_FlipFluidBasis::GetReference(int i)
{
	return pb2;
}

void OP_FlipFluidBasis::SetReference(int i, RefTargetHandle rtarg)
{
	pb2 = (IParamBlock2*)rtarg;
}

int OP_FlipFluidBasis::NumSubs()
{
	return 1;
}

Animatable* OP_FlipFluidBasis::SubAnim(int i)
{
	return pb2;
}

TSTR OP_FlipFluidBasis::SubAnimName(int i)
{
	return _T("Parameter");
}

void OP_FlipFluidBasis::BeginEditParams(IObjParam *ip, ULONG flags, Animatable *prev)
{
	if (pmap)
	{
		if (dlgProc)
			dlgProc->SetObject(this);
		else {
			dlgProc = new OP_FlipFluidBasis_DlgProc(this);
			pmap->SetUserDlgProc(dlgProc);
		}

		pmap->SetParamBlock(pb2);
	}
	else
	{
		dlgProc = new OP_FlipFluidBasis_DlgProc(this);
		pmap = CreateRParamMap2(pb2, GetRightIRendParams(), hInstance, MAKEINTRESOURCE(IDD_OP_FLIPFLUIDBASIS_UI), GetName(), 0, dlgProc);
	}
}

void OP_FlipFluidBasis::EndEditParams(IObjParam *ip, ULONG flags, Animatable *next)
{
	if (flags & END_EDIT_REMOVEUI)
	{
		if (pmap)
		{
			DestroyRParamMap2(pmap);
			pmap = NULL;
		}
		dlgProc = NULL;
	}
}



int OP_FlipFluidBasis::Type()
{
	return DYN_TYPE_OPERATOR;
}

Class_ID OP_FlipFluidBasis::ClassID()
{
	return OP_FLIPFLUIDBASIS_CLASS_ID;
}

void OP_FlipFluidBasis::GetClassName(TSTR& s)
{
	s = GetString(IDS_OP_FLIPFLUIDBASIS_CLASS, hInstance);
}

int OP_FlipFluidBasis::NumParamBlocks()
{
	return 1;
}

IParamBlock2* OP_FlipFluidBasis::GetParamBlock(int i)
{
	return pb2;
}

IParamBlock2* OP_FlipFluidBasis::GetParamBlockByID(short id)
{
	return id ? NULL : pb2;
}


void OP_FlipFluidBasis::DeleteThis()
{
	delete this;
}



void OP_FlipFluidBasis::SetPGroup(PGroup *group)
{
	pb2->SetValue(FLIP_PGROUP, 0, (ReferenceTarget*)group);
}

PGroup *OP_FlipFluidBasis::GetPGroup()
{
	return (PGroup*)pb2->GetReferenceTarget(FLIP_PGROUP, 0);
}

void OP_FlipFluidBasis::SetSPGroup(PGroup *group)
{
	pb2->SetValue(FLIP_SPGROUP, 0, (ReferenceTarget*)group);
}

PGroup *OP_FlipFluidBasis::GetSPGroup()
{
	return (PGroup*)pb2->GetReferenceTarget(FLIP_SPGROUP, 0);
}



OP_FlipFluidBasis_DlgProc::OP_FlipFluidBasis_DlgProc(OP_FlipFluidBasis *p)
{
	map = NULL;
	op = p;
	SBox = NULL;
	GBox = NULL;
}

void OP_FlipFluidBasis_DlgProc::SetObject(OP_FlipFluidBasis *p)
{
	op = p;
}

void OP_FlipFluidBasis_DlgProc::DeleteThis()
{
	delete this;
}



void OP_FlipFluidBasis_DlgProc::UpdateGroups()
{
	Tab<PGroup*> gtab;

	SetWindowRedraw(GBox, FALSE);
	SetWindowRedraw(SBox, FALSE);

	SendMessage(GBox, CB_RESETCONTENT, 0, 0);
	SendMessage(SBox, CB_RESETCONTENT, 0, 0);

	SendMessage(GBox, CB_ADDSTRING, 0, (LPARAM)_T("None"));
	SendMessage(SBox, CB_ADDSTRING, 0, (LPARAM)_T("None"));

	info.groupmanager->EnumDyn(NULL, DYN_MSG_GROUP_GETALL, (void*)&gtab);

	for (int i = 0; i < gtab.Count(); i++)
	{
		SendMessage(GBox, CB_ADDSTRING, 0, (LPARAM)gtab[i]->GetName());
		SendMessage(GBox, CB_SETITEMDATA, (WPARAM)gtab[i]->GetGroupID() + 1, (LPARAM)gtab[i]);
		SendMessage(SBox, CB_ADDSTRING, 0, (LPARAM)gtab[i]->GetName());
		SendMessage(SBox, CB_SETITEMDATA, (WPARAM)gtab[i]->GetGroupID() + 1, (LPARAM)gtab[i]);
	}


	if (op->GetPGroup())
		SendMessage(GBox, CB_SETCURSEL, op->GetPGroup()->GetGroupID() + 1, 0);
	else
		SendMessage(GBox, CB_SETCURSEL, 0, 0);

	if (op->GetSPGroup())
		SendMessage(SBox, CB_SETCURSEL, op->GetSPGroup()->GetGroupID() + 1, 0);
	else
		SendMessage(SBox, CB_SETCURSEL, 0, 0);


	SetWindowRedraw(GBox, TRUE);
	SetWindowRedraw(SBox, TRUE);
}



INT_PTR OP_FlipFluidBasis_DlgProc::DlgProc(TimeValue t, IParamMap2 *map, HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	switch (msg) {
	case WM_INITDIALOG:
	{
		this->map = map;

		GBox = GetDlgItem(hWnd, IDC_OP_FLIPFLUIDBASIS_PGROUP);
		SBox = GetDlgItem(hWnd, IDC_OP_FLIPFLUIDBASIS_SPGROUP);


		op->NotifyDependents(FOREVER, (PartID)&info, REFMSG_DYN_GETINFO);


		UpdateGroups();
		return TRUE;
	}
	case WM_DESTROY:
		break;

	case WM_COMMAND:
		switch (LOWORD(wParam))
		{
		case IDC_OP_FLIPFLUIDBASIS_PGROUP:
			switch (HIWORD(wParam))
			{
			case CBN_SELCHANGE:
			{
				int i = ComboBox_GetCurSel(GBox);

				PGroup *group = (PGroup*)ComboBox_GetItemData(GBox, i);

				op->SetPGroup(i ? group : NULL);
			}
			break;
			}
			break;

		case IDC_OP_FLIPFLUIDBASIS_SPGROUP:
			switch (HIWORD(wParam))
			{
			case CBN_SELCHANGE:
			{
				int i = ComboBox_GetCurSel(SBox);

				PGroup *group = (PGroup*)ComboBox_GetItemData(SBox, i);

				op->SetSPGroup(i ? group : NULL);
			}
			break;
			}
			break;
		}
		break;
	}
	return FALSE;
}

///////////