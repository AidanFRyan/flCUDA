/**********************************************************************
 *<
	FILE: DllEntry.cpp

	DESCRIPTION:Contains the Dll Entry stuff

	CREATED BY: 

	HISTORY: 

 *>	Copyright (c) 1997, All Rights Reserved.
 **********************************************************************/
#include "Thinking.h"
#include "resource.h"


extern ClassDesc* GetOP_FlipFluidBasis_Desc();

HINSTANCE hInstance;
static int controlsInit = FALSE;
static Tab<ClassDesc*> classes;

int isInSlaveMode = FALSE;
int dllinit       = -1;


void RegisterInterfaces()
{
	
	ISubClassManager *mgr = GetSubClassManager();
	if(mgr)
	{
		
		// Conditions
		//mgr->RegisterInterfaces(I_DYNIF,
		//	                    GetIF_Distance_Desc(), SCM_REGEND);
		
		  
		// Operators
		mgr->RegisterInterfaces(I_DYNOP,
			
			GetOP_FlipFluidBasis_Desc(),
								SCM_REGEND);
		
		// Helpers
		/*
		mgr->RegisterInterfaces(I_DYNHELP,
			                    GetHP_AND_Desc(), 
								GetHP_OneToX_Desc(),
								GetHP_ParticleData_Desc(),
								SCM_REGEND);
*/
	}
}


BOOL WINAPI DllMain(HINSTANCE hinstDLL,ULONG fdwReason,LPVOID lpvReserved)
{
BOOL ret = TRUE;

	hInstance = hinstDLL;				// Hang on to this DLL's instance handle.
	
	 switch( fdwReason ) 
   { 
     case DLL_PROCESS_ATTACH:
				if(!controlsInit)
				{
					controlsInit = TRUE;
#if GET_MAX_RELEASE(VERSION_3DSMAX) < 13900
					InitCustomControls(hInstance);	// Initialize MAX's custom controls
#endif
					InitCustButton(hInstance);		// LightLine CustButton
					InitGradientControl(hInstance);	// LightLine Gradients
					InitCustStatus(hInstance);
					InitCustRollup(hInstance);
		
					RegisterInterfaces();
		
					InitCommonControls();			// Initialize Win95 controls

					Interface *ip = GetCOREInterface();
#if GET_MAX_RELEASE(VERSION_3DSMAX) >= 15000
					isInSlaveMode = (ip->IsNetworkRenderServer()==1)?1:0;
#else
					isInSlaveMode = (ip->InSlaveMode()==1)?1:0;
#endif
				}
        break;

     case DLL_THREAD_ATTACH:
        // Do thread-specific initialization.
        break;

     case DLL_THREAD_DETACH:
       // Do thread-specific cleanup.
        break;

     case DLL_PROCESS_DETACH:
	      break;
   }
  
return ret;
}

__declspec(dllexport) const TCHAR* LibDescription()
{
	return GetString(IDS_LIBDESCRIPTION, hInstance);
}

__declspec(dllexport) int LibNumberClasses()
{
	int num;
	
	if(classes.Count()) return classes.Count();
	
	ClassDesc *desc = GetOP_FlipFluidBasis_Desc();

	if (IsClassPlugged(desc->SuperClassID(), desc->ClassID()) == FALSE)
	{
		num = classes.Count();
		classes.SetCount(num + 1);
		classes[num] = desc;
	}
	return classes.Count();
}

__declspec(dllexport) ClassDesc* LibClassDesc(int i)
{
	if(!classes.Count()) LibNumberClasses();
	
	return classes[i];
}

__declspec(dllexport) ULONG LibVersion()
{
	return VERSION_3DSMAX;
}

__declspec(dllexport) ULONG CanAutoDefer()
{
 return 0;
}


TCHAR *GetString(int id)
{
	return GetString(id, hInstance);
}
