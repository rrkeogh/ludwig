/*****************************************************************************
 *
 *  wall.c
 *
 *  Static solid objects (porous media).
 *
 *  Special case: boundary walls. The two issues might be sepatated.
 *
 *  Edinburgh Soft Matter and Statistical Physics and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2011-2016 The University of Edinburgh
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kernel.h"
#include "lb_model_s.h"
#include "map_s.h"
#include "physics.h"
#include "util.h"
#include "wall.h"

typedef enum wall_init_enum {WALL_INIT_COUNT_ONLY,
			     WALL_INIT_ALLOCATE} wall_init_enum_t;

typedef enum wall_uw_enum {WALL_UZERO = 0,
			   WALL_UWTOP,
			   WALL_UWBOT,
			   WALL_UWMAX} wall_uw_enum_t;

struct wall_s {
  pe_t * pe;             /* Parallel environment */
  cs_t * cs;             /* Reference to coordinate system */
  map_t * map;           /* Reference to map structure */
  lb_t * lb;             /* Reference to LB information */
  wall_t * target;       /* Device memory */

  wall_param_t * param;  /* parameters */
  int   nlink;           /* Number of links */
  int * linki;           /* outside (fluid) site indices */
  int * linkj;           /* inside (solid) site indices */
  int * linkp;           /* LB basis vectors for links */
  int * linku;           /* Link wall_uw_enum_t (wall velocity) */
	int * links;           /* flag inside (solid) site indices are slip or no-slip RYAN EDIT*/
  double fnet[3];        /* Momentum accounting for source/sink walls */
};

int wall_init_boundaries(wall_t * wall, wall_init_enum_t init);
int wall_init_map(wall_t * wall);
int wall_init_uw(wall_t * wall);

__global__ void wall_setu_kernel(wall_t * wall, lb_t * lb);
__global__ void wall_bbl_kernel(wall_t * wall, lb_t * lb, map_t * map);

static __constant__ wall_param_t static_param;

/*****************************************************************************
 *
 *  wall_create
 *
 *****************************************************************************/

__host__ int wall_create(pe_t * pe, cs_t * cs, map_t * map, lb_t * lb,
			 wall_t ** p) {

  int ndevice;
  wall_t * wall = NULL;

  assert(pe);
  assert(cs);
  assert(p);

  wall = (wall_t *) calloc(1, sizeof(wall_t));
  if (wall == NULL) pe_fatal(pe, "calloc(wall_t) failed\n");

  wall->param = (wall_param_t *) calloc(1, sizeof(wall_param_t));
  if (wall->param == NULL) pe_fatal(pe, "calloc(wall_param_t) failed\n");

  wall->pe = pe;
  wall->cs = cs;
  wall->map = map;
  wall->lb = lb;

  cs_retain(cs);

  /* Target copy */

  targetGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    wall->target = wall;
  }
  else {
    wall_param_t * tmp = NULL;

    targetCalloc((void **) &wall->target, sizeof(wall_t));
    targetConstAddress((void **) &tmp, static_param);
    copyToTarget(&wall->target->param, &tmp, sizeof(wall_param_t *));
  }

  *p = wall;

  return 0;
}

/*****************************************************************************
 *
 *  wall_free
 *
 *****************************************************************************/

__host__ int wall_free(wall_t * wall) {

  assert(wall);

  if (wall->target != wall) {
    int * tmp;
    copyFromTarget(&tmp, &wall->target->linki, sizeof(int *));
    targetFree(tmp);
    copyFromTarget(&tmp, &wall->target->linkj, sizeof(int *));
    targetFree(tmp);
    copyFromTarget(&tmp, &wall->target->linkp, sizeof(int *));
    targetFree(tmp);
    copyFromTarget(&tmp, &wall->target->linku, sizeof(int *));
    targetFree(tmp);
		copyFromTarget(&tmp, &wall->target->links, sizeof(int *));		/*RYAN EDIT*/
    targetFree(tmp);
    targetFree(wall->target);
  }

  cs_free(wall->cs);
  free(wall->param);
  if (wall->linki) free(wall->linki);
  if (wall->linkj) free(wall->linkj);
  if (wall->linkp) free(wall->linkp);
  if (wall->linku) free(wall->linku);
	if (wall->links) free(wall->links);								/*RYAN EDIT*/
  free(wall);

  return 0;
}

/*****************************************************************************
 *
 *  wall_commit
 *
 *****************************************************************************/

__host__ int wall_commit(wall_t * wall, wall_param_t param) {

  assert(wall);

  *wall->param = param;

  wall_init_map(wall);
  wall_init_boundaries(wall, WALL_INIT_COUNT_ONLY);
  wall_init_boundaries(wall, WALL_INIT_ALLOCATE);
  wall_init_uw(wall);

  /* As we have initialised the map on the host, ... */
  map_memcpy(wall->map, cudaMemcpyHostToDevice);

  return 0;
}

/*****************************************************************************
 *
 *  wall_info
 *
 *  Note a global communication.
 *
 *****************************************************************************/

__host__ int wall_info(wall_t * wall) {

  int nlink;
  pe_t * pe = NULL;
  MPI_Comm comm;

  assert(wall);

  pe = wall->pe;

  pe_mpi_comm(pe, &comm);
  MPI_Reduce(&wall->nlink, &nlink, 1, MPI_INT, MPI_SUM, 0, comm);

  if (wall->param->iswall) {
    pe_info(pe, "\n");
    pe_info(pe, "Boundary walls\n");
    pe_info(pe, "--------------\n");
    pe_info(pe, "Boundary walls:                  %1s %1s %1s\n",
	    (wall->param->isboundary[X] == 1) ? "X" : "-",
	    (wall->param->isboundary[Y] == 1) ? "Y" : "-",
	    (wall->param->isboundary[Z] == 1) ? "Z" : "-");
		pe_info(pe, "--------------\n");
	   pe_info(pe, "Slip walls:                  %1s %1s %1s\n",
		   (wall->param->isslip[X] == 1) ? "X" : "-",
		   (wall->param->isslip[Y] == 1) ? "Y" : "-",
		   (wall->param->isslip[Z] == 1) ? "Z" : "-");
    pe_info(pe, "Boundary speed u_x (bottom):    %14.7e\n",
	    wall->param->ubot[X]);
    pe_info(pe, "Boundary speed u_x (top):       %14.7e\n",
	    wall->param->utop[X]);
    pe_info(pe, "Boundary normal lubrication rc: %14.7e\n",
	    wall->param->lubr_rc[X]);

    pe_info(pe, "Wall boundary links allocated:   %d\n", nlink);
    pe_info(pe, "Memory (total, bytes):           %d\n", 4*nlink*sizeof(int));
    pe_info(pe, "Boundary shear initialise:       %d\n",
	    wall->param->initshear);
  }

  if (wall->param->isporousmedia) {
    pe_info(pe, "\n");
    pe_info(pe, "Porous Media\n");
    pe_info(pe, "------------\n");
    pe_info(pe, "Wall boundary links allocated:   %d\n", nlink);
    pe_info(pe, "Memory (total, bytes):           %d\n", 4*nlink*sizeof(int));
  }

  return 0;
}

/*****************************************************************************
 *
 *  wall_target
 *
 *****************************************************************************/

__host__ int wall_target(wall_t * wall, wall_t ** target) {

  assert(wall);
  assert(target);

  *target = wall->target;

  return 0;
}

/*****************************************************************************
 *
 *  wall_param_set
 *
 *****************************************************************************/

__host__ int wall_param_set(wall_t * wall, wall_param_t values) {

  assert(wall);

  *wall->param = values;

  return 0;
}

/*****************************************************************************
 *
 *  wall_param
 *
 *****************************************************************************/

__host__ int wall_param(wall_t * wall, wall_param_t * values) {

  assert(wall);
  assert(values);

  *values = *wall->param;

  return 0;
}

/*****************************************************************************
 *
 *  wall_init_boundaries
 *
 *  To be called twice:
 *     1. with WALL_INIT_COUNT_ONLY
 *     2. with WALL_INIT_ALLOCATE
 *
 *****************************************************************************/

__host__ int wall_init_boundaries(wall_t * wall, wall_init_enum_t init) {

  int ic, jc, kc;
  int ic1, jc1, kc1;
	int ic2, jc2, kc2;
	int isslip;
  int indexi, indexj, indexjPrime;
  int p;
  int nlink;
  int nlocal[3];
  int status;
  int ndevice;

  assert(wall);

  targetGetDeviceCount(&ndevice);

  if (init == WALL_INIT_ALLOCATE) {
    wall->linki = (int *) calloc(wall->nlink, sizeof(int));
    wall->linkj = (int *) calloc(wall->nlink, sizeof(int));
    wall->linkp = (int *) calloc(wall->nlink, sizeof(int));
    wall->linku = (int *) calloc(wall->nlink, sizeof(int));
		wall->links = (int *) calloc(wall->nlink, sizeof(int));				/*RYAN EDIT*/
		assert(wall->linki);
		assert(wall->linkj);
		assert(wall->linkp);
		assert(wall->linku);
		assert(wall->links);
	  if (wall->linki == NULL) pe_fatal(wall->pe,"calloc(wall->linki) failed\n");
    if (wall->linkj == NULL) pe_fatal(wall->pe,"calloc(wall->linkj) failed\n");
    if (wall->linkp == NULL) pe_fatal(wall->pe,"calloc(wall->linkp) failed\n");
    if (wall->linku == NULL) pe_fatal(wall->pe,"calloc(wall->linku) failed\n");
		if (wall->links == NULL) pe_fatal(wall->pe,"calloc(wall->links) failed\n");		/*RYAN EDIT*/
    if (ndevice > 0) {
      int tmp;
      targetMalloc((void **) &tmp, wall->nlink*sizeof(int));
      copyToTarget(&wall->target->linki, &tmp, sizeof(int *));
      targetMalloc((void **) &tmp, wall->nlink*sizeof(int));
      copyToTarget(&wall->target->linkj, &tmp, sizeof(int *));
      targetMalloc((void **) &tmp, wall->nlink*sizeof(int));
      copyToTarget(&wall->target->linkp, &tmp, sizeof(int *));
      targetMalloc((void **) &tmp, wall->nlink*sizeof(int));
      copyToTarget(&wall->target->linku, &tmp, sizeof(int *));
			targetMalloc((void **) &tmp, wall->nlink*sizeof(int));
      copyToTarget(&wall->target->links, &tmp, sizeof(int *));			/*RYAN EDIT*/
    }
  }

  nlink = 0;
  cs_nlocal(wall->cs, nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	indexi = cs_index(wall->cs, ic, jc, kc);
	map_status(wall->map, indexi, &status);
	if (status != MAP_FLUID) continue;

	/* Look for non-solid -> solid links */
	for (p = 1; p < NVEL; p++) {
		/* Calculates coordinates for solid nodes */
	  ic1 = ic + cv[p][X];
	  jc1 = jc + cv[p][Y];
	  kc1 = kc + cv[p][Z];

	  indexjPrime = cs_index(wall->cs, ic1, jc1, kc1); //Calculates temporary indexjPrime (indexjPrime = indexj)
	  map_status(wall->map, indexjPrime, &status);

		/* Setting new coordinates ic2, jc2, kc2 for solid sites to capture perfect-slip.		**
		** New j index for perfect-slip is calculated such that the intermediary solid nodes**
		** where the distribution is passed from is directly adjacent to origin fluid node  **
		** This should allow for proper spectral reflection 																*/
		if (status == MAP_BOUNDARY) {
			isslip = 0;								//Set flag for bounceback
			ic2 = ic1;								//Default ic2, jc2, kc2 values are no-slip (ic2 = ic1 etc.)
			jc2 = jc1;
			kc2 = kc1;

			/* Redefining ic2, jc2, kc2 for perfect-slip on walls in the x-direction */
			if ( wall->param->isslip[0] && ( ic == 1 || ic == nlocal[X] ) ){	//if x-walls are slip and for nodes located at the extremes:
				isslip = 1;																											//Set flag for perfect-slip in x-direction
				if (ic == 1) ic2 = 0;																						//Set ic2 to for x = 1 to be in the solid domain
				else if (ic == nlocal[X]) ic2 = nlocal[X] + 1;									//Set ic2 for x = max x to be in the solid domain

				if ( jc > 1 && jc < nlocal[Y] && kc > 1 && kc < nlocal[Z] ){		//General case for non-edges on x-walls
					jc2 = jc;																											//Set jc2 and kc2 to be same size as fluid domain
					kc2 = kc;																											//All solid indices should now be directly adjacent to equivalent fluid index
				}
				else if ( jc == nlocal[Y] ){																		//Top left & right edges of x-walls
					jc2 = jc1;																										//Allows for diagonal indices of solid sites at edges
					kc2 = kc;
					if ( p == 14 || p == 1) isslip = 0;														//If link -> edge, do bounceback
				}
				else if ( jc == 1 ){																						//Bottom left & right edges
					jc2 = jc1;
					kc2 = kc;
					if ( p == 18 || p == 5 ) isslip = 0;
				}
				else if ( kc == nlocal[Z] ){																		//Front left & right edges
					jc2 = jc;
					kc2 = kc1;
					if ( p == 15 || p == 2) isslip = 0;
				}
				else if ( kc == 1 ){																						//Back left & right edges
					jc2 = jc;
					kc2 = kc1;
					if ( p == 17 || p == 4 ) isslip = 0;
				}

				/* Accounting for PBCs in perfect slip on x-walls by reverting back to slip conditions at PBC-slip wall edges*/
				if ( wall->param->isboundary[Y] == 0 ){															//y-walls are PBC
					if ( ( jc == 1 || jc == nlocal[Y] ) && ( kc > 1 && kc < nlocal[Z] ) ) isslip = 1;
				}

				else if ( wall->param->isboundary[Z] == 0 ){												//z-walls are PBC
					if ( ( kc == 1 || kc == nlocal[Z] ) && ( jc > 1 && jc < nlocal[Y] ) ) isslip = 1;
				}
			}
			/* Redefining ic2, jc2, kc2 for perfect-slip on walls in the y-direction */
			if ( wall->param->isslip[1] && ( jc == 1 || jc == nlocal[Y] ) ){			//Perfect slip on y-walls
				isslip = 2;
				if ( jc == 1 ) jc2 = 0;																							//Set jc2 for bottom wall
				else if ( jc == nlocal[Y] ) jc2 = nlocal[Y]+1;											//Set jc2 for top wall

				if ( ic > 1 && ic < nlocal[X] && kc > 1 && kc < nlocal[Z] ){				//General case for non-edges on y-walls
					ic2 = ic;
					kc2 = kc;
				}
				else if ( ic == 1 ){																								//Top & bottom left edge
					ic2 = ic1;
					kc2 = kc;
					if ( p == 14 || p == 18 ) isslip = 0;
				}
				else if ( ic == nlocal[X] ){																				//Top & bottom right edg
					ic2 = ic1;
					kc2 = kc;
					if ( p == 1 || p == 5 ) isslip = 0;
				}
				else if ( kc == 1 ){																								//Top & bottom back edge
					ic2 = ic;
					kc2 = kc1;
					if ( p == 8 || p == 13 ) isslip = 0;
				}
				else if ( kc == nlocal[Z] ){																				//Top & bottom front edge
					ic2 = ic;
					kc2 = kc1;
					if ( p == 11 || p == 6 ) isslip = 0;
				}
				/* Accounting for PBCs in perfect slip on y-walls by reverting back to slip conditions at PBC-slip wall edges*/
				if ( wall->param->isboundary[X] == 0 ){															//x-walls are PBC
					if ( ( ic == 1 || ic == nlocal[X] ) && ( kc > 1 && kc < nlocal[Z] ) ) isslip = 2;
				}

			 	else if ( wall->param->isboundary[Z] == 0 ){												//z-walls are PBC
					if ( ( kc == 1 || kc == nlocal[Z] ) && ( ic > 1 && ic < nlocal[X] ) ) isslip = 2;
				}
			}
			/* Redefining ic2, jc2, kc2 for perfect-slip on walls in the y-direction */
			if ( wall->param->isslip[2] && ( kc == 1 || kc == nlocal[Z] ) ){			//Perfect slip on z-walls
				isslip = 3;
				if ( kc == 1 ) kc2 = 0;																							//Set kc2 for back wall
				else if ( kc == nlocal[Z] ) kc2 = nlocal[Z] + 1;										//Set kc2 for front wall

				if ( ic > 1 && ic < nlocal[X] && jc > 1 && jc < nlocal[Y] ){				//General case for non-edges on z-walls
					ic2 = ic;
					jc2 = jc;
				}
				else if ( ic == 1 ){																								//Front & back left edges
					ic2 = ic1;
					jc2 = jc;
					if ( p == 15 || p == 17 ) isslip = 0;
				}
				else if ( ic == nlocal[X] ){																				//Front & back right edges
					ic2 = ic1;
					jc2 = jc;
					if ( p == 2 || p == 4 ) isslip = 0;
				}
				else if ( jc == 1 ){																								//Front & back bottom edges
					ic2 = ic;
					jc2 = jc1;
					if ( p == 13 || p == 11) isslip = 0;
				}
				else if ( jc == nlocal[Y] ){																				//Front & back top edges
					ic2 = ic;
					jc2 = jc1;
					if ( p == 8 || p == 6 ) isslip = 0;
				}
				/* Accounting for PBCs in perfect slip on x-walls by reverting back to slip conditions at PBC-slip wall edges*/
				if ( wall->param->isboundary[X] == 0 ){															//x-walls are PBC
					if ( ( ic == 1 || ic == nlocal[X] ) && ( jc > 1 && jc < nlocal[Y] ) ) isslip = 3;
				}

				else if ( wall->param->isboundary[Y] == 0 ){												//y-walls are PBC
					if ( ( jc == 1 || jc == nlocal[Y] ) && ( ic > 1 && ic < nlocal[X] ) ) isslip = 3;
				}
			}

			indexj = cs_index(wall->cs, ic2, jc2, kc2);		//Calculating new solid site index j via ic2, jc2, kc2 (default no-slip ic2 = ic1 etc.)

			/*Setting site indices i and j to linki and linkj respectively. Setting vector p and linkp, wall velocity and flag for perfect-slip condition */
	    if (init == WALL_INIT_ALLOCATE) {
	      wall->linki[nlink] = indexi;
	      wall->linkj[nlink] = indexj;
	      wall->linkp[nlink] = p;
	      wall->linku[nlink] = WALL_UZERO;
				wall->links[nlink] = isslip;
	    }
	    nlink += 1;
	  }
	}

	/* Next site */
      }
    }
  }

  if (init == WALL_INIT_ALLOCATE) {
    assert(nlink == wall->nlink);
    wall_memcpy(wall, cudaMemcpyHostToDevice);
  }
  wall->nlink = nlink;

  return 0;
}

/*****************************************************************************
 *
 *  wall_memcpy
 *
 *****************************************************************************/

__host__ int wall_memcpy(wall_t * wall, int flag) {

  int ndevice;

  assert(wall);

  targetGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    assert(wall->target == wall);
  }
  else {

    int * tmp = NULL;
    int nlink;

    nlink = wall->nlink;

    switch (flag) {
    case cudaMemcpyHostToDevice:
      copyToTarget(&wall->target->nlink, &wall->nlink, sizeof(int));
      copyToTarget(wall->target->fnet, wall->fnet, 3*sizeof(double));

      /* In turn, linki, linkj, linkp, linku */
      copyFromTarget(&tmp, &wall->target->linki, sizeof(int *));
      copyToTarget(tmp, wall->linki, nlink*sizeof(int));

      copyFromTarget(&tmp, &wall->target->linkj, sizeof(int *));
      copyToTarget(tmp, wall->linkj, nlink*sizeof(int));

      copyFromTarget(&tmp, &wall->target->linkp, sizeof(int *));
      copyToTarget(tmp, wall->linkp, nlink*sizeof(int));

      copyFromTarget(&tmp, &wall->target->linku, sizeof(int *));
      copyToTarget(tmp, wall->linku, nlink*sizeof(int));

			copyFromTarget(&tmp, &wall->target->links, sizeof(int *));
      copyToTarget(tmp, wall->links, nlink*sizeof(int));
      break;
    case cudaMemcpyDeviceToHost:
      assert(0); /* Not required */
      break;
    default:
      pe_fatal(wall->pe, "Should definitely not be here\n");
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  wall_init_uw
 *
 *  Only the simple case of one set of walls is handled at present.
 *
 *****************************************************************************/

__host__ int wall_init_uw(wall_t * wall) {

  int n;
  int iw;
  int nwall;

  assert(wall);

  nwall = wall->param->isboundary[X] + wall->param->isboundary[Y]
    + wall->param->isboundary[Z];

  if (nwall == 1) {
    /* All links are either top or bottom */
    if (wall->param->isboundary[X]) iw = X;
    if (wall->param->isboundary[Y]) iw = Y;
    if (wall->param->isboundary[Z]) iw = Z;

    for (n = 0; n < wall->nlink; n++) {
      if (cv[wall->linkp[n]][iw] == -1) wall->linku[n] = WALL_UWBOT;
      if (cv[wall->linkp[n]][iw] == +1) wall->linku[n] = WALL_UWTOP;
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  wall_set_wall_distribution
 *
 *  Driver routine.
 *
 *****************************************************************************/

__host__ int wall_set_wall_distributions(wall_t * wall) {

  dim3 nblk, ntpb;

  assert(wall);
  assert(wall->target);

  if (wall->nlink == 0) return 0;

  kernel_launch_param(wall->nlink, &nblk, &ntpb);

  __host_launch(wall_setu_kernel, nblk, ntpb, wall->target, wall->lb->target);

  targetDeviceSynchronise();

  return 0;
}

/*****************************************************************************
 *		RYAN EDIT
 *  wall_setu_kernel
 *
 *  Set distribution at solid sites to reflect the solid body velocity.
 *  This allows 'solid-solid' exchange of distributions between wall
 *  and colloids.
 *
 *****************************************************************************/

__global__ void wall_setu_kernel(wall_t * wall, lb_t * lb) {

  int n;
  const double rcs2 = 3.0; /* macro? */

  assert(wall);
  assert(lb);

  __target_simt_parallel_for(n, wall->nlink, 1) {

		int ij, is;
    int p;              /* Outward going component of link velocity */
    double fp;          /* f = w_p (rho0 + (1/cs2) u_a c_pa) No sdotq */
    double ux = 0.0;    /* PENDING initialisation */

		ij = wall->linkp[n];
		is = wall->links[n];
    p = NVEL - ij;
		if (is == 1){																								/*x-perfect slip condition*/
			if (ij == 1 || ij == 2 || ij == 4 || ij == 5) p = ij+13;
			else if (ij == 14|| ij == 15|| ij == 17|| ij == 18) p = ij-13;
		}
		else if (is == 2){																							/*y-perfect slip condition*/
			if (ij == 6 || ij == 8) p = ij+5;
			else if (ij == 11 || ij ==13) p = ij-5;
			else if (ij == 1 || ij == 14) p = ij+4;
			else if (ij == 5 || ij == 18) p = ij-4;
		}
		else if (is == 3){																							/*z-perfect slip condition*/
			if (ij == 2|| ij == 6|| ij == 11|| ij == 15) p = ij+2;
			else if (ij == 4|| ij == 8|| ij == 13|| ij == 17) p = ij-2;
			// printf("ij ji: %d \n\t %d\n", ij, ji );
		}

		fp = lb->param->wv[p]*(lb->param->rho0 + rcs2*ux*lb->param->cv[p][X]);
    lb_f_set(lb, wall->linkj[n], p, LB_RHO, fp);
  }

  return;
}

/*****************************************************************************
 *
 *  wall_bbl
 *
 *  Driver routine.
 *
 *****************************************************************************/

__host__ int wall_bbl(wall_t * wall) {

  dim3 nblk, ntpb;

  assert(wall);
  assert(wall->target);

  if (wall->nlink == 0) return 0;

  /* Update kernel constants */
  copyConstToTarget(&static_param, wall->param, sizeof(wall_param_t));

  kernel_launch_param(wall->nlink, &nblk, &ntpb);

  __host_launch(wall_bbl_kernel, nblk, ntpb, wall->target, wall->lb->target,
		wall->map->target);

  targetDeviceSynchronise();

  return 0;
}

/*****************************************************************************
 *
 *  wall_bbl_kernel
 *
 *  Bounce-back on links for the walls.
 *  A reduction is required to tally the net momentum transfer.
 *
 *****************************************************************************/

__global__ void wall_bbl_kernel(wall_t * wall, lb_t * lb, map_t * map) {

  int n;
  int ib;
  double uw[WALL_UWMAX][3];

  __shared__ double fx[TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double fy[TARGET_MAX_THREADS_PER_BLOCK];
  __shared__ double fz[TARGET_MAX_THREADS_PER_BLOCK];
  const double rcs2 = 3.0;

  assert(wall);
  assert(lb);
  assert(map);

  /* Load the current wall velocities into the uw table */

  for (ib = 0; ib < 3; ib++) {
    uw[WALL_UZERO][ib] = 0.0;
    uw[WALL_UWTOP][ib] = wall->param->utop[ib];
    uw[WALL_UWBOT][ib] = wall->param->ubot[ib];
  }

  __target_simt_parallel_region() {

    int tid;
    double fxb, fyb, fzb;

    __target_simt_threadIdx_init();
    tid = threadIdx.x;

    fx[tid] = 0.0;
    fy[tid] = 0.0;
    fz[tid] = 0.0;

    __target_simt_for(n, wall->nlink, 1) {

      int i, j, ij, ji, ia, is, jk;
      int status;
      double rho, cdotu;
      double fp, fp0, fp1;
      double force;

      i  = wall->linki[n];		/* Original fluid site i */
      j  = wall->linkj[n];		/* Intermediate solid site j */
	//Target fluid site is k (for bounceback k = i) -- never need to know k
      ij = wall->linkp[n];   	/* Link index direction solid->fluid */
      ia = wall->linku[n];   	/* Wall velocity lookup */
			is = wall->links[n];
			ji = NVEL - ij;        	/* Opposite direction index */

			/* Calculating new link between solid and target fluid node */
			jk = ji;																												/* Link between solid to target fluid site - if bounceback then jk = ji*/
			if (is == 1){																										/*x-perfect slip condition*/
				if (ij == 1 || ij == 2 || ij == 4 || ij == 5) jk = ij+13;
				else if (ij == 14|| ij == 15|| ij == 17|| ij == 18) jk = ij-13;
			}
			else if (is == 2){																							/*y-perfect slip condition*/
				if (ij == 6 || ij == 8) jk = ij+5;
				else if (ij == 11 || ij == 13) jk = ij-5;
				else if (ij == 1 || ij == 14) jk = ij+4;
				else if (ij == 5 || ij == 18) jk = ij-4;
			}
			else if (is == 3){																							/*z-perfect slip condition*/
				if (ij == 2|| ij == 6|| ij == 11|| ij == 15) jk = ij+2;
				else if (ij == 4|| ij == 8|| ij == 13|| ij == 17) jk = ij-2;
			}

			/* Adjust for moving walls -- if stationary cdotu = 0 */
      cdotu = lb->param->cv[ij][X]*uw[ia][X] +
	      			lb->param->cv[ij][Y]*uw[ia][Y] +
              lb->param->cv[ij][Z]*uw[ia][Z];

			/* Determine if fluid node is adjacent to colloid or wall boundary */
      map_status(map, i, &status);

      if (status == MAP_COLLOID) {
				/* This matches the momentum exchange in colloid BBL. */
				/* This only affects the accounting (via anomaly, as below) */
				lb_f(lb, i, ij, LB_RHO, &fp0);
				lb_f(lb, j, ji, LB_RHO, &fp1);		/* Because of this, colloids limited to bounceback ONLY */
				fp = fp0 + fp1;

				fx[tid] += (fp - 2.0*lb->param->wv[ij])*lb->param->cv[ij][X];
				fy[tid] += (fp - 2.0*lb->param->wv[ij])*lb->param->cv[ij][Y];
				fz[tid] += (fp - 2.0*lb->param->wv[ij])*lb->param->cv[ij][Z];
      }
      else {

				/* This is the momentum. To prevent accumulation of round-off
				 * in the running total (fnet_), we subtract the equilibrium
				 * wv[ij]. This is ok for walls where there are exactly
				 * equal and opposite links at each side of the system. */

				lb_f(lb, i, ij, LB_RHO, &fp);  //Gets distribution from origin fluid site i in direction ij
				lb_0th_moment(lb, i, LB_RHO, &rho); //Gets density from origin fluid site i

				force = 2.0*fp - 2.0*rcs2*lb->param->wv[ij]*lb->param->rho0*cdotu;
				if (is == 0) {
					fx[tid] += (force - 2.0*lb->param->wv[ij])*lb->param->cv[ij][X];
					fy[tid] += (force - 2.0*lb->param->wv[ij])*lb->param->cv[ij][Y];
					fz[tid] += (force - 2.0*lb->param->wv[ij])*lb->param->cv[ij][Z];
				}
				else if (is == 1) fx[tid] += (force - 2.0*lb->param->wv[ij])*lb->param->cv[ij][X];
				else if (is == 2) fy[tid] += (force - 2.0*lb->param->wv[ij])*lb->param->cv[ij][Y];
				else if (is == 3) fz[tid] += (force - 2.0*lb->param->wv[ij])*lb->param->cv[ij][Z];
				fp = fp - 2.0*rcs2*lb->param->wv[ij]*lb->param->rho0*cdotu;

				/* Setting distribution at intermediary solid site j in direction jk (where jk = ji for bounceback) */
				lb_f_set(lb, j, jk, LB_RHO, fp);

				/* For multi-phase fluids */
				if (lb->param->ndist > 1) {
				  /* Order parameter */
				  lb_f(lb, i, ij, LB_PHI, &fp);
				  lb_0th_moment(lb, i, LB_PHI, &rho);

				  fp = fp - 2.0*rcs2*lb->param->wv[ij]*lb->param->rho0*cdotu;
				  lb_f_set(lb, j, ji, LB_PHI, fp);
				}
      }
      /* Next link */
    }

    /* Reduction for momentum transfer */

		fxb = target_block_reduce_sum_double(fx);
	  fyb = target_block_reduce_sum_double(fy);
	  fzb = target_block_reduce_sum_double(fz);


    if (tid == 0) {
      target_atomic_add_double(&wall->fnet[X], fxb);
      target_atomic_add_double(&wall->fnet[Y], fyb);
      target_atomic_add_double(&wall->fnet[Z], fzb);
    }
  }

  return;
}

/*****************************************************************************
 *
 *  wall_init_map
 *
 *****************************************************************************/

__host__ int wall_init_map(wall_t * wall) {

  int ic, jc, kc, index;
  int ic_global, jc_global, kc_global;
  int ntotal[3];
  int nlocal[3];
  int noffset[3];
  int nextra;

  assert(wall);

  cs_ntotal(wall->cs, ntotal);
  cs_nlocal(wall->cs, nlocal);
  cs_nlocal_offset(wall->cs, noffset);
  cs_nhalo(wall->cs, &nextra);

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	/* If this is an appropriate periodic boundary, set to solid */

	ic_global = ic + noffset[X];
	jc_global = jc + noffset[Y];
	kc_global = kc + noffset[Z];

	if (wall->param->isboundary[Z]) {
	  if (kc_global == 0 || kc_global == ntotal[Z] + 1) {
	    index = cs_index(wall->cs, ic, jc, kc);
	    map_status_set(wall->map, index, MAP_BOUNDARY);
	  }
	}

	if (wall->param->isboundary[Y]) {
	  if (jc_global == 0 || jc_global == ntotal[Y] + 1) {
	    index = cs_index(wall->cs, ic, jc, kc);
	    map_status_set(wall->map, index, MAP_BOUNDARY);
	  }
	}

	if (wall->param->isboundary[X]) {
	  if (ic_global == 0 || ic_global == ntotal[X] + 1) {
	    index = cs_index(wall->cs, ic, jc, kc);
	    map_status_set(wall->map, index, MAP_BOUNDARY);
	  }
	}
	/* next site */
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  wall_momentum_add
 *
 *****************************************************************************/

__host__ int wall_momentum_add(wall_t * wall, const double f[3]) {

  assert(wall);

  wall->fnet[X] += f[X];
  wall->fnet[Y] += f[Y];
  wall->fnet[Z] += f[Z];

  return 0;
}

/*****************************************************************************
 *
 *  wall_momentum
 *
 *  If a global total is required, the caller is responsible for
 *  any MPI reduction. This is the local contribution.
 *
 *****************************************************************************/

__host__ int wall_momentum(wall_t * wall, double f[3]) {

  int ndevice;
  double ftmp[3];

  assert(wall);

  /* Some care at the moment. Accumulate the device total to the
   * host and zero the device fnet so that we don't double-count
   * it next time. */

  /* This is required as long as some contributions are made on
   * the host via wall_momentum_add() and others are on the
   * device. */

  targetGetDeviceCount(&ndevice);

  if (ndevice > 0) {
    copyFromTarget(ftmp, wall->target->fnet, 3*sizeof(double));
    wall->fnet[X] += ftmp[X];
    wall->fnet[Y] += ftmp[Y];
    wall->fnet[Z] += ftmp[Z];
    ftmp[X] = 0.0; ftmp[Y] = 0.0; ftmp[Z] = 0.0;
    copyToTarget(wall->target->fnet, ftmp, 3*sizeof(double));
  }

  /* Return the current net */

  f[X] = wall->fnet[X];
  f[Y] = wall->fnet[Y];
  f[Z] = wall->fnet[Z];

  return 0;
}

/*****************************************************************************
 *
 *  wall_is_pm
 *
 *****************************************************************************/

__host__ __device__ int wall_is_pm(wall_t * wall, int * ispm) {

  assert(wall);

  *ispm = wall->param->isporousmedia;

  return 0;
}

/*****************************************************************************
 *
 *  wall_present
 *
 *****************************************************************************/

__host__ __device__ int wall_present(wall_t * wall) {

  wall_param_t * wp = NULL;

  assert(wall);

  wp = wall->param;
  return (wp->isboundary[X] || wp->isboundary[Y] || wp->isboundary[Z]);
}

/*****************************************************************************
 *
 *  wall_present_dim
 *
 *****************************************************************************/

__host__ __device__ int wall_present_dim(wall_t * wall, int iswall[3]) {

  assert(wall);

  iswall[X] = wall->param->isboundary[X];
  iswall[Y] = wall->param->isboundary[Y];
  iswall[Z] = wall->param->isboundary[Z];

  return 0;
}

/*****************************************************************************
 *
 *  wall_shear_init
 *
 *  Initialise the distributions to be consistent with a linear shear
 *  profile for the given top and bottom wall velocities.
 *
 *  This is only relevant for walls at z = 0 and z = L_z.
 *
 *****************************************************************************/

__host__ int wall_shear_init(wall_t * wall) {

  int ic, jc, kc, index;
  int ia, ib, p;
  int nlocal[3];
  int noffset[3];
  double rho, u[3], gradu[3][3];
  double eta;
  double gammadot;
  double f;
  double cdotu;
  double sdotq;
  double uxbottom;
  double uxtop;
  double ltot[3];
  physics_t * phys = NULL;

  assert(wall);

  /* One wall constraint */
  uxtop = wall->param->utop[X];
  uxbottom = wall->param->ubot[X];

  cs_ltot(wall->cs, ltot);

  /* Shear rate */
  gammadot = (uxtop - uxbottom)/ltot[Z];

  pe_info(wall->pe, "Initialising linear shear profile for walls\n");
  pe_info(wall->pe, "Speed at top u_x    %14.7e\n", uxtop);
  pe_info(wall->pe, "Speed at bottom u_x %14.7e\n", uxbottom);
  pe_info(wall->pe, "Overall shear rate  %14.7e\n", gammadot);

  /* Initialise the density, velocity, gradu; ghost modes are zero */

  physics_ref(&phys);
  physics_rho0(phys, &rho);
  physics_eta_shear(phys, &eta);

  cs_nlocal(wall->cs, nlocal);
  cs_nlocal_offset(wall->cs, noffset);

  for (ia = 0; ia < 3; ia++) {
    u[ia] = 0.0;
    for (ib = 0; ib < 3; ib++) {
      gradu[ia][ib] = 0.0;
    }
  }

  /* Shear rate */
  gradu[X][Z] = gammadot;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	/* Linearly interpolate between top and bottom to get velocity;
	 * the - 1.0 accounts for kc starting at 1. */
	u[X] = uxbottom + (noffset[Z] + kc - 0.5)*(uxtop - uxbottom)/ltot[Z];

        index = cs_index(wall->cs, ic, jc, kc);

        for (p = 0; p < NVEL; p++) {

	  cdotu = 0.0;
	  sdotq = 0.0;

          for (ia = 0; ia < 3; ia++) {
            cdotu += cv[p][ia]*u[ia];
            for (ib = 0; ib < 3; ib++) {
              sdotq += (rho*u[ia]*u[ib] - eta*gradu[ia][ib])*q_[p][ia][ib];
            }
          }
          f = wv[p]*rho*(1.0 + rcs2*cdotu + 0.5*rcs2*rcs2*sdotq);
          lb_f_set(wall->lb, index, p, 0, f);
        }
        /* Next site */
      }
    }
  }

  return 0;
}

/******************************************************************************
 *
 *  wall_lubrication
 *
 *  This returns the normal lubrication correction for colloid of hydrodynamic
 *  radius ah at position r near a flat wall in dimension dim (if present).
 *  This is based on the analytical expression for a sphere.
 *
 *  The result should be added to the appropriate diagonal element of
 *  the colloid's drag matrix in the implicit update. There is, therefore,
 *  no velocity appearing here (wall assumed to have no velocity).
 *  This is therefore closely related to BBL in bbl.c.
 *
 *  This operates in parallel by computing the absolute distance between
 *  the side of the system (walls nominally at Lmin and (Lmax + Lmin)),
 *  and applying the cutoff.
 *
 *  Normal force is added to the diagonal of drag matrix \zeta^FU_xx etc
 *  (No tangential force would be added to \zeta^FU_xx and \zeta^FU_yy)
 *
 *****************************************************************************/

__host__ int wall_lubr_sphere(wall_t * wall, double ah, const double r[3],
			      double * drag) {

  double hlub;
  double h;
  double eta;
  double lmin[3];
  double ltot[3];
  physics_t * phys = NULL;
  PI_DOUBLE(pi);

  drag[X] = 0.0;
  drag[Y] = 0.0;
  drag[Z] = 0.0;

  if (wall == NULL) return 0; /* PENDING prefer assert()? */

  cs_lmin(wall->cs, lmin);
  cs_ltot(wall->cs, ltot);

  physics_ref(&phys);
  physics_eta_shear(phys, &eta);

  /* Lower, then upper wall X, Y, and Z */

  if (wall->param->isboundary[X]) {
    hlub = wall->param->lubr_rc[X];
    h = r[X] - lmin[X] - ah;
    if (h < hlub) drag[X] = -6.0*pi*eta*ah*ah*(1.0/h - 1.0/hlub);
    h = lmin[X] + ltot[X] - r[X] - ah;
    if (h < hlub) drag[X] = -6.0*pi*eta*ah*ah*(1.0/h - 1.0/hlub);
  }

  if (wall->param->isboundary[Y]) {
    hlub = wall->param->lubr_rc[Y];
    h = r[Y] - lmin[Y] - ah;
    if (h < hlub) drag[Y] = -6.0*pi*eta*ah*ah*(1.0/h - 1.0/hlub);
    h = lmin[Y] + ltot[Y] - r[Y] - ah;
    if (h < hlub) drag[Y] = -6.0*pi*eta*ah*ah*(1.0/h - 1.0/hlub);
  }

  if (wall->param->isboundary[Z]) {
    hlub = wall->param->lubr_rc[Z];
    h = r[Z] - lmin[Z] - ah;
    if (h < hlub) drag[Z] = -6.0*pi*eta*ah*ah*(1.0/h - 1.0/hlub);
    h = lmin[Z] + ltot[Z] - r[Z] - ah;
    if (h < hlub) drag[Z] = -6.0*pi*eta*ah*ah*(1.0/h - 1.0/hlub);
  }

  return 0;
}
