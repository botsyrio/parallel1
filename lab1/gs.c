//the function int calc() and several other minor edits (noted in the comments) were written by David A. Foley on 3/15

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

/*** Skeleton for Lab 1 ***/

/***** Globals ******/
float **a; /* The coefficients */
float *x;  /* The unknowns */
float *b;  /* The constants */
float err; /* The absolute relative error */
int num = 0;  /* number of unknowns */
int my_rank, comm_sz;

/****** Function declarations */
void check_matrix(); /* Check whether the matrix will converge */
void get_input();  /* Read input from file */

int calc(); 

/********************************/



/* Function definitions: functions are ordered alphabetically ****/
/*****************************************************************/

/* 
   Conditions for convergence (diagonal dominance):
   1. diagonal element >= sum of all other elements of the row
   2. At least one diagonal element > sum of all other elements of the row
 */
void check_matrix()
{
  int bigger = 0; /* Set to 1 if at least one diag element > sum  */
  int i, j;
  float sum = 0;
  float aii = 0;
  
  for(i = 0; i < num; i++)
  {
    sum = 0;
    aii = fabs(a[i][i]);
    
    for(j = 0; j < num; j++)
       if( j != i)
	 sum += fabs(a[i][j]);
       
    if( aii < sum)
    {
      printf("The matrix will not converge.\n");
      exit(1);
    }
    
    if(aii > sum)
      bigger++;
    
  }
  
  if( !bigger )
  {
     printf("The matrix will not converge\n");
     exit(1);
  }
}


/******************************************************/
/* Read input from file */
/* After this function returns:
 * a[][] will be filled with coefficients and you can access them using a[i][j] for element (i,j)
 * x[] will contain the initial values of x
 * b[] will contain the constants (i.e. the right-hand-side of the equations
 * num will have number of variables
 * err will have the absolute error that you need to reach
 */
void get_input(char filename[])
{
  FILE * fp;
  int i,j;  
 
  fp = fopen(filename, "r");
  if(!fp)
  {
    printf("Cannot open file %s\n", filename);
    exit(1);
  }

 fscanf(fp,"%d ",&num);
 fscanf(fp,"%f ",&err);

 /* Now, time to allocate the matrices and vectors */
 a = (float**)malloc(num * sizeof(float*));
 if( !a)
  {
	printf("Cannot allocate a!\n");
	exit(1);
  }

 for(i = 0; i < num; i++) 
  {
    a[i] = (float *)malloc(num * sizeof(float)); 
    if( !a[i])
  	{
		printf("Cannot allocate a[%d]!\n",i);
		exit(1);
  	}
  }
 
 x = (float *) malloc(num * sizeof(float));
 if( !x)
  {
	printf("Cannot allocate x!\n");
	exit(1);
  }


 b = (float *) malloc(num * sizeof(float));
 if( !b)
  {
	printf("Cannot allocate b!\n");
	exit(1);
  }

 /* Now .. Filling the blanks */ 

 /* The initial values of Xs */
 for(i = 0; i < num; i++)
	fscanf(fp,"%f ", &x[i]);
 
 for(i = 0; i < num; i++)
 {
   for(j = 0; j < num; j++)
     fscanf(fp,"%f ",&a[i][j]);
   
   /* reading the b element */
   fscanf(fp,"%f ",&b[i]);
 }
 
 fclose(fp); 

}


/************************************************************/
int calc(){
	
	int recvCounts[comm_sz];
	for(int i=0; i<comm_sz; i++){
		int count = num/comm_sz;
		if(i<num%comm_sz)
			count++;
		recvCounts[i] = count;
	}
	
	int displs[comm_sz];
	displs[0]=0;
	for(int i=1; i<comm_sz; i++)
		displs[i]=displs[i-1]+recvCounts[i-1];
	
	//printf("process %d has constructed both arrays\n", my_rank);
	float xNew[recvCounts[my_rank]];
	
	int locUnf = 0;//contains the number of Xs calculated by the current process which have not yet met the error parameters
	int gloUnf = 1;//contains the number of Xs across all processes which have not yet met the error parameters -- initialized to 1
	int numIt = 0;
	
	while(gloUnf !=0){		
		for(int i=displs[my_rank]; i<displs[my_rank]+recvCounts[my_rank]; i++){
			float localSum=0;
			for(int j=0; j<num; j++){
				if(j!=i)
					localSum +=(a[i][j]*x[j]);
			}
			xNew[i-displs[my_rank]]=(b[i]-localSum)/a[i][i];
			
		}
		//printf("process %d has completed its local calculations\n", my_rank);
		locUnf=0;
		float error;
		for(int i =displs[my_rank]; i<displs[my_rank]+recvCounts[my_rank]; i++){
			error = ((xNew[i-displs[my_rank]]-x[i])/xNew[i-displs[my_rank]]);
			if(error<0)
				error = -1*error;
			if(error>err)
				locUnf++;
		}
		//printf("process %d has completed its error \n", my_rank);
		MPI_Allreduce(&locUnf, &gloUnf, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		//printf("process %d has completed all reduce, locUnf is %d, gloUnf is %d \n", my_rank, locUnf, gloUnf);
		if(gloUnf!=0)
			MPI_Allgatherv(xNew, recvCounts[my_rank], MPI_FLOAT, x, (const int *)recvCounts, (const int*)displs, MPI_FLOAT, MPI_COMM_WORLD);
			
		//printf("process %d has completed all Gather\n", my_rank);
		numIt++;
	}
	MPI_Gatherv(xNew, recvCounts[my_rank], MPI_FLOAT, x, (const int *)recvCounts, (const int*)displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
	return numIt;
}

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
		
	int i;
	int nit = 0; /* number of iterations */
	FILE * fp;
	char output[100] ="";
	  
	if( argc != 2)
	{
		printf("Usage: ./gsref filename\n");
		exit(1);
	}
	  
	/* Read the input file and fill the global data structure above */ 
	
	get_input(argv[1]);
	 
	/* Check for convergence condition */
	/* This function will exit the program if the coffeicient will never converge to 
	 * the needed absolute error. 
	 * This is not expected to happen for this programming assignment.
	 */
	check_matrix();
		
		
	
	//MPI_Bcast((void*)&num, 1, MPI_INT, 0, MPI_COMM_WORLD);
	//MPI_Bcast((void*)*a, num*num, MPI_FLOAT, 0, MPI_COMM_WORLD);
	//MPI_Bcast((void*)x, num, MPI_FLOAT, 0, MPI_COMM_WORLD);
	//MPI_Bcast((void*)b, num, MPI_FLOAT, 0, MPI_COMM_WORLD);
	//MPI_Bcast((void*)&err, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
	
	nit = calc();
	
	if(my_rank==0){
		/* Writing results to file */
		sprintf(output,"%d.sol",num);
		fp = fopen(output,"w");
		if(!fp)
		{
		printf("Cannot create the file %s\n", output);
		exit(1);
		}
			
		for( i = 0; i < num; i++)
			fprintf(fp,"%f\n",x[i]);
		 
		printf("total number of iterations: %d\n", nit);
		 
		fclose(fp);
	}
	
	MPI_Finalize();
	exit(0);

}
