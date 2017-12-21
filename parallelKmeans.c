/*
//  parallelKmeans.c
//
//  Manisha Siddartha Nalla & Bharath Kumar Kande
*/

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define  MASTER	0

void gendata(int startindex, int endindex, double *data, int dimensions);
void second_Centroid(int startindex, int endindex, double *max, int *indexofc, int dimensions, double **cluster_centroid, double *data);
void initial_kClusters(int startindex, int endindex, int i, int noofclusters, double *max, int *indexofc, int dimensions, double *data, double **cluster_centroid);
int assigning_Points(int start_index, int end_index, int dimensions, int noofclusters, double *data, double **cluster_centroid, int *cluster_assign, int *cluster_size);
void clusterSizeZeroCondition(int start_index, int end_index, int emptytest, int *indexofnew, double *max, int noofclusters, int dimensions, double *data, double **cluster_centroid);
void clusterSort(int start_index, int end_index, int noofclusters, int dimensions, double *data, int* cluster_assign);
void newCentroids(int start_index, int end_index, int dimensions, int *global_cluster_size, double **cluster_centroid, double *data, int *cluster_assign);
void calculate_ClusterRadius(int start_index, int end_index, int noofclusters, int dimensions, int *cluster_assign, double **cluster_centroid, double *cluster_radius, double *data);
void exhaustiveSearch(int start_index, int end_index, int index, int *visits, double *min, int dimensions, double *query, double *data, int *cluster_assign);


int main(int argc, char *argv[]) {
	int dimensions, noofpoints, noofclusters, ndata, *cluster_start, *cluster_size, *cluster_assign, i, j, k, l, x, indexofc = -1, index, glob_sum = 0;
	double *data, *query, *cluster_radius, *boundary_distance, **global_cluster_centroid, *global_cluster_radius, **cluster_centroid, sum = 0.0, max = -1, min, distance;
	int   numtasks, taskid, tag1, tag2, tag3, chunksize, emptytest = -1, indexofnew = -1, visits = 0, visits1 = 0, count, search_times;

	MPI_Status status;

	// Initializations
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
	printf("MPI task %d has started...\n", taskid);


	dimensions = atoi(argv[1]);
	ndata = atoi(argv[2]);
	noofclusters = atoi(argv[3]);
	noofpoints = ndata*dimensions;

	data = (double *)malloc(noofpoints * sizeof(double));
	query = (double *)malloc(dimensions * sizeof(double));
	cluster_size = (int *)calloc(noofclusters, sizeof(int));
	boundary_distance = malloc(noofclusters * sizeof(double));
	cluster_start = (int *)malloc(noofclusters * sizeof(int));
	cluster_radius = (double *)calloc(noofclusters, sizeof(double));
	global_cluster_radius = (double *)calloc(noofclusters, sizeof(double));
	cluster_assign = (int *)malloc(ndata * sizeof(int));
	cluster_centroid = (double **)malloc(noofclusters * sizeof(double *));
	for (i = 0; i<noofclusters; i++)
		*(cluster_centroid + i) = (double *)malloc(dimensions * sizeof(double));
	global_cluster_centroid = (double **)malloc(noofclusters * sizeof(double *));
	for (i = 0; i<noofclusters; i++)
		*(global_cluster_centroid + i) = (double *)malloc(dimensions * sizeof(double));
	int *global_cluster_size = (int *)calloc(noofclusters, sizeof(int));


	chunksize = (ndata / numtasks);
	tag2 = 1;
	tag1 = 2;
	tag3 = 3;
	int loopcondition;
	int *recvbufsize = malloc(numtasks * sizeof(int));
	int *displs = calloc(numtasks, sizeof(int));
	double global_max;
	int *check = malloc(numtasks * sizeof(int));
	double *final_distance = malloc(numtasks * sizeof(double));
	int seeds[4] = { 1,2,3,4 };

	// End of: Initializations

	if (taskid == MASTER) {
		/* initializing data array*/
		if (numtasks == 1) {
			for (i = 0; i < 4; i++) {
				srand(seeds[i]);
				gendata(i*(chunksize*dimensions) / 4, (i + 1)*(chunksize*dimensions) / 4, data, dimensions);
			}
		}
		else if (numtasks == 2) {
			for (i = 0; i < 2; i++) {
				srand(seeds[i]);
				gendata(i*(chunksize*dimensions) / 2, (i + 1)*(chunksize*dimensions) / 2, data, dimensions);
			}
		}
		else if (numtasks == 4) {
			srand(seeds[taskid]);
			gendata(0, (chunksize*dimensions), data, dimensions);
		}
		else
			printf(" \n Please choose the number of tasks equal to 1,2,4 \n");

		/* Calculating first centroid*/
		for (i = 0; i<dimensions; i++)
			cluster_centroid[0][i] = data[i];

		/* broadcasting first centroid*/
		MPI_Bcast(&cluster_centroid[0][0], dimensions, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

		/* Calculating second centroid*/
		second_Centroid(0, chunksize, &max, &indexofc, dimensions, cluster_centroid, data);
		MPI_Allreduce(&max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		if (global_max == max)
		{
			for (j = 0; j < dimensions; j++)
				cluster_centroid[1][j] = data[(indexofc*dimensions) + j];
		}
		else {
			MPI_Recv(cluster_centroid[1], dimensions, MPI_DOUBLE, MPI_ANY_SOURCE, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		/* broadcasting second centroid*/
		MPI_Bcast(cluster_centroid[1], dimensions, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

		/*finding remaining k-2 clusters */
		for (i = 2; i < noofclusters; i++) {
			initial_kClusters(0, chunksize, i, noofclusters, &max, &indexofc, dimensions, data, cluster_centroid);
			MPI_Allreduce(&max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
			if (global_max == max)
			{
				for (j = 0; j < dimensions; j++)
					cluster_centroid[i][j] = data[(indexofc*dimensions) + j];
			}
			else {
				MPI_Recv(cluster_centroid[i], dimensions, MPI_DOUBLE, MPI_ANY_SOURCE, tag3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
			MPI_Bcast(cluster_centroid[i], dimensions, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
		}

		/* looping untill cluster assigns doesnt change*/

		while (1) {

			/*Assigning points to cluster centroids  */
			for (i = 0; i < noofclusters; i++)
				cluster_size[i] = 0;
			check[taskid] = assigning_Points(0, chunksize, dimensions, noofclusters, data, cluster_centroid, cluster_assign, cluster_size);

			/* loop break condition. It breaks when cluster assign values doesnt change from previous iteration*/
			MPI_Allgather(&check[taskid], 1, MPI_INT, check, 1, MPI_INT, MPI_COMM_WORLD);
			sum = 0;
			for (i = 0; i < numtasks; i++)
				sum += check[i];
			if (sum == 0)
				break;

			/* Calculating cluster sizes*/
			MPI_Allreduce(cluster_size, global_cluster_size, noofclusters, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

			/*Calculate cluster start based on cluster size and broadcasting it*/
			cluster_start[0] = 0;
			for (i = 1; i < noofclusters; i++) {
				cluster_start[i] = cluster_start[i - 1] + global_cluster_size[i - 1];
			}
			MPI_Bcast(&cluster_start[0], noofclusters, MPI_INT, MASTER, MPI_COMM_WORLD);

			/* As we are using gatherv condition we have to calculate the size of data array recieved from various processes and stride value*/
			/* We are using gatherv because last process might contain more data when the total data is not a multiple of number of processes.*/
			for (i = 0; i < numtasks; i++) {
				if (i != (numtasks - 1))
					recvbufsize[i] = chunksize;
				else
					recvbufsize[i] = (ndata - ((numtasks - 1)*chunksize));
				displs[i] = i*chunksize;
			}

			/* Broadcasting stride and recvbuffsize array*/
			MPI_Bcast(&recvbufsize[0], numtasks, MPI_INT, MASTER, MPI_COMM_WORLD);
			MPI_Bcast(&displs[0], numtasks, MPI_INT, MASTER, MPI_COMM_WORLD);

			/* gathering cluster assign values from all process*/
			MPI_Allgatherv(&cluster_assign[0], chunksize, MPI_INT, cluster_assign, recvbufsize, displs, MPI_INT, MPI_COMM_WORLD);

			/* Checking if any cluster is empty and broadcasting its index*/
			for (i = 0; i < noofclusters; i++) {
				if (global_cluster_size[i] == 0) {
					emptytest = i;
					break;
				}
			}
			MPI_Bcast(&emptytest, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

			/* if any cluster size is 0 calculate new centroid*/
			if (emptytest != -1) {
				clusterSizeZeroCondition(0, chunksize, emptytest, &indexofnew, &max, noofclusters, dimensions, data, cluster_centroid);
				MPI_Allreduce(&max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
				if (global_max == max)
				{
					for (j = 0; j < dimensions; j++)
						cluster_centroid[emptytest][j] = data[(indexofnew*dimensions) + j];
				}
				else {
					tag3++;
					MPI_Recv(cluster_centroid[emptytest], dimensions, MPI_DOUBLE, MPI_ANY_SOURCE, tag3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
				/*Broadcasting new centroid*/
				MPI_Bcast(cluster_centroid[emptytest], dimensions, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
			}

			/* If no empty clusters */
			else {
				/*call local sort*/
				clusterSort(0, chunksize, noofclusters, dimensions, data, cluster_assign);

				/*performing gather for cluster assign after sort*/
				MPI_Allgatherv(&cluster_assign[0], chunksize, MPI_INT, cluster_assign, recvbufsize, displs, MPI_INT, MPI_COMM_WORLD);

				/* calculating recv buffer sizes and stripe values for data array*/
				for (i = 0; i < numtasks; i++) {
					if (i != (numtasks - 1))
						recvbufsize[i] = chunksize*dimensions;
					else
						recvbufsize[i] = (ndata - ((numtasks - 1)*chunksize))*dimensions;
					displs[i] = i*chunksize*dimensions;
				}
				/*Broadcasting the above calculated values */
				MPI_Bcast(&recvbufsize[0], numtasks, MPI_INT, MASTER, MPI_COMM_WORLD);
				MPI_Bcast(&displs[0], numtasks, MPI_INT, MASTER, MPI_COMM_WORLD);

				/* performing gather for data array after sort*/
				MPI_Allgatherv(&data[0], chunksize*dimensions, MPI_DOUBLE, data, recvbufsize, displs, MPI_DOUBLE, MPI_COMM_WORLD);

				/* calling global sort*/
				clusterSort(0, ndata, noofclusters, dimensions, data, cluster_assign);

				/*Broadcasting sorted data array*/
				MPI_Bcast(&cluster_assign[0], ndata, MPI_INT, MASTER, MPI_COMM_WORLD);
				MPI_Bcast(&data[0], ndata*dimensions, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

				/* Recalculating centroids by initializing them to zero*/
				for (i = 0; i < noofclusters; i++) {
					for (j = 0; j < dimensions; j++) {
						cluster_centroid[i][j] = 0.0;
					}
				}

				/* calling recalculation function */
				newCentroids(0, chunksize, dimensions, global_cluster_size, cluster_centroid, data, cluster_assign);
				for (i = 0; i < noofclusters; i++)
					MPI_Allreduce(cluster_centroid[i], global_cluster_centroid[i], dimensions, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			}
		}

		/*Claculating cluster radius*/
		calculate_ClusterRadius(0, chunksize, noofclusters, dimensions, cluster_assign, global_cluster_centroid, cluster_radius, data);
		MPI_Allreduce(cluster_radius, global_cluster_radius, noofclusters, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

		/*query point*/
		for (search_times = 0; search_times < 1; search_times++) {
			visits = 0;
			visits1 = 0;
			srand(5 + search_times * 100);
			for (i = 0; i < dimensions; i++)
				query[i] = ((double)rand()*(101)) / (double)RAND_MAX;
			MPI_Bcast(&query[0], dimensions, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

			/*calculate cluster boundaries*/
			int count = 0;
			min = -1.0;
			for (x = 0; x < noofclusters; x++) {
				distance = 0.0;
				for (i = 0; i < dimensions; i++) {
					distance = distance + pow((query[i] - global_cluster_centroid[x][i]), 2);
				}
				distance = (global_cluster_radius[x] - sqrt(distance));
				if (distance <= 0) {
					index = x;
					count++;
					distance = -distance;
				}
				else if ((min == -1.0 || min > distance) && count == 0) {
					min = distance;
					index = x;
				}
				boundary_distance[x] = distance;
			}

			/*broadcasting boundary distances*/
			MPI_Bcast(&boundary_distance[0], noofclusters, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

			/* exhaustive search in the closest cluster and finds closest point to query point*/
			distance = 0.0;
			min = -1.0;
			for (i = cluster_start[index]; i < (cluster_start[index] + global_cluster_size[index]); i++) {
				distance = 0.0;
				visits++;
				for (j = 0; j < dimensions; j++)
					distance = distance + pow((query[j] - data[(i*dimensions) + j]), 2);
				distance = sqrt(distance);
				if (min == -1.0 || min > distance)
					min = distance;
			}

			/* minimum distance */
			MPI_Bcast(&min, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
			MPI_Bcast(&visits, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
			MPI_Bcast(&index, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

			/* exhaustive search */
			for (x = 0; x < noofclusters; x++) {
				if (boundary_distance[x] < min && index != x) {
					exhaustiveSearch(0, chunksize, x, &visits1, &min, dimensions, query, data, cluster_assign);
				}
			}
			MPI_Reduce(&visits1, &glob_sum, 1, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);
			visits = visits + glob_sum;

			MPI_Gather(&min, 1, MPI_DOUBLE, final_distance, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
			for (i = 0; i < numtasks; i++) {
				if (min == -1.0)
					min = final_distance[i];
				else if (final_distance[i] != -1.0 && min > final_distance[i])
					min = final_distance[i];
			}
			printf("\n minimum distance : %lf \n", min);
		}
	}


	if (taskid > MASTER) {
		/*If size of data is not a multiple of number of process the last process will get excess remaining points*/
		loopcondition = ((taskid*chunksize) + chunksize);
		if (taskid == (numtasks - 1))
			loopcondition = (ndata);

		/*Generating data*/
		if (numtasks == 2) {
			for (i = 2; i < 4; i++) {
				srand(seeds[i]);
				gendata((i*(chunksize)*dimensions) / 2, (i + 1)*(loopcondition * dimensions) / 4, data, dimensions);
			}
		}
		else if (numtasks == 4) {
			srand(taskid + 1);
			gendata((taskid*chunksize)*dimensions, (loopcondition * dimensions), data, dimensions);
		}

		/*recieving fst centroid*/
		MPI_Bcast(&cluster_centroid[0][0], dimensions, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);


		/* Calculating second centroid*/
		second_Centroid(taskid*chunksize, loopcondition, &max, &indexofc, dimensions, cluster_centroid, data);
		MPI_Allreduce(&max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		if (global_max == max)
		{
			for (j = 0; j < dimensions; j++)
				cluster_centroid[1][j] = data[(indexofc*dimensions) + j];
			MPI_Send(cluster_centroid[1], dimensions, MPI_DOUBLE, MASTER, 5, MPI_COMM_WORLD);
		}

		/*receiving second centroid*/
		MPI_Bcast(cluster_centroid[1], dimensions, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

		/*finding remaining k-2 clusters */
		for (i = 2; i < noofclusters; i++) {
			initial_kClusters(taskid*chunksize, loopcondition, i, noofclusters, &max, &indexofc, dimensions, data, cluster_centroid);
			MPI_Allreduce(&max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
			if (global_max == max)
			{
				for (j = 0; j < dimensions; j++)
					cluster_centroid[i][j] = data[(indexofc*dimensions) + j];
				MPI_Send(cluster_centroid[i], dimensions, MPI_DOUBLE, MASTER, tag3, MPI_COMM_WORLD);
			}
			MPI_Bcast(cluster_centroid[i], dimensions, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
		}

		/*looping untill cluster centroids dont change*/
		while (1) {
			for (i = 0; i < noofclusters; i++)
				cluster_size[i] = 0;

			/*Assigning points to clusters*/
			check[taskid] = assigning_Points(taskid*chunksize, loopcondition, dimensions, noofclusters, data, cluster_centroid, cluster_assign, cluster_size);

			/*loop break condition*/
			MPI_Allgather(&check[taskid], 1, MPI_INT, check, 1, MPI_INT, MPI_COMM_WORLD);
			sum = 0;
			for (i = 0; i < numtasks; i++)
				sum += check[i];
			if (sum == 0)
				break;

			/*sum cluster sizes calculated in each process*/
			MPI_Allreduce(cluster_size, global_cluster_size, noofclusters, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

			/*Recieving cluster start array*/
			MPI_Bcast(&cluster_start[0], noofclusters, MPI_INT, MASTER, MPI_COMM_WORLD);

			/*recieving cluster assign relaated stride and recv buff sizes*/
			MPI_Bcast(&recvbufsize[0], numtasks, MPI_INT, MASTER, MPI_COMM_WORLD);
			MPI_Bcast(&displs[0], numtasks, MPI_INT, MASTER, MPI_COMM_WORLD);

			/*gathering cluster assign*/
			if (taskid != (numtasks - 1))
				MPI_Allgatherv(&cluster_assign[(taskid*chunksize)], chunksize, MPI_INT, cluster_assign, recvbufsize, displs, MPI_INT, MPI_COMM_WORLD);
			else
				MPI_Allgatherv(&cluster_assign[(taskid*chunksize)], (ndata - ((numtasks - 1)*chunksize)), MPI_INT, cluster_assign, recvbufsize, displs, MPI_INT, MPI_COMM_WORLD);

			/* recieving cluster empty information and calculating new centroids*/
			MPI_Bcast(&emptytest, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
			if (emptytest != -1) {
				clusterSizeZeroCondition(taskid*chunksize, loopcondition, emptytest, &indexofnew, &max, noofclusters, dimensions, data, cluster_centroid);
				MPI_Allreduce(&max, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
				if (global_max == max)
				{
					for (j = 0; j < dimensions; j++)
						cluster_centroid[emptytest][j] = data[(indexofnew*dimensions) + j];
					tag3++;
					MPI_Send(cluster_centroid[emptytest], dimensions, MPI_DOUBLE, MASTER, tag3, MPI_COMM_WORLD);
				}
				MPI_Bcast(cluster_centroid[emptytest], dimensions, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
			}

			/* if no non empty clusters*/
			else {

				/*Call sort*/
				clusterSort(taskid*chunksize, loopcondition, noofclusters, dimensions, data, cluster_assign);

				/* All gather for cluster assign array array*/
				if (taskid != (numtasks - 1))
					MPI_Allgatherv(&cluster_assign[(taskid*chunksize)], chunksize, MPI_INT, cluster_assign, recvbufsize, displs, MPI_INT, MPI_COMM_WORLD);
				else
					MPI_Allgatherv(&cluster_assign[(taskid*chunksize)], (ndata - ((numtasks - 1)*chunksize)), MPI_INT, cluster_assign, recvbufsize, displs, MPI_INT, MPI_COMM_WORLD);

				MPI_Bcast(&recvbufsize[0], numtasks, MPI_INT, MASTER, MPI_COMM_WORLD);
				MPI_Bcast(&displs[0], numtasks, MPI_INT, MASTER, MPI_COMM_WORLD);

				/* All gather for data array*/
				if (taskid != (numtasks - 1))
					MPI_Allgatherv(&data[(taskid*chunksize*dimensions)], chunksize*dimensions, MPI_DOUBLE, data, recvbufsize, displs, MPI_DOUBLE, MPI_COMM_WORLD);
				else
					MPI_Allgatherv(&data[(taskid*chunksize*dimensions)], (ndata - ((numtasks - 1)*chunksize))*dimensions, MPI_DOUBLE, data, recvbufsize, displs, MPI_DOUBLE, MPI_COMM_WORLD);

				/* Recieving data array and cluster assign*/
				MPI_Bcast(&cluster_assign[0], ndata, MPI_INT, MASTER, MPI_COMM_WORLD);
				MPI_Bcast(&data[0], ndata*dimensions, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

				/* Recalculating centroids by initializing them to zero*/
				for (i = 0; i < noofclusters; i++) {
					for (j = 0; j < dimensions; j++) {
						cluster_centroid[i][j] = 0.0;
					}
				}
				newCentroids(taskid*chunksize, loopcondition, dimensions, global_cluster_size, cluster_centroid, data, cluster_assign);
				for (i = 0; i < noofclusters; i++)
					MPI_Allreduce(cluster_centroid[i], global_cluster_centroid[i], dimensions, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

			}
		}

		/*cluster radius*/
		calculate_ClusterRadius(taskid*chunksize, loopcondition, noofclusters, dimensions, cluster_assign, global_cluster_centroid, cluster_radius, data);
		MPI_Allreduce(cluster_radius, global_cluster_radius, noofclusters, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

		for (search_times = 0; search_times < 1; search_times++) {
			visits1 = 0;
			/*recieving query*/
			MPI_Bcast(&query[0], dimensions, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

			/*broadcasting boundary distances*/
			MPI_Bcast(&boundary_distance[0], noofclusters, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

			/* minimum distance */
			MPI_Bcast(&min, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
			MPI_Bcast(&visits, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
			MPI_Bcast(&index, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

			/* exhaustive search */
			for (x = 0; x < noofclusters; x++) {
				if (boundary_distance[x] < min && index != x) {
					exhaustiveSearch(taskid*chunksize, loopcondition, x, &visits1, &min, dimensions, query, data, cluster_assign);
				}
			}
			MPI_Reduce(&visits1, &glob_sum, 1, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);
			MPI_Gather(&min, 1, MPI_DOUBLE, final_distance, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
		}
	}
	MPI_Finalize();
	return 0;
}

void gendata(int startindex, int endindex, double *data, int dimensions) {
	int i;
	for (i = startindex; i < endindex; i++) {
		data[i] = ((double)rand()*(101)) / (double)RAND_MAX;
	}
}

void second_Centroid(int startindex, int endindex, double *max, int *indexofc, int dimensions, double **cluster_centroid, double *data) {
	int i, j;
	double sum, local_max = -1, local_index;
	for (i = startindex; i<endindex; i++) {
		sum = 0.0;
		for (j = 0; j<dimensions; j++)
			sum += pow(fabs(cluster_centroid[0][j] - data[(i*dimensions) + j]), 2);
		sum = sqrt(sum);
		if (sum > local_max) {
			local_max = sum;
			local_index = i;
		}
	}
	*max = local_max;
	*indexofc = local_index;
}

void initial_kClusters(int startindex, int endindex, int i, int noofclusters, double *max, int *indexofc, int dimensions, double *data, double **cluster_centroid) {
	int j, l, k, local_index;
	double min, local_max, sum;

	local_max = -1.0;
	for (j = startindex; j < endindex; j++) {
		min = -1.0;
		for (l = 0; l < i; l++) {
			sum = 0.0;
			for (k = 0; k < dimensions; k++)
				sum += pow((data[(j*dimensions) + k] - cluster_centroid[l][k]), 2);
			sum = sqrt(sum);
			if (sum < min || min == -1.0) {
				min = sum;
			}
		}
		if (min > local_max) {
			local_max = min;
			local_index = j;
		}

	}
	*max = local_max;
	*indexofc = local_index;
}

int assigning_Points(int start_index, int end_index, int dimensions, int noofclusters, double *data, double **cluster_centroid, int *cluster_assign, int *cluster_size) {
	start_index = start_index * dimensions;
	end_index = end_index*dimensions;

	double min, distance;
	int  i, j, k, nearestc, ccounter = 0;
	for (i = start_index; i< end_index; i = i + dimensions) {
		min = -1.0;
		for (j = 0; j<noofclusters; j++) {
			distance = 0.0;
			for (k = 0; k<dimensions; k++)
				distance += pow((data[i + k] - cluster_centroid[j][k]), 2);
			distance = sqrt(distance);
			if ((distance < min) || (min == -1.0)) {
				min = distance;
				nearestc = j;
			}
		}
		if (cluster_assign[(i / dimensions)] != nearestc)
			ccounter++;
		cluster_assign[(i / dimensions)] = nearestc;
		cluster_size[nearestc]++;
	}
	return ccounter;
}

void clusterSizeZeroCondition(int start_index, int end_index, int emptytest, int *indexofnew, double *max, int noofclusters, int dimensions, double *data, double **cluster_centroid) {
	double localmax, distance, min;
	int local_index, i, j, k;
	localmax = -1.0;
	start_index = start_index * dimensions;
	end_index = end_index*dimensions;
	for (i = start_index; i<end_index; i = i + dimensions) {
		min = -1.0;
		for (j = 0; j<noofclusters; j++) {
			if (j == emptytest)
				continue;
			distance = 0.0;
			for (k = 0; k<dimensions; k++)
				distance += pow((data[i + k] - cluster_centroid[j][k]), 2);
			distance = sqrt(distance);
			if (distance<min || min == -1.0)
				min = distance;
		}
		if (min > localmax) {
			localmax = min;
			local_index = i;
		}
	}
	*max = localmax;
	*indexofnew = local_index;
}

void clusterSort(int start_index, int end_index, int noofclusters, int dimensions, double *data, int* cluster_assign) {
	int i, j, k;
	double temp, temp1;
	int left = start_index;
	start_index = start_index * dimensions;
	end_index = end_index*dimensions;
	for (i = 0; i<noofclusters; i++) {
		for (j = start_index; j< end_index; j = j + dimensions) {
			if (cluster_assign[(j / dimensions)] == i) {
				temp = cluster_assign[left];
				cluster_assign[left] = cluster_assign[(j / dimensions)];
				cluster_assign[(j / dimensions)] = temp;
				for (k = 0; k<dimensions; k++) {
					temp1 = data[(left*dimensions) + k];
					data[(left*dimensions) + k] = data[j + k];
					data[j + k] = temp1;
				}
				left++;
			}
		}
	}
}

void newCentroids(int start_index, int end_index, int dimensions, int *global_cluster_size, double **cluster_centroid, double *data, int *cluster_assign) {
	int j, k;
	start_index = start_index*dimensions;
	end_index = end_index*dimensions;
	for (j = start_index; j<end_index; j = j + dimensions) {
		for (k = 0; k<dimensions; k++)
			cluster_centroid[cluster_assign[j / dimensions]][k] += data[j + k] / global_cluster_size[cluster_assign[j / dimensions]];
	}
}

void calculate_ClusterRadius(int start_index, int end_index, int noofclusters, int dimensions, int *cluster_assign, double **global_cluster_centroid, double *cluster_radius, double *data) {
	double  distance;
	int i, j, k;
	start_index = start_index*dimensions;
	end_index = end_index*dimensions;
	for (j = start_index; j<end_index; j += dimensions) {
		distance = 0.0;
		for (k = 0; k<dimensions; k++) {
			distance += pow((global_cluster_centroid[cluster_assign[j / dimensions]][k] - data[j + k]), 2);
		}
		distance = sqrt(distance);
		if (distance > cluster_radius[cluster_assign[j / dimensions]]) {
			cluster_radius[cluster_assign[j / dimensions]] = distance;
		}
	}
}

void exhaustiveSearch(int start_index, int end_index, int index, int *visits1, double *min, int dimensions, double *query, double *data, int *cluster_assign) {
	int i, j;
	double distance;
	for (i = start_index; i < end_index; i++) {
		if (cluster_assign[i] == index) {
			distance = 0.0;
			*visits1 = *visits1 + 1;
			for (j = 0; j < dimensions; j++)
				distance = distance + pow((query[j] - data[(i*dimensions) + j]), 2);
			distance = sqrt(distance);
			if (*min > distance)
				*min = distance;
		}
	}
}