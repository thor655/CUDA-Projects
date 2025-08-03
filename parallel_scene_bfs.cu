/*
	Do not make any changes to the boiler plate code or the other files in the folder.
	Use cudaFree to deallocate any memory not in usage.
	Optimize as much as possible.
 */

#include "SceneNode.h"
#include <queue>
#include "Renderer.h"
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <chrono>
#include <utility>
#include <algorithm>

#define BLOCK_ID (blockIdx.x * gridDim.y * gridDim.z + blockIdx.y * gridDim.z + blockIdx.z)
#define THREAD_TOTAL (blockDim.x * blockDim.y * blockDim.z) 
#define THREAD_ID (threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z)

void readFile (const char *fileName, std::vector<SceneNode*> &scenes, std::vector<std::vector<int> > &edges, std::vector<std::vector<int> > &translations, int &frameSizeX, int &frameSizeY) {
	/* Function for parsing input file*/

	FILE *inputFile = NULL;
	// Read the file for input. 
	if ((inputFile = fopen (fileName, "r")) == NULL) {
		printf ("Failed at opening the file %s\n", fileName) ;
		return ;
	}

	// Input the header information.
	int numMeshes ;
	fscanf (inputFile, "%d", &numMeshes) ;
	fscanf (inputFile, "%d %d", &frameSizeX, &frameSizeY) ;
	

	// Input all meshes and store them inside a vector.
	int meshX, meshY ;
	int globalPositionX, globalPositionY; // top left corner of the matrix.
	int opacity ;
	int* currMesh ;
	for (int i=0; i<numMeshes; i++) {
		fscanf (inputFile, "%d %d", &meshX, &meshY) ;
		fscanf (inputFile, "%d %d", &globalPositionX, &globalPositionY) ;
		fscanf (inputFile, "%d", &opacity) ;
		currMesh = (int*) malloc (sizeof (int) * meshX * meshY) ;
		for (int j=0; j<meshX; j++) {
			for (int k=0; k<meshY; k++) {
				fscanf (inputFile, "%d", &currMesh[j*meshY+k]) ;
			}
		}
		//Create a Scene out of the mesh.
		SceneNode* scene = new SceneNode (i, currMesh, meshX, meshY, globalPositionX, globalPositionY, opacity) ; 
		scenes.push_back (scene) ;
	}

	// Input all relations and store them in edges.
	int relations;
	fscanf (inputFile, "%d", &relations) ;
	int u, v ; 
	for (int i=0; i<relations; i++) {
		fscanf (inputFile, "%d %d", &u, &v) ;
		edges.push_back ({u,v}) ;
	}

	// Input all translations.
	int numTranslations ;
	fscanf (inputFile, "%d", &numTranslations) ;
	std::vector<int> command (3, 0) ;
	for (int i=0; i<numTranslations; i++) {
		fscanf (inputFile, "%d %d %d", &command[0], &command[1], &command[2]) ;
		translations.push_back (command) ;
	}
}


void writeFile (const char* outputFileName, int *hFinalPng, int frameSizeX, int frameSizeY) {
	/* Function for writing the final png into a file.*/
	FILE *outputFile = NULL; 
	if ((outputFile = fopen (outputFileName, "w")) == NULL) {
		printf ("Failed while opening output file\n") ;
	}
	
	for (int i=0; i<frameSizeX; i++) {
		for (int j=0; j<frameSizeY; j++) {
			fprintf (outputFile, "%d ", hFinalPng[i*frameSizeY+j]) ;
		}
		fprintf (outputFile, "\n") ;
	}
}

// __global__ 
// void tree_traversal(int V, int *x_trans, int *y_trans, int *hOffset, int*hCsr, int *hGlobalCoordinatesX, int *hGlobalCoordinatesY, volatile bool *vis){
// 	unsigned id = BLOCK_ID * THREAD_TOTAL + THREAD_ID;
// 	if(id%32==0 && id/32<V){
// 		id = id/32;
// 		if(id == 0) vis[0] = true;
// 		// printf("%d says hi\n",id);
// 		while(!vis[id]){
// 			//stall
// 		}
// 		// printf("%d unlocked\n",id);
// 		int child_id;
// 		for(int j = hOffset[id]; j<hOffset[id+1]; ++j){
// 			child_id = hCsr[j];
// 			x_trans[child_id] += x_trans[id];
// 			y_trans[child_id] += y_trans[id];
// 			vis[child_id] = true;
// 			// printf("from %d unlock %d\n",id,child_id);
// 		}
// 		hGlobalCoordinatesX[id]+=x_trans[id];
// 		hGlobalCoordinatesY[id]+=y_trans[id];
// 	}
// }

// __global__
// void upd_scene(int indx, int *hFinalPng, int *hMesh, int *hGlobalCoordinatesX, int *hGlobalCoordinatesY, int framesizeX, int framesizeY){
// 	int idx = blockIdx.x;
// 	int idy = threadIdx.x;
// 	int curx = idx + hGlobalCoordinatesX[indx];	
// 	int cury = idy + hGlobalCoordinatesY[indx];	
// 	if(curx>=0 && curx<framesizeX && cury>=0 && cury<framesizeY){
// 		hFinalPng[curx*framesizeY+cury] = hMesh[idx*blockDim.x+idy];
// 	}
// }
 
__global__
void upd_scene2(int indx, int *hFinalPng, int *hMesh, int *hGlobalCoordinatesX, int *hGlobalCoordinatesY, int framesizeX, int framesizeY, int *cur_opcty, int opc){
	int idx = blockIdx.x;
	int idy = threadIdx.x;
	int curx = idx + hGlobalCoordinatesX[indx];	
	int cury = idy + hGlobalCoordinatesY[indx];	
	if(curx>=0 && curx<framesizeX && cury>=0 && cury<framesizeY){
		if(opc >= cur_opcty[curx*framesizeY+cury]){
			hFinalPng[curx*framesizeY+cury] = hMesh[idx*blockDim.x+idy];
			cur_opcty[curx*framesizeY+cury] = opc;
		}
	}
}


// __global__
// void comparator(int *p1, int *p2, int fx, int fy){
// 	unsigned id = BLOCK_ID * THREAD_TOTAL + THREAD_ID;
// 	if(id<fx*fy){
// 		if(p1[id]!=p2[id]){
// 			printf("%d -> (%d) & (%d)\n",id, p1[id], p2[id]);
// 		}
// 	}
// }

// __global__
// void printframe(int *p, int fx, int fy){
// 	printf("%dx%d\n",fx,fy);
// 	for(int i=0; i<fx; i++){
// 		for(int j=0; j<fy; j++){
// 			printf("%d ",p[i*fy+j]);
// 		}
// 		printf(" <- \n");
// 	}
// }

__global__
void tree_bfs_levelwise(int V, bool *to_visit, int *x_trans, int *y_trans, int *hOffset, int*hCsr, int *hGlobalCoordinatesX, int *hGlobalCoordinatesY, bool *to_continue){
	unsigned id = BLOCK_ID * THREAD_TOTAL + THREAD_ID;
	if(to_visit[id] && id<V){
		// printf("Node %d says hi\n",id);
		int child_id;
		for(int j = hOffset[id]; j<hOffset[id+1]; ++j){
			child_id = hCsr[j];
			x_trans[child_id] += x_trans[id];
			y_trans[child_id] += y_trans[id];
			to_visit[child_id] = true;
			to_continue[0] = true;
		}
		to_visit[id] = false;
		hGlobalCoordinatesX[id] += x_trans[id];
		hGlobalCoordinatesY[id] += y_trans[id];
	}
}


int main (int argc, char **argv) {
	
	// Read the scenes into memory from File.
	const char *inputFileName = argv[1] ;
	int* hFinalPng ; 

	int frameSizeX, frameSizeY ;
	std::vector<SceneNode*> scenes ;
	std::vector<std::vector<int> > edges ;
	std::vector<std::vector<int> > translations ;
	readFile (inputFileName, scenes, edges, translations, frameSizeX, frameSizeY) ;
	hFinalPng = (int*) malloc (sizeof (int) * frameSizeX * frameSizeY) ;
	
	// Make the scene graph from the matrices.
    Renderer* scene = new Renderer(scenes, edges) ;

	// Basic information.
	int V = scenes.size () ;
	int E = edges.size () ;
	int numTranslations = translations.size () ;

	// Convert the scene graph into a csr.
	scene->make_csr () ; // Returns the Compressed Sparse Row representation for the graph.
	int *hOffset = scene->get_h_offset () ;  
	int *hCsr = scene->get_h_csr () ;
	int *hOpacity = scene->get_opacity () ; // hOpacity[vertexNumber] contains opacity of vertex vertexNumber.
	int **hMesh = scene->get_mesh_csr () ; // hMesh[vertexNumber] contains the mesh attached to vertex vertexNumber.
	int *hGlobalCoordinatesX = scene->getGlobalCoordinatesX () ; // hGlobalCoordinatesX[vertexNumber] contains the X coordinate of the vertex vertexNumber.
	int *hGlobalCoordinatesY = scene->getGlobalCoordinatesY () ; // hGlobalCoordinatesY[vertexNumber] contains the Y coordinate of the vertex vertexNumber.
	int *hFrameSizeX = scene->getFrameSizeX () ; // hFrameSizeX[vertexNumber] contains the vertical size of the mesh attached to vertex vertexNumber.
	int *hFrameSizeY = scene->getFrameSizeY () ; // hFrameSizeY[vertexNumber] contains the horizontal size of the mesh attached to vertex vertexNumber.
	

	auto start = std::chrono::high_resolution_clock::now () ;


	// Code begins here.
	// Do not change anything above this comment.


	//DGB
	// printf("Hello World\n");
	// printf("%d Nodes and %d Edges\n", V, E);
	// for(int i=0; i<=V; i++) printf("%d ",hOffset[i]); printf("\n");
	// for(int i=0; i<hOffset[V]; i++) printf("%d ",hCsr[i]); printf("\n");

	//Evaluate net Translations on each Node
	int *x_trans, *y_trans; 
	x_trans = (int*) malloc(sizeof(int) * V);
	y_trans = (int*) malloc(sizeof(int) * V);
	memset (x_trans, 0, V * sizeof(int));
	memset (y_trans, 0, V * sizeof(int));
	for(int t=0; t<numTranslations; ++t){
		int curN = translations[t][0];
		int curC = translations[t][1];
		int curA = translations[t][2];
		if(curC==0) x_trans[curN]-=curA; 
		if(curC==1) x_trans[curN]+=curA; 
		if(curC==2) y_trans[curN]-=curA; 
		if(curC==3) y_trans[curN]+=curA; 
	}// Try later for parallelism
	// for(int i=0; i<V; ++i) printf("%d ",x_trans[i]); printf("\n");
	// for(int i=0; i<V; ++i) printf("%d ",y_trans[i]); printf("\n");
	
	//Now transfer these net translations to a GPU Array
	int *x_trans_gpu, *y_trans_gpu;
	cudaMalloc(&x_trans_gpu, V * sizeof(int));
	cudaMemcpy(x_trans_gpu, x_trans, V * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc(&y_trans_gpu, V * sizeof(int));
	cudaMemcpy(y_trans_gpu, y_trans, V * sizeof(int), cudaMemcpyHostToDevice);

	//Transfer data to GPU
	int *hOffset_gpu, *hCsr_gpu, *hGlobalCoordinatesX_gpu, *hGlobalCoordinatesY_gpu;
	// bool *vis_gpu;

	cudaMalloc(&hOffset_gpu, (V+1) * sizeof(int));
	cudaMemcpy(hOffset_gpu, hOffset, (V+1) * sizeof(int), cudaMemcpyHostToDevice);
	
	cudaMalloc(&hCsr_gpu, hOffset[V] * sizeof(int));
	cudaMemcpy(hCsr_gpu, hCsr, hOffset[V] * sizeof(int), cudaMemcpyHostToDevice);
	
	cudaMalloc(&hGlobalCoordinatesX_gpu, V * sizeof(int));
	cudaMemcpy(hGlobalCoordinatesX_gpu, hGlobalCoordinatesX, V * sizeof(int), cudaMemcpyHostToDevice);
	
	cudaMalloc(&hGlobalCoordinatesY_gpu, V * sizeof(int));
	cudaMemcpy(hGlobalCoordinatesY_gpu, hGlobalCoordinatesY, V * sizeof(int), cudaMemcpyHostToDevice);

	// cudaMalloc(&vis_gpu, V * sizeof(bool));
	// cudaMemset(vis_gpu,false,sizeof(bool) * V);

	//Implement Tree Traversal
	bool *to_visit;
	cudaMalloc(&to_visit, sizeof(bool) * V);
	cudaMemset(to_visit, false, sizeof(bool) * V);
	cudaDeviceSynchronize();
	cudaMemset(to_visit, true, sizeof(bool));
	cudaDeviceSynchronize();

	bool *to_continue_gpu;
	cudaMalloc(&to_continue_gpu, sizeof(bool));
	bool *to_continue = (bool*) malloc(sizeof(bool)) ;

	dim3 block(1024, 1, 1);
	dim3 grid((V+1023)/1024, 1, 1);

	// printf("hi before launching kernel\n");

	while(true){
	// for(int p=0; p<2; ++p){
		cudaMemset(to_continue_gpu, false, sizeof(bool));
		cudaDeviceSynchronize();
		tree_bfs_levelwise<<<grid,block>>>(V, to_visit, x_trans_gpu, y_trans_gpu, hOffset_gpu, hCsr_gpu, hGlobalCoordinatesX_gpu, hGlobalCoordinatesY_gpu, to_continue_gpu);
		cudaDeviceSynchronize();
		cudaMemcpy(to_continue, to_continue_gpu, sizeof(bool), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		// printf(" -> %d \n",to_continue[0]);
		if(to_continue[0]==false) break;
		cudaDeviceSynchronize();
	}

	cudaDeviceSynchronize();

	cudaFree(hOffset_gpu);
	free(hOffset);
	cudaFree(hCsr_gpu);
	free(hCsr);
	cudaFree(x_trans_gpu);
	free(x_trans);
	cudaFree(y_trans_gpu);
	free(y_trans);

	// dim3 block(1024, 1, 1); // Use only 32 of these
	// dim3 grid((V+31)/32, 1, 1);

	// tree_traversal<<<grid,block>>>(V, x_trans_gpu, y_trans_gpu, hOffset_gpu, hCsr_gpu, hGlobalCoordinatesX_gpu, hGlobalCoordinatesY_gpu, vis_gpu);


	//Printig for DBG
	// cudaMemcpy(x_trans, x_trans_gpu, V * sizeof(int), cudaMemcpyDeviceToHost);
	// cudaMemcpy(y_trans, y_trans_gpu, V * sizeof(int), cudaMemcpyDeviceToHost);

	// for(int i=0; i<V; i++) printf("%d ", x_trans[i]); printf("\n");
	// for(int i=0; i<V; i++) printf("%d ", y_trans[i]); printf("\n");

	//Opacity Order
	// std::pair<int,int> * pr_list = new std::pair<int,int> [V];
	// for(int i=0; i<V; ++i) pr_list[i]={hOpacity[i],i};
	// sort(pr_list,pr_list+V);
	
	// for(int i=0; i<min(10,V); i++){
	// 	printf("%d %d\n",pr_list[i].first,pr_list[i].second);
	// }
	// int counter0=0;
	// for(int i=1; i<V; i++){
	// 	if(pr_list[i].first==pr_list[i-1].first) counter0++;
	// }
	// printf("counter0 -> %d\n",counter0);
	// printf("st %d ed %d\n",pr_list[0].first,pr_list[V-1].first);

	//Updating the Frames
	// int *hFinalPng_gpu, 
	// cudaMalloc(&hFinalPng_gpu, sizeof(int) * frameSizeX * frameSizeY);
	// cudaMemset(hFinalPng_gpu, 0, sizeof(int) * frameSizeX * frameSizeY);
	
	int *hMesh_gpu;
	
	int *hFinalPng_gpu2;
	cudaMalloc(&hFinalPng_gpu2, sizeof(int) * frameSizeX * frameSizeY);
	cudaMemset(hFinalPng_gpu2, 0, sizeof(int) * frameSizeX * frameSizeY);
	cudaDeviceSynchronize();

	int *cur_opcty_gpu;
	cudaMalloc(&cur_opcty_gpu, sizeof(int) * frameSizeX * frameSizeY);
	cudaMemset(cur_opcty_gpu, 0, sizeof(int) * frameSizeX * frameSizeY);
	cudaDeviceSynchronize();

	for(int i=0; i<V; ++i){
		// int curNode = pr_list[i].second;
		// cudaMalloc(&hMesh_gpu, sizeof(int) * hFrameSizeX[curNode] * hFrameSizeY[curNode]);
		// cudaMemcpy(hMesh_gpu, hMesh[curNode], sizeof(int) * hFrameSizeX[curNode] * hFrameSizeY[curNode], cudaMemcpyHostToDevice);
		// upd_scene<<<hFrameSizeX[curNode],hFrameSizeY[curNode]>>>(curNode, hFinalPng_gpu, hMesh_gpu, hGlobalCoordinatesX_gpu, hGlobalCoordinatesY_gpu, frameSizeX, frameSizeY);
		// cudaFree(hMesh_gpu);
		cudaMalloc(&hMesh_gpu, sizeof(int) * hFrameSizeX[i] * hFrameSizeY[i]);
		cudaMemcpy(hMesh_gpu, hMesh[i], sizeof(int) * hFrameSizeX[i] * hFrameSizeY[i], cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		upd_scene2<<<hFrameSizeX[i],hFrameSizeY[i]>>>(i, hFinalPng_gpu2, hMesh_gpu, hGlobalCoordinatesX_gpu, hGlobalCoordinatesY_gpu, frameSizeX, frameSizeY, cur_opcty_gpu, hOpacity[i]);
		cudaDeviceSynchronize();
		cudaFree(hMesh_gpu);
	}

	cudaMemcpy(hFinalPng,hFinalPng_gpu2, sizeof (int) * frameSizeX * frameSizeY, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	// comparator<<<(frameSizeX*frameSizeY+1023)/1024,1024>>>(hFinalPng_gpu,hFinalPng_gpu2,frameSizeX,frameSizeY);
	// printframe<<<1,1>>>(hFinalPng_gpu,frameSizeX,frameSizeY);
	// printframe<<<1,1>>>(hFinalPng_gpu2,frameSizeX,frameSizeY);


	cudaFree(hGlobalCoordinatesX_gpu);
	free(hGlobalCoordinatesX);
	cudaFree(hGlobalCoordinatesY_gpu);
	free(hGlobalCoordinatesY);
	// cudaFree(hFinalPng_gpu);
	cudaFree(hFinalPng_gpu2);
	

	// Do not change anything below this comment.
	// Code ends here.

	auto end  = std::chrono::high_resolution_clock::now () ;

	std::chrono::duration<double, std::micro> timeTaken = end-start;

	printf ("execution time : %f\n", timeTaken) ;
	// Write output matrix to file.
	const char *outputFileName = argv[2] ;
	writeFile (outputFileName, hFinalPng, frameSizeX, frameSizeY) ;	

}