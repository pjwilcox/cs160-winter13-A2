#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <assert.h>
#include <math.h>
#include "particle.h"
#include "common.h"
#include "Plotting.h"
#ifdef  _OPENMP
#include "omp.h"
#endif
using namespace std;

extern double dt;
int cmdLine( int argc, char **argv, int&n, int& nt, int& chunk, int& nsteps, int& nplot, int& sd, bool& imbal, char* savename);
void RepNorms(double uMax,double vMax,double uL2,double vL2);
void SimulateParticles(int nsteps, particle_t *particles, int n, int nt, int chunk, int nplot, bool imbal, double &uMax, double &vMax, double &uL2, double &vL2, Plotter *plotter, FILE *fsave );
int main( int argc, char **argv )
{    
// Command line arguments
    int nt; 	// number of threads
    int n;	// # of particles
    int nsteps;	// # of timesteps
    int nplot;  // Plotting frequency
    int sd;	// Random seed
    char *savename;	// Name of file to save output
    bool imbal;	// Use an imbalanced initial particle distribition
    int chunk;  // Chunk size for dynamic scheduling
    		// When chunk=0, we use static BLOCK scheduling

    // Get command line arguments
    int OK =  cmdLine( argc, argv, n, nt, chunk, nsteps, nplot, sd, imbal, savename);

    // If there was a parsing error, exit
    if (!OK){
        exit(-1);
    }
    cout << endl << "dt: " << dt << endl;
    
    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;

    particle_t *particles = new particle_t[ n ];
    assert(particles);
    set_size( n );
    init_particles( n, particles, sd, imbal );
    double uMax, vMax, uL2, vL2;
    Plotter *plotter = NULL;
    if (nplot){
        plotter = new Plotter();
        assert(plotter);
        VelNorms(particles,n,uMax, vMax, uL2, vL2);
        plotter->updatePlot(particles,n,0,uMax,vMax,uL2,vL2);
    }
    cout << "# particles : " << n << endl;
    if (chunk == -1)
       cout << "Using static BLOCK decomposition" << endl;
    else
       cout << "Using DYNAMIC scheduling with a chunk size of " << chunk << endl;
    if (imbal)
       cout << " Irregular particle distribution" << endl;
    cout << "Nsteps: " << nsteps << endl;
#ifdef  _OPENMP
   printf("Compiled with openmp enabled\n");
   omp_set_num_threads(nt);
#pragma omp parallel 
  {
    int tid = omp_get_thread_num();
    if (tid == 0) {
	int nthreads = omp_get_num_threads();
	cout << "Number of openMP threads:  " << nthreads << endl;
    }
  }
#ifdef DYN
    cout << "Compiled with dynamic scheduling.";
#ifdef CHUNK
    cout << "  Chunksize = " << CHUNK << endl;
#else
    cout << endl;
#endif
#endif
#endif
    
    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    SimulateParticles(nsteps,particles,n,nt,chunk, nplot, imbal, uMax, vMax, uL2, vL2, plotter, fsave );
    simulation_time = read_timer( ) - simulation_time;
   
    cout << endl;
    cout <<  "n = " << n << ", nsteps = " << nsteps << endl;
    VelNorms(particles,n,uMax,vMax,uL2,vL2);
    RepNorms(uMax,vMax,uL2,vL2);
    cout <<  "Running time = " << simulation_time << " sec.\n";

    if( fsave )
        fclose( fsave );
    
    if (nplot)
        delete plotter;
    delete [ ] particles;
    
    return 0;
}
