// All the parallelism  arises in code within this file
//
// The following two functions update the state of the particles,
// and will perform the computation in parallel across multiple threads
// The openMP code specifies how you'll parallelize the code
// including options (set on the make command line) to
// enable dynamic scheduling and chunking
// Note that dynamic scheduling and chunking are specified at compile time
// in OpenMP, but at run time in your multithreaded code
// Keep this in mind when building the code and don't use the compile time
// flags dyn= chunk= unless you are using OpenMP
// You should not enable OpenMP and C++11 threads in the same code
// as the behavior is unpredictable
//
//    apply_forces( )
//    move_particles( )
//
// Your performance optimizations may change the order in
// which arithmetic gets done, affecting the results
// This is acceptable, as discussed in class, so long as the differences
// are to within roundoff errors 
// To assess correctness, examine the values of vMax and vL2
// reported at the simulation's end
///
//
#include <cstdlib>
#include <stdio.h>
#include <math.h>
#include "particle.h"
#include "common.h"
#include "Plotting.h"
#include <iostream>
#ifdef  _OPENMP
#include "omp.h"
#endif
using namespace std;

extern double size;

extern double dt;

void imbal_particles(particle_t *particles, int n);
//
//  compute force between two particles
//  You should not modify the numerical computations
//  other than to optimize them
//  Be careful in optimizing, as some optimizations
//  can subtly affect the computed answers
//
void apply_forces( particle_t* particles, int n){

#ifdef DYN
#if CHUNK
#pragma omp parallel for shared(particles,n) schedule(dynamic,CHUNK)
#else
#pragma omp parallel for shared(particles,n) schedule(dynamic)
#endif
#else
#pragma omp parallel for shared(particles,n)
#endif
    for( int i = 0; i < n; i++ ) {
        particles[i].ax = particles[i].ay = 0;
	if ((particles[i].vx != 0) || (particles[i].vx != 0)){
	    for (int j = 0; j < n; j++ ){
		if (i==j)
		    continue;
		double dx = particles[j].x - particles[i].x;
		double dy = particles[j].y - particles[i].y;
		double r2 = dx * dx + dy * dy;
		if( r2 > cutoff*cutoff )
		    continue;
		r2 = fmax( r2, min_r*min_r );
		double r = sqrt( r2 );

		//
		//  very simple short-range repulsive force
		//
		double coef = ( 1 - cutoff / r ) / r2 / mass;
		particles[i].ax += coef * dx;
		particles[i].ay += coef * dy;

	    }
        }
    }
}

//
//  integrate the ODE, advancing the positions of the particles
//
void move_particles( particle_t* particles, int n)
{
    for( int i = 0; i < n; i++ ) {
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
        particles[i].vx += particles[i].ax * dt;
        particles[i].vy += particles[i].ay * dt;
        particles[i].x  += particles[i].vx * dt;
        particles[i].y  += particles[i].vy * dt;

    //
    //  bounce off the walls
    //
        while( particles[i].x < 0 || particles[i].x > size ) {
            particles[i].x  = particles[i].x < 0 ? -particles[i].x : 2*size-particles[i].x;
            particles[i].vx = -particles[i].vx;
        }
        while( particles[i].y < 0 || particles[i].y > size ) {
            particles[i].y  = particles[i].y < 0 ? -particles[i].y : 2*size-particles[i].y;
            particles[i].vy = -particles[i].vy;
        }
    }
}

// This is the main driver routine that runs the simulation
void SimulateParticles(int nsteps, particle_t *particles, int n, int nt, int chunk, int nplot, bool imbal, double &uMax, double &vMax, double &uL2, double &vL2, Plotter *plotter, FILE *fsave ){
    for( int step = 0; step < nsteps; step++ ) {
    //
    //  compute forces
    //
	apply_forces(particles,n);

	// If we asked for an imbalanced distribution
	if (imbal)
	    imbal_particles(particles,n);
//     Debugging output
//      list_particles(particles,n);
    
    //
    //  move particles
    //
	move_particles(particles,n);


	if (nplot && ((step % nplot ) == 0)){

	// Computes the absolute maximum velocity 
	    VelNorms(particles,n,uMax,vMax,uL2,vL2);
	    plotter->updatePlot(particles,n,step,uMax,vMax,uL2,vL2);
	}

	VelNorms(particles,n,uMax,vMax,uL2,vL2);
    //
    //  save if necessary
    //
	if( fsave && (step%SAVEFREQ) == 0 )
	    save( fsave, n, particles );
    }
}

