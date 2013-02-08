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
#include <thread>
#include <atomic>
#include <mutex>
#include <atomic>
using namespace std;

/*
    Variables for BARRIER
    arrival = UNLOCKED
    departure =  LOCKED
    **Lock departure immediately in SimulateParticles()
*/
/*
mutex arrival;
mutex departure;
int count;
*/

//for thread function
struct SimulateArgs{
    //calculated in original callback
    int chunkSize;
    //from original callback
    int nsteps;
    particle_t *particles;
    int n;
    int nt;
    int chunk;
    int nplot;
    bool imbal;
    double uMax;
    double vMax;
    double uL2;
    double vL2;
    Plotter *plotter;
    FILE *fsave;
};

class Barrier {
private:
    int maxThreads;
    std::mutex in;
    std::mutex out;
    std::atomic<int> numThreads;
public:
    Barrier(int mt) : maxThreads (mt) {
        out.lock();
        numThreads.store(0);
    }
    void barrier() {
        in.lock();
        numThreads.fetch_add(1);
        if(numThreads.load() == maxThreads) {
            out.unlock();
        }
        else {
            in.unlock();
        }
        out.lock();
        numThreads.fetch_sub(1);
        if(numThreads.load() == 0) {
            in.unlock();
        }
        else {
            out.unlock();
        }
    }
};

//flags
bool block = false;

//for dynamic partitioning
int numOfParticles;
atomic<int> counted;
atomic<int> particlesTraversed;

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

    int base;
    while(particlesTraversed.load() < numOfParticles){
        //set up for next chunk of particles thread will compute
        base = particlesTraversed.fetch_add(n);

        for( int i = base; i < base + n; ++i ) {
            particles[i].ax = particles[i].ay = 0;
            if ((particles[i].vx != 0) || (particles[i].vy != 0)){
                for (int j = 0; j < n; ++j ){
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
        //BLOCK partitioning
        //return after one chunk (aka a block) is computed
        if(block)
            return;
    }
    return;
}

//
//  integrate the ODE, advancing the positions of the particles
//
void move_particles( particle_t* particles, int n)
{
    int base;
    while(particlesTraversed.load() < numOfParticles){
        //set up for next chunk of particles thread will compute
        base = particlesTraversed.fetch_add(n);

        for( int i = base; i < base + n; ++i) {
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
        //BLOCK partitioning
        //return after one chunk (aka a block) is computed
        if(block)
            return;
    }
    return;
}

//thread function
void _SimulateParticles(SimulateArgs *args, int thread_id, Barrier * barrier){
    for( int step = 0; step < args->nsteps; ++step ) {
        //set particlesTraversed to 0 fr every loop

        //
        //  compute forces
        //
        apply_forces(args->particles,args->chunkSize);
        barrier->barrier();
        // If we asked for an imbalanced distribution

        
        particlesTraversed.store(0);
        barrier->barrier();
        //
        //  move particles
        //
        move_particles(args->particles,args->chunkSize);
        barrier->barrier();
        if(thread_id == 0) {
            VelNorms(args->particles,args->n,args->uMax,args->vMax,args->uL2,args->vL2);
        }
        //Thread 0 does this work ALONE
        if(thread_id == 0) {
            if (args->nplot && ((step % args->nplot ) == 0)){
                // Computes the absolute maximum velocity
                VelNorms(args->particles,args->n,args->uMax,args->vMax,args->uL2,args->vL2);
                args->plotter->updatePlot(args->particles,args->n,step,args->uMax,args->vMax,args->uL2,args->vL2);
            }
              VelNorms(args->particles,args->n,args->uMax,args->vMax,args->uL2,args->vL2);   
        }
        barrier->barrier();
        //
        //  save if necessary
        //
        if( args->fsave && (step%SAVEFREQ) == 0 ){
            if(thread_id == 0) 
                save( args->fsave, args->n, args->particles);
            barrier->barrier();
        }
        particlesTraversed.store(0);
    }    
    return;
}

// This is the main driver routine that runs the simulation
void SimulateParticles(int nsteps, particle_t *particles, 
                        int n, int nt, int chunk, int nplot, 
                        bool imbal, double &uMax, double &vMax, 
                        double &uL2, double &vL2, Plotter *plotter, 
                        FILE *fsave )
{   
    SimulateArgs sargs;
    thread t[nt];
    Barrier * barrier = new Barrier(nt);
    numOfParticles = n;
    particlesTraversed.store(0);

    //intilaize struct feilds
    sargs.nsteps = nsteps;
    sargs.particles = particles;
    sargs.n = n;
    sargs.nt = nt;
    sargs.chunk = chunk;
    sargs.nplot = nplot;
    sargs.imbal = imbal;
    sargs.plotter = plotter;
    sargs.fsave = fsave;
    

    //BLOCK
    if(chunk == -1){
        block = true;
        sargs.chunkSize = n / nt;
    }
    //DYNAMIC
    else{
        block = false;
        sargs.chunkSize = chunk;
    }
    for(int i = 0; i < nt; ++i){
        t[i] = thread(_SimulateParticles, &sargs, i, ref(barrier));
    }

    for(int i = 0; i < nt; ++i){
        t[i].join();
    }

    uMax = sargs.uMax;
    vMax = sargs.vMax;
    uL2 = sargs.uL2;
    vL2 = sargs.vL2;

    return;


}

/*
void Barrier(int numThreads){
    arrival.lock( );                // atomically count the
    count++;                        // waiting threads
    if (count < numThreads) arrival.unlock( );
    else departure.unlock( );       // last processor enables all to go

    departure.lock( );
    count--;                        // atomically decrement
    if (count > 0) departure.unlock( );
    else arrival.unlock( );         // last processor resets state
}
*/
