// 
// Various help functions that used to process particles
//
// Don't modify any code in this file
//
//
//
#include <cstdlib>
#include <stdio.h>
#include <math.h>
#include "particle.h"
#include "common.h"
#include <iostream>
using namespace std;

double size;

//
//  Tuned constants
//  Don't modify these values
//
double dt;


//
//  keep density constant
//
void set_size( int n )
{
    size = sqrt( density * n );
    cout << "Simulation box size: " <<  size << endl;
}

void imbal_particles(  particle_t *particles, int n){
    for( int i = 0; i < n/4; i++ ) {
	particles[i].ax = 0;
	particles[i].ay = 0;
	particles[i].vx = 0;
	particles[i].vy = 0;
    }
}
//
//  Initialize the particle positions and velocities
//
void init_particles( int n, particle_t *p, int seed, bool imbal )
{
    long int sd;
    if (!seed)
        sd = time(NULL);
    else
        sd = seed;

    srand48( sd );
    cout << "Random seed is " << sd << endl;
        
    int sx = (int)ceil(sqrt((double)n));
    int sy = (n+sx-1)/sx;
    
    int *shuffle = new int[n];
    for( int i = 0; i < n; i++ )
        shuffle[i] = i;
    
    for( int i = 0; i < n; i++ ) 
    {
        //
        //  make sure particles are not spatially sorted
        //
        int j = lrand48()%(n-i);
        int k = shuffle[j];
        shuffle[j] = shuffle[n-i-1];
        
        //
        //  distribute particles evenly to ensure proper spacing
        //
	double x = size*(1.+(k%sx))/(1+sx);
	double y = size*(1.+(k/sx))/(1+sy);
        p[i].x = x;
        p[i].y = y;

        //
        // Assign random velocities within a bound
        //
	p[i].vx = drand48()*2-1;
	p[i].vy = drand48()*2-1;
    }
	// If we asked for an imbalanced distribution,
        // zero the velocities for the first 1/4 of the particles
    if (imbal)
	imbal_particles( p, n);
    delete [] shuffle;
}

// Output max velocity and L2 norm of x and y components of velocity

void VelNorms(particle_t *particles, int n,
              double& uMax, double& vMax, double& uL2, double& vL2)
{
    uMax = vMax = -1e10;
    uL2 = vL2 = 0;
    for( int i = 0; i < n; i++ ) {
        double vx = fabs(particles[i].vx);
        double vy = fabs(particles[i].vy);
        uMax = fmax(uMax,vx);
        vMax = fmax(vMax,vy);
        uL2 += vx*vx;
        vL2 += vy*vy;
    }
    uL2 = sqrt (uL2/ (double) n);
    vL2 = sqrt (vL2/ (double) n);
}
//
//  I/O routines
//
void save( FILE *f, int n, particle_t *p )
{
    static bool first = true;
    if( first )
    {
        fprintf( f, "%d %g\n", n, size );
        first = false;
    }
    for( int i = 0; i < n; i++ )
        fprintf( f, "%g %g\n", p[i].x, p[i].y );
}

ostream & operator<<(ostream &os, const particle_t& p) {
    os << "Particle(x=" << p.x << ", y=" << p.y << ", vx=" << p.vx << ", vy=" << p.vy << ", ax=" << p.ax << ", ay=" << p.ay << ")";
    return os;
}

void list_particles(const particle_t* p, int n){
    for (int i=0; i<n; i++)
        cout << i << ": " << p[i] << endl;
    cout << endl;
}
void RepNorms(double uMax,double vMax,double uL2,double vL2){
    cout <<  "(u,vMax) = ";
    cout.precision(8);
    cout << scientific;
//    cout << fixed;
    cout << "( " << uMax << ", " << vMax << " )" << endl;
    cout <<  "(u,vL2 ) = ";
    cout << "( " << uL2 << ", " << vL2 << " )" << endl;
}
