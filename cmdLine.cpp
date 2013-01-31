#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <assert.h>
#include "math.h"
#include "common.h"

using namespace std;

extern double dt;
//
// Parse command line arguments
//
//
int cmdLine( int argc, char **argv, int&n, int& nt, int& chunk, int& nsteps, int& nplot, int& sd, bool& imbal, char* savename)
{    
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        cout << "Options:\n";
        cout << "-h to see this help\n";
        cout << "-n <int> to set the number of particles [default = 1024]\n";
        cout << "-t <int> to set the number of timesteps i [1000]\n";
        cout << "-d to scale dt by sqrt(n)" << endl;
        cout << "-b to get an imbalanced initial particle distribution" << endl;
	// This command line flag doesn't have any effect on the
	// provided code; you will use it to control the granularity
	// of dynamic scheduling in the threaded code you write
        cout << "-c <int> chunk size" << endl;
        cout << "-p <int> plot interval [off]\n";
        cout << "-s <int> to specify random seed [use time of day]\n";
        cout << "-o <filename> to specify the output file name\n";
	cout << "-nt <number of threads> \n";
        return 0;
    }
    

    dt     =  0.0005;
    // Get command line arguments
    // Final argument is the default value
    n = read_int( argc, argv, "-n", 1024 );
    if( find_option( argc, argv, "-d" ) >= 0 ){
        dt /= sqrt((double) (n/1000));
        cout << "Scaling dt by sqrt(n)\n";
    }
    nsteps = read_int( argc, argv, "-t", 1000 );
    // By default, we use BLOCK partitioning (chunk= -1)
    chunk = read_int( argc, argv, "-c", -1 );
    nplot = read_int( argc, argv, "-p", 0 );
    sd = read_int( argc, argv, "-s", 0 );
    imbal = false;
    if( find_option( argc, argv, "-b" ) >= 0 ){
	imbal = true;
    }

    savename = read_string( argc, argv, "-o", NULL );
    

    nt = read_int( argc, argv, "-nt", 1 );
    
    return 1;
    
}
