#include <stdlib.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <cmath>
#include "opennn_4.0/opennn/opennn.h"
#include "opennn_4.0/opennn/data_set.h"

using namespace OpenNN;
using namespace std;

int main(void)
{
	 

   std :: cout<<"its made\n";
    
   DataSet data_set;
   data_set("iris_flowers.csv",',', true);
   data_set.set_columns_uses({"Input", "Input", "Input", "Input", "Target"});

	return 0;
}