#include<stdlib.h>
#include<string>
#include<iostream>
#include "vector.h"
#include "data_set.h"
#include "matrix.h"
#include "vector.h"

int main(int argc, char const *argv[])
{
	std :: cout<<"its made\n";
	DataSet :: DataSet data_set("iris_flowers.csv",',',true);
	return 0;
}