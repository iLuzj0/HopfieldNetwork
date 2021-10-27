#include "Hopfield.h"

#define MarkSize 8

int main()
{
	//Eigen::MatrixXd inputMark(MarkSize, MarkSize);
	Eigen::VectorXd inputMark(MarkSize * MarkSize);
	inputMark <<
		1, 1, 1, 1, 1, 1, 1, 1,
		1, 0, 0, 0, 0, 0, 0, 1,
		1, 0, 0, 0, 0, 0, 0, 1,
		1, 0, 0, 1, 1, 0, 0, 1,
		1, 0, 0, 1, 1, 0, 0, 1,
		1, 0, 0, 0, 0, 0, 0, 1,
		1, 0, 0, 0, 0, 0, 0, 1,
		1, 1, 1, 1, 1, 1, 1, 1;

	Hopfield Network(MarkSize);

	Eigen::MatrixXd Mark(MarkSize, MarkSize);
	Mark = Network.ConvertVectorToMatrix(inputMark);
	std::cout << "Input Mark: " << std::endl;
	std::cout << Mark << std::endl; 

	
	Network.GenerateWeights(inputMark);

	std::cout << "Weights: " << std::endl;
	Network.PrintGeneratedWeights();
	Eigen::VectorXd testingMark(MarkSize * MarkSize);
	Eigen::MatrixXd testingMarkM(MarkSize,MarkSize);
	//testingMark <<
	//	1, 1, 1, 1, 1, 1, 1, 1,
	//	1, 0, 0, 0, 0, 0, 0, 1,
	//	1, 0, 0, 0, 0, 1, 0, 1,
	//	1, 0, 0, 1, 0, 0, 0, 1,
	//	1, 0, 0, 0, 1, 0, 1, 1,
	//	1, 0, 0, 0, 0, 0, 0, 1,
	//	1, 0, 0, 0, 0, 0, 0, 1,
	//	1, 1, 1, 1, 1, 1, 1, 1;
	//testingMark = Eigen::VectorXd::Zero(MarkSize * MarkSize);
	testingMark <<
		0, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 1, 1, 1, 1, 1, 0,
		0, 1, 1, 1, 1, 1, 1, 0,
		0, 1, 1, 0, 0, 1, 1, 0,
		0, 1, 1, 0, 0, 1, 1, 0,
		0, 1, 1, 1, 1, 1, 1, 0,
		0, 1, 1, 1, 1, 1, 1, 0,
		0, 0, 0, 0, 0, 0, 0, 0;
	testingMarkM = Network.ConvertVectorToMatrix(testingMark);

	std::cout << "Testing Mark: " << std::endl;
	std::cout << testingMarkM << std::endl;
	
	std::cout << "Mark after activation: " << std::endl;
	std::cout << Network.Activation(testingMark);

	return 0;
}