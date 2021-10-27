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

	//Eigen::MatrixXd test(MarkSize, MarkSize);
	//test = Eigen::MatrixXd::Zero(MarkSize, MarkSize);


	Hopfield Network(MarkSize);

	Eigen::MatrixXd Mark(MarkSize, MarkSize);
	Mark = Network.ConvertVectorToMatrix(inputMark);

	std::cout << Mark << std::endl;

	Network.GenerateWeights(inputMark);
	Network.PrintGeneratedWeights();
	Eigen::VectorXd testingMark(MarkSize * MarkSize);
	testingMark <<
		1, 1, 1, 1, 1, 1, 1, 1,
		1, 0, 0, 0, 0, 0, 0, 1,
		1, 0, 0, 0, 0, 1, 0, 1,
		1, 0, 0, 1, 0, 0, 0, 1,
		1, 0, 0, 0, 1, 0, 1, 1,
		1, 0, 0, 0, 0, 0, 0, 1,
		1, 0, 0, 0, 0, 0, 0, 1,
		1, 1, 1, 1, 1, 1, 1, 1;
	std::cout << Network.ConvertVectorToMatrix((Network.weightsMatrix * testingMark));
	//std::cout << (inputMark * Network.weightsMatrix);

	return 0;
}