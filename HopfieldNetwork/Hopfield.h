#pragma once
#include <Eigen/Dense>
#include <iostream>

class Hopfield
{

public:
	Hopfield(int x);


	Eigen::MatrixXd ConvertVectorToMatrix(Eigen::VectorXd inputMark);
	void GenerateWeights(Eigen::VectorXd MarkAsVector);
	void PrintGeneratedInputMark();
	void PrintGeneratedWeights();

	Eigen::MatrixXd Activation(Eigen::VectorXd testVector);

	Eigen::MatrixXd inputMatrix;
	Eigen::MatrixXd weightsMatrix;

private:
	int inputMatrixSize;
	bool isInitialized;
};

