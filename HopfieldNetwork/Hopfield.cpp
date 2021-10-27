#include "Hopfield.h"

Hopfield::Hopfield(int x)
{
	inputMatrixSize = x;
	inputMatrix = Eigen::MatrixXd::Zero(inputMatrixSize, inputMatrixSize);
	weightsMatrix = Eigen::MatrixXd::Zero(inputMatrixSize * inputMatrixSize, inputMatrixSize * inputMatrixSize);
	isInitialized = true;
}
Eigen::MatrixXd Hopfield::ConvertVectorToMatrix(Eigen::VectorXd inputMark)
{
	if (inputMatrixSize * inputMatrixSize == inputMark.size() && isInitialized)
	{
		for (int i = 0; i < inputMatrixSize; i++)
		{
			for (int j = 0; j < inputMatrixSize; j++)
			{
				inputMatrix(i, j) = inputMark((i * inputMatrixSize) + j);
			}
		}
	}
	return inputMatrix;
}

void Hopfield::PrintGeneratedInputMark()
{
	if (isInitialized)
	{
		std::cout << "Input mark as matrix: " << std::endl;
		std::cout << inputMatrix << std::endl;
	}
}

void Hopfield::GenerateWeights(Eigen::VectorXd MarkAsVector)
{
	Eigen::MatrixXd test(8, 8);
	test = Eigen::MatrixXd::Zero(8, 8);
	if (isInitialized)
	{
		for (int i = 0; i < inputMatrixSize * inputMatrixSize; i++)
		{
			for (int j = 0; j < inputMatrixSize * inputMatrixSize; j++)
			{
				if (i == j)
				{
					weightsMatrix(i, j) = 0;
				}
				else
				{
					weightsMatrix(i, j) = (2 * MarkAsVector(i) - 1) + (2 * MarkAsVector(j) - 1);
				}
			}
		}
	}
}

void Hopfield::PrintGeneratedWeights()
{
	if (isInitialized)
	{
		std::cout << weightsMatrix << std::endl;
	}
}

Eigen::MatrixXd Hopfield::Activation(Eigen::VectorXd testVector)
{
	if (inputMatrixSize * inputMatrixSize == testVector.size() && isInitialized)
	{
		for (int i = 0; i < inputMatrixSize; i++)
		{
			for (int j = 0; j < inputMatrixSize; j++)
			{
				if (testVector((i * inputMatrixSize) + j) >= 0)
				{
					testVector((i * inputMatrixSize) + j) = 1;
				}


			}
		}
	}
	return inputMatrix;
}




