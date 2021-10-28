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
					weightsMatrix(i, j) = (2 * MarkAsVector(i) - 1) * (2 * MarkAsVector(j) - 1);
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
	Eigen::MatrixXd Holder(inputMatrixSize, inputMatrixSize);
	Eigen::MatrixXd TestVectorM(inputMatrixSize, inputMatrixSize);
	TestVectorM = ConvertVectorToMatrix(testVector);
	Holder = ConvertVectorToMatrix(weightsMatrix * testVector);

	for (int i = 0; i < inputMatrixSize; i++)
	{
		for (int j = 0; j < inputMatrixSize; j++)
		{
			if(Holder(i, j) > 0)
			{
				Holder(i, j) = 1;
			}
			else if(Holder(i, j) < 0)
			{
				Holder(i, j) = 0;
			}
			else {
				Holder(i, j) = TestVectorM(i, j);
			}
		}
	}
	
	return Holder;
}




