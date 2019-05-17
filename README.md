## PyTorch Binary Neural Network with Data Export

Source code used in my research concerning Logical Formalization of Binary Neural Neutworks. It is the reproduction of methods used in the resarch: [Verifying Properties of Binarized Deep Neural Networks
](https://arxiv.org/abs/1709.06662)

### Features
 - Pytorch BNN
 - BNN Data Export
	 - Weights
	 - Biases
	 - Running Mean
	 - Standard Deviation

### Installation
 1. Clone repository
 2. Install [Anaconda](https://www.anaconda.com)
 3. [Import](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) enviroment in Anaconda from other/environment.yml
 4. Install [CUDA](https://developer.nvidia.com/cuda-10.0-download-archive) *(for GPU Support)*
 5. Run program

### Exportation
Exportation happens automatically after Test phase.
The code of exportation is at the end of the main py. 
Comment out unnecessary lines.

### Exported File Structure
**mean.txt:**
[µ] BatchNorm.RunningMean

**std.txt:**
[σ] BatchNorm.RunningVariance.SquareRoot

**weights.txt:**
[A] InnerLinearization.Weight
[α] BatchNorm.Weight - BatchNorm.Scale
[A] OutputLinearization.Weight

**biases.txt:**
[b] InnerLinearization.Bias
[γ]	BatchNorm.Bias - BatchNorm.Shift
[b] OutputLinearization.Bias


#### *All of this was created under the supervision of Dr. Kovásznai Gergely.*
