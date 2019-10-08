# CSNN Implementation
This is the original implementation of the paper “CSNNs: Unsupervised, Backpropagation-Free Convolutional Neural Networks for Representation Learning” (TODO: insert link and reference when reviewed) in TensorFlow r1.13 and Keras delivered with this version.
## Usage
### 1. Configuration
There are two .json configuration files: ```controllerConfig.json``` and ```experimetSchedule.json``` to configure a run. The default settings execute all experiments presented in the paper on 4 GPUs and 28 CPU cores (executing all experiments takes a long time).
Cifar10 and Cifar100 will be automaticly downloaded. [Tiny ImageNet](https://tiny-imagenet.herokuapp.com/) and [ImageNet](https://www.kaggle.com/c/imagenet-object-localization-challenge/data) must be manually downloaded and placed in the ```data``` folder.
#### 1.1. Setup Configuration: ```controllerConfig.json```
 Here you can configure the number of GPUs and CPU cores to use for execution:
```json
{
    "hardware": {
        "numGPUs": 4,
        "numCPUCores": 28
    }
}
```
#### 1.2. Choose the Experiments to run.
Here you can choose the experiments to execute (by inserting the name of the IExperiment class) and set the .json configuration file(s) for these experiments:
```json
{
    "mode": "sequential",
    "experimentsToRun": ["CsnnOfmExperiment", "CsnnPerformancesExperiment", ...],
    "experimentConfigsToRun": {
        "CsnnOfmExperiment": ["csnnOfmExperiment.json"],
        "CsnnPerformancesExperiment": ["csnnMethodInfluences1Experiment.json", "csnnMethodInfluences2Experiment.json", "csnnPerformanceExperiment.json"]
            ...
    }
}
```
Experiments can be found in ```Experiment_Component//Experiments``` and their configurations can be found in ```Experiment_Component//ExperimentConfigs```
### 2. Run the Experiments.
Simply run the ```main.py```. 
## Experiment Configuration
For each experiment .json configuration files must be defined to configure the CSNN models and their training parameters. Furthermore, the number of runs (e.g., for cross-validation), the datasets and the classifiers to train, evaluate and test on may be defined here (depending on the experiment). You can use the original configuration files of the experiments from the paper as template to configure your own experiments. These original configuration files will be used by default, but you can register your own configuration files in the ```experimetSchedule.json``` file.
### Correspondence between the Paper and the Experiments
1. Experiment ```CsnnOfmExperiment``` with the configuration ```ofmExperiment.json``` corresponds to Figure 4 in the paper 
2. Experiment ```CsnnPerformancesExperiment``` with the configuration ```csnnMethodInfluences1Experiment.json``` corresponds to the left side of Table 1 in the paper 
3. Experiment ```CsnnPerformancesExperiment``` with the configuration ```csnnMethodInfluences2Experiment.json``` corresponds to the right side of Table 1 in the paper 
4. Experiment ```CsnnPerformancesExperiment``` with the configuration ```csnnPerformanceExperiment.json``` corresponds to the models, which are presented in Table 2 for the genaralization test and used for the Figures in the paper 
5. Experiment ```CsnnPerformancesExperiment``` with the configuration ```csnnPerformanceAugExperiment.json``` corresponds to the augmented classifier accuracies presented in Table 1 and 2 in the paper
6. Experiment ```CsnnPerformancesExperiment``` with the configuration ```csnnFewShotExperiment.json``` corresponds to the models presented in the lower part of Figure 5 and to the 10-fold cross-validation accuracies of D-CSNN-B1 and S2-CSNN-B32 in Table 1
7. Experiment ```CsnnGeneralizationExperiment``` with the configuration ```csnnGeneralizationExperiment.json``` corresponds to the generalization test in Table 2 of the paper
8. Experiment ```CsnnReconstructionExperiment``` with the configuration ```csnnReconstuctionExperiment.json``` corresponds to the reconstructions presented in Figure 6 of the paper
9. Experiment ```CsnnVisualizationExperiment``` with the configuration ```CsnnVisualizationExperiment.json``` corresponds to the average representations per class presented in Figure 6 of the paper

## The CSNN 
This work combines Convolutional Neural Networks (CNNs), clustering via Self-Organizing Maps (SOMs) and Hebbian Learning to propose the building blocks of Convolutional Self-Organizing Neural Networks (CSNNs), which learn representations in an unsupervised and Backpropagation-free manner. Our approach replaces the learning of traditional convolutional layers from CNNs with the competitive learning procedure of SOMs and simultaneously learns local masks between those layers with separate Hebbian-like learning rules to overcome the problem of disentangling factors of variation when filters are learned through clustering. We investigate the learned representation by designing two simple models with our building blocks, achieving comparable performance to many methods which use Backpropagation, while we reach comparable performance on Cifar10 and give baseline performances on Cifar100, Tiny ImageNet and a small subset of ImageNet for Backpropagation-free methods.

Please read our paper for more details.

## License
[MIT](https://choosealicense.com/licenses/mit/)
