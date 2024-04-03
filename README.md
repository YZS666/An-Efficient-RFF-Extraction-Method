# An-Efficient-RFF-Extraction-Method

<p align="center">
  <img src="https://github.com/YZS666/An-Efficient-RFF-Extraction-Method/blob/main/Visualization/AMAE-Based%20RFFE.jpg?raw=true" width="480">
</p>

This is a PyTorch/GPU implementation of the paper [An Efficient RFF Extraction Method Using Asymmetric Masked Auto-Encoder](https://ieeexplore.ieee.org/abstract/document/10460605). If using relevant content, please cite this paper:
```
@INPROCEEDINGS{10460605,
  author={Yao, Zhisheng and Fu, Xue and Wang, Shufei and Wang, Yu and Gui, Guan and Mao, Shiwen},
  booktitle={2023 28th Asia Pacific Conference on Communications (APCC)}, 
  title={An Efficient RFF Extraction Method Using Asymmetric Masked Auto-Encoder}, 
  year={2023},
  volume={},
  number={},
  pages={364-368},
  keywords={Convolutional codes;Wireless communication;Training;Convolution;Fingerprint recognition;Feature extraction;Transceivers;Radio frequency fingerprint (RFF);unsupervised learning;asymmetric masked auto-encoder (AMAE)},
  doi={10.1109/APCC60132.2023.10460605}}

```

* Attention, you need to manually create three folders and name them as 'model_weight' and 'test_result'.
* In addition, the dataset link used in this demo is https://ieee-dataport.org/open-access/lorarffidataset.

### Catalog

- [x] Training and Visualization code

### Training
* Start training by running the train_FS-AMAE.py file.
* After the training is completed, the Visualization.py file can be run to visualize the features of the trained model, and the trained model can be evaluated using unsupervised clustering indicators.
