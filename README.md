# frame-pred-mask

To set up on HPC, create a new conda environment with Python 3.8 in Singluarity with the requirements.txt file. Unsquash the dataset in your scratch folder before you begin.
Then, change the *.slurm files in this repo based on your created environment.
Change the config.json to point to folders you have access to.
All models will get saved in models/
All logs will get saved in logs/

Train the video prediction model with the framepred.slurm file. Train the segmentation model with the segpred.slurm file. 
To run the infer file, change the 'fp_model_name' key in config.json to point to your Lightning checkpoint. Then, run the gpuinfer.slurm file. Based on the logs, find the segmentation model which gave the best validation IOU on the whole task. Write this model's name in the 'seg_model_name' key in config.json file. And for the final test test, run the following on a GPU node.
```python infer.py --test``` 


By default, config.json is used for hyperparams and metadata. For your own custom config, you can use the following command:

```python next_frame_prediction.py --cfg "{PATH}/config_custom.json"```