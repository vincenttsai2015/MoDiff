# MoDiff - Graph Generation with Motif-Aware Diffusion Model
Code for the paper MoDiff-Graph Generation with Motif-Aware Diffusion Model (SIGKDD 2025).

## Dependencies
Please use the following command to install the requirements:
```bash
pip install -r requirements.txt
```

## Running Experiments
**Data Preparation**

The subgraphs for training are preprocessed with motif detection and stored in different files in this code.
We provide StackOverflow as an example. The 1h scale can be used directly, while the 5h and 1k scales need to be decompressed.
More data could be found here. We are also working on merging them in one graph, and will update that later.

**Train model**

```bash
CUDA_VISIBLE_DEVICES=${gpu_ids} python main.py --type train_comp --scale ${scale} --config_folder ${folder_name} --config_prefix ${config_prefix} 
```

for example,
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --type train_comp -scale 1h --config_folder Stack --config_prefix SO
python main.py --type train_comp  #Default settings for a quick start
```

**Evaluation model**

To generate graphs using the trained score models, run the following command.
```bash
CUDA_VISIBLE_DEVICES=${gpu_ids} python main.py --type eval_comp --scale ${scale} --config_folder ${folder_name} --config_prefix ${config_prefix} --ckpt_name ${ckpt_name}
```

for example,
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --type eval_comp -scale 1h --config_folder Stack --config_prefix SO --ckpt_name Sep30_SO1h_comp
python main.py --type eval_comp  #Default settings for a quick start
```

We provide trained models on StackOverflow for evaluation.
Currently, the density of the generated subgraph can be adjusted manually or inferred from the training data. 
We are also developing a simple module to automatically predict the target density based on historical density patterns.


## Citation

If you found the provided code with our paper useful in your work, we kindly request that you cite our work.
