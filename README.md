## PlantDid
Information Transfer Diffusion Model-Based Synthetic Plant Diseases for Precise Plant Disease Identification

## Environment
You can create a new Conda environment by running the following command:
```
    conda env create -f environment.yml 
```
In the environment, we still use other networks. If it does not work, please configure the environment of other networks first.
## PreTrained Model
The pre-trained PlantDid model is linked below, you can download it.
<li>PlantDid:https://drive.google.com/file/d/1VMu7AMv6SYp9HXWpjXmERAsmhh8aahFY/view?usp=drive_link
<li>And we use GroundingDINO, the download is:
https://drive.google.com/drive/folders/1z7Xxm-ZpTC0k4qyaCjyEJqh_q1rou1Ba?usp=drive_link

## Test

If you want to batch test, you can use `leaf_diseases/inference_bench_gd_auto.py`. For example,
```
    python leaf_diseases/inference_bench_gd_auto.py 
    --config configs/leaf-diseases_with_disease.yaml
    --test_path TestDataset 
    --outdir Result/inferenceTest 
    --ddim_steps 200 --ddim_eta 1.0 --scale 4 --seed 250 
    --ckpt models/last.ckpt 
    --max_size 250
```

