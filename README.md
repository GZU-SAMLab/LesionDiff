## Environment
You can create a new Conda environment by running the following command:
```
    conda env create -f environment.yml 
```
## Test
To sample one image from our model, you can use `leaf_diseases/inference.py`. For example, 
```
    python leaf_diseases/inference.py 
    --config configs/leaf-diseases_with_disease.yaml
    --image_path TestDataset/image/example_1.jpg
    --reference_path TestDataset/ref/example_1.jpg
    --outdir Result/inferenceTest 
    --ddim_steps 200 --ddim_eta 1.0 --scale 4 --seed 250 
    --ckpt models/last.ckpt 
    --mask_rcnn_pth output/model_final.pth --mask_rcnn_cfg output/config.pickle
```
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


