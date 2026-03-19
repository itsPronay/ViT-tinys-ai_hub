import argparse
import pandas as pd
import torch
import utils
import ai_hub
import colab
import utils.wandb
import os
from model.vitfs import *

parser = argparse.ArgumentParser(description='Run the benchmark script for in colab gpu and cpu')

parser.add_argument('--runs', type=int, default=10, help='Number of runs to perform for benchmarking')
# parser.add_argument('--run_warmup', action='store_true', help='Whether to run warmup runs before benchmarking')
parser.add_argument('--warmup_runs', type=int, default=10, help='Number of warmup runs to perform before benchmarking')
parser.add_argument("--image_sizes", nargs="+", type=int, default=[224, 448], help="List of image sizes to run the benchmark on")
parser.add_argument("--ai_hub_device", default='Samsung Galaxy S25 (Family)', help="Physical ai_hub device to run the benchmark on")
parser.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online', help='WandB logging mode')
parser.add_argument("--device", choices=['cpu', 'cuda', 'ai_hub', 'all'], default='all', help="Device to run the benchmark on")
parser.add_argument("--models", nargs="+", default=[
    "vit_tiny_patch16_224",
    "mobilevitv2_100",
    "mobilevitv2_125",
    "tiny_vit_5m_224",
    "vitfs_tiny_patch16_gap_reg4_dinov2_bn_init",
    "vitfs_tiny_patch16_gap_reg4_dinov2_init"
    ],
    help="List of model names"
)

args = parser.parse_args()

def main():

    all_results = []
    
    for model_name in args.models:
        for image_size in args.image_sizes:

            print(f"Running benchmark for Model: {model_name} and Image Size: {image_size}")
            model = utils.get_model(model_name, image_size)

            if args.device == 'all':
                ai_hub_results = process_ai_hub(model, model_name, image_size)
                all_results.append(ai_hub_results)

                colab_results = process_colab(model, "cpu", model_name, image_size)
                all_results.append(colab_results)

                if torch.cuda.is_available():
                    colab_results = process_colab(model, "cuda", model_name, image_size)
                    all_results.append(colab_results)
                else:
                    print("GPU not available, running on CPU only.")

            elif args.device == 'ai_hub':
                ai_hub_results = process_ai_hub(model, model_name, image_size)
                all_results.append(ai_hub_results)

            elif args.device == 'cpu':
                colab_results = process_colab(model, "cpu", model_name, image_size)
                all_results.append(colab_results)

            elif args.device == 'cuda':
                colab_results = process_colab(model, "cuda", model_name, image_size)
                all_results.append(colab_results)

            else:
                raise ValueError(f"Invalid device option: {args.device}")
            
    df = pd.DataFrame(all_results)

    # Reorder columns: model_name, image_size, device, then the rest
    preferred_order = [col for col in ['model_name', 'image_size', 'device'] if col in df.columns]
    other_cols = [col for col in df.columns if col not in preferred_order]
    df = df[preferred_order + other_cols]
    print(df)
    # Ensure results directory exists if needed
    # os.makedirs("results", exist_ok=True)
    df.to_csv("benchmark_results.csv", index=False)


def process_ai_hub(model, model_name, image_size):

    name = f"Model: {model_name}, Image Size: {image_size}, Device: {args.ai_hub_device}"
    utils.wandb.setup(
        name = name,
        config = vars(args),
        mode=args.wandb_mode
    )

    metrics = {
        'model_name' : model_name,
        'image_size' : "1, 3, {0}, {0}".format(image_size),
        'device' : args.ai_hub_device,
    }
    
    result = ai_hub.run(model, (1, 3, image_size, image_size), args.ai_hub_device)
    metrics.update(result)

    utils.wandb.log(metrics, mode=args.wandb_mode)

    return metrics


def process_colab(model, device, model_name, image_size):

    name = f"Model: {model_name}, Image Size: {image_size}, Device: {device}"
    utils.wandb.setup(
        name = name,
        config = vars(args),
        mode=args.wandb_mode
    )

    metrics = colab.run(model, device, image_size, runs=args.runs, warmup_runs=args.warmup_runs)
    
    metrics.update(
        {
            'model_name' : model_name,
            'image_size' : "1, 3, {0}, {0}".format(image_size),
            'device' : device,
        }
    )

    utils.wandb.log(metrics, mode=args.wandb_mode)
    
    return metrics


if __name__ == "__main__":
    main()
    


     
     
     

                

                
        



