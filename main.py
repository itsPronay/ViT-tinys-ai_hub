import utils 
import argparse
import timm
import qai_hub as hub
import wandb

parser = argparse.ArgumentParser('ViT-TinyS Benchmarking')

parser.add_argument('--model',type=str, choices= ["vit_tiny_patch16_224", "mobilevitv2_100", "mobilevitv2_125", "tiny_vit_5m_224"] ,default='vit_tiny_patch16_224')
parser.add_argument('--image_size', type=int, default=448)
parser.add_argument('--device', type=str, default='Samsung Galaxy S25 (Family)')
parser.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default='online', help='WandB logging mode')
parser.add_argument('--wandb_project', type=str, default='vit-tinys-results', help='WandB project name')

args = parser.parse_args()


def main():

    # these models do not support 448, so skipping it
    if args.image_size != 224 and (args.model == 'vit_tiny_patch16_224' or args.model=='tiny_vit_5m_224'):
        print('Image size and Model mismatch, skipping ')
        return
    
    model = timm.create_model(args.model, pretrained=False)
    model = model.to("cpu").eval()

    input_shape = (1, 3, args.image_size, args.image_size)
    print(f"Running benchmark for input shape: {input_shape} , Model: {args.model} on Device: {args.device}")

    device = hub.Device(args.device)

    if args.wandb_mode != 'disabled':
        wandb.init(
            project = args.wandb_project,
            name = f"Input shape: {input_shape}, Model: {args.model}",
            config = vars(args)
        )

    traced_model = utils.get_traced_model(input_shape, model)

    compile_job = utils.run_compile(traced_model, device, input_shape)

    profile_job = utils.run_profile(compile_job, device)
    profile_data = profile_job.download_profile()

    metrices = {
        'input_shape' : str(input_shape),
        'device' : str(args.device),
        **utils.extract_metrics_from_profile(profile_data),
    }

    if args.wandb_mode != 'disabled':
        wandb.log(metrices)

        wandb.log({"log_op_type_table": utils.log_op_type_table(profile_data)})
        wandb.log({"log_top15_table": utils.log_top15_table(profile_data)})

        wandb.finish()

if __name__ == '__main__':
    main()
