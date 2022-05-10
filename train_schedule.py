from train import train
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", type=int, default=0, help="GPU device index")
    args = parser.parse_args()

    #default values
    data_dir="/mnt/datasets" # Path to data root dir
    gpu_device=0 # GPU device index
    alpha=1e4 # Weight of content loss
    beta=1e5 # Weight of style loss
    gamma=1e-5 # Weight of style loss
    lambda_f=1e5 # Weight of feature temporal loss
    lambda_o=2e5 # Weight of output temporal loss
    epochs=1 # Number of epochs
    lr=1e-3 # Learning rate
    frn=False # Output model file path
    style_image_name="5_horses" # Use Filter Response Normalization and TLU
    style_image_path=f"./styles/{style_image_name}.jpg" # Path to style image

    if args.list==0:
        alpha=1e2 # Weight of content loss
        beta=1e5 # Weight of style loss
        frn=True # Output model file path
        output_path = f"./models/{style_image_name}#alpha{alpha:.1E}#beta{beta:.1E}#gamma{gamma:.1E}#lf{lambda_f:.1E}#lo{lambda_o:.1E}#ep{epochs}#lr{lr:.1E}#frn{frn}.pth"
        train(data_dir, gpu_device , alpha , beta , gamma, lambda_f , lambda_o , epochs , lr, output_path, frn, style_image_path)
    
    # if args.list==1:
        # alpha=1e2 # Weight of content loss
        # beta=1e5 # Weight of style loss
        # gamma=1e-4 # Weight of style loss
        # output_path = f"./models/{style_image_name}-{alpha}-{beta}-{gamma}-{lambda_f}-{lambda_o}-{epochs}-{lr}-{frn}.pth"
        # train(data_dir, gpu_device , alpha , beta , gamma, lambda_f , lambda_o , epochs , lr, output_path, frn, style_image_path)
