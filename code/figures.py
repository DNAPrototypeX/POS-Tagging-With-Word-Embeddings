import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pypalettes import load_cmap

def plot_model_performance_ieee(root_dir="resources/taggers", output_file="norwegian_training_summary.pdf"):

    width = 3.5
    height = 5.0 
    
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 8,       
        "axes.labelsize": 9,   
        "axes.titlesize": 9,    
        "legend.fontsize": 7,    
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.6,   
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.2    
    })

    loss_files = list(Path(root_dir).glob("*-norwegian-upos-adam/loss.tsv"))
    
    if not loss_files:
        print(f"No loss.tsv files found in '{root_dir}'.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width, height), sharex=True)
    
    cmap = load_cmap("Fun")
    colors = [cmap(i / max(1, len(loss_files) - 1)) for i in range(len(loss_files))]

    for i, file_path in enumerate(loss_files):
        model_name = file_path.parent.name
        color = colors[i % len(colors)]
        
        try:
            df = pd.read_csv(file_path, sep='\t')
            
            # Loss plot (Train dashed, Dev solid)
            ax1.plot(df['EPOCH'], df['TRAIN_LOSS'], label=f'{model_name} (Train)', 
                     color=color, linestyle='--', alpha=0.5, linewidth=0.8)
            ax1.plot(df['EPOCH'], df['DEV_LOSS'], label=f'{model_name} (Dev)', 
                     color=color)
            
            # F1 plot
            ax2.plot(df['EPOCH'], df['DEV_ACCURACY'], label=f'{model_name}', 
                         color=color)
                
        except Exception as e:
            print(f"Error processing {model_name}: {e}")

    ax1.set_title('Training vs Development Loss')
    ax1.set_ylabel('Loss Value')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')

    ax2.set_title('Development Accuracy')
    ax2.set_xlabel('Training Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black')

    plt.tight_layout()
    
    plt.savefig(output_file, dpi=600, bbox_inches='tight', pad_inches=0.02)
    print(f"IEEE formatted plot saved to {output_file}")

if __name__ == "__main__":
    plot_model_performance_ieee()
