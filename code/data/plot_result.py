import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def combined():
    # Load the images
    image1 = Image.open("./1.png")
    image2 = Image.open("./2.png")
    image3 = Image.open("./3.png")

    # Determine the size of the combined image
    total_width = image1.width + image2.width + image3.width
    max_height = max(image1.height, image2.height, image3.height)

    # Create a new empty image with the size to fit all three images
    combined_image = Image.new('RGB', (total_width, max_height))

    # Paste the images next to each other
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (image1.width, 0))
    combined_image.paste(image3, (image1.width + image2.width, 0))

    # Save the combined image
    combined_image_path = "combined_image.png"
    combined_image.save(combined_image_path)


# Modified code with increased font sizes for the bar chart

def pilar():
    # Set the random seed for reproducibility
    np.random.seed(1)

    # Generate some data
    categories = ['Vanderbilt', 'CATS', 'OpenACC', 'Ohio', 'Waymo', 'Argoverse']
    algorithms = ['Acceleration', 'Speed', 'Space']
    data = np.random.randint(0, 150, size=(len(algorithms), len(categories)))
    data = np.array([[10, 11, 8, 10, 9, 8], [20, 22, 25, 22, 16, 17], [15, 17, 20, 18, 15, 16]])
    # [[10, 20, 15], [11, 22, 17], [8, 25, 20],
    # [10, 22, 18], [9, 16, 15], [8, 17, 16]]

    # Create a bar chart
    fig, ax = plt.subplots(figsize=(16, 8))

    # Set the positions and width for the bars
    positions = np.arange(len(categories))
    width = 0.25  # Increase width for clarity

    # Plot bars for each algorithm
    for i in range(len(algorithms)):
        ax.bar(positions - width + (i * width), data[i], width=width, label=algorithms[i])

    # Customize font sizes
    plt.rcParams.update({'font.size': 14})  # Update default rc settings for font size

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    # ax.set_ylabel('Scores', fontsize=16)
    # ax.set_title('Scores by category and algorithm', fontsize=18)
    ax.set_xticks(positions)
    ax.set_xticklabels(categories, fontsize=18)
    # ax.set_ylim(0, 160)
    ax.legend(fontsize=18)

    # Display the bar chart
    plt.savefig('./scores.png')


def scatter(input_path):
    df = pd.read_csv(input_path)

    for i in range(200, 400, 100):
        colors = [(42 / 255, 157 / 255, 140 / 255),
                  (233 / 255, 196 / 255, 107 / 255),
                  (230 / 255, 111 / 255, 81 / 255)]

        plt.figure(figsize=(15, 5))

        # for name, group in df.groupby('ID_FAV'):
        plt.scatter(df['Space_Gap'][i:i + 100], df['Acc_FAV'][i:i + 100], color=colors[0], s=100)
        # plt.title('Acceleration and Space Gap', fontsize=20)
        plt.xlabel('$g$', fontsize=40)
        plt.ylabel('$a^{\mathrm{f}}$', fontsize=40)
        plt.tick_params(axis='x', labelbottom=False)  # 不显示x轴刻度标签
        plt.tick_params(axis='y', labelleft=False)  # 不显示y轴刻度标签
        plt.tight_layout()
        plt.savefig(f'{i}_Space_Gap.png')  # _{name}
        plt.close()

        plt.figure(figsize=(15, 5))

        plt.scatter(df['Speed_FAV'][i:i + 100], df['Acc_FAV'][i:i + 100], color=colors[1], s=100)
        plt.xlabel('$v^{\mathrm{f}}$', fontsize=40)
        plt.ylabel('$a^{\mathrm{f}}$', fontsize=40)
        plt.tick_params(axis='x', labelbottom=False)  # 不显示x轴刻度标签
        plt.tick_params(axis='y', labelleft=False)  # 不显示y轴刻度标签
        plt.tight_layout()
        plt.savefig(f'{i}_Speed_FAV.png')  # _{name}
        plt.close()

        plt.figure(figsize=(15, 5))

        plt.scatter(df['Speed_Diff'][i:i + 100], df['Acc_FAV'][i:i + 100], color=colors[2], s=100)
        # plt.title('Acceleration and Speed Difference', fontsize=20)
        plt.xlabel('$\Delta v$', fontsize=40)
        plt.ylabel('$a^{\mathrm{f}}$', fontsize=40)
        plt.tick_params(axis='x', labelbottom=False)  # 不显示x轴刻度标签
        plt.tick_params(axis='y', labelleft=False)  # 不显示y轴刻度标签
        plt.tight_layout()
        plt.savefig(f'{i}_Speed_Diff.png')  # _{name}
        plt.close()


def read_date():
    original_data_path = './Dataset/Ohio/data/Advanced_Driver_Assistance_System__ADAS_-Equipped_Single-Vehicle_Data_for_Central_Ohio.csv'
    df = pd.read_csv(original_data_path)
    unique_dates = df['date'].drop_duplicates()
    print(unique_dates)

    print('-----')

    original_data_path = './Dataset/Ohio/data/Advanced_Driver_Assistance_System__ADAS_-Equipped_Two-Vehicle_Data_for_Central_Ohio.csv'
    df = pd.read_csv(original_data_path)
    unique_dates = df['date'].drop_duplicates()
    print(unique_dates)

# scatter('./Dataset/OpenACC/output/step1_ASta_merge.csv')
# pilar()
# read_date()