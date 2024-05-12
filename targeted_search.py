from PIL import Image
from skimage.metrics import structural_similarity as ssim
from utils.Evaluator import Evaluator
from utils.img_read_save import image_read_cv2
import os
import shutil


def calculate_ssim(image_path1, image_path2):
    image1 = Image.open(image_path1).convert('L')  # turn to gray image
    image2 = Image.open(image_path2).convert('L')  # turn to gray image
    return ssim(image1, image2)

def find_max_function_images(vis_folder, ir_folder, target_folders, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vis_images = sorted(os.listdir(vis_folder))

    for image_name in vis_images:
        max_metric = -1
        max_metric_image_path = None

        for target_folder in target_folders:
            target_image_path = os.path.join(target_folder, image_name)
            vis_image_path = os.path.join(vis_folder, image_name)
            ir_image_path = os.path.join(ir_folder, image_name)

            ir = image_read_cv2(ir_image_path, 'GRAY')
            vi = image_read_cv2(vis_image_path, 'GRAY')
            fi = image_read_cv2(target_image_path, 'GRAY')
            ssim_value = Evaluator.SSIM(fi, ir, vi)
            vif_value = Evaluator.VIFF(fi, ir, vi)
            Qabf = Evaluator.Qabf(fi, ir, vi)

            metric_value = 2 * ssim_value + vif_value + 3 * Qabf

            if metric_value > max_metric:
                max_metric = metric_value
                max_metric_image_path = target_image_path

        if max_metric_image_path:
            output_image_path = os.path.join(output_folder, image_name)
            shutil.copy(max_metric_image_path, output_image_path)
            print(f"Selected {max_metric_image_path} {image_name} and saved to {output_image_path}")


if __name__ == "__main__":
    #----------------Infrared and Visibile Dataset   VIF---------#
    vis_folder = " "  # reference folder  vis image
    ir_folder = " "  # reference folder  ir image
    target_folders = [
        "./dataset/fusion_knowledge_prior/Knowledge_DDFM",  # fusion knowledge folded 1
        "./dataset/fusion_knowledge_prior/Knowledge_MetaFusion",  # fusion knowledge folded 2
        "./dataset/fusion_knowledge_prior/Knowledge_U2Fusion",  # fusion knowledge folded 3
        "./dataset/fusion_knowledge_prior/Knowledge_TarDAL",  # fusion knowledge folded 4
        #"./dataset/fusion_knowledge_prior/Knowledge_MoE-Fusion",  # fusion knowledge folded 5
        #"./dataset/fusion_knowledge_prior/Knowledge_",  # ...  your fusion methods
    ]

    output_folder = "./dataset/your_dataset/train/Fusion_K"  # save the targeted search images (ready to distillation)
    find_max_function_images(vis_folder, ir_folder, target_folders, output_folder)