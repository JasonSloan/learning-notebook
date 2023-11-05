from count_numbers_of_each_channel import count_numbers_of_each_channel



if __name__ == '__main__':
    images_path = '/root/work/real-esrgan/train/datasets/landsea/raw_data/hr_sole_psnr18'
    images_counts = count_numbers_of_each_channel(images_path)
    print()