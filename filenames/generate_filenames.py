import os
from glob import glob


def gen_kitti_2015():
    data_dir = 'data/KITTI/kitti_2015/data_scene_flow'

    train_file = 'KITTI_2015_train.txt'
    val_file = 'KITTI_2015_val.txt'

    # Split the training set with 4:1 raito (160 for training, 40 for validation)
    with open(train_file, 'w') as train_f, open(val_file, 'w') as val_f:
        dir_name = 'image_2'
        left_dir = os.path.join(data_dir, 'training', dir_name)
        left_imgs = sorted(glob(left_dir + '/*_10.png'))

        print('Number of images: %d' % len(left_imgs))

        for left_img in left_imgs:
            right_img = left_img.replace(dir_name, 'image_3')
            disp_path = left_img.replace(dir_name, 'disp_occ_0')

            img_id = int(os.path.basename(left_img).split('_')[0])

            if img_id % 5 == 0:
                val_f.write(left_img.replace(data_dir + '/', '') + ' ')
                val_f.write(right_img.replace(data_dir + '/', '') + ' ')
                val_f.write(disp_path.replace(data_dir + '/', '') + '\n')
            else:
                train_f.write(left_img.replace(data_dir + '/', '') + ' ')
                train_f.write(right_img.replace(data_dir + '/', '') + ' ')
                train_f.write(disp_path.replace(data_dir + '/', '') + '\n')

def gen_instereo():
    data_dir = '/data/users/zz/data/InStereo2K/'

    train_file = 'instereo_train.txt'
    test_file = 'instereo_test.txt'
    with open(train_file,'w') as train_f, open(test_file,'w') as test_f:
        for i in os.listdir(data_dir):
            if i=='line_segment':
                continue
            for j in os.listdir(data_dir+i):
                for k in os.listdir(data_dir+i+'/'+j):
                    left_img = data_dir+i+'/'+j+'/'+k+'/left.png'
                    right_img = left_img.replace('left','right')
                    left_disp = left_img.replace('left','left_disp')
                    if i == 'train':
                        train_f.write(left_img+' ')
                        train_f.write(right_img+' ')
                        train_f.write(left_disp+'\n')
                    if i == 'test':
                        test_f.write(left_img+' ')
                        test_f.write(right_img+' ')
                        test_f.write(left_disp+'\n')

if __name__ == '__main__':
    gen_instereo()
