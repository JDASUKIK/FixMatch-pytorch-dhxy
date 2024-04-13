import matplotlib.pyplot as plt
import sys
sys.path.append('..')
# from FixMatch-pytorch import denormalization

# def show_pics(pics):
#     if not isinstance(pics, list):
#         plt.imshow(denormalization(tensor=pics))
#         plt.show()
#     crop_num = len(pics[1])
#     fig = plt.figure()
#     plt.axis('off')
#     for i in range(3):
#         if i == 0:
#             plt.subplot(3, crop_num, 1)
#         for j in range(crop_num):
#             plt.subplot(3, crop_num, i*crop_num+j+1)
#             plt.imshow(denormalization(tensor=pics[i][j]))
#     plt.show()