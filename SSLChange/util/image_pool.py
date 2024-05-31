import random
import torch

# Image pool/buffer机制是为了存储之前生成的图像，重复给鉴别器进行加深印象，类似强化学习中的Experience Replay Buffer
class ImagePool():
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size  # 图像缓冲区大小，默认为50
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0   # 图像张数
            self.images = []    # 图像张量

    def query(self, images):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images

        # 存储返回图像的list
        return_images = []

        # 对于新一批batch中每一张生成图像image
        for image in images:
            image = torch.unsqueeze(image.data, 0)  # 扩展增加一个第0维度

            # 如果缓冲区中的图像数量小于pool_size (50)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1   # buffer图像+1
                self.images.append(image)   # 继续将新生成的图像存入images
                return_images.append(image) # 并将新生成的图像同时存入到return_images中

            # 若缓冲区中已存储大于50张图像
            else:
                # 返回一个(0, 1)的任意值作为概率p
                p = random.uniform(0, 1)

                # 50%的概率下，随机抽取一个已存储的id为[0, 49]之间的任意整数的图像，将选中图像和新图像进行调换，存储新图像，返回旧图像到鉴别器
                # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()    # 将选中位置的先前存储的图像复制一份给tmp
                    self.images[random_id] = image      # 将新图像插入到列表选中位置
                    return_images.append(tmp)   # 将选中的旧图像返回

                # 50%的概率下，直接将新生成的图像直接返回给鉴别器
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)

        # 将所有返回图像concat
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images
