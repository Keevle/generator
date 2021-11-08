import yaml
import csv
import glob
import random
import imageio
import numpy as np
from pathlib import Path

yaml_file_names = glob.glob('./build/yaml-data/*.yaml')
rows_to_write = []


class Generator:

    def __init__(self, config, config_normal):
        self.config = config
        self.config_normal = config_normal
        # Initialize matrix of generated trait combinations
        self.trait_matrix = np.zeros((0, len(self.config_normal)))
        # Create build directory
        self.build = Path(self.config['build'])
        self.build.mkdir(exist_ok=True)
        # create large image director
        self.ImagesNormal = Path(self.config['Images-normal'])
        self.ImagesNormal.mkdir(exist_ok=True)
        # create large image director
        self.ImagesLarge = Path(self.config['Images-large'])
        self.ImagesLarge.mkdir(exist_ok=True)
       # yaml files
        self.YamlData = Path(self.config['yaml-files'])
        self.YamlData.mkdir(exist_ok=True)

    def generate_normal(self, img_id, traits, rec=0):
        # Raise exception if no unique image could be create after 200 tries
        # Add some more traits or lower the number of generated images
        if rec > 200:
            raise Exception('Could not create another unique image')
        desc = {}

        # initialize image and iterate through layers
        ids = np.zeros((len(self.config_normal),))
        image = np.zeros(
            (self.config['image_height'], self.config['image_width'], 3))
        for idx, (layer_id, layer) in enumerate(self.config_normal.items()):
            image, ids[idx], name, value, trait = self.add_layer(
                image, layer_id, layer, traits)
            desc[layer_id] = {'value': value}
            desc["name"] = img_id

        # make sure that each two doggos differ in at least some traits
        diff = self.trait_matrix - ids
        diff = np.abs(diff)
        diff = np.clip(diff, 0, 1)
        diff = np.sum(diff, axis=-1)
        if self.trait_matrix.shape[0] > 0 and np.any(diff < self.config['distance']):
            return self.generate_normal(img_id, traits, rec + 1)

        # add this id to matrix of all previous ids
        ids = np.expand_dims(ids, axis=0)
        self.trait_matrix = np.concatenate((self.trait_matrix, ids))

        # Save description
        name = f'{self.config["prefix"]}{img_id:02d}'
        self.save_desc(name, desc)
        # Save image in original and upscaled resolution
        self.save_img(name, image)
        image = self.scale_img(image, self.config['image_scale'])
        self.save_large(f'{name}', image)

    def generate_legend(self, legend):
        descl = {}
        print(legend)
        descl["background"] = "Legendary"
        descl["clothes"] = "Legendary"
        descl["ear"] = "Legendary"
        descl["fur"] = "Legendary"
        descl["hat"] = "Legendary"
        descl["mouth"] = "Legendary"
        descl["Legendary"] = legend['image_id']
        # Load image of legendary doggo
        image = imageio.imread(legend['src'])[:, :, :3]
        # Save unscaled version in build folder
        name = f'{self.config["prefix"]}{legend["image_id"]:02d}'
        self.save_img(name, image)
        # Scale and save high res version
        image = self.scale_img(image, self.config['image_scale'])
        self.save_large(f'{name}', image)
        # Save description
        self.save_desc(f'{name}', descl)

    def generate_collage(self):
        # Get image size and initialize matrix
        c_width = self.config['collage_width']
        c_height = self.config['collage_height']
        i_width = self.config['image_width']
        i_height = self.config['image_height']
        collage = np.zeros((c_height * i_height, c_width * i_width, 3))

        # Load single images and add them to the collage
        for i in range(c_height):
            for j in range(c_width):
                # Load single image (ignore alpha channel)
                image_id = i * c_width + j + 1
                path = self.ImagesNormal / \
                    f'{self.config["prefix"]}{image_id:02d}.png'
                image = imageio.imread(path)[:, :, :3]
                # Add image to collage
                x1, x2 = j * i_width, (j + 1) * i_width
                y1, y2 = i * i_height, (i + 1) * i_height
                collage[y1:y2, x1:x2] = image

        # scale and save collage
        collage = self.scale_img(collage, self.config['collage_scale'])
        self.save_img('collage', collage)

    def bulk_yaml_to_csv(self):
        rows_to_write.append(
            ["Name", "Background", "Clothes", "Ear", "Fur", "Hat", "Mouth", "Class"])
        for idx, each_yaml_file in enumerate(yaml_file_names):
            print("Processing file ", idx+1, "of",
                  len(yaml_file_names), "file name:", each_yaml_file)
            with open(each_yaml_file) as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
                try:
                    data["Legendary"]
                except KeyError:
                    if self.config['hasProperties']:
                        rows_to_write.append(
                            [data["name"],
                             data["background"]["value"],
                             data["clothes"]["value"],
                             data["ear"]["value"],
                             data["fur"]["value"],
                             data["hat"]["value"],
                             data["mouth"]["value"]])
                    else:
                        print("Has no elements")
                        rows_to_write.append([data["name"]])
                else:
                    if self.config['hasProperties']:
                        rows_to_write.append(
                            [f'{0}{data["Legendary"]}',
                             data["background"],
                             data["clothes"],
                             data["ear"],
                             data["fur"],
                             data["hat"],
                             data["mouth"]
                             ])
                    else:
                        print("Has no elements")
                        rows_to_write.append([f'{0}{data["Legendary"]}'])

        with open('output_csv_file.csv', 'w', newline='') as out:
            csv_writer = csv.writer(out, delimiter=',', quotechar=' ')
            csv_writer.writerows(rows_to_write)
            print("Output file output_csv_file.csv created")

    def add_layer(self, image, key_c, value_c, traits):
        # get random trait or use given one
        cat_name, values = self.get_category(value_c)
        idx, trait_id = self.get_random(values)
        if key_c in traits:
            trait_id = traits.get(key_c)

        # get trait data and read image
        val_name, src, cat_name = self.get_value(values, trait_id, cat_name)
        img = imageio.imread(src)
        alpha = img[:, :, 3:] / 255
        layer = img[:, :, :3]
        # Stack new layer onto current image
        image = alpha * layer + (1 - alpha) * image
        return image, idx, cat_name, val_name, trait_id

    def save_img(self, name, image):
        image_path = self.ImagesNormal / (name + '.png')
        alpha = np.full((image.shape[0], image.shape[1], 1), 255)
        image = np.concatenate((image, alpha), axis=-1)
        imageio.imwrite(image_path, image)

    def save_large(self, name, image):
        image_path = self.ImagesLarge / (name + '.png')
        alpha = np.full((image.shape[0], image.shape[1], 1), 255)
        image = np.concatenate((image, alpha), axis=-1)
        imageio.imwrite(image_path, image)

    def save_desc(self, name, desc):
        desc_path = self.YamlData / (name + '.yaml')
        with open(desc_path, 'w') as desc_file:
            desc_file.write(yaml.dump(desc))

    @staticmethod
    def scale_img(image, scale):
        # increases the image represented as numpy array by the scale
        width, height = image.shape[0], image.shape[1]
        scaled_img = np.zeros((width * scale, height * scale, 3))
        map_width = np.array([p for p in range(width) for _ in range(scale)])
        map_height = np.array([p for p in range(height) for _ in range(scale)])
        for x in range(width * scale):
            for y in range(height * scale):
                scaled_img[x, y] = image[map_width[x], map_height[y]]
        return scaled_img

    @staticmethod
    def get_category(category):
        name = category.get('category', None)
        values = category['values']
        return name, values

    @staticmethod
    def get_random(category):
        rnd, accu = random.random(), 0
        for idx, (name, val) in enumerate(category.items()):
            accu += val['prob']
            if rnd <= accu:
                return idx, name
        raise Exception('Probabilities does not sum up to 1')

    @staticmethod
    def get_value(category, value, cat_name=None):
        name = category[value].get('name', None)
        src = category[value].get('src', None)
        cat_name = category[value].get('category', cat_name)
        return name, src, cat_name


def add_mistakes(idx):
    # Fixed the script over time but some mistakes found their way onto the blockchain.
    # This will reproduce these mistakes while the rest of the script is fixed now.
    # Try to get along without this function for you own collection ;D
    props = {}
    if idx in [56, 87, 621]:
        props['eyes'] = 'sun_pink_eyes'
    return props


def main():
    # Load config with collection properties
    with open('config.yaml') as config:
        config = yaml.load(config, Loader=yaml.FullLoader)
    # Load config for the normal doggos
    with open('config_normal.yaml') as config_normal:
        config_normal = yaml.load(config_normal, Loader=yaml.FullLoader)
    # Load config for the legendary doggos
    with open('config_legend.yaml') as config_legend:
        config_legend = yaml.load(config_legend, Loader=yaml.FullLoader)

    # Set a seed to make the generation reproducible
    random.seed(config['seed'])
    gen = Generator(config, config_normal)

    # Generate normal doggos (leave out legendary positions)
    for i in range(1, config['number'] + 1):
        mistakes = {}
        # Comment this in to recreate the original cyberdoggos
        # mistakes = add_mistakes(i)
        gen.generate_normal(i, mistakes)

    # Generate legendary doggos
    for item in config_legend:
        gen.generate_legend(item)

    # Create single picture of all doggos
    gen.generate_collage()

    gen.bulk_yaml_to_csv()


if __name__ == "__main__":
    main()
