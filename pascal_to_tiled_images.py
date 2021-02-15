import glob
import os
import xmltodict
import pprint
from pascal_voc_writer import Writer

from PIL import Image
from image_slicer.image_slicer import slice_overlapped

image_divisor = 5   # output image sizes compared to the original Note actual output number will depend on overlap
image_overlap = 0.5 # overlap percentage
start_dir = "/media/kurt/win1/labelimg/shade-tank3-left-01"

# Printing results
pp = pprint.PrettyPrinter(indent=4)


# Look for XML files and parses then as if they were Pascal VOC Files
def main():
    # Finds all XML files on data/ and append to list
    pascal_voc_contents = []
    os.chdir(start_dir)

    print("Found {} files in data directory!".format(
        str(len(glob.glob("*.xml")))))
    for file in glob.glob("*.xml"):
        f_handle = open(file, 'r')
        print("Parsing file '{}'...".format(file))
        pascal_voc_contents.append(xmltodict.parse(f_handle.read()))

    # Process each file individually
    for index in pascal_voc_contents:
        image_file = index['annotation']['filename']
        # If there's a corresponding file in the folder,
        # process the images and save to output folder
        if os.path.isfile(image_file):
            extractDataset(index['annotation'])
        else:
            print("Image file '{}' not found, skipping file...".format(image_file))


# Extract image samples and save to output dir
def extractDataset(dataset):
    print("Found {} objects on image '{}'...".format(
        len(dataset['object']), dataset['filename']))

    # Open image and get ready to process
    img = Image.open(dataset['filename'])

    # Create output directory
    save_dir = dataset['filename'].split('.')[0]
    try:
        os.mkdir(save_dir)
    except:
        pass
    # Image name preamble
    sample_preamble = save_dir + "/" + dataset['filename'].split('.')[0] + "_"

    # generate the tiles from the source image
    im_w, im_h = img.size
    window_size = int(im_w / image_divisor), int(im_h / image_divisor)
    
    image_tiles = slice_overlapped(
    dataset['filename'],
    window_size = window_size,
    window_overlap = image_overlap,
    directory = save_dir,
    )

    # print(image_tiles)

    #process create the xml for each tiled image and include the relevent objects from the source image 
    for tile in image_tiles:

        image_bb = [tile.coords[0], tile.coords[1], tile.coords[0] + window_size[0], tile.coords[1] + window_size[1]]

        # print("basename: ", tile.basename)
        # print("position: ", tile.position)
        # print("co-ords: ", tile.coords)
        # print("Image BB: ", image_bb)

        # Run through each object, check if it is in the sliced image and append after recalulating coords for object bb

        # Writer(path, width, height)
        writer = Writer((os.path.join(start_dir, save_dir) + "/" + tile.basename + ".png"), window_size[0], window_size[1])
        has_object = False
        for item in dataset['object']:
            object_bb_dict = dict([(a, int(b)) for (a, b) in item['bndbox'].items()])
            object_bb = list(object_bb_dict.values())
            
            if rectContains(image_bb, object_bb):
                # item["bndbox"] = collections.OrderedDict([
                #     ('xmin', str(object_bb[0] - image_bb[0])), 
                #     ('ymin', str(object_bb[1] - image_bb[1])), 
                #     ('xmax', str(object_bb[2] - image_bb[0])), 
                #     ('ymax', str(object_bb[3] - image_bb[1]))])
                # ::addObject(name, xmin, ymin, xmax, ymax)
                has_object = True
                writer.addObject(item["name"],
                    object_bb[0] - image_bb[0], 
                    object_bb[1] - image_bb[1], 
                    object_bb[2] - image_bb[0], 
                    object_bb[3] - image_bb[1])
               
        if has_object:
            writer.save((os.path.join(start_dir, save_dir) + "/" +  tile.basename + ".xml"))


def rectContains(image_bb,object_bb):
    start_object_bb = image_bb[0] < object_bb[0] < image_bb[2] and image_bb[1] < object_bb[1] < image_bb[3]
    end_object_bb = image_bb[0] < object_bb[2] < image_bb[2] and image_bb[1] < object_bb[3] < image_bb[3]
    return start_object_bb and end_object_bb


if __name__ == '__main__':
    print("\n------------------------------------")
    print("----- PascalVOC-to-Images v0.1 -----")
    print("Created by Giovanni Cimolin da Silva")
    print("------------------------------------\n")
    main()
