Instagram Influencer Dataset

1. influencers.txt
    - This file contains a list of influencers with their Instagram username, category, the number of followers, followees, and posts. There are eight major influencer categories as follows: Beauty, Family, Fashion, Fitness, Food, Interior, Pet, and Travel. Influencers who were not classified into one of the eight categories are labeled as 'Other'.

2. Post metadata (JSON files)
    - This directory has zip files. You can download all zip files and extract it to get post metadata files which are in JSON format. Each file name starts with a username followed by a specific post ID. The post JSON files have various information such as captions, likes, comments, timestamps, sponsorship, usertags, etc.

3. Post images (JPG files)
    - This directory contains the image files that correspond to the post meta data. All image files are resized for your convenience.

4. sample_images.zip
    - Due to the huge size of image files (189GB in total), I also uploaded sample image files. The sample_images.zip contains a set of image files from one family influencer from the dataset. 

5. JSON-Image_files_mapping.txt
    - This file contains a list of posts where each line shows a file name of post metadata (JSON file) and its corresponding image files. Since a post can have more than one image file, the names of JSON and image files are not always the same. Use this txt file to map the names of JSON files and their image files.
