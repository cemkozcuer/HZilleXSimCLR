"""
This script generates an static HTML webpage where all photographs
of Heinrich Zille can be browsed side by side,
separated by known and unknown authorship.

The static HTML can be found "generated_pages" folder and can be opened
in any browser in order to browse the images.

Additionally, also static pages with only known and unknown get created.
"""

import jinja2
import pandas as pd

title = 'Heinrich Zille'

templates_folder = 'HZille_browser/templates'
output_path = 'HZille_browser/generated_pages'

outputs_templates = [
    ('HZille_browser.html', 'HZille_browser_template.html'),
    ('HZille_browser_known.html', 'HZille_browser_known_template.html'),
    ('HZille_browser_unknown.html', 'HZille_browser_unknown_template.html')
]

data_columns = ['id', 'title', 'date', 'author', 'technique', 'measurement_1', 'measurement_2', 'measurement_3', 'measurement_4', 'measurement_5']
df = pd.read_csv('data/parsed_image_meta_data.csv', usecols=data_columns)

data = {
    'known_author_images': [],
    'unknown_author_images': []
}

# parse data into dict in order to use with the html templates
for image_id in df['id']:
    image_row = df.loc[df['id'] == image_id]

    image_id = image_row['id'].item()
    title = image_row['title'].item()
    date = image_row['date'].item()
    technique = image_row['technique'].item()
    measurement_1 = image_row['measurement_1'].item()
    measurement_2 = image_row['measurement_2'].item()
    measurement_3 = image_row['measurement_3'].item()
    measurement_4 = image_row['measurement_4'].item()
    measurement_5 = image_row['measurement_5'].item()

    author = image_row['author'].item()
    # get first char of author. 'Z' from 'Zille, Heinrich' and 'U' from 'Unknown'
    author_prefix = author[0]

    image_path = f'/Users/cem/Documents/Beuth_DataScience/2021-22_WiSe/Learning_from_Images/python-shell-lfi/HZille/data/HZille_small_cropped/{image_id}.jpg'

    image_data = {
        'id': image_id,
        'title': title,
        'date': date,
        'technique': technique,
        'measurement_1': measurement_1,
        'measurement_2': measurement_2,
        'measurement_3': measurement_3,
        'measurement_4': measurement_4,
        'measurement_5': measurement_5,
        'image_path': image_path
    }

    if author_prefix == 'Z':
        data['known_author_images'].append(image_data)
    elif author_prefix == 'U':
        data['unknown_author_images'].append(image_data)
    else:
        print(f'Not able to assign image: {image_path}')

print(f'{len(data["known_author_images"])} images by known author, {len(data["unknown_author_images"])} images by unknown author')


# generate final HTML files
for output_file, template_file in outputs_templates:
    print(f'Writing to {output_file} with {template_file}...')

    subs = jinja2.Environment(
        loader=jinja2.FileSystemLoader(templates_folder)
    ).get_template(template_file).render(title=title, data=data)

    with open(f'{output_path}/{output_file}', 'w') as f:
        f.write(subs)


print('Done.')
