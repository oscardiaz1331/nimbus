# l-Sky Imager Cloud Segmentation Dataset Almería

### Authors/Creators

- [Fabel, Yann](https://zenodo.org/search?q=metadata.creators.person_or_org.name:%22Fabel,+Yann%22)[](https://orcid.org/0000-0002-1892-5701)
- [Magiera, David](https://zenodo.org/search?q=metadata.creators.person_or_org.name:%22Magiera,+David%22)
- [Nouri, Bijan](https://zenodo.org/search?q=metadata.creators.person_or_org.name:%22Nouri,+Bijan%22)[](https://orcid.org/0000-0002-9891-1974)
- [Blum, Niklas](https://zenodo.org/search?q=metadata.creators.person_or_org.name:%22Blum,+Niklas%22)[](https://orcid.org/0000-0002-1541-7234)
- [Zarzalejo, Luis F.](https://zenodo.org/search?q=metadata.creators.person_or_org.name:%22Zarzalejo,+Luis+F.%22)[](https://orcid.org/0000-0003-4522-6815)

### Dataset DOI

- [Zenodo record 16647156](https://zenodo.org/records/16647156)
- DOI: [10.5281/zenodo.16647156](https://doi.org/10.5281/zenodo.16647156)

### Recommended citation

Fabel, Y., Magiera, D., Nouri, B., Blum, N., & Zarzalejo, L. F. (2022). Applying self-supervised learning for semantic cloud segmentation of all-sky images. *Atmos. Meas. Tech.*, 15, 797-810. DOI: [10.5194/amt-15-797-2022](https://doi.org/10.5194/amt-15-797-2022)

## Description

The l-Sky Imager Cloud Segmentation Dataset Almería contains 818 all-sky imagery samples acquired at a solar research facility in southern Spain. Each sky image is paired with a manually refined semantic segmentation mask. The dataset supports research and model development for cloud classification and cloud layer segmentation in all-sky camera images.

The dataset is organized into two main subsets:

- **Training & Validation Set (`kontas_2017`)**: 770 sky images captured in 2017 by the Cloud_Cam_Kontas imager.
- **Test Set**: 48 sky images captured in 2021 by four different imagers located at the same facility:
  - `Cloud_Cam_Kontas`
  - `Cloud_Cam_Metas`
  - `Cloud_Cam_PVot_Q71`
  - `Cloud_Cam_PVotSky`

Images were manually selected to cover a wide range of cloud conditions and solar elevation angles.

Segmentation masks distinguish cloud layers by base height and include a mask label for camera occlusions.

## Labels and classes

The dataset uses the following class labels in the grayscale segmentation masks:

- `0`: camera mask (occluded or invalid pixels)
- `1`: sky (cloudless)
- `2`: low-layer clouds
- `3`: mid-layer clouds
- `4`: high-layer clouds

This label mapping is defined in `datasets/16647156/classes.yaml`.

## Data format and repository layout

All dataset files are stored under `datasets/16647156/` in this repository.

### Training & Validation Set

- Images: `datasets/16647156/kontas_2017/images/`
- Segmentation masks: `datasets/16647156/kontas_2017/seg_masks/`
- Validation split: `datasets/16647156/kontas_2017/validation.csv`

### Test Set

- Images: `datasets/16647156/test_set/images/`
- Segmentation masks: `datasets/16647156/test_set/seg_masks/`

### Metadata

- `datasets/16647156/meta_data.yaml`: camera location, altitude, and timezone information for each imager.
- `datasets/16647156/classes.yaml`: label definitions used by the segmentation masks.

### File conventions

- Sky images are JPEG files named using acquisition timestamp conventions, for example:
  - `kontas_2017/images/asi_001_170328164030.jpg`
- Segmentation masks are grayscale PNG files using the same filename as the corresponding sky image, for example:
  - `kontas_2017/seg_masks/asi_001_170328164030.png`

## Notes for repository users

- Use the provided `validation.csv` split for reproducible training/validation experiments.
- Respect the original dataset citation and license terms from the Zenodo record when publishing results.
- The dataset is intended for semantic cloud segmentation research using all-sky imagery.

---

# 2-SWIMSEG: Singapore Whole sky IMaging SEGmentation Database

### Authors/Creators

- Dev, S.
- Lee, Y. H.
- Winkler, S. (Contacto: Stefan.Winkler@adsc.com.sg)

### Recommended citation

Dev, S., Lee, Y. H., & Winkler, S. (2017). Color-based segmentation of sky/cloud images from ground-based cameras. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, vol. 10, no. 1, pp. 231-242. 

## Description

The SWIMSEG dataset contains 1013 daytime images of sky/cloud patches accompanied by their corresponding binary ground truth maps, annotated in consultation with experts from the Singapore Meteorological Services.

All images have a dimension of 600x600 pixels, covering a viewing angle of approximately 62°. They were captured over a 22-month period (October 2013 to July 2015) using the WAHRSIS ground-based whole-sky imager at Nanyang Technological University, Singapore (1.34°N, 103.68°E). The subset features a broad diversity of visual characteristics, including different times of day, acquisition dates, cloud types, and cloud coverage percentages. Images are free from lens distortion due to an applied ray-tracing correction method.

## Labels and classes

The dataset utilizes binary ground truth masks for sky/cloud segmentation:

- **Black (0)**: Daytime sky / cloudless regions
- **White (255 / Object)**: Clouds

## Data format and repository layout

All dataset files are stored under `datasets/swimseg/` in this repository.

### Dataset Files

- Images: `datasets/swimseg/images/`
- Segmentation masks (Ground Truth): `datasets/swimseg/GTmaps/`

### Metadata

- `datasets/swimseg/metadata.csv`: A comma-separated values file containing image metadata (including camera settings and virtual camera angles).
- `datasets/swimseg/license.html`: Licensing terms and information.

### File conventions

- Sky images are PNG files named using an incremental numeric index, for example:
  - `images/0001.png`
- Segmentation masks are binary PNG files using a hyphen suffix before the extension, for example:
  - `GTmaps/0001-GT.png`

## Notes for repository users

- **License**: This dataset is released under a Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). Commercial use of these assets is prohibited.
- Please ensure you cite the recommended paper in any publication or research project utilizing the SWIMSEG database.

---

# 3-SWINSEG: Singapore Whole sky Nighttime Imaging SEGmentation Database

### Authors/Creators

- Dev, S.
- Savoy, F. M.
- Lee, Y. H.
- Winkler, S. (Contacto: Stefan.Winkler@adsc.com.sg)

### Recommended citation

Dev, S., Savoy, F. M., Lee, Y. H., & Winkler, S. (2017). Nighttime sky/cloud image segmentation. *Proc. IEEE International Conference on Image Processing (ICIP)*.

## Description

The SWINSEG dataset contains 115 nighttime images of sky/cloud patches along with their corresponding binary ground truth maps, generated in consultation with experts from the Singapore Meteorological Services. 

All images have a dimension of 500x500 pixels and were captured over a 12-month period (January 2016 to December 2016) using WAHRSIS, a calibrated ground-based whole-sky imager located at Nanyang Technological University, Singapore (1.34°N, 103.68°E). The images were selected to ensure diversity in terms of capture date, time, and cloud percentage. They have been geometrically undistorted using a ray-tracing approach to project the sky hemisphere onto a flat plane.

## Labels and classes

The dataset provides binary segmentation maps where the classes distinguish between sky and cloud pixels:

- **Black (0)**: Nighttime sky / cloudless regions
- **White (255 / Object)**: Clouds

## Data format and repository layout

All dataset files are stored under `datasets/swinseg/` in this repository.

### Dataset Files

- Images: `datasets/swinseg/images/`
- Segmentation masks (Ground Truth): `datasets/swinseg/GTmaps/`

### Metadata

- `datasets/swinseg/metadata.txt`: A tab-delimited file containing detailed metadata for each image.
- `datasets/swinseg/license.html`: Licensing terms and information.

### File conventions

- Sky images are JPEG files named using an incremental numeric index, for example:
  - `images/0001.jpg`
- Segmentation masks are binary JPEG files using an underscore suffix before the extension, for example:
  - `GTmaps/0001_GT.jpg`

## Notes for repository users

- **License**: The dataset is released under a Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). You are free to share and adapt the material for non-commercial purposes, provided appropriate credit is given.
- Respect the original dataset citation and license terms when publishing results or using this data.
