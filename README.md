# dataset

This repo is part of the final project for concluding my bachelor's degree in Computer Science. It holds the dataset I've developed and used for machine learning stuff elsewhere. 

There are 4 folders containing the dataset images:
- `original-images` with the unedited images taken directly from the phone used.
- `dataset-raw` contains the set of unedited images that generated the dataset. Some images from the above didn't cut it.
- `dataset-cropped` is the raw dataset with a crop focusing on the middle of the image. This is used for processing.
- `dataset-processed` is, well, the final processed dataset.

These roughly represent the steps taken during the processing to get the raw images ready for usage. Code related to it is found in the `image-processing.py` file. But basically, it's a Hough Transform for finding the circles and some math to crop the images when needed. Nothing fancy:

<p align="center">
  <img src="https://user-images.githubusercontent.com/29930410/147713543-8688b2d4-69c1-42ce-95b3-70dc73fa92ba.png" />
</p>  


## usage
As an initial and exploratory project, the dataset is public for future usage. There's no need for asking for permission before using the contents of this repository, but I'd appreciate a shoutout if it's of help. For good measure, and at least in spirit, everything in here is under the [DO WHATEVER THE FUCK YOU WANT TO PUBLIC LICENSE](http://www.wtfpl.net/about/).

Copyright © 2021 Gabriel Corrêa Bunselmeyer Ferreira <gabrielbunselmeyer@gmail.com>
This work is free. You can redistribute it and/or modify it under the
terms of the Do What The Fuck You Want To Public License, Version 2,
as published by Sam Hocevar. See the COPYING file for more details.
