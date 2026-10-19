SYDE 671 Assignment 2 — Yanbo Cheng, 21287485

Run: python A2.py
Dependencies: numpy, scipy, matplotlib, opencv-python, imageio, Pillow.
The script opens figures interactively. For batch output, set MPLBACKEND=Agg.
Folder convention matches A1: project_dir / data and project_dir / filtered_output.
Input files are in data/. Generated figures go to filtered_output/; runtimes print in the terminal.
All sections are inline in the main block.
The selfie is read in grayscale at its original 1350 x 1800 resolution without resizing.
Timing values are one measured run on this machine, not repeated benchmark averages.

Sources and modifications
Selfie: supplied by Yanbo Cheng.
Architecture: Don Cochran / Eastman Kodak, Kodak Image 01 (JN1206), stone building in the Bahamas.
https://r0k.us/graphics/kodak/kodim01.html
Native 768 x 512 input; Gaussian blur and unsharp masking applied for this report.
Derek/Nutmeg, Taj Mahal, and burt_apple/burt_orange: user-supplied assignment examples.
Cameraman: classical camera.png from scikit-image v0.17.2 (not the newer replacement):
https://github.com/scikit-image/scikit-image/blob/v0.17.2/skimage/data/camera.png
https://github.com/scikit-image/scikit-image/issues/3927
Obama: Pete Souza / White House official portrait, 2009, public domain.
https://commons.wikimedia.org/wiki/File:Official_portrait_of_Barack_Obama.jpg
Jobs: Matthew Yohe; crop by MetalGearLiquid, CC BY-SA 3.0.
https://commons.wikimedia.org/wiki/File:Steve_Jobs_Headshot_2010-CROP2.jpg
https://creativecommons.org/licenses/by-sa/3.0/
Jobs-derived images are offered under CC BY-SA 3.0.
Biden: Adam Schultz / White House official portrait, 2021, public domain.
https://commons.wikimedia.org/wiki/File:Joe_Biden_presidential_portrait.jpg
Trump: Shealah Craighead / White House official portrait, 2017, public domain.
https://commons.wikimedia.org/wiki/File:Donald_Trump_official_portrait.jpg
Portraits were resized before this work, manually landmark-aligned, cropped, filtered,
converted to grayscale for the baseline hybrids, and combined for the displayed results.
These are additional selected image pairs, not photographs taken by Yanbo.
Cutoffs are visually selected settings, not a mathematically optimized result.
Circle centres and radii are manually selected after alignment and cropping.
