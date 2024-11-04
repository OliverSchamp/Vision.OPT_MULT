# Vision.OPT_MULT

Project for automatically marking multiple choice questionnaires. 

Will be used in Top of the Bench. 

Will be a project for the personal website. 

## Plan

Use the hough transform (or a segmentation model) to detect the grid. 
Using the grid, crop out each question individually. 
Run a detection model on the cropped out questions to determine if the student has answered. 
If only one detection is found, compare this to the markscheme. 

There should also be a method of detecting the markscheme. But this comes after. At the start, have a vritual markscheme. 