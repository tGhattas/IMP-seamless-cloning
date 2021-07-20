Regular gaussian pyramid blending:
![Alt text](https://github.com/tGhattas/IMP-seamless-cloning/blob/master/exports/pyramid-monkey.png?raw=true)

Poisson based image cloning:
![Alt text](https://github.com/tGhattas/IMP-seamless-cloning/blob/master/exports/poisson-monkey.png?raw=true)



#################
Running Manual:
#################

python run_clone.py [options]
        Options:
        	-h	 Flag to specify a brief help message and exits.
        	-s	(Required) Specify a source image.
        	-t	(Required) Specify a target image.
        	-m	(Optional) Specify a mask image with the object in white and other part in  black, ignore this option if you plan to draw it later.
            -x	(Optional) Flag to specify a mode, either 'poisson' or 'shepard'. default is poisson.


#################
Running Examples:
#################

- Poisson based solver:
python run_clone.py -s external/blend-1.jpg -t external/main-1.jpg

- Shepard's interpolation:
python run_clone.py -s external/blend-1.jpg -t external/main-1.jpg -x

#################
Modules:
#################
clone.py - have the main logic of both Shepard's interpolation & Poisson based solver.
demo.py - includes examples.
gui.py - implements cv2 based interactive GUI for marking mask on on source image.
run_clone.py - CLI for the solution
utils.py - includes basic auxiliary functions for image processing and a gaussian pyramid based blending implementation.

######################
Author: Tamer Ghattas
######################


