#################
Running Manual:
#################

python run_clone.py [options]
        Options:
        	-h	 Flag to specify a brief help message and exits.
        	-s	(Required) Specify a source image.
        	-t	(Required) Specify a target image.
        	-m	(Optional) Specify a mask image with the object in white and other part in  black, ignore this option if you plan to draw it later.
            -x	(Optional) Flag to specify a mode, either 'possion' or 'shepard'. default is possion.


#################
Running Examples:
#################

- Poisson based solver:
python run_clone.py -s external/blend-1.jpg -t external/main-1.jpg

- Shepard's interpolation:
python run_clone.py -s external/blend-1.jpg -t external/main-1.jpg -x