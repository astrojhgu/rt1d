#! /bin/bash
# -----------------------------------------
# Run starburst99
# -----------------------------------------

# Current directory
export run=/Users/jmirocha/Scratch/starburst_test

# Name of input parameter file
export pf=s99.in

# Name and extension number of output files:
#  --> files will be: noutput.colornext, noutput.quantnext etc.
export noutput=s99.out
export outputdir=output

# Directory where code is
export STARBURST99=/Users/jmirocha/Work/mods/starburst99

# Name of executable
export exe=galaxy

# START
cd $run

# Tracks + Spectral type calibration
if [ ! -e tracks ]; then 
    ln -s $STARBURST99/tracks/ tracks
fi

# Atmosphere models
if [ ! -e lejeune ]; then 
    ln -s $STARBURST99/lejeune/ lejeune
fi

# Spectral libraries
if [ ! -e auxil ]; then 
    ln -s $STARBURST99/auxil/ auxil
fi

# Link input file
if [ -e fort.1 ]; then 
    rm fort.1
fi
if [ -e $pf ]; then 
    ln -s $pf fort.1
fi

echo "Executing starburst99..."

# Call starburst99
$STARBURST99/$exe

echo "Saving output..."

# Save output
$STARBURST99/save_output $outputdir/$noutput 1

echo "Done."