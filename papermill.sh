#!/bin/bash
#SBATCH --job-name='papermill'
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=232GB
#SBATCH --output=papermill-%j.log
#SBATCH --time=24:00:00


if [ -z $1 ]
then
  echo No input args
  exit
fi

if [ -z $2 ]
then  
  output=`echo $1 | sed s/\.ipynb/\.milled/`
else
  output=$2
fi



date

papermill $1 $output
status=$?

cp $output $1     # cp contents of milled to original notebook
sleep 5
echo $1 `date` > $output   # milled file contains something

if [ $status -ne 0 ]
then
	rm $output   # milled file is removed, which indicates the milling failed
fi

date

exit $status
