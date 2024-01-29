#!/bin/bash
#SBATCH --job-name='papermill'
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=28GB
#SBATCH --output=papermill-%j.log
#SBATCH --time=4:00:00


if [ -z $1 ]
then
  echo No input args
  exit
fi

if [ -z $2 ]
then  
  output=`echo $1 | sed s/\.ipynb/_out\.ipynb/`
else
  output=$2
fi



date

papermill $1 $output
status=$?

mv $output $1


date

exit $status