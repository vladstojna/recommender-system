#!/bin/bash

sample_dir=$(pwd)/samples
bin=$(pwd)/matFact.out

serial='########### Serial ###########'
atomic='########### Atomic ###########'
reduction='########### Reduction ###########'
sep=$(printf '=%.0s' {1..50})
sep_small=$(printf '=%.0s' {1..20})

tries=1
threads=16

echo "Starting performance dump"
echo $sep

# Test serial

echo $serial
echo $sep

make clean && make

echo $sep

for f in $(ls $sample_dir/*.in)
do
	echo $f | sed 's|.*/||'
	echo $sep_small

	for _ in $(seq $tries)
	do
		$bin $f | tail -1 | sed 's|.*: ||'
		sleep 1
	done

	echo $sep_small

done

# Test atomic

echo $atomic
echo $sep

make CFLAGS="-Wall -Wextra -fopenmp -DREDUCTION=0" clean omp

echo $sep

for f in $(ls $sample_dir/*.in)
do
	echo $f | sed 's|.*/||'
	echo $sep_small

	for t in $(seq $threads)
	do
		echo "-- threads: $t --"
		export OMP_NUM_THREADS=$t 
		for _ in $(seq $tries)
		do
			$bin $f | tail -1 | sed 's|.*: ||'
			sleep 1
		done
	done

	echo $sep_small

done

# Test reduction

echo $reduction
echo $sep

make CFLAGS="-Wall -Wextra -fopenmp -DREDUCTION=1" clean omp

echo $sep

for f in $(ls $sample_dir/*.in)
do
	echo $f | sed 's|.*/||'
	echo $sep_small

	for t in $(seq $threads)
	do
		echo "-- threads: $t --"
		export OMP_NUM_THREADS=$t
		for _ in $(seq $tries)
		do
			$bin $f | tail -1 | sed 's|.*: ||'
			sleep 1
		done
	done

	echo $sep_small

done
