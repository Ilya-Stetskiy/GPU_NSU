#!/bin/bash
#SBATCH -J lab4 # Название задачи 
#SBATCH -p gpuserv # Название очереди (либо «compclass», либо «compclass_unstable»)
#SBATCH -e myjob.%j.err # название файла с потоком ошибок stderr (%j заменится на номер задачи)
#SBATCH -o myjob.%j.out # название файла с потоком вывода stdout (%j заменится на номер задачи)
#SBATCH -N 1 # Количество требуемых для задачи вычислительных узлов
#SBATCH -n 1 # Количество требуемых MPI-процессов
#SBATCH -c 1 # Количество процессорных ядер на задачу
#SBATCH -t 00:01:00 # Требуемое для задачи время работы в формате hh:mm:ss – в
# данном случае 1 минута
module load nvidia/cuda 
nvcc multipli_matrix.cu -o global -Xcompiler -fopenmp `pkg-config --cflags --libs opencv`
./global
nvcc multipli_matrix_shared.cu -o shared -Xcompiler -fopenmp `pkg-config --cflags --libs opencv`
./shared