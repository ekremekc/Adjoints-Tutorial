# mesh
python3 mesh.py -nopopup 
# serial
python3 -u main.py |tee ResultsDir/solution.log
# parallel
mpirun -np 8 python3 -u main.py |tee ResultsDir/parallelSolution.log