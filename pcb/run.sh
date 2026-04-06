# mesh
python3 mesh.py -nopopup 
# serial
python3 -u main_direct.py |tee ResultsDir/solution_direct.log
python3 -u main_adjoint.py |tee ResultsDir/solution_adjoint.log
python3 -u optimization.py |tee ResultsDir/optimization.log
# parallel
mpirun -np 8 python3 -u main.py |tee ResultsDir/parallelSolution.log