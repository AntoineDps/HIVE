environment = surrogate

# test script run.py
python -m script.run
python -m plot.test.test_latex

# handle cases
python -m script.run_handle_data --case cases.json

# update 25/04/26
repertory structure has been reforged
-> matlab script are bugy
-> plot_paracoord.py is bugy
-> PI and MCR related code is bugy

# optuna command

# mlflow command