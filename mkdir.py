import os


path = os.path.abspath('.')
results_dir = os.path.join(path, "results")
if not os.path.exists(results_dir):
     os.mkdir(results_dir)
     print("results:",os.path.exists(results_dir))
logs_dir = os.path.join(path, "logs")
if not os.path.exists(logs_dir):
    os.mkdir(logs_dir)
    print("logs:",os.path.exists(logs_dir))
    log = os.path.join(logs_dir, "log")
    with open(log, "w"):
        print(log)
    print("log:",os.path.exists(log))

