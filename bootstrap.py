import os
import uuid
import argparse

from distutils.dir_util import copy_tree

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="A M(achine) L(earning) T(emplate)",
                                     description="MLT helps to build your initial codebase \
                                        for your machine learning project")

    parser.add_argument("-n", "--name",
                        type=str,
                        default=str(uuid.uuid4()),
                        help="Define project name. Default is a UUID4")
    
    parser.add_argument("-p", "--path",
                        type=str,
                        default=os.path.join(*os.getcwd().split(os.sep)[:-1]),
                        help="Define project path. Default is the parent of the current path")
    
    args = parser.parse_args()

    project_path = os.path.join(args.path, args.name)

    if not os.path.exists():
        os.makedirs(project_path)
        
    os.chdir("..")
    copy_tree("MLT", project_path)

    os.remove("README.md")
    os.remove("bootstrap.py")
    os.rename("machinelearningtemplate", args.name)