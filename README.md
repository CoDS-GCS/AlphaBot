# AlphaBot: A Weak Supervision-based Approach to Improve Chatbots for Code Repositories

This is the repository that contains (Code + Dataset + Results) of the paper "A Weak Supervision-based Approach to Improve Chatbots for Code Repositories". The README file for Rasa Platform can be found in the `Rasa-Project/README.md` file in this repo. Also, the README file for datasets can be found in the `Dataset/README.md` file in this repository. The README file for resutls can be found in the `Results/README.md` file in this repository. Finally, all the directories in this repo have their own README.md file.

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Important Note About Google Dialogflow](#important-note-about-google-dialogflow)
- [Features](#features)
- [FAQ](#faq)
- [Contributing](#contributing)
- [License](#license)

---

## Requirements

This project is written in **Python 3.7.9** and **Shell** scripts. However, to install and run the programs and scripts in this repository, you **DO NOT** need any requirements on your system except the following items. To clone this repo, install the dependencies, and run the programs, you need:
* **Required packages**:
  * **Git** (2.24.x or newer)
  * **Oracle VM VirtualBox** (6.1.x or above)
  * **Oracle VM VirtualBox Extension Pack** (6.1.x or above)
  * **HashiCorp Vagrant** (2.2.x or above)

All the above tools are **free** to download and no licenses are required for usages.

* **Optional packages**:
  * **PyCharm Professional** (2021.2.x or above)

You can directly run the program and Python scripts by logging to the virtual machine using SSH. However, if you want to develop, view, and run the program in `PyCharm Professional`, you have to install it. `PyCharm Professional` can run the Python scripts and programs using the Reproducible Development Environments (by integrating `Vagrant`).  This can be done in `PyCharm Professional` by defining a **remote** Python interpreter. This remote interpreter is installed in the Virtual Machine defined in this repository.

The `PyCharm Professional` is not free. However, by registering in `JetBrains` using your **University email address**, you can have a 1-year license for all the products of `JetBrains` including `PyCharm Professional`. Also, the license could be extended for at least one another year.

---

## Installation

After installing the required tools mentioned in the [Requirements](#requirements) section, clone this repo and run the following command in the main directory of the repo (where the `Vagrantfile` exists):

```shell
$ vagrant up
```

**Note: You need at least 30 GB of free disk space and 3 GB free RAM on your machine for a successful installation.**

This command starts installing a Virtual Machine on your system. We wrote the full shell scripts regarding the installation and customization of this virtual machine. So, the full installation will take about **2~3 hours** depending on your **internet speed**. The installed virtual machine has the following main specifications:
- Ubuntu 20.04 LTS (Focal Fossa).
- Python 3.7.9 compiled from source using asdf CLI tool.
- 3 GB RAM and 2 Cores (could be changed in the `Vagrantfile` file whenever needed).
- 1 GB Swap Memory (could be changed in the `bootstrap.sh` file).
- Installing the `Rasa X` and `Facebook Duckling` local servers. 
- Forwarded the port `5002` in the virtual machine to port `5010` in the host to use for the `Rasa X` web application.
- Forwarded the port `8000` in the virtual machine to port `8010` in the host to use for the `Facebook Duckling` server.
- Two Python Virtual Environments using `python3 venv` package. One is used for our platform (Snorkel-based), and the other is used for the environment needed for `Rasa`.
- Installing the required dependencies for the above-mentioned virtual environments using `pip3`.
- Setting a default password for `Rasa X` as follows: `123456`.
- Installing other required packages in Ubuntu using the `apt` package manager.

The following message must be shown after a successful installation of the virtual machine:
```text
****** We are done !!! ********
```

---

## Usage

**Step 1 (Optional: if you want to use `PyCharm Professional`, then follow this step.)**: After finishing the installation of the Vagrant VM, you have to define this remote interpreter in `PyCharm Professional`. To read more about this feature and how to use it, follow the instructions in this **[LINK](https://www.jetbrains.com/help/pycharm/vagrant-support.html)**.
```shell
Remote Python 3.7.9 Vagrant VM (/home/vagrant/venv-3.7.9/bin/python3)
```

**Step 2**: Login to your virtual machine using the `vagrant ssh` command in the main directory of the repository:
```shell
$ vagrant ssh
```
**Note:** To read more about the Vagrant commands and how to use them, follow this tutorial: **[LINK](https://learn.hashicorp.com/tutorials/vagrant/getting-started-index?in=vagrant/getting-started)**.

**Step 3**: To run the `Facebook Duckling`, After logging in to your virtual machine in the previous step, follow these commands:
```shell
$ cd ~/duckling/
$ stack exec duckling-example-exe
```
The `Facebook Duckling` server will be accessible using `http://0.0.0.0:8010/parse/` in the host machine. **Note: An active Facebook Duckling server is needed to train the Rasa models.**

**Step 4**: To run the `Rasa X`, login to your virtual machine using the `vagrant ssh` command in the main directory of the repository and follow these commands:
```shell
$ cd /vagrant/Rasa-Project/
$ source ~/venv-3.7.9-rasa/bin/activate
$ rasa x
```
Rasa X will ask you to agree on their license agreement. Basically, you should enter "Y" to the two questions that appear after executing the above commands. Finally, the `Rasa X` web application will be accessible using `localhost:5010/` in your browser with the password: `123456`.

**Step 5**: All the Python scripts in this repository can be executed using the Python remote interpreter defined in the Pycharm Professional. However, if you want to run the Python scripts in the virtual machine, log in to your virtual machine using the `vagrant ssh` command in the main directory of the repository and follow these commands:
```shell
$ cd /vagrant/
$ source ~/venv-3.7.9/bin/activate
$ python test-environment.py
```
The last command executes the `test-environment.py` script and if everything is installed correctly, it shows the following message:
```text
Hi, AlphaBot!
All Good!
```

**Step 6**: You can run the RQ1 for Rasa by logging in to your virtual machine using the `vagrant ssh` command in the main directory of the repository and follow these commands::
```shell
$ cd /vagrant/
$ source ~/venv-3.7.9/bin/activate
$ python Rasa-RQ1.py
```
This script asks you the parameters for (Training-Testing-Validation) splits (as presented in Table 3 of the paper). We used 0.4, 0.3 and 0.3 for the default values of these parameters. You can change these parameters at the beginning of each run.  After specifying the input parameters, the study can be conducted and the results will be saved in the `Output` directory.

**Step 7**: You can run the RQ2 for Rasa by logging in to your virtual machine using the `vagrant ssh` command in the main directory of the repository and follow these commands::
```shell
$ cd /vagrant/
$ source ~/venv-3.7.9/bin/activate
$ python Rasa-RQ2.py
```
This script asks you the to choose the baseline models (Training-Testing-Validation splits as discussed in the paper), the number of experiments (different shuffles of LFs), and the number of iteration per experiment. It is important to note that even with selecting **one** baseline model,
**one** experiment and **one** iteration, this script is a time-consuming one, due to training Rasa for very LF in the system. After specifying the input parameters, the study can be conducted and the results will be saved in the `Output` directory. The gathered values from all LFs will be saved as CSV files with `rasa-rq2` prefix in the destination directory.

**Step 8**: You can run the RQ1 for Dialogflow by logging in to your virtual machine using the `vagrant ssh` command in the main directory of the repository and follow these commands::
```shell
$ cd /vagrant/
$ source ~/venv-3.7.9/bin/activate
$ python Dialogflow-RQ1.py
```
This script asks you the parameters for (Training-Testing-Validation) splits (as presented in Table 3 of the paper). We used 0.4, 0.3 and 0.3 for the default values of these parameters. You can change these parameters at the beginning of each run.  After specifying the input parameters, the study can be conducted and the results will be saved in the `Output` directory.

**Step 9**: You can run the RQ2 for Dialogflow by logging in to your virtual machine using the `vagrant ssh` command in the main directory of the repository and follow these commands::
```shell
$ cd /vagrant/
$ source ~/venv-3.7.9/bin/activate
$ python Dialogflow-RQ2.py
```
This script asks you the to choose the baseline models (Training-Testing-Validation splits as discussed in the paper), the number of experiments (different shuffles of LFs), and the number of iteration per experiment. It is important to note that even with selecting **one** baseline model,
**one** experiment and **one** iteration, this script is a time-consuming one, due to training Dialogflow for very LF in the system. After specifying the input parameters, the study can be conducted and the results will be saved in the `Output` directory. The gathered values from all LFs will be saved as CSV files with `Dialogflow-rq2` prefix in the destination directory.

---

## Folder Structure

This directory contains all the classes that we wrote to create our framework. You do not need to run these
scripts directly. The main Python scripts uses these classes to perform the main tasks. So, keep these classes
unchanged.

* **Files and related classes:**
  * **Bash:** This directory contains all the bash scripts that are needed as a part of Python scripts. You do not need to run these
scripts directly. The main Python scripts will run them when needed. So, keep these scripts unchanged.
  * **classes:** This directory contains all the classes that we wrote to create our framework. You do not need to run these scripts directly. The main Python scripts uses these classes to perform the main tasks. So, keep these classes
unchanged.
  * **Dataset:** This directory contains the dataset we used in our study. The main dataset that we used to conduct all the experiments
can be found in the **Paper** directory.
  * **Dialogflow-Baseline-Results:** This directory contains the Dialogflow baseline results that we trained and used in our study. These results used as the baseline
for answering the **RQ2** for Dialogflow.
  * **keys:** This directory is used to save the keys that are needed to connect to Dialogflow API. The keys are ignored in the
.gitignore file. You have to change the key name in `Dialogflow-RQ1.py` and `Dialogflow-RQ2.py` files based on your key
file.
  * **Output:** All the results from the main scripts will be saved in this directory. The content of this directory is ignored in the
.gitignore file.
  * **Rasa-Baseline-Results:** This directory contains the Rasa baseline results that we trained and used in our study. These results used as the baseline
for answering the **RQ2** for Rasa.
  * **Rasa-Project:** This directory is used by our framework to install Rasa on the virtual machine. Please keep this directory and its files
unchanged. You do not need to work directly with this directory.
  * **Results:** This directory contains the results of our study.
  * **Vagrant:** We wrote some scripts to install and config the virtual machine. This directory contains these scripts.

---

## Features

- Developing using Reproducible Development Environments (by integrating `Vagrant`).
- Customizing the `ubuntu/focal64` for this platform.
- Creating two separated Python virtual environments for the platform, and the `Rasa` project.
- Pre-installing the `Rasa X` and `Facebook Duckling` servers that are needed in both Rasa, and our model.
- OOP design for developing the system.  
- Integrating Google Dialogflow into the pipeline.

---

## FAQ

- **Why do you use *Vagrant and Virtual Machine* in the development cycle of this platform?**
    - Vagrant is a tool for building and managing virtual machine environments in a single workflow. With an easy-to-use workflow and focus on automation, Vagrant lowers development environment setup time, increases production parity, and makes the “works on my machine” excuse a relic of the past. Vagrant is convenient to share virtual environment setup and configurations.
  
---

## Contributing

### Step 1

- **Option 1**
    - 🍴 Fork this repo!

- **Option 2**
    - 👯 Clone this repo to your local machine.

### Step 2

- **HACK AWAY!** 🔨🔨🔨

### Step 3

- 🔃 Create a new pull request. To read more about the standard pull requests, follow the instructions here: **[LINK](https://github.com/susam/gitpr)**.

---

## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- This project is licensed under the **[MIT license](http://opensource.org/licenses/mit-license.php)**.

---

This README was written with ❤️.
