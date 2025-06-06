# This reposetory is created to generate the results for F-Net and EF-Net 

# Note
- Due to memory constrain the electricity dataset is saved in google drive. The electricity dataset can be downloaded from the foloowing link : 

# Code
Codes 

# Generate the results
Directory Structure and Dataset Requirements:
Ensure that Model_FNet_96.py and Test_FNet_EFNet_96.py are located in the same main directory.

The dataset files should be organized in the following structure within the main project directory:

For ETT Datasets (ETTh1, ETTh2, ETTm1, ETTm2)
Create a folder named ETT-small in the main directory. Place the following CSV files inside it:

## Dataset Setup and Usage Instructions

### Directory Structure and Dataset Requirements:

Ensure that Model_FNet_96.py and Test_FNet_EFNet_96.py are located in the **same main directory**.

The dataset files must be organized as follows in the main project directory:

#### For ETT Datasets (`ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`)
Create a folder named `ETT-small` in the main directory and place the following files inside it:



```bash
├── app
│   ├── css
│   │   ├── **/*.css
│   ├── favicon.ico
│   ├── images
│   ├── index.html
│   ├── js
│   │   ├── **/*.js
│   └── partials/template
├── dist (or build)
├── node_modules
├── bower_components (if using bower)
├── test
├── Gruntfile.js/gulpfile.js
├── README.md
├── package.json
├── bower.json (if using bower)
└── .gitignore
```


<!-- TREEVIEW START -->
Main
└── dataset
    └──  ETT-small
      └── ETTh1.csv
      └── ETTh2.csv
      └── ETTh2.csv
      └── ETTh2.csv

<!-- TREEVIEW END -->
│   ├── ETTh2.csv


│   ├── ETTm1.csv


│   └── ETTm2.csv
