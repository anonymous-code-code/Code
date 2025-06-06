## This reposetory is created to generate the results for F-Net and EF-Net 

## Note
Due to memory constrain the electricity dataset is saved in google drive. The electricity dataset can be downloaded from the following link
https://drive.google.com/file/d/1OGU2TcuKw9nfchNZ91JUIroH-h8dQXR9/view?usp=drive_link




#### Directory Structure:

Ensure that Model_FNet_HorizonLength.py and Test_FNet_EFNet_HorizonLength.py are located in the **same main directory**.

The dataset and code files must be organized as follows in the main project directory:


```bash
├── Main directory
│   ├── dataset
│   │   ├── ETT-small
│   │   │   ├── ETTh1.csv
│   │   │   ├── ETTh2.csv
│   │   │   ├── ETTm1.csv
│   │   │   ├── ETTm2.csv
│   │   ├── electricity
│   │   │   ├── electricity.csv
│   │   ├── weather
│   │   │   ├── weather.csv
│   ├── Model_FNet_96.py
│   ├── Test_FNet_EFNet_96.py
│   ├── FNet_Weight_96.tf
│   ├── EFNet_Weight_96.tf
```

#### Generate the results
Run following command
```bash
python3 Test_FNet_EFNet_96.py
```
