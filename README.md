## This reposetory is created to generate the results for F-Net and EF-Net 

## Note
Due to memory constrain the electricity dataset and the model weights are saved in the google drive . 

Google drive link:
https://drive.google.com/drive/folders/1JOCTAPNvjONCdmYBbYYO6Ten8_lsR1LZ?usp=sharing




#### Directory Structure:

Ensure that Model_FNet_96.py and Test_FNet_EFNet_96.py are located in the **same main directory**.

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


The `.tf` folders should be placed in the **main directory**.

---

#### Running Experiments

##### Example:

To generate results for the **ETTh1** dataset with a **forecast horizon of 96**, run:

```bash
python3 Test_FNet_EFNet_96.py
```

To use a different dataset (e.g., ETTh2, ETTm1, electricity, weather, etc.), simply change the dataset name in the corresponding test script (Test_FNet_EFNet_96.py, Test_FNet_EFNet_192.py, etc.).

