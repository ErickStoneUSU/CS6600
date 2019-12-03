# Dependencies:
### Python library Dependencies:
- python -m pip install pickle random pandas seaborn sklearn matplotlib

### Data Dependecies
I've included the data as part of the submission so you don't have to though.
But, if you must download it, it is a small download from: Download the data from: https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass

### Running:
Simply unzip and run the final.py file.
At the bottom of final.py are three commands and one setting.
- display_data() generates charts to understand the dispersion of the data.
- train() trains the models and generates pickle files of the models 
- validate() evaluates the models using cross-validation
- use_birad = False -- This indicates that we should use the professional assessment as well.

I've commented out the training and display lines as I've simply included these generated files. 

# Report
I took mammogram data and classified it on several classifiers, 
then ensembled them together. Initially, I had to actually learn what the 
data meant. I came up with the following simplified definitions:

## Data descriptions
- BIRADS: Professional assessment from 1 to 5 of severity post mammogram, before biopsy
- Age: persons age
- Margin: How definied are the borders - Circumscribed, Microlobulated, Obscured, Ill-defined, Spiculated
- Density: Fat to breast ratio - High, iso, low, fat-containing(ordinal)
- Severity: Is it going to spread? - Benign (Good), Malginant (Bad)
- Shape: Tumor Shape - round, oval, lobular, irregular 

## Other Definitions
- Biopsy: Physical Surgery which extracts tissue
- Mammogram: A compression prossess that takes xray images of breasts to identify tumors and cancer.
- Lobular: relating to breast glands
- Ductal: relating to the milk ducts (paths from milk glands to nipples)
- Circumscribed: Solid
- Microbulated: Vein like
- Spiculated: Spikey tumor segments
- High Density: Low fat, high gland/lobule density
- Iso: Standard comparable gland to fat ratio
- Scattered: Fat cells and glands are fairly dispersed
- Fatty: High fat to low gland/lobule density
 
My initial results were terrible. I slowly worked through each of the input 
parameters for each of the classifiers choosing the best values to optimize 
them. After the initial classifiers were built, I noticed something very odd.
The ensemble did worse than most of the other classifiers and the Decision Tree
had a significant discrepancy between the train and test data. (Train|Test)
- Logistic Regression: 82.1%|81.3%
- Support Vector Machine: 82.8%|83.7%
- Decision Tree: 84.2%|79.5%
- Random Forest: 84.1%|83.7%
- Ensemble: 82.9%|82.5%

It seems that decision tree is likely overfitted. This was when I studied 
the data. I found that the BIRADS score was a "professional" assessment, so I tried 
taking it out. I took it out I found that the accuracy decreased by roughly 4%.
The fact that the expert opinion accounted for 4% tells me that more data factors 
are needed than what the data has in order to predict more accurately.  
- Logistic Regression: 80.4%|75.3%
- Support Vector Machine: 80.8%|77.7%
- Decision Tree: 79.1%|76.5%
- Random Forest: 81.1%|78.9%
- Ensemble: 81.8%|77.7%
  
