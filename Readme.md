# Medical Image Sequencing Using Generative Artificial Intelligence.

The project aims to bridge healthcare gaps by predicting disease progression using a generative AI system that analyzes the sequence of medical images and then generates the next image in sequence. This system is crucial for enhancing diagnosis and treatment planning. Despite challenges like limited data and technology access, the project utilizes pre-trained Vision Transformers (ViTs) with encoder, and Variational Auto Encoders with encoder and decoder layers. These layers forecast subsequent images in a patient's sequence, aiding healthcare professionals in treatment planning and accurate disease diagnosis. The project's composite model approach facilitates early disease detection, personalized treatment plans, and enhanced medical education for clinicians and radiologists. This leads to better patient management and optimized use of healthcare resources.

## Getting Started

### System Dataflow
* The below picture depicts the projects dataflow step by step.

![Alt text](./Docs/DataConditioning.png?raw=true "Title")


### System Architecture

* The following architecture is the model architecture of the project.

![Alt text](./Docs/SystemArchitecture_drawio.png?raw=true "Title")

### Dependencies

* Using a CPU can take a lot of time to train the model. So, use GPU with configuration: 40 GB and 16 cores.
* Use the pre-trained Vit as mentioned in the above system architecture.

### Help

* Replace the datsets source path in the code with your respective path.

### Installing

* Use either .ipynb or .py of Capstone_DrStrange_GenerativeMedicalImageSequencing. We used Jupyter Notebook so the packages are installed according to our environment. Make sure you install all the neccessary packages and import modules.
* We have divided our dataset into Training and Validation. Our validation data has an image count of less than 20.

### Team Members

Parth Dalal, Ajith Kumar Jalagam, Richa Saraf, Karthick Balaje Sasirekha Eswaramoorthi, Moazzam Mansoob, Bhuvana Yadavalli


### Version History

* Initial Release

