# projectTwo.cu

The goal of this code was to calculate the histogram of a large data set. A histogram is like a tally chart that counts how many times each value (or range of values) appears in the dataset.

## Platform
Project created on Windows X64

## IDE Used
Visual Studio 2022

## Build the code 
-> Extract the zip folder on a Windows machine 
-> Build using Visual Studio 2022 [use the 2022 version]
-> Now open the terminal and traverse to the Debug directory.

## Usage
Sample command to create 
Bin Size: 4
Vector Size: 1000
******************************************
```bash
cd .\x64\
cd .\Debug\

# Now in the Debug directory
./Project_two -i 2 1000
```
******************************************
## NOTE
Please make sure that the project settings are also imported, if not then follow this
******************************************
--> Open Solution Explorer 
--> Navigate to projectOne properties page 
--> Under VC++ Directories 
--> Find External Included Directories 
--> Add the Project_one\Common folder to the list(Already present in the Zipped folder)
