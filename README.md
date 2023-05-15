# TATE
tensorflow implementation for TATE
## Requirements
   tensorflow-gpu==1.14.0 <br>
   torch == 1.2.0 <br>
   transformers==2.10.0 <br>
   numpy==1.15.4 <br>
   sklearn <br>
   python 3.6.13 <br>
   
## Usage
  
   
* download processed features from [Google Drive](https://drive.google.com/drive/folders/14U5t9Y8VsiF3P0ICrZVSwUYlpZOg4H7I?usp=sharing) or [Baidu Disk]( https://pan.baidu.com/s/1BJZkg8nNElFg9KSXdrLlyw?pwd=2qiu) 

* or process file by [OpenSimle2.0](https://github.com/audeering/opensmile), Bert, and [Librosa](https://github.com/librosa/librosa). Refer to utils folder.


* generate train.pkl and test.pkl
```python 
python Dataset.py
```
* train (can modify settings in Settings.py) <br>
```python 
python train.py  
``` 
* As for the IEMOCAP dataset, replacing the Datset.py, network.py and train.py in the iemocap folder.

* test <br>
```python 
python train.py --train=False
```
