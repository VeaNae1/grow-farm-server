# Flow Plot
[Mobile App]    
   │   
   │ POST /detect (Upload Image)   
   ▼   
[FastAPI Server]    
   │   
   │ YOLOv5 model inference   
   │ Save detection results to DB (using DB_URL)   
   ▼    
[Database]    
   │   
   │ Save / Lookup Results   
   ▼    
[FastAPI Server]    
   │   
   │ JSON Response Return   
   ▼   
[Mobile App]      
   │   
   │ Display detection results (bound box, object information, etc.)   
## Working on it
* Add a new file each time you modify it
* When completed, the rest except for the final file will be moved to the tmp folder separately
## Progress and Planning
* ~The basic frame~
* ~Test frame~
* ~The frame for deployment~
* Model applied and tested -> until complete
* framework for deployment Referencing, combining
* Tested, modified, and finally completed
