# Conver_SAT

## Contributors
* Ridha Alkhabaz (ridhama2@illinois.edu/ ridha.alkhabaz@gmail.com)
* Hari Umesh (humesh2@illinois.edu)
* Vibha Nanda (vibhan2@illinois.edu)
* Khushi Sidana (ksidana2@illinois.edu)
* Priyam Mazumdar (priyamm2@illinois.edu)
* Aiman Soliman (asoliman@illinois.edu)


## Project overview
Here, we study different methods to expediate and decrease the cost of information retrieval of geospatial imagery in urban planning applications. 
## Software Details:
make sure that you have `geopandas==0.10.2`. Also, for apple silicon users, vscode has some low-level compiling issues, try to use other text-editors to run and debug your code. 

### Relevant Publications
* https://www.researchgate.net/publication/370635187_Autonomous_GIS_the_next-generation_AI-powered_GIS
* https://arxiv.org/abs/1703.02529
* https://www.microsoft.com/en-us/research/uploads/prod/2017/12/s18_cr3.pdf
* https://vldb.org/pvldb/vol14/p2341-kang.pdf

### Relevant git repository:
* TBD



## File structure:

* **Data**:
  There are a few caveats to using the data. The data is in our `centers.csv` are sorted in the same manners our folder of images. Furthermore, we need to project the data using `.set_crs(3443, allow_override=True)` because there were some data corruption issues. Then, to get the accurate measures for our data, we need to project them using `to_crs(4326)`.
  For our indexing, we will use precision=10 for local screening, and precision=5 for searching. 

* **Running**:

After downloading the data from [this link](https://uofi.box.com/s/nzdbbatyhousoretgh6lxhdjqflq07sj), you can run the following commands for the following experiments:
```
# for log filtering at 10% ratio sampling and specialized neural nets for a detect 
python main.py 
```

* **Analysis**
TBD 


## Results
TBD


