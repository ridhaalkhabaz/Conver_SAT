
geojson files are geo-datafarmes that can be accessed using geopandas 
csv are regular pandas dataframes   

------------------------------
samples_bld.geojson

- id : unique id
- FID:  count of individual houses 
- area: sum of area square meters 

------------------------------

samples.geojson

location of the samples 
- unique id 


------------------------------

samples_buildings.csv

- id : unique id 
- FID: count of individual buildings 
- area: sum of building area within a sample [m2]
- percentage: ratio of building area to total area of the samples [ratio]
- x longitude coordinate of the sample center    
- y latitude coordinate of the sample center 


