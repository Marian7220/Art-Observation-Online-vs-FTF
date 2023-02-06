This is a guide on how to run the many analysis types of the project. For each analysis, we show a sample of the visualization outputted.  
  
  
  
#################################### Mean Shift Clustering #######################################  
In <MeanShift.py>:    
Given a radius, and a list of (x,y) points we will estimate the clustering of the data. (Areas of Interests - denoted AOI)  

    
  
You can tweak the parameters as you like. The explanations are provided in comments within the file.  
  
################################### Cluster Neighboring Matrices #######################################  
  
In <MeanShift.py>:  
For each subject we calculate the number of times they switched from cluster A to cluster B (the clusters as were calculated in Mean Shift).  
In the matrix, the cell (i, j) means how many times the subject has shifted his look from area "i" to area "j".  
The numbers of i, j are shown in the figure above in Mean Shift.   
 
  
Note: In order to assist with visualization, the diagonal was set to 0 (people tend to move within the same area a lot). Otherwise, you'd have a really strong green in the  
diagonal line and pretty weak green everywhere else in the matrix.  
    
################################## Cluster Histogram ########################################  
  
In <MeanShift.py>:  
This is a general Histogram - for each area it will show how many points (in percentages) fall within the same cluster.  
The color codes within the histogram are the same as the one in the Mean Shift Clustering.  


  
  
  
  
  
  